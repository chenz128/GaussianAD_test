import math

import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmengine.registry import MODELS

from .planner import VADHead


@MODELS.register_module()
class VADHeadTimeAlignedGaussian(VADHead):
    """Predict once from time-aligned ego and future-Gaussian tokens.

    Single-branch time-aligned planner warm-started from FutAttn.  The cross
    attention folds the future frames into the batch dimension exactly like
    FutAttn (each ego frame attends ONLY its own frame's gaussians), so there
    is no cross-frame dilution and no attention-bias hack is needed.  At
    initialization (both gates == 0) the module is mathematically identical to
    FutAttn:
      * query = scene + fut_pos (continuous-time injection gated to zero),
      * future key/value = future_content (identical to FutAttn),
      * per-frame folded cross-attention == FutAttn's per-frame attention.
    Training then opens a single gate to inject a continuous (Fourier) time
    code into the ego query.  Note: in the per-frame folded attention the time
    signal is constant across a frame's gaussians, so injecting it into the
    key/value is a no-op (softmax is shift-invariant); the only effective place
    for the time enhancement is the query.
    """

    def __init__(self, *args, fut_self_decoder=None,
                 ego_fut_gaussian_decoder=None,
                 time_interval=0.5, num_fourier_bands=4,
                 **kwargs):
        self.fut_self_decoder = fut_self_decoder
        self.ego_fut_gaussian_decoder = ego_fut_gaussian_decoder
        self.time_interval = time_interval
        self.num_fourier_bands = num_fourier_bands
        super().__init__(*args, **kwargs)

    def _init_layers(self):
        super()._init_layers()
        # Single-branch design: the base VADHead ego_fut_decoder is unused.
        del self.ego_fut_decoder

        self.fut_gaussian_fus_mlp = nn.Sequential(
            nn.Linear(28, self.embed_dims, bias=True),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims, bias=True))
        self.ego_to_fut = nn.Linear(
            self.embed_dims * 3, self.embed_dims, bias=True)

        self.fut_pos = nn.Embedding(self.fut_ts, self.embed_dims)
        # Learnable temporal self-attention bias (zero-init -> no effect at
        # init, so exact FutAttn equivalence is preserved).
        self.relative_time_bias = nn.Embedding(2 * self.fut_ts - 1, 1)

        minimum_period = self.time_interval * 2.0
        maximum_period = self.time_interval * self.fut_ts * 2.0
        periods = torch.logspace(
            math.log10(minimum_period), math.log10(maximum_period),
            steps=self.num_fourier_bands)
        self.register_buffer(
            'continuous_time_frequencies', periods.reciprocal())
        self.continuous_time_mlp = nn.Sequential(
            nn.Linear(1 + 2 * self.num_fourier_bands, self.embed_dims),
            nn.SiLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        self.time_encoding_norm = nn.LayerNorm(self.embed_dims)
        # Single gate (init 0) so it receives a first-order gradient and can
        # actually open.  query_cont_gate scales the continuous-time code that
        # is injected into the ego query (the only effective place for it under
        # per-frame folded attention).
        self.query_cont_gate = nn.Parameter(torch.zeros(()))

        if self.fut_self_decoder is not None:
            self.fut_self_decoder = build_transformer_layer_sequence(
                self.fut_self_decoder)
        if self.ego_fut_gaussian_decoder is not None:
            self.ego_fut_gaussian_decoder = build_transformer_layer_sequence(
                self.ego_fut_gaussian_decoder)

        self.fut_out_mlp = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.ego_fut_mode * 2),
        )

    def init_weights(self):
        super().init_weights()
        for decoder in (
                self.fut_self_decoder, self.ego_fut_gaussian_decoder):
            if decoder is not None:
                for parameter in decoder.parameters():
                    if parameter.dim() > 1:
                        nn.init.xavier_uniform_(parameter)

        nn.init.xavier_uniform_(self.fut_pos.weight)
        # Self-attention temporal bias starts neutral (learnable).
        nn.init.zeros_(self.relative_time_bias.weight)

    def _build_continuous_time_code(self, reference):
        """Continuous (Fourier) per-frame time code, LayerNorm-normalized.

        Returns a pure additive time signal of shape (fut_ts, D); it does NOT
        contain fut_pos, so it can be gated independently.
        """
        times = torch.arange(
            1, self.fut_ts + 1, device=reference.device,
            dtype=reference.dtype) * self.time_interval
        frequencies = self.continuous_time_frequencies.to(
            device=reference.device, dtype=reference.dtype)
        phases = 2.0 * math.pi * times[:, None] * frequencies[None, :]
        normalized_time = (
            times / (self.time_interval * self.fut_ts))[:, None]
        continuous_features = torch.cat(
            [normalized_time, phases.sin(), phases.cos()], dim=-1)
        code = self.continuous_time_mlp(continuous_features)
        return self.time_encoding_norm(code)

    def _build_future_gaussians(self, results):
        gaussian_output = results['gaussian_output']
        batch, num_gaussians = gaussian_output.shape[:2]
        offset = results.get('offset')
        if offset is None:
            future_offset = gaussian_output.new_zeros(
                (batch, self.fut_ts, num_gaussians, 2))
        else:
            future_offset = offset.reshape(
                batch, num_gaussians, self.fut_ts, 2).permute(0, 2, 1, 3)
            future_offset = torch.nan_to_num(
                future_offset, nan=0.0, posinf=0.0, neginf=0.0)

        future_xy = gaussian_output[:, None, :, :2] + future_offset
        future_rest = gaussian_output[:, None, :, 2:].expand(
            -1, self.fut_ts, -1, -1)
        future_features = torch.cat([future_xy, future_rest], dim=-1)
        future_content = self.fut_gaussian_fus_mlp(future_features)
        # Key == value == future_content (exactly FutAttn).  A per-frame time
        # signal on the key would be shift-invariant under the folded softmax
        # and therefore a no-op, so no time is injected here.
        return future_content

    def _self_attention_bias(self, reference):
        time_index = torch.arange(self.fut_ts, device=reference.device)
        relative_index = (
            time_index[:, None] - time_index[None, :] + self.fut_ts - 1)
        return self.relative_time_bias(relative_index).squeeze(-1).to(
            dtype=reference.dtype)

    def forward(self, results):
        agent_query, agent_mask = self.prepare_agent_query(results)
        map_query = self.prepare_map_query(results)
        gaussian_query = self.gaussian_fus_mlp(results['gaussian_output'])

        batch = agent_query.shape[0]
        ego_query = self.ego_query.weight.unsqueeze(0).expand(batch, -1, -1)
        ego_agent_query = self.ego_agent_decoder(
            query=ego_query.permute(1, 0, 2),
            key=agent_query.permute(1, 0, 2),
            value=agent_query.permute(1, 0, 2),
            key_padding_mask=agent_mask)
        ego_map_query = self.ego_map_decoder(
            query=ego_agent_query,
            key=map_query.permute(1, 0, 2),
            value=map_query.permute(1, 0, 2))
        ego_gaussian_query = self.ego_gaussian_decoder(
            query=ego_map_query,
            key=gaussian_query.permute(1, 0, 2),
            value=gaussian_query.permute(1, 0, 2))

        current_features = torch.cat([
            ego_agent_query.permute(1, 0, 2),
            ego_map_query.permute(1, 0, 2),
            ego_gaussian_query.permute(1, 0, 2),
        ], dim=-1)
        scene_token = self.ego_to_fut(current_features)
        time_code = self._build_continuous_time_code(scene_token)
        future_content = self._build_future_gaussians(results)
        num_gaussians = future_content.shape[2]

        # ---- Time-aligned path ----
        # query = scene + fut_pos (always) + gated continuous code.
        # gate=0 -> exactly the FutAttn query.
        time_query = scene_token.expand(-1, self.fut_ts, -1)
        time_query = (
            time_query
            + self.fut_pos.weight[None, :, :]
            + self.query_cont_gate.tanh() * time_code[None]).permute(1, 0, 2)
        if self.fut_self_decoder is not None:
            time_query = self.fut_self_decoder(
                query=time_query, key=time_query, value=time_query,
                attn_masks=[self._self_attention_bias(time_query)])

        # ---- Per-frame folded cross-attention (== FutAttn structure) ----
        # Fold the future frames into the batch dimension so every ego frame
        # attends ONLY its own frame's gaussians: query (1, B*T, D),
        # key/value (G, B*T, D).  No cross-frame dilution, memory O(B*T*G).
        folded_query = time_query.permute(1, 0, 2).reshape(
            batch * self.fut_ts, self.embed_dims).unsqueeze(0)
        folded_kv = future_content.reshape(
            batch * self.fut_ts, num_gaussians,
            self.embed_dims).permute(1, 0, 2)
        if self.ego_fut_gaussian_decoder is not None:
            folded_query = self.ego_fut_gaussian_decoder(
                query=folded_query, key=folded_kv, value=folded_kv)

        aligned_features = folded_query.squeeze(0).reshape(
            batch, self.fut_ts, self.embed_dims)
        prediction = self.fut_out_mlp(aligned_features).reshape(
            batch, self.fut_ts, self.ego_fut_mode, 2).permute(0, 2, 1, 3)
        return {
            'ego_fut_preds': prediction,
            'ego_fut_aligned_preds': prediction,
        }
