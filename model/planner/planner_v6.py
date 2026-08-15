import math

import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmengine.registry import MODELS

from .planner import VADHead


@MODELS.register_module()
class VADHeadTimeQueryResidual(VADHead):
    """Fuse a global future-Gaussian plan with supervised time queries."""

    def __init__(self,
                 *args,
                 ego_fut_gaussian_decoder=None,
                 time_self_decoder=None,
                 time_gaussian_decoder=None,
                 time_interval=0.5,
                 num_fourier_bands=8,
                 **kwargs):
        self.ego_fut_gaussian_decoder = ego_fut_gaussian_decoder
        self.time_self_decoder = time_self_decoder
        self.time_gaussian_decoder = time_gaussian_decoder
        self.time_interval = time_interval
        self.num_fourier_bands = num_fourier_bands
        super().__init__(*args, **kwargs)

    def _init_layers(self):
        super()._init_layers()

        self.fut_gaussian_fus_mlp = nn.Sequential(
            nn.Linear(28, self.embed_dims, bias=True),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims, bias=True))

        if self.ego_fut_gaussian_decoder is not None:
            self.ego_fut_gaussian_decoder = build_transformer_layer_sequence(
                self.ego_fut_gaussian_decoder)
        if self.time_self_decoder is not None:
            self.time_self_decoder = build_transformer_layer_sequence(
                self.time_self_decoder)
        if self.time_gaussian_decoder is not None:
            self.time_gaussian_decoder = build_transformer_layer_sequence(
                self.time_gaussian_decoder)

        self.learned_time_pos = nn.Embedding(self.fut_ts, self.embed_dims)
        periods = torch.logspace(
            math.log10(self.time_interval),
            math.log10(self.time_interval * self.fut_ts * 2),
            steps=self.num_fourier_bands)
        self.register_buffer('time_frequencies', periods.reciprocal())
        self.time_fourier_proj = nn.Linear(
            1 + 2 * self.num_fourier_bands, self.embed_dims)
        self.time_encoding_norm = nn.LayerNorm(self.embed_dims)

        global_dim = self.embed_dims * 4
        self.ego_fut_decoder = nn.Sequential(
            nn.Linear(global_dim, global_dim),
            nn.ReLU(),
            nn.Linear(global_dim, global_dim),
            nn.ReLU(),
            nn.Linear(global_dim, self.ego_fut_mode * self.fut_ts * 2))

        self.time_context_proj = nn.Linear(global_dim, self.embed_dims)
        self.time_aux_decoder = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.ego_fut_mode * 2))

        # tanh(0) keeps the initial main prediction exactly on the global path.
        self.time_fusion_gate = nn.Parameter(torch.zeros(self.fut_ts))

    def init_weights(self):
        super().init_weights()
        for decoder in (
                self.ego_fut_gaussian_decoder,
                self.time_self_decoder,
                self.time_gaussian_decoder):
            if decoder is not None:
                for parameter in decoder.parameters():
                    if parameter.dim() > 1:
                        nn.init.xavier_uniform_(parameter)
        nn.init.xavier_uniform_(self.learned_time_pos.weight)

    def _build_time_encoding(self, reference):
        times = torch.arange(
            1, self.fut_ts + 1, device=reference.device,
            dtype=reference.dtype) * self.time_interval
        phases = 2 * math.pi * times[:, None] * self.time_frequencies.to(
            device=reference.device, dtype=reference.dtype)[None, :]
        normalized_time = (times / (self.time_interval * self.fut_ts))[:, None]
        fourier_features = torch.cat(
            [normalized_time, phases.sin(), phases.cos()], dim=-1)
        continuous_encoding = self.time_fourier_proj(fourier_features)
        time_indices = torch.arange(self.fut_ts, device=reference.device)
        learned_encoding = self.learned_time_pos(time_indices)
        return self.time_encoding_norm(
            continuous_encoding + learned_encoding).to(reference.dtype)

    def _build_future_gaussian_features(self, results):
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
        time_encoding = self._build_time_encoding(future_content)
        future_key = future_content + time_encoding[None, :, None, :]
        return future_content, future_key, time_encoding

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

        future_content, future_key, time_encoding = \
            self._build_future_gaussian_features(results)
        num_gaussians = future_content.shape[2]
        global_future_query = self.ego_fut_gaussian_decoder(
            query=ego_gaussian_query,
            key=future_key.reshape(
                batch, self.fut_ts * num_gaussians,
                self.embed_dims).permute(1, 0, 2),
            value=future_content.reshape(
                batch, self.fut_ts * num_gaussians,
                self.embed_dims).permute(1, 0, 2))

        global_features = torch.cat([
            ego_agent_query.permute(1, 0, 2),
            ego_map_query.permute(1, 0, 2),
            ego_gaussian_query.permute(1, 0, 2),
            global_future_query.permute(1, 0, 2),
        ], dim=-1)
        global_prediction = self.ego_fut_decoder(global_features).reshape(
            batch, self.ego_fut_mode, self.fut_ts, 2)

        time_query = self.time_context_proj(global_features).expand(
            -1, self.fut_ts, -1)
        time_query = time_query + time_encoding[None, :, :]
        time_query = time_query.permute(1, 0, 2)
        if self.time_self_decoder is not None:
            time_query = self.time_self_decoder(
                query=time_query, key=time_query, value=time_query)

        folded_query = time_query.permute(1, 0, 2).reshape(
            batch * self.fut_ts, self.embed_dims).unsqueeze(0)
        folded_key = future_key.reshape(
            batch * self.fut_ts, num_gaussians,
            self.embed_dims).permute(1, 0, 2)
        folded_value = future_content.reshape(
            batch * self.fut_ts, num_gaussians,
            self.embed_dims).permute(1, 0, 2)
        if self.time_gaussian_decoder is not None:
            folded_query = self.time_gaussian_decoder(
                query=folded_query, key=folded_key, value=folded_value)

        time_features = folded_query.squeeze(0).reshape(
            batch, self.fut_ts, self.embed_dims)
        auxiliary_prediction = self.time_aux_decoder(time_features).reshape(
            batch, self.fut_ts, self.ego_fut_mode, 2).permute(0, 2, 1, 3)

        fusion_gate = self.time_fusion_gate.tanh().view(1, 1, self.fut_ts, 1)
        main_prediction = global_prediction + fusion_gate * (
            auxiliary_prediction - global_prediction)

        return {
            'ego_fut_preds': main_prediction,
            'ego_fut_aux_preds': auxiliary_prediction,
        }
