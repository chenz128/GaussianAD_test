import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmengine.registry import MODELS

from .planner import VADHead


@MODELS.register_module()
class VADHeadDualTimeResidual(VADHead):
    """Fuse an exact costime global path with per-frame residual features."""

    def __init__(self,
                 *args,
                 ego_fut_gaussian_decoder=None,
                 fut_self_decoder=None,
                 time_gaussian_decoder=None,
                 **kwargs):
        self.ego_fut_gaussian_decoder = ego_fut_gaussian_decoder
        self.fut_self_decoder = fut_self_decoder
        self.time_gaussian_decoder = time_gaussian_decoder
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
        if self.fut_self_decoder is not None:
            self.fut_self_decoder = build_transformer_layer_sequence(
                self.fut_self_decoder)
        if self.time_gaussian_decoder is not None:
            self.time_gaussian_decoder = build_transformer_layer_sequence(
                self.time_gaussian_decoder)

        self.fut_time_pos = nn.Embedding(self.fut_ts, self.embed_dims)
        self.fut_pos = nn.Embedding(self.fut_ts, self.embed_dims)
        self.relative_time_bias = nn.Embedding(2 * self.fut_ts - 1, 1)

        global_dim = self.embed_dims * 4
        self.ego_fut_decoder = nn.Sequential(
            nn.Linear(global_dim, global_dim),
            nn.ReLU(),
            nn.Linear(global_dim, global_dim),
            nn.ReLU(),
            nn.Linear(global_dim, self.ego_fut_mode * self.fut_ts * 2))

        self.ego_to_fut = nn.Linear(
            self.embed_dims * 3, self.embed_dims, bias=True)
        self.global_context_proj = nn.Linear(
            global_dim, self.embed_dims, bias=True)

        fusion_dim = self.embed_dims * 3
        self.residual_mlp = nn.Sequential(
            nn.Linear(fusion_dim, self.embed_dims, bias=True),
            nn.ReLU(),
            nn.Linear(
                self.embed_dims, self.ego_fut_mode * 2, bias=True))
        self.gate_mlp = nn.Sequential(
            nn.Linear(fusion_dim, self.embed_dims, bias=True),
            nn.ReLU(),
            nn.Linear(self.embed_dims, 1, bias=True))

    def init_weights(self):
        super().init_weights()
        for decoder in (
                self.ego_fut_gaussian_decoder,
                self.fut_self_decoder,
                self.time_gaussian_decoder):
            if decoder is not None:
                for parameter in decoder.parameters():
                    if parameter.dim() > 1:
                        nn.init.xavier_uniform_(parameter)

        nn.init.xavier_uniform_(self.fut_time_pos.weight)
        nn.init.xavier_uniform_(self.fut_pos.weight)
        nn.init.zeros_(self.relative_time_bias.weight)

        # The model starts as the exact global costime path. The residual and
        # scene-dependent gate are learned only through the final PlanLoss.
        nn.init.zeros_(self.residual_mlp[-1].weight)
        nn.init.zeros_(self.residual_mlp[-1].bias)
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.constant_(self.gate_mlp[-1].bias, -2.0)

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

        # Exact costime semantics: learned time position is present in both K
        # and V before flattening all frames into one global attention set.
        future_global_kv = future_content + self.fut_time_pos.weight[
            None, :, None, :]
        return future_content, future_global_kv

    def _relative_time_attn_bias(self, reference):
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

        future_content, future_global_kv = \
            self._build_future_gaussian_features(results)
        num_gaussians = future_content.shape[2]

        global_future_query = self.ego_fut_gaussian_decoder(
            query=ego_gaussian_query,
            key=future_global_kv.reshape(
                batch, self.fut_ts * num_gaussians,
                self.embed_dims).permute(1, 0, 2),
            value=future_global_kv.reshape(
                batch, self.fut_ts * num_gaussians,
                self.embed_dims).permute(1, 0, 2))

        global_features = torch.cat([
            current_features,
            global_future_query.permute(1, 0, 2),
        ], dim=-1)
        global_prediction = self.ego_fut_decoder(global_features).reshape(
            batch, self.ego_fut_mode, self.fut_ts, 2)

        # Futattn semantics: one query per future frame, temporal interaction,
        # then same-frame Gaussian cross-attention with content-only K/V.
        time_query = self.ego_to_fut(current_features).expand(
            -1, self.fut_ts, -1)
        time_query = time_query + self.fut_pos.weight[None, :, :]
        time_query = time_query.permute(1, 0, 2)
        if self.fut_self_decoder is not None:
            time_query = self.fut_self_decoder(
                query=time_query,
                key=time_query,
                value=time_query,
                attn_masks=[self._relative_time_attn_bias(time_query)])

        folded_query = time_query.permute(1, 0, 2).reshape(
            batch * self.fut_ts, self.embed_dims).unsqueeze(0)
        folded_kv = future_content.reshape(
            batch * self.fut_ts, num_gaussians,
            self.embed_dims).permute(1, 0, 2)
        if self.time_gaussian_decoder is not None:
            folded_query = self.time_gaussian_decoder(
                query=folded_query, key=folded_kv, value=folded_kv)

        time_features = folded_query.squeeze(0).reshape(
            batch, self.fut_ts, self.embed_dims)
        future_summary = future_content.mean(dim=2)
        global_context = self.global_context_proj(global_features).expand(
            -1, self.fut_ts, -1)
        fusion_features = torch.cat(
            [global_context, time_features, future_summary], dim=-1)

        residual = self.residual_mlp(fusion_features).reshape(
            batch, self.fut_ts, self.ego_fut_mode, 2).permute(0, 2, 1, 3)
        gate = self.gate_mlp(fusion_features).sigmoid()
        main_prediction = global_prediction + gate[:, None, :, :] * residual

        return {'ego_fut_preds': main_prediction}
