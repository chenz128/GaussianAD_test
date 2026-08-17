import math

import torch
import torch.nn as nn
from mmengine.registry import MODELS

from .planner_v7 import VADHeadDualTimeResidual


@MODELS.register_module()
class VADHeadHybridTimePositionResidual(VADHeadDualTimeResidual):
    """Exact v7 dual-time anchor with supervised bounded position residuals."""

    def __init__(self, *args, time_interval=0.5, num_fourier_bands=8,
                 position_residual_scale=(
                     0.35, 0.65, 0.95, 1.25, 1.60, 2.00),
                 initial_gate=0.2, anchor_gradient_scale=0.25,
                 hybrid_feature_gradient_scale=0.25, **kwargs):
        self.time_interval = time_interval
        self.num_fourier_bands = num_fourier_bands
        self.position_residual_scale_values = position_residual_scale
        self.initial_gate = initial_gate
        self.anchor_gradient_scale = anchor_gradient_scale
        self.hybrid_feature_gradient_scale = hybrid_feature_gradient_scale
        super().__init__(*args, **kwargs)

    def _init_layers(self):
        super()._init_layers()
        if len(self.position_residual_scale_values) != self.fut_ts:
            raise ValueError(
                'position_residual_scale must contain one value per step')
        if not 0.0 < self.initial_gate < 1.0:
            raise ValueError('initial_gate must be in (0, 1)')
        if not 0.0 <= self.anchor_gradient_scale <= 1.0:
            raise ValueError('anchor_gradient_scale must be in [0, 1]')
        if not 0.0 <= self.hybrid_feature_gradient_scale <= 1.0:
            raise ValueError(
                'hybrid_feature_gradient_scale must be in [0, 1]')

        periods = torch.logspace(
            math.log10(self.time_interval),
            math.log10(self.time_interval * self.fut_ts * 2),
            steps=self.num_fourier_bands)
        self.register_buffer('hybrid_time_frequencies', periods.reciprocal())
        self.time_fourier_proj = nn.Linear(
            1 + 2 * self.num_fourier_bands, self.embed_dims)
        self.hybrid_time_norm = nn.LayerNorm(self.embed_dims)

        fusion_dim = self.embed_dims * 3
        self.residual_context_proj = nn.Sequential(
            nn.Linear(fusion_dim, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        self.proposal_position_proj = nn.Sequential(
            nn.Linear(2, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        self.mode_embedding = nn.Embedding(
            self.ego_fut_mode, self.embed_dims)
        self.position_residual_mlp = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, 2),
        )
        self.position_gate_mlp = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, 1),
        )

        residual_scale = torch.as_tensor(
            self.position_residual_scale_values, dtype=torch.float32)
        self.register_buffer(
            'position_residual_scale',
            residual_scale.view(1, 1, self.fut_ts, 1))

    def init_weights(self):
        super().init_weights()
        nn.init.xavier_uniform_(self.time_fourier_proj.weight)
        nn.init.zeros_(self.time_fourier_proj.bias)
        nn.init.normal_(self.mode_embedding.weight, std=0.02)
        nn.init.zeros_(self.position_residual_mlp[-1].weight)
        nn.init.zeros_(self.position_residual_mlp[-1].bias)
        nn.init.zeros_(self.position_gate_mlp[-1].weight)
        gate_bias = math.log(self.initial_gate / (1.0 - self.initial_gate))
        nn.init.constant_(self.position_gate_mlp[-1].bias, gate_bias)

    def _build_hybrid_time_encoding(self, reference):
        times = torch.arange(
            1, self.fut_ts + 1, device=reference.device,
            dtype=reference.dtype) * self.time_interval
        frequencies = self.hybrid_time_frequencies.to(
            device=reference.device, dtype=reference.dtype)
        phases = 2.0 * math.pi * times[:, None] * frequencies[None, :]
        normalized_time = (
            times / (self.time_interval * self.fut_ts))[:, None]
        fourier_features = torch.cat(
            [normalized_time, phases.sin(), phases.cos()], dim=-1)
        continuous_encoding = self.time_fourier_proj(fourier_features)
        return self.hybrid_time_norm(
            self.fut_pos.weight + continuous_encoding).to(reference.dtype)

    
    @staticmethod
    def _scale_gradient(value, scale):
        return value.detach() + scale * (value - value.detach())

    def _forward_time_features(
            self, current_features, future_content, time_encoding):
        batch, _, num_gaussians, _ = future_content.shape
        time_query = self.ego_to_fut(current_features).expand(
            -1, self.fut_ts, -1)
        time_query = (time_query + time_encoding[None]).permute(1, 0, 2)
        if self.fut_self_decoder is not None:
            time_query = self.fut_self_decoder(
                query=time_query, key=time_query, value=time_query,
                attn_masks=[self._relative_time_attn_bias(time_query)])
        folded_query = time_query.permute(1, 0, 2).reshape(
            batch * self.fut_ts, self.embed_dims).unsqueeze(0)
        folded_kv = future_content.reshape(
            batch * self.fut_ts, num_gaussians,
            self.embed_dims).permute(1, 0, 2)
        if self.time_gaussian_decoder is not None:
            folded_query = self.time_gaussian_decoder(
                query=folded_query, key=folded_kv, value=folded_kv)
        return folded_query.squeeze(0).reshape(
            batch, self.fut_ts, self.embed_dims)

    @staticmethod
    def _positions_to_displacements(positions):
        origin = positions.new_zeros((*positions.shape[:2], 1, 2))
        return torch.diff(torch.cat([origin, positions], dim=2), dim=2)

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
        anchor_time_features = self._forward_time_features(
            current_features, future_content, self.fut_pos.weight)
        hybrid_time_encoding = self._build_hybrid_time_encoding(future_content)
        hybrid_time_features = self._forward_time_features(
            current_features, future_content, hybrid_time_encoding)
        future_summary = future_content.mean(dim=2)
        global_context = self.global_context_proj(global_features).expand(
            -1, self.fut_ts, -1)
        anchor_fusion_features = torch.cat(
            [global_context, anchor_time_features, future_summary], dim=-1)
        hybrid_fusion_features = torch.cat(
            [global_context, hybrid_time_features, future_summary], dim=-1)

        anchor_residual = self.residual_mlp(anchor_fusion_features).reshape(
            batch, self.fut_ts, self.ego_fut_mode, 2).permute(0, 2, 1, 3)
        anchor_gate = self.gate_mlp(anchor_fusion_features).sigmoid()
        anchor_prediction = (
            global_prediction
            + anchor_gate[:, None, :, :] * anchor_residual)
        anchor_positions_raw = anchor_prediction.cumsum(dim=2)

        hybrid_fusion_features = self._scale_gradient(
            hybrid_fusion_features, self.hybrid_feature_gradient_scale)
        condition = self.residual_context_proj(hybrid_fusion_features)[:, None]
        normalized_positions = (
            anchor_positions_raw.detach() / self.position_residual_scale)
        condition = condition + self.proposal_position_proj(normalized_positions)
        condition = condition + self.mode_embedding.weight[None, :, None, :]

        raw_residual = self.position_residual_mlp(condition)
        normalized_residual = raw_residual.tanh()
        position_residual = normalized_residual * self.position_residual_scale
        gate = self.position_gate_mlp(condition).sigmoid()
        applied_residual = gate * position_residual

        anchor_positions = (
            anchor_positions_raw.detach()
            + self.anchor_gradient_scale
            * (anchor_positions_raw - anchor_positions_raw.detach()))
        final_positions = anchor_positions + applied_residual
        auxiliary_positions = (
            anchor_positions_raw.detach() + position_residual)
        return {
            'ego_fut_preds': self._positions_to_displacements(final_positions),
            'ego_fut_global_preds': global_prediction,
            'ego_fut_anchor_preds': anchor_prediction,
            'ego_fut_position_aux_preds': self._positions_to_displacements(
                auxiliary_positions),
            'ego_fut_applied_residual': applied_residual,
            'ego_fut_applied_residual_normalized': gate * normalized_residual,
            'ego_fut_position_gate': gate,
        }
