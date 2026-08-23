"""Baseline-conditioned residual diffusion planner for GaussianAD.

The trusted v12 planner remains the deterministic reference.  This module
predicts a bounded, per-horizon normalized residual in cumulative-position
space and never generates a complete trajectory from pure noise.  Training
uses clean-residual (x0) prediction under a truncated VP path.  Inference uses
deterministic DDIM with a fixed noise bank and always keeps the unmodified v12
trajectory as candidate zero.

The implementation intentionally avoids a continuous fusion gate.  Candidate
selection first applies Gaussian/dynamics feasibility checks and then uses a
small learned quality score.  A feasible baseline wins whenever the best
generated candidate does not improve the total cost by a configured margin.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.registry import MODELS

from .planner_v12 import VADHeadFutAttnGlobalResidual
from ..utils.utils import get_rotation_matrix


def _positions_to_displacements(positions):
    origin = positions.new_zeros((*positions.shape[:-2], 1, 2))
    return torch.diff(torch.cat([origin, positions], dim=-2), dim=-2)


class SinusoidalTimeEmbedding(nn.Module):
    """Continuous diffusion-time embedding."""

    def __init__(self, embed_dims):
        super().__init__()
        if embed_dims % 2:
            raise ValueError('time embedding dimension must be even')
        self.embed_dims = int(embed_dims)

    def forward(self, timestep):
        half = self.embed_dims // 2
        work_timestep = timestep.float()
        exponent = -math.log(10000.0) * torch.arange(
            half, device=timestep.device, dtype=torch.float32)
        exponent = exponent / max(half - 1, 1)
        frequency = exponent.exp()
        phase = 1000.0 * work_timestep[:, None] * frequency[None, :]
        embedding = torch.cat([phase.sin(), phase.cos()], dim=-1)
        return embedding.to(dtype=timestep.dtype)


class ResidualDiTBlock(nn.Module):
    """Small DiT block with temporal and time-aligned Gaussian attention."""

    def __init__(self, embed_dims, num_heads, dropout):
        super().__init__()
        self.norm_temporal = nn.LayerNorm(
            embed_dims, elementwise_affine=False)
        self.norm_gaussian = nn.LayerNorm(
            embed_dims, elementwise_affine=False)
        self.norm_ffn = nn.LayerNorm(embed_dims, elementwise_affine=False)
        self.temporal_attention = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=dropout, batch_first=True)
        self.gaussian_attention = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, embed_dims * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dims * 4, embed_dims))
        # shift, scale and residual gate for three sublayers.
        self.modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(embed_dims, embed_dims * 9))

    @staticmethod
    def _modulate(value, shift, scale):
        return value * (1.0 + scale[:, None, :]) + shift[:, None, :]

    def forward(self, waypoint, gaussian, condition):
        modulation = self.modulation(condition).chunk(9, dim=-1)
        shift_t, scale_t, gate_t = modulation[0:3]
        shift_g, scale_g, gate_g = modulation[3:6]
        shift_f, scale_f, gate_f = modulation[6:9]

        query = self._modulate(
            self.norm_temporal(waypoint), shift_t, scale_t)
        attended = self.temporal_attention(
            query, query, query, need_weights=False)[0]
        waypoint = waypoint + gate_t[:, None, :] * attended

        batch_branches, timesteps, embed_dims = waypoint.shape
        gaussian_count = gaussian.shape[-2]
        query = self._modulate(
            self.norm_gaussian(waypoint), shift_g, scale_g)
        query = query.reshape(batch_branches * timesteps, 1, embed_dims)
        gaussian = gaussian.reshape(
            batch_branches * timesteps, gaussian_count, embed_dims)
        attended = self.gaussian_attention(
            query, gaussian, gaussian, need_weights=False)[0]
        attended = attended.reshape(batch_branches, timesteps, embed_dims)
        waypoint = waypoint + gate_g[:, None, :] * attended

        ffn_input = self._modulate(
            self.norm_ffn(waypoint), shift_f, scale_f)
        waypoint = waypoint + gate_f[:, None, :] * self.ffn(ffn_input)
        return waypoint


@MODELS.register_module()
class VADHeadFutAttnResidualDDIM(VADHeadFutAttnGlobalResidual):
    """v12 planner plus a Gaussian-conditioned residual DDIM proposal head.

    All trajectory tensors exposed to the legacy GaussianAD losses remain step
    displacements.  Diffusion is performed internally on normalized residuals
    between cumulative v12 and GT positions.
    """

    def __init__(
            self,
            *args,
            residual_hidden_dims=192,
            residual_num_layers=4,
            residual_num_heads=8,
            residual_dropout=0.1,
            residual_scale=1.0,
            residual_clip=8.0,
            diffusion_sigma_max=0.5,
            diffusion_train_t_min=0.02,
            diffusion_sample_steps=2,
            num_inference_samples=4,
            fixed_noise_seed=3407,
            gaussian_topk=128,
            gaussian_corridor_radius=0.75,
            gaussian_importance_floor=0.1,
            dynamic_semantic_dims=10,
            risk_margin=0.5,
            risk_uncertainty_growth=0.15,
            planner_gaussian_grad_scale=1.0,
            planner_offset_grad_scale=1.0,
            detach_baseline=False,
            detach_residual_reference=True,
            detach_gaussian_context=True,
            keep_baseline_eval=False,
            time_interval=0.5,
            ego_width=1.85,
            ego_length=4.084,
            selector_risk_weight=4.0,
            selector_residual_weight=0.05,
            selector_dynamics_weight=0.02,
            selector_learned_weight=0.25,
            selector_baseline_margin=0.05,
            selector_risk_threshold=0.45,
            selector_max_normalized_residual=6.0,
            selector_max_acceleration=8.0,
            selector_max_jerk=15.0,
            **kwargs):
        self.residual_hidden_dims = int(residual_hidden_dims)
        self.residual_num_layers = int(residual_num_layers)
        self.residual_num_heads = int(residual_num_heads)
        self.residual_dropout = float(residual_dropout)
        self.residual_clip = float(residual_clip)
        self.diffusion_sigma_max = float(diffusion_sigma_max)
        self.diffusion_train_t_min = float(diffusion_train_t_min)
        self.diffusion_sample_steps = int(diffusion_sample_steps)
        self.num_inference_samples = int(num_inference_samples)
        self.fixed_noise_seed = int(fixed_noise_seed)
        self.gaussian_topk = int(gaussian_topk)
        self.gaussian_corridor_radius = float(gaussian_corridor_radius)
        self.gaussian_importance_floor = float(gaussian_importance_floor)
        self.dynamic_semantic_dims = int(dynamic_semantic_dims)
        self.risk_margin = float(risk_margin)
        self.risk_uncertainty_growth = float(risk_uncertainty_growth)
        self.planner_gaussian_grad_scale = float(
            planner_gaussian_grad_scale)
        self.planner_offset_grad_scale = float(planner_offset_grad_scale)
        self.detach_baseline = bool(detach_baseline)
        self.detach_residual_reference = bool(detach_residual_reference)
        self.detach_gaussian_context = bool(detach_gaussian_context)
        self.keep_baseline_eval = bool(keep_baseline_eval)
        self.time_interval = float(time_interval)
        self.ego_width = float(ego_width)
        self.ego_length = float(ego_length)
        self.selector_risk_weight = float(selector_risk_weight)
        self.selector_residual_weight = float(selector_residual_weight)
        self.selector_dynamics_weight = float(selector_dynamics_weight)
        self.selector_learned_weight = float(selector_learned_weight)
        self.selector_baseline_margin = float(selector_baseline_margin)
        self.selector_risk_threshold = float(selector_risk_threshold)
        self.selector_max_normalized_residual = float(
            selector_max_normalized_residual)
        self.selector_max_acceleration = float(selector_max_acceleration)
        self.selector_max_jerk = float(selector_max_jerk)

        if self.residual_hidden_dims % self.residual_num_heads:
            raise ValueError(
                'residual_hidden_dims must be divisible by residual_num_heads')
        if not 0.0 < self.diffusion_sigma_max < 1.0:
            raise ValueError('diffusion_sigma_max must be in (0, 1)')
        if not 0.0 <= self.diffusion_train_t_min < 1.0:
            raise ValueError('diffusion_train_t_min must be in [0, 1)')
        if self.diffusion_sample_steps < 1:
            raise ValueError('diffusion_sample_steps must be positive')
        if self.num_inference_samples < 1:
            raise ValueError('num_inference_samples must be positive')
        if self.gaussian_topk < 1:
            raise ValueError('gaussian_topk must be positive')

        super().__init__(*args, time_interval=time_interval, **kwargs)

        scale = torch.as_tensor(residual_scale, dtype=torch.float32)
        if scale.numel() == 1:
            scale = scale.expand(self.fut_ts, 2).clone()
        else:
            scale = scale.reshape(self.fut_ts, 2)
        if not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError('residual_scale must contain finite positive values')
        self.register_buffer('residual_scale', scale)

        generator = torch.Generator(device='cpu')
        generator.manual_seed(self.fixed_noise_seed)
        fixed_noise = torch.randn(
            self.num_inference_samples, self.fut_ts, 2,
            generator=generator)
        self.register_buffer('fixed_residual_noise', fixed_noise)

    @staticmethod
    def _residual_child_names():
        return {
            'noisy_residual_encoder',
            'reference_encoder',
            'gaussian_context_proj',
            'gaussian_relative_encoder',
            'horizon_embedding',
            'mode_embedding',
            'diffusion_time_embedding',
            'diffusion_time_mlp',
            'residual_dit_blocks',
            'residual_final_norm',
            'residual_output',
            'candidate_quality_mlp',
        }

    def train(self, mode=True):
        super().train(mode)
        if mode and self.detach_baseline and self.keep_baseline_eval:
            residual_children = self._residual_child_names()
            for name, module in self.named_children():
                if name not in residual_children:
                    module.eval()
        return self

    def _init_layers(self):
        super()._init_layers()
        hidden = self.residual_hidden_dims
        self.noisy_residual_encoder = nn.Sequential(
            nn.Linear(2, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.reference_encoder = nn.Sequential(
            nn.Linear(4, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.gaussian_context_proj = nn.Linear(self.embed_dims, hidden)
        # relative xy, log-scale xy, opacity, dynamic probability and distance.
        self.gaussian_relative_encoder = nn.Sequential(
            nn.Linear(7, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.horizon_embedding = nn.Embedding(self.fut_ts, hidden)
        self.mode_embedding = nn.Embedding(self.ego_fut_mode, hidden)
        self.diffusion_time_embedding = SinusoidalTimeEmbedding(hidden)
        self.diffusion_time_mlp = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.residual_dit_blocks = nn.ModuleList([
            ResidualDiTBlock(
                hidden, self.residual_num_heads, self.residual_dropout)
            for _ in range(self.residual_num_layers)
        ])
        self.residual_final_norm = nn.LayerNorm(hidden)
        self.residual_output = nn.Linear(hidden, 2)

        # Per-timestep features are [normalized residual xy, risk, speed, accel].
        quality_input_dims = self.fut_ts * 5
        self.candidate_quality_mlp = nn.Sequential(
            nn.Linear(quality_input_dims, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1))

    def init_weights(self):
        super().init_weights()
        residual_modules = [
            self.noisy_residual_encoder,
            self.reference_encoder,
            self.gaussian_context_proj,
            self.gaussian_relative_encoder,
            self.diffusion_time_mlp,
            self.residual_dit_blocks,
            self.residual_final_norm,
            self.residual_output,
            self.candidate_quality_mlp,
        ]
        for root in residual_modules:
            for module in root.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.LayerNorm):
                    if module.elementwise_affine:
                        nn.init.ones_(module.weight)
                        nn.init.zeros_(module.bias)
        nn.init.normal_(self.horizon_embedding.weight, std=0.02)
        nn.init.normal_(self.mode_embedding.weight, std=0.02)
        for block in self.residual_dit_blocks:
            nn.init.zeros_(block.modulation[-1].weight)
            nn.init.zeros_(block.modulation[-1].bias)
        # Exact identity invariant: zero clean residual means v14 == v12.
        nn.init.zeros_(self.residual_output.weight)
        nn.init.zeros_(self.residual_output.bias)
        nn.init.zeros_(self.candidate_quality_mlp[-1].weight)
        nn.init.zeros_(self.candidate_quality_mlp[-1].bias)

    @staticmethod
    def _straight_through_grad_scale(value, scale):
        if value is None or scale >= 1.0:
            return value
        if scale <= 0.0:
            return value.detach()
        return value.detach() + scale * (value - value.detach())

    def _planner_results(self, results):
        planner_results = dict(results)
        if results.get('gaussian_output') is not None:
            planner_results['gaussian_output'] = (
                self._straight_through_grad_scale(
                    results['gaussian_output'],
                    self.planner_gaussian_grad_scale))
        if results.get('offset') is not None:
            planner_results['offset'] = self._straight_through_grad_scale(
                results['offset'], self.planner_offset_grad_scale)
        return planner_results

    @staticmethod
    def _metadata_tensor(results, key):
        value = results.get(key)
        if torch.is_tensor(value):
            return value
        metas = results.get('metas')
        if isinstance(metas, dict):
            value = metas.get(key)
            return value if torch.is_tensor(value) else None
        if isinstance(metas, (list, tuple)) and metas:
            values = [item.get(key) for item in metas if isinstance(item, dict)]
            if len(values) == len(metas) and all(
                    torch.is_tensor(item) for item in values):
                return torch.stack(values)
        return None

    @staticmethod
    def _squeeze_annotation(value, target_dims):
        while value.dim() > target_dims and value.shape[1] == 1:
            value = value.squeeze(1)
        return value

    def _diffusion_schedule(self, timestep):
        sigma = self.diffusion_sigma_max * torch.sin(
            0.5 * math.pi * timestep)
        alpha = (1.0 - sigma.square()).clamp_min(1e-6).sqrt()
        return alpha, sigma

    def _build_gaussian_scene(self, results):
        if self.detach_gaussian_context:
            with torch.no_grad():
                future_content, _, _ = self._build_future_gaussians(results)
        else:
            future_content, _, _ = self._build_future_gaussians(results)
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
        scale_xy = gaussian_output[:, None, :, 3:5].clamp_min(0.05)
        scale_xy = scale_xy.expand(-1, self.fut_ts, -1, -1)
        opacity = gaussian_output[:, None, :, 10:11].clamp(0.0, 1.0)
        opacity = opacity.expand(-1, self.fut_ts, -1, -1)

        semantics = gaussian_output[..., 11:]
        dynamic_dims = min(self.dynamic_semantic_dims, semantics.shape[-1])
        if dynamic_dims:
            dynamic_probability = semantics[..., :dynamic_dims].sum(
                dim=-1, keepdim=True).clamp(0.0, 1.0)
        else:
            dynamic_probability = opacity[:, 0].new_ones(
                (batch, num_gaussians, 1))
        dynamic_probability = dynamic_probability[:, None].expand(
            -1, self.fut_ts, -1, -1)
        importance = opacity * (
            self.gaussian_importance_floor
            + (1.0 - self.gaussian_importance_floor)
            * dynamic_probability)

        rotation = get_rotation_matrix(gaussian_output[..., 6:10])
        rotation_xy = rotation[..., :2, :2]
        scene = {
            'content': future_content,
            'future_xy': future_xy,
            'scale_xy': scale_xy,
            'opacity': opacity,
            'dynamic_probability': dynamic_probability,
            'importance': importance,
            'rotation_xy': rotation_xy,
        }
        if self.detach_gaussian_context:
            scene = {key: value.detach() for key, value in scene.items()}
        return scene

    def _select_gaussian_context(
            self, scene, reference_position, return_tokens=True):
        """Select time-aligned Gaussians around each candidate trajectory."""
        future_xy = scene['future_xy']
        num_gaussians = future_xy.shape[2]
        topk = min(self.gaussian_topk, num_gaussians)
        if topk < 1:
            raise RuntimeError('residual planner received no future Gaussians')

        delta = reference_position[:, :, :, None, :] - future_xy[:, None]
        local_delta = torch.einsum(
            'bgij,bctgj->bctgi', scene['rotation_xy'], delta)
        rotation_abs = scene['rotation_xy'].abs()
        projected_ego_x = (
            0.5 * self.ego_length * rotation_abs[..., 0, 0]
            + 0.5 * self.ego_width * rotation_abs[..., 0, 1])
        projected_ego_y = (
            0.5 * self.ego_length * rotation_abs[..., 1, 0]
            + 0.5 * self.ego_width * rotation_abs[..., 1, 1])
        projected_ego = torch.stack(
            [projected_ego_x, projected_ego_y], dim=-1)[:, None, None]
        future_time = torch.arange(
            1, self.fut_ts + 1,
            device=reference_position.device,
            dtype=reference_position.dtype).reshape(1, 1, self.fut_ts, 1, 1)
        risk_radius = (
            scene['scale_xy'][:, None]
            + projected_ego
            + self.risk_margin
            + self.risk_uncertainty_growth * future_time)
        selection_radius = risk_radius + self.gaussian_corridor_radius
        selection_distance2 = (
            local_delta / selection_radius.clamp_min(0.1)
        ).square().sum(dim=-1)
        risk_distance2 = (
            local_delta / risk_radius.clamp_min(0.1)
        ).square().sum(dim=-1)
        importance = scene['importance'][..., 0][:, None]
        relevance = -0.5 * selection_distance2 + 0.25 * (
            importance + 1e-4).log()
        indices = relevance.topk(topk, dim=-1).indices

        density = torch.exp(-0.5 * risk_distance2) * importance
        selected_density = density.gather(3, indices)
        gaussian_risk = (
            0.7 * selected_density[..., 0]
            + 0.3 * selected_density.mean(dim=-1)).clamp(0.0, 1.0)
        if not return_tokens:
            return None, gaussian_risk

        branches = reference_position.shape[1]

        def gather_scene(value):
            expanded = value[:, None].expand(
                -1, branches, -1, -1, -1)
            gather_index = indices[..., None].expand(
                -1, -1, -1, -1, value.shape[-1])
            return expanded.gather(3, gather_index)

        selected_content = gather_scene(scene['content'])
        selected_xy = gather_scene(scene['future_xy'])
        selected_scale = gather_scene(scene['scale_xy'])
        selected_opacity = gather_scene(scene['opacity'])
        selected_dynamic = gather_scene(scene['dynamic_probability'])
        selected_distance = selection_distance2.gather(
            3, indices).sqrt()[..., None]
        relative = torch.cat([
            (selected_xy - reference_position[:, :, :, None, :]) / 30.0,
            selected_scale.clamp_min(1e-3).log(),
            selected_opacity,
            selected_dynamic,
            selected_distance / 10.0,
        ], dim=-1)
        tokens = (
            self.gaussian_context_proj(selected_content)
            + self.gaussian_relative_encoder(relative))
        return tokens, gaussian_risk

    def _predict_clean_residual(
            self, noisy_residual, timestep, reference_displacement,
            reference_position, mode_ids, scene, context_position=None):
        batch, branches, timesteps = noisy_residual.shape[:3]
        if timesteps != self.fut_ts:
            raise ValueError('unexpected planning horizon')
        if context_position is None:
            context_position = reference_position
        gaussian, context_risk = self._select_gaussian_context(
            scene, context_position, return_tokens=True)

        reference_features = torch.cat([
            reference_displacement,
            reference_position / 30.0,
        ], dim=-1)
        waypoint = (
            self.noisy_residual_encoder(noisy_residual)
            + self.reference_encoder(reference_features))
        horizon = self.horizon_embedding.weight.reshape(
            1, 1, self.fut_ts, self.residual_hidden_dims)
        mode = self.mode_embedding(mode_ids).reshape(
            1, branches, 1, self.residual_hidden_dims)
        waypoint = waypoint + horizon + mode

        condition = self.diffusion_time_mlp(
            self.diffusion_time_embedding(timestep.reshape(-1)))
        condition = condition + self.mode_embedding(
            mode_ids)[None].expand(batch, -1, -1).reshape(
                batch * branches, self.residual_hidden_dims)
        waypoint = waypoint.reshape(
            batch * branches, self.fut_ts, self.residual_hidden_dims)
        gaussian = gaussian.reshape(
            batch * branches, self.fut_ts,
            gaussian.shape[-2], self.residual_hidden_dims)
        for block in self.residual_dit_blocks:
            waypoint = block(waypoint, gaussian, condition)
        raw_residual = self.residual_output(
            self.residual_final_norm(waypoint))
        raw_residual = raw_residual.reshape(
            batch, branches, self.fut_ts, 2)
        if self.residual_clip > 0:
            clean_residual = self.residual_clip * torch.tanh(
                raw_residual / self.residual_clip)
        else:
            clean_residual = raw_residual
        return clean_residual, context_risk

    def _training_target(self, results, baseline_position):
        target = self._metadata_tensor(results, 'ego_fut_trajs')
        if target is None:
            return None, None
        target = self._squeeze_annotation(target, 3).to(
            device=baseline_position.device, dtype=baseline_position.dtype)
        target_position = target[..., :2].cumsum(dim=-2)
        target_position = target_position[:, None].expand(
            -1, self.ego_fut_mode, -1, -1)
        scale = self.residual_scale.to(
            device=baseline_position.device,
            dtype=baseline_position.dtype)[None, None]
        residual = (target_position - baseline_position) / scale
        if self.residual_clip > 0:
            residual = residual.clamp(-self.residual_clip, self.residual_clip)

        mask = self._metadata_tensor(results, 'ego_fut_masks')
        if mask is not None:
            mask = self._squeeze_annotation(mask, 2).to(
                device=residual.device, dtype=residual.dtype)
            residual = residual * mask[:, None, :, None]
        return residual, mask

    def _candidate_quality(
            self, candidate_position, baseline_position, candidate_risk):
        scale = self.residual_scale.to(
            device=candidate_position.device,
            dtype=candidate_position.dtype)
        while scale.dim() < candidate_position.dim():
            scale = scale.unsqueeze(0)
        residual = (candidate_position - baseline_position) / scale
        displacement = _positions_to_displacements(candidate_position)
        speed = torch.linalg.norm(displacement, dim=-1) / self.time_interval
        acceleration = torch.diff(speed, dim=-1) / self.time_interval
        acceleration = F.pad(acceleration, (1, 0))
        features = torch.cat([
            residual,
            candidate_risk[..., None],
            (speed / 15.0)[..., None],
            (acceleration / 8.0)[..., None],
        ], dim=-1)
        # Ranking trains the selector, not the proposal generator.  The
        # generator receives trajectory and safety gradients from dedicated
        # objectives whose semantics do not change with selector calibration.
        features = features.detach()
        return self.candidate_quality_mlp(
            features.flatten(start_dim=-2)).squeeze(-1)

    def _candidate_costs(
            self, candidate_position, baseline_position,
            candidate_risk, quality_logits):
        scale = self.residual_scale.to(
            device=candidate_position.device,
            dtype=candidate_position.dtype)
        scale = scale.reshape(
            *((1,) * (candidate_position.dim() - 2)), self.fut_ts, 2)
        residual_size = torch.linalg.norm(
            (candidate_position - baseline_position) / scale,
            dim=-1).mean(dim=-1)
        residual_max = torch.linalg.norm(
            (candidate_position - baseline_position) / scale,
            dim=-1).max(dim=-1).values

        horizon_weight = torch.linspace(
            0.5, 1.5, self.fut_ts,
            device=candidate_risk.device,
            dtype=candidate_risk.dtype)
        risk_mean = (
            (candidate_risk * horizon_weight).sum(dim=-1)
            / horizon_weight.sum())
        risk_max = candidate_risk.max(dim=-1).values

        displacement = _positions_to_displacements(candidate_position)
        velocity = displacement / self.time_interval
        if self.fut_ts > 1:
            acceleration = torch.diff(
                velocity, dim=-2) / self.time_interval
            acceleration_max = torch.linalg.norm(
                acceleration, dim=-1).max(dim=-1).values
        else:
            acceleration = velocity[..., :0, :]
            acceleration_max = risk_mean.new_zeros(risk_mean.shape)
        if self.fut_ts > 2:
            jerk = torch.diff(acceleration, dim=-2) / self.time_interval
            jerk_max = torch.linalg.norm(
                jerk, dim=-1).max(dim=-1).values
        else:
            jerk_max = risk_mean.new_zeros(risk_mean.shape)

        acceleration_violation = F.relu(
            acceleration_max - self.selector_max_acceleration)
        jerk_violation = F.relu(jerk_max - self.selector_max_jerk)
        dynamics_cost = acceleration_violation + 0.25 * jerk_violation
        analytic_cost = (
            self.selector_risk_weight * risk_mean
            + self.selector_residual_weight * residual_size
            + self.selector_dynamics_weight * dynamics_cost)
        feasible = (
            (risk_max < self.selector_risk_threshold)
            & (residual_max < self.selector_max_normalized_residual)
            & (acceleration_max < self.selector_max_acceleration)
            & (jerk_max < self.selector_max_jerk))
        residual_violation = F.relu(
            residual_max - self.selector_max_normalized_residual)
        violation = (
            F.relu(risk_max - self.selector_risk_threshold)
            + residual_violation / max(
                self.selector_max_normalized_residual, 1e-3)
            + acceleration_violation / max(self.selector_max_acceleration, 1e-3)
            + jerk_violation / max(self.selector_max_jerk, 1e-3))
        cost = (
            analytic_cost
            - self.selector_learned_weight * quality_logits.clamp(-10.0, 10.0)
            + 1000.0 * violation)
        return cost, feasible, risk_mean, acceleration_max, jerk_max

    def _select_candidates(
            self, candidate_position, baseline_position,
            candidate_risk, quality_logits):
        baseline_expanded = baseline_position[:, :, None].expand_as(
            candidate_position)
        costs, feasible, risk_mean, acceleration_max, jerk_max = (
            self._candidate_costs(
                candidate_position, baseline_expanded,
                candidate_risk, quality_logits))
        selected = costs.argmin(dim=-1)
        selected_cost = costs.gather(-1, selected[..., None]).squeeze(-1)
        baseline_cost = costs[..., 0]
        keep_baseline = (
            feasible[..., 0]
            & (baseline_cost <= selected_cost + self.selector_baseline_margin))
        selected = torch.where(keep_baseline, selected.new_zeros(()), selected)
        return {
            'selected': selected,
            'costs': costs,
            'feasible': feasible,
            'risk_mean': risk_mean,
            'acceleration_max': acceleration_max,
            'jerk_max': jerk_max,
        }

    def _teacher_forward(
            self, results, baseline_displacement, baseline_position, scene):
        if self.detach_residual_reference:
            reference_displacement = baseline_displacement.detach()
            reference_position = baseline_position.detach()
        else:
            reference_displacement = baseline_displacement
            reference_position = baseline_position
        target_residual, _ = self._training_target(
            results, reference_position)
        if target_residual is None:
            return None
        batch = reference_position.shape[0]
        timestep = torch.rand(
            batch, self.ego_fut_mode,
            device=reference_position.device,
            dtype=reference_position.dtype)
        timestep = (
            self.diffusion_train_t_min
            + (1.0 - self.diffusion_train_t_min) * timestep)
        alpha, sigma = self._diffusion_schedule(timestep)
        noise = torch.randn_like(target_residual)
        noisy_residual = (
            alpha[..., None, None] * target_residual
            + sigma[..., None, None] * noise)
        mode_ids = torch.arange(
            self.ego_fut_mode, device=reference_position.device)
        predicted_residual, _ = self._predict_clean_residual(
            noisy_residual, timestep,
            reference_displacement, reference_position,
            mode_ids, scene)
        scale = self.residual_scale.to(
            device=reference_position.device,
            dtype=reference_position.dtype)[None, None]
        # The legacy PlanLoss keeps its original gradient path through the
        # v12 trajectory.  New residual-specific losses use the numerically
        # identical detached-reference trajectory below and therefore cannot
        # alter any baseline parameter through their target/reference path.
        generated_position = baseline_position + scale * predicted_residual
        generated_displacement = _positions_to_displacements(
            generated_position)
        residual_position = reference_position + scale * predicted_residual
        residual_displacement = _positions_to_displacements(
            residual_position)

        candidate_position = torch.stack(
            [reference_position, residual_position], dim=2)
        flat_candidate_position = candidate_position.reshape(
            batch, self.ego_fut_mode * 2, self.fut_ts, 2)
        _, flat_risk = self._select_gaussian_context(
            scene, flat_candidate_position, return_tokens=False)
        candidate_risk = flat_risk.reshape(
            batch, self.ego_fut_mode, 2, self.fut_ts)
        baseline_for_candidates = reference_position[:, :, None].expand_as(
            candidate_position)
        quality_logits = self._candidate_quality(
            candidate_position, baseline_for_candidates, candidate_risk)
        candidate_displacement = _positions_to_displacements(
            candidate_position)
        selection = self._select_candidates(
            candidate_position, reference_position,
            candidate_risk, quality_logits)
        return {
            # Training uses the generated branch directly; argmin must not
            # block the denoiser gradient.  Inference uses hard selection.
            'ego_fut_preds': generated_displacement,
            'ego_fut_residual_preds': residual_displacement,
            'ego_fut_ddim_preds': predicted_residual,
            'ego_fut_ddim_targets': target_residual,
            'ego_fut_ddim_t': timestep,
            'ego_fut_candidates': candidate_displacement,
            'ego_fut_candidate_risk': candidate_risk,
            'ego_fut_candidate_quality_logits': quality_logits,
            'ego_fut_candidate_costs': selection['costs'],
            'ego_fut_candidate_feasible': selection['feasible'],
            'ego_fut_selected_index': torch.ones(
                batch, self.ego_fut_mode,
                device=reference_position.device, dtype=torch.long),
            'ego_fut_generated_risk': candidate_risk[:, :, 1],
        }

    def _ddim_sample(self, baseline_displacement, baseline_position, scene):
        batch = baseline_position.shape[0]
        modes = self.ego_fut_mode
        samples = self.num_inference_samples
        branches = modes * samples
        mode_ids = torch.arange(
            modes, device=baseline_position.device)[:, None].expand(
                modes, samples).reshape(branches)

        reference_displacement = baseline_displacement[:, :, None].expand(
            -1, -1, samples, -1, -1).reshape(
                batch, branches, self.fut_ts, 2)
        reference_position = baseline_position[:, :, None].expand(
            -1, -1, samples, -1, -1).reshape(
                batch, branches, self.fut_ts, 2)
        noise = self.fixed_residual_noise.to(
            device=baseline_position.device,
            dtype=baseline_position.dtype)
        noise = noise[None, None].expand(
            batch, modes, -1, -1, -1).reshape(
                batch, branches, self.fut_ts, 2)
        start_t = baseline_position.new_ones((batch, branches))
        _, start_sigma = self._diffusion_schedule(start_t)
        residual_state = start_sigma[..., None, None] * noise

        solver_times = torch.linspace(
            1.0, 0.0, self.diffusion_sample_steps + 1,
            device=baseline_position.device,
            dtype=baseline_position.dtype)
        context_position = reference_position
        for step_index in range(self.diffusion_sample_steps):
            current_t = solver_times[step_index].expand(batch, branches)
            next_t = solver_times[step_index + 1].expand(batch, branches)
            clean_residual, _ = self._predict_clean_residual(
                residual_state, current_t,
                reference_displacement, reference_position,
                mode_ids, scene, context_position=context_position)
            alpha, sigma = self._diffusion_schedule(current_t)
            next_alpha, next_sigma = self._diffusion_schedule(next_t)
            estimated_noise = (
                residual_state - alpha[..., None, None] * clean_residual
            ) / sigma[..., None, None].clamp_min(1e-6)
            residual_state = (
                next_alpha[..., None, None] * clean_residual
                + next_sigma[..., None, None] * estimated_noise)
            # Re-pool future Gaussians around the first clean trajectory.
            context_position = (
                reference_position
                + self.residual_scale.to(
                    device=reference_position.device,
                    dtype=reference_position.dtype)[None, None]
                * clean_residual).detach()

        generated_residual = residual_state.reshape(
            batch, modes, samples, self.fut_ts, 2)
        generated_position = (
            baseline_position[:, :, None]
            + self.residual_scale.to(
                device=baseline_position.device,
                dtype=baseline_position.dtype)[None, None, None]
            * generated_residual)
        candidate_position = torch.cat([
            baseline_position[:, :, None], generated_position,
        ], dim=2)
        candidate_count = samples + 1
        flat_candidate_position = candidate_position.reshape(
            batch, modes * candidate_count, self.fut_ts, 2)
        _, flat_risk = self._select_gaussian_context(
            scene, flat_candidate_position, return_tokens=False)
        candidate_risk = flat_risk.reshape(
            batch, modes, candidate_count, self.fut_ts)
        baseline_for_candidates = baseline_position[:, :, None].expand_as(
            candidate_position)
        quality_logits = self._candidate_quality(
            candidate_position, baseline_for_candidates, candidate_risk)
        selection = self._select_candidates(
            candidate_position, baseline_position,
            candidate_risk, quality_logits)

        selected_position = candidate_position.gather(
            2, selection['selected'][..., None, None, None].expand(
                -1, -1, 1, self.fut_ts, 2)).squeeze(2)
        candidate_displacement = _positions_to_displacements(
            candidate_position)
        selected_displacement = _positions_to_displacements(
            selected_position)
        return {
            'ego_fut_preds': selected_displacement,
            'ego_fut_candidates': candidate_displacement,
            'ego_fut_candidate_risk': candidate_risk,
            'ego_fut_candidate_quality_logits': quality_logits,
            'ego_fut_candidate_costs': selection['costs'],
            'ego_fut_candidate_feasible': selection['feasible'],
            'ego_fut_selected_index': selection['selected'],
            'ego_fut_ddim_nfe': selected_displacement.new_full(
                (), self.diffusion_sample_steps),
        }

    def forward(self, results):
        planner_results = self._planner_results(results)
        if self.detach_baseline:
            with torch.no_grad():
                baseline_outputs = super().forward(planner_results)
        else:
            baseline_outputs = super().forward(planner_results)
        baseline_displacement = torch.nan_to_num(
            baseline_outputs['ego_fut_preds'],
            nan=0.0, posinf=0.0, neginf=0.0)
        if self.detach_baseline:
            baseline_displacement = baseline_displacement.detach()
        baseline_position = baseline_displacement.cumsum(dim=-2)

        if self.detach_baseline:
            outputs = {
                key: value.detach() if torch.is_tensor(value) else value
                for key, value in baseline_outputs.items()
            }
        else:
            outputs = dict(baseline_outputs)
        outputs['ego_fut_base_preds'] = baseline_displacement
        scene = self._build_gaussian_scene(planner_results)
        teacher_outputs = self._teacher_forward(
            planner_results, baseline_displacement, baseline_position, scene)
        if teacher_outputs is not None:
            outputs.update(teacher_outputs)

        if self.training:
            if teacher_outputs is None:
                raise KeyError(
                    'training residual DDIM requires '
                    "results['metas']['ego_fut_trajs']")
        else:
            outputs.update(self._ddim_sample(
                baseline_displacement, baseline_position, scene))
        return outputs
