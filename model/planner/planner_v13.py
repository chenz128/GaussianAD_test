"""Risk-aware extension of the v12 per-frame/global residual planner.

This module is intentionally self-contained.  It inherits the proven v12
architecture but does not modify v12 or the planner package exports.  The
corresponding config loads it through ``custom_imports``.

The v12 fusion gate only sees latent features and shares one scalar among the
three navigation modes.  This variant adds a differentiable, time-aligned risk
measurement from the predicted future Gaussians and predicts one gate for each
navigation mode and future timestep:

    main = per_frame + safe_gate * (global - per_frame)

The final gate layer remains zero-initialized, hence the initial forward output
is exactly the v12 collision-safe per-frame branch.
"""

import torch
import torch.nn as nn
from mmengine.registry import MODELS

from .planner_v12 import VADHeadFutAttnGlobalResidual


@MODELS.register_module()
class VADHeadFutAttnRiskAwareGlobalResidual(
        VADHeadFutAttnGlobalResidual):
    """v12 planner with explicit Gaussian risk and mode-specific fusion.

    Args:
        risk_topk: Number of most relevant Gaussians used by each candidate
            waypoint.  The implementation never rasterizes a dense BEV map.
        risk_margin: Extra metric inflation around the ego footprint.
        risk_uncertainty_growth: Extra Gaussian radius per future step.  This
            makes the safety estimate conservative when flow becomes uncertain.
        risk_safety_temperature: Controls how strongly a riskier global branch
            is suppressed by the deterministic safety factor.
        planner_gaussian_grad_scale: Straight-through gradient scale from the
            planner to current Gaussian parameters.
        planner_offset_grad_scale: Straight-through gradient scale from the
            planner to future flow offsets.
    """

    def __init__(
            self,
            *args,
            risk_topk=32,
            risk_margin=0.5,
            risk_uncertainty_growth=0.15,
            risk_safety_temperature=8.0,
            planner_gaussian_grad_scale=0.1,
            planner_offset_grad_scale=0.1,
            dynamic_semantic_dims=10,
            ego_width=1.85,
            ego_length=4.084,
            **kwargs):
        self.risk_topk = int(risk_topk)
        self.risk_margin = float(risk_margin)
        self.risk_uncertainty_growth = float(risk_uncertainty_growth)
        self.risk_safety_temperature = float(risk_safety_temperature)
        self.planner_gaussian_grad_scale = float(
            planner_gaussian_grad_scale)
        self.planner_offset_grad_scale = float(planner_offset_grad_scale)
        self.dynamic_semantic_dims = int(dynamic_semantic_dims)
        self.ego_width = float(ego_width)
        self.ego_length = float(ego_length)
        super().__init__(*args, **kwargs)

    def _init_layers(self):
        super()._init_layers()

        global_dim = self.embed_dims * 4
        # risk features per mode/timestep:
        # [per-frame risk, global risk, risk delta,
        #  candidate disagreement, normalized time]
        self.risk_feature_encoder = nn.Sequential(
            nn.Linear(5, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.ReLU())

        # One logit for every navigation mode rather than one shared gate.
        self.refine_gate_mlp = nn.Sequential(
            nn.Linear(
                global_dim + self.embed_dims + self.embed_dims,
                self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.ego_fut_mode))

    def init_weights(self):
        super().init_weights()
        # super() initializes the replaced gate as well, but keep this explicit
        # invariant next to the new implementation: main == per_frame at step 0.
        nn.init.zeros_(self.refine_gate_mlp[-1].weight)
        nn.init.zeros_(self.refine_gate_mlp[-1].bias)

    @staticmethod
    def _straight_through_grad_scale(value, scale):
        if value is None or scale >= 1.0:
            return value
        if scale <= 0.0:
            return value.detach()
        return value.detach() + scale * (value - value.detach())

    def _planner_results(self, results):
        """Preserve forward values while limiting planner/perception conflict."""
        planner_results = dict(results)
        planner_results['gaussian_output'] = self._straight_through_grad_scale(
            results['gaussian_output'], self.planner_gaussian_grad_scale)
        if results.get('offset') is not None:
            planner_results['offset'] = self._straight_through_grad_scale(
                results['offset'], self.planner_offset_grad_scale)
        return planner_results

    def _future_geometry(self, results):
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
        scales_xy = gaussian_output[:, None, :, 3:5].clamp_min(0.05)
        opacity = gaussian_output[:, None, :, 10:11].clamp(0.0, 1.0)

        semantics = gaussian_output[..., 11:]
        dynamic_dims = min(self.dynamic_semantic_dims, semantics.shape[-1])
        if dynamic_dims > 0:
            dynamic_probability = semantics[..., :dynamic_dims].sum(
                dim=-1, keepdim=True).clamp(0.0, 1.0)
        else:
            dynamic_probability = opacity.new_ones(
                opacity.shape[0], opacity.shape[2], 1)
        importance = opacity * dynamic_probability[:, None, :, :]
        return future_xy, scales_xy, importance

    def _trajectory_risk(self, results, prediction):
        """Evaluate candidate trajectories against time-aligned Gaussians.

        ``prediction`` stores step displacements, so it is accumulated before
        measuring an anisotropic Gaussian distance.  The ego footprint and an
        increasing temporal uncertainty radius are folded into each Gaussian's
        XY scale.  Top-k aggregation avoids dilution by the 25,600 mostly
        irrelevant Gaussians.
        """
        future_xy, scales_xy, importance = self._future_geometry(results)
        position = prediction.cumsum(dim=2)

        delta = position[:, :, :, None, :] - future_xy[:, None, :, :, :]
        time = torch.arange(
            1, self.fut_ts + 1, device=position.device,
            dtype=position.dtype).reshape(1, 1, self.fut_ts, 1, 1)
        ego_extent = position.new_tensor([
            0.5 * self.ego_length,
            0.5 * self.ego_width,
        ]).reshape(1, 1, 1, 1, 2)
        radius = (
            scales_xy[:, None, :, :, :]
            + ego_extent
            + self.risk_margin
            + self.risk_uncertainty_growth * time)
        normalized_distance2 = (delta / radius.clamp_min(0.1)).square().sum(
            dim=-1)
        density = torch.exp(-0.5 * normalized_distance2)
        weighted_density = density * importance[:, None, :, :, 0]
        weighted_density = torch.nan_to_num(
            weighted_density, nan=0.0, posinf=1.0, neginf=0.0)

        topk = min(self.risk_topk, weighted_density.shape[-1])
        if topk <= 0:
            return weighted_density.new_zeros(weighted_density.shape[:-1])
        top_values = weighted_density.topk(topk, dim=-1).values
        # A max-dominated but still spatially stable summary in [0, 1].
        risk = 0.7 * top_values[..., 0] + 0.3 * top_values.mean(dim=-1)
        return risk.clamp(0.0, 1.0)

    def forward(self, results):
        results = self._planner_results(results)
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

        future_content, future_key, _ = self._build_future_gaussians(results)
        num_gaussians = future_content.shape[2]

        # v12 per-frame collision-safe branch.
        fut_ego = self.ego_to_fut(current_features)
        fut_ego = fut_ego.expand(-1, self.fut_ts, -1)
        fut_ego = fut_ego + self.fut_pos.weight[None, :, :]
        fut_query = fut_ego.permute(1, 0, 2)
        if self.fut_self_decoder is not None:
            fut_query = self.fut_self_decoder(
                query=fut_query, key=fut_query, value=fut_query)
        folded_query = fut_query.permute(1, 0, 2).reshape(
            batch * self.fut_ts, self.embed_dims).unsqueeze(0)
        folded_kv = future_content.reshape(
            batch * self.fut_ts, num_gaussians,
            self.embed_dims).permute(1, 0, 2)
        if self.ego_fut_gaussian_decoder is not None:
            folded_query = self.ego_fut_gaussian_decoder(
                query=folded_query, key=folded_kv, value=folded_kv)
        per_frame_features = folded_query.squeeze(0).reshape(
            batch, self.fut_ts, self.embed_dims)
        per_frame_prediction = self.fut_out_mlp(per_frame_features).reshape(
            batch, self.fut_ts, self.ego_fut_mode, 2).permute(0, 2, 1, 3)

        # v12 global low-L2 branch.
        global_key = future_key.reshape(
            batch, self.fut_ts * num_gaussians,
            self.embed_dims).permute(1, 0, 2)
        global_value = future_content.reshape(
            batch, self.fut_ts * num_gaussians,
            self.embed_dims).permute(1, 0, 2)
        if self.global_fut_gaussian_decoder is not None:
            global_future_query = self.global_fut_gaussian_decoder(
                query=ego_gaussian_query, key=global_key, value=global_value)
        else:
            global_future_query = ego_gaussian_query
        global_features = torch.cat([
            current_features,
            global_future_query.permute(1, 0, 2),
        ], dim=-1)
        global_prediction = self.global_shape_mlp(global_features).reshape(
            batch, self.ego_fut_mode, self.fut_ts, 2)

        # Explicit candidate-conditioned risk for both experts.
        per_frame_risk = self._trajectory_risk(results, per_frame_prediction)
        global_risk = self._trajectory_risk(results, global_prediction)
        disagreement = torch.linalg.norm(
            global_prediction.cumsum(dim=2)
            - per_frame_prediction.cumsum(dim=2), dim=-1)
        normalized_time = torch.arange(
            1, self.fut_ts + 1, device=global_risk.device,
            dtype=global_risk.dtype) / self.fut_ts
        normalized_time = normalized_time.reshape(
            1, 1, self.fut_ts).expand_as(global_risk)
        risk_features = torch.stack([
            per_frame_risk,
            global_risk,
            global_risk - per_frame_risk,
            disagreement,
            normalized_time,
        ], dim=-1)
        risk_embedding = self.risk_feature_encoder(risk_features)

        # Broadcast latent features to (B, mode, time, channels).
        latent_gate_input = torch.cat([
            global_features[:, None, :, :].expand(
                -1, self.ego_fut_mode, self.fut_ts, -1),
            per_frame_features[:, None, :, :].expand(
                -1, self.ego_fut_mode, -1, -1),
            risk_embedding,
        ], dim=-1)
        # The MLP emits all modes; select its matching mode on the diagonal.
        all_gate_logits = self.refine_gate_mlp(latent_gate_input)
        gate_logits = all_gate_logits.diagonal(
            dim1=1, dim2=3).permute(0, 2, 1)

        gate = gate_logits.tanh()
        time_scale = 0.2 + 0.8 * normalized_time
        # Preserve v12 behavior when risks are equal.  Suppress the global
        # residual only when it is explicitly riskier than the per-frame base.
        safety_factor = (
            2.0 * torch.sigmoid(
                -self.risk_safety_temperature
                * (global_risk - per_frame_risk))).clamp(max=1.0)
        gate = gate * time_scale * safety_factor
        main_prediction = per_frame_prediction + gate[..., None] * (
            global_prediction - per_frame_prediction)

        return {
            'ego_fut_preds': main_prediction,
            'ego_fut_aux_preds': global_prediction,
            'ego_fut_per_frame_preds': per_frame_prediction,
            'ego_fut_gate_logits': gate_logits,
            'ego_fut_gate': gate,
            'ego_fut_global_risk': global_risk,
            'ego_fut_per_frame_risk': per_frame_risk,
        }
