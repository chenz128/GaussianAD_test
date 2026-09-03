"""Frozen-OCC, anchor-conditioned truncated diffusion planner.

This planner is designed for the strong V3-SE3 OCC checkpoint.  It does not
change or back-propagate through image, Gaussian, OCC, detection, map, or the
already trained deterministic planner modules.  The inherited planner supplies
three deterministic anchors (main/per-frame/global).  A small residual DiT
then denoises bounded perturbations around the main anchor and a conservative
selector may choose a generated proposal only when predicted full-footprint
OCC risk is non-regressive.

Unlike the v14/v15 implementation, the scene used by diffusion and selection
is the exact ``planner_future_gaussians`` bank produced by the strong OCC
frontend.  GT trajectories and boxes remain loss-only data and never enter
this forward method.
"""

import torch

from mmengine.registry import MODELS

from .planner_v12_future_gaussian_isolated import (
    VADHeadFutAttnGlobalResidualFutureGaussianIsolated,
)
from .planner_v14_residual_ddim import (
    VADHeadFutAttnResidualDDIM,
    _positions_to_displacements,
)
from ..utils.utils import get_rotation_matrix


@MODELS.register_module()
class VADHeadFrozenOccTruncatedResidualDDIM(
        VADHeadFutAttnResidualDDIM,
        VADHeadFutAttnGlobalResidualFutureGaussianIsolated):
    """Residual diffusion proposals protected by a frozen strong-OCC anchor."""

    _ANCHOR_CANDIDATE_COUNT = 3

    def __init__(
            self,
            *args,
            obstacle_semantic_indices=(2, 3, 4, 5, 6, 7, 9, 10),
            footprint_longitudinal_samples=5,
            footprint_lateral_samples=3,
            footprint_gaussian_margin=0.35,
            truncated_start_t=0.25,
            truncated_noise_scale=1.0,
            safety_guidance_scale=0.20,
            safety_guidance_clip=0.25,
            safety_guidance_activation=0.15,
            guidance_gaussian_topk=32,
            selector_min_risk_improvement=0.03,
            selector_risk_tolerance=0.01,
            selector_quality_improvement=0.05,
            freeze_deterministic_anchor=True,
            **kwargs):
        self.obstacle_semantic_indices = tuple(
            int(index) for index in obstacle_semantic_indices)
        self.footprint_longitudinal_samples = int(
            footprint_longitudinal_samples)
        self.footprint_lateral_samples = int(footprint_lateral_samples)
        self.footprint_gaussian_margin = float(footprint_gaussian_margin)
        self.truncated_start_t = float(truncated_start_t)
        self.truncated_noise_scale = float(truncated_noise_scale)
        self.safety_guidance_scale = float(safety_guidance_scale)
        self.safety_guidance_clip = float(safety_guidance_clip)
        self.safety_guidance_activation = float(
            safety_guidance_activation)
        self.guidance_gaussian_topk = int(guidance_gaussian_topk)
        self.selector_min_risk_improvement = float(
            selector_min_risk_improvement)
        self.selector_risk_tolerance = float(selector_risk_tolerance)
        self.selector_quality_improvement = float(
            selector_quality_improvement)
        self.freeze_deterministic_anchor = bool(
            freeze_deterministic_anchor)

        if not self.obstacle_semantic_indices:
            raise ValueError('at least one obstacle semantic index is required')
        if self.footprint_longitudinal_samples < 2:
            raise ValueError('footprint_longitudinal_samples must be >= 2')
        if self.footprint_lateral_samples < 2:
            raise ValueError('footprint_lateral_samples must be >= 2')
        if not 0.0 < self.truncated_start_t < 1.0:
            raise ValueError('truncated_start_t must be in (0, 1)')
        if self.guidance_gaussian_topk < 1:
            raise ValueError('guidance_gaussian_topk must be positive')

        # These defaults implement the no-regression contract.  They can still
        # be stated explicitly in the config for checkpoint auditing.
        kwargs.setdefault('detach_baseline', True)
        kwargs.setdefault('detach_residual_reference', True)
        kwargs.setdefault('detach_gaussian_context', True)
        kwargs.setdefault('keep_baseline_eval', True)
        super().__init__(*args, **kwargs)

        longitudinal = torch.linspace(
            -0.5 * self.ego_length + 0.5,
            0.5 * self.ego_length + 0.5,
            self.footprint_longitudinal_samples)
        lateral = torch.linspace(
            -0.5 * self.ego_width,
            0.5 * self.ego_width,
            self.footprint_lateral_samples)
        grid_x, grid_y = torch.meshgrid(
            longitudinal, lateral, indexing='ij')
        footprint = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
        self.register_buffer('ego_footprint_samples', footprint)

        if self.freeze_deterministic_anchor:
            trainable_children = self._residual_child_names()
            for name, module in self.named_children():
                if name not in trainable_children:
                    module.requires_grad_(False)

    def _build_gaussian_scene(self, results, future_content=None):
        """Decode the exact GT-free future-Gaussian planner bank."""
        future = results.get('planner_future_gaussians')
        padding_mask = results.get('planner_future_gaussian_mask')
        if future is None or padding_mask is None:
            raise KeyError(
                'v16 requires planner_future_gaussians and its padding mask')
        if future.dim() != 4 or future.shape[1] != self.fut_ts:
            raise ValueError(
                'planner_future_gaussians must be (B,T,G,28), got '
                f'{tuple(future.shape)}')
        if future.shape[-1] != 28:
            raise ValueError('v16 expects 28-D decoded Gaussian attributes')
        if padding_mask.shape != future.shape[:3]:
            raise ValueError('future Gaussian mask shape mismatch')
        if future_content is None:
            projected = self._build_future_gaussians(results)
            future_content = projected[0]
        if future_content.shape[:3] != future.shape[:3]:
            raise ValueError('future Gaussian content/attribute mismatch')

        future = torch.nan_to_num(
            future, nan=0.0, posinf=0.0, neginf=0.0)
        semantics = future[..., 11:]
        if max(self.obstacle_semantic_indices) >= semantics.shape[-1]:
            raise ValueError(
                'obstacle semantic index exceeds Gaussian semantic width')
        # The strong OCC bank can contain either softmax probabilities or the
        # non-negative ``softplus`` evidence used by the direct-future head.
        # Normalizing evidence is identity for probabilities and avoids
        # treating every high-magnitude direct Gaussian as probability one.
        semantic_evidence = semantics.clamp_min(0.0)
        semantic_probability = semantic_evidence / semantic_evidence.sum(
            dim=-1, keepdim=True).clamp_min(1e-6)
        semantic_index = torch.as_tensor(
            self.obstacle_semantic_indices,
            device=semantics.device,
            dtype=torch.long)
        obstacle_probability = semantic_probability.index_select(
            -1, semantic_index).sum(dim=-1, keepdim=True).clamp(0.0, 1.0)
        opacity = future[..., 10:11].clamp(0.0, 1.0)
        importance = opacity * (
            self.gaussian_importance_floor
            + (1.0 - self.gaussian_importance_floor)
            * obstacle_probability)
        obstacle_importance = opacity * obstacle_probability

        scene = {
            'content': future_content,
            'future_xy': future[..., :2],
            'scale_xy': future[..., 3:5].clamp_min(0.05),
            'opacity': opacity,
            'dynamic_probability': obstacle_probability,
            'importance': importance,
            'obstacle_importance': obstacle_importance,
            'rotation_xy': get_rotation_matrix(
                future[..., 6:10])[..., :2, :2],
            'padding_mask': padding_mask.bool(),
        }
        # The strong OCC representation is an immutable condition.  This also
        # prevents a planning loss from changing any frontend tensor through a
        # future refactor of the segmentor.
        return {key: value.detach() for key, value in scene.items()}

    @staticmethod
    def _gather_future(value, indices):
        """Gather a (B,T,G,...) tensor with (B,C,T,K) indices."""
        branches = indices.shape[1]
        expanded = value[:, None].expand(-1, branches, *value.shape[1:])
        tail_dims = value.dim() - 3
        index = indices.reshape(
            *indices.shape, *((1,) * tail_dims)).expand(
                *indices.shape, *value.shape[3:])
        return torch.gather(expanded, 3, index)

    def _selection_geometry(self, scene, reference_position):
        """Return relevance, normalized distance and world-frame delta."""
        delta = reference_position[..., None, :] - scene['future_xy'][:, None]
        rotation = scene['rotation_xy'][:, None]
        local_delta = torch.einsum(
            'bctgij,bctgj->bctgi', rotation, delta)
        rotation_abs = rotation.abs()
        projected_x = (
            0.5 * self.ego_length * rotation_abs[..., 0, 0]
            + 0.5 * self.ego_width * rotation_abs[..., 0, 1])
        projected_y = (
            0.5 * self.ego_length * rotation_abs[..., 1, 0]
            + 0.5 * self.ego_width * rotation_abs[..., 1, 1])
        projected_ego = torch.stack([projected_x, projected_y], dim=-1)
        future_time = torch.arange(
            1, self.fut_ts + 1,
            device=reference_position.device,
            dtype=reference_position.dtype).reshape(1, 1, self.fut_ts, 1, 1)
        radius = (
            scene['scale_xy'][:, None]
            + projected_ego
            + self.risk_margin
            + self.risk_uncertainty_growth * future_time)
        normalized_distance2 = (
            local_delta / radius.clamp_min(0.1)).square().sum(dim=-1)
        importance = scene['importance'][..., 0][:, None]
        relevance = (
            -0.5 * normalized_distance2
            + 0.25 * (importance + 1e-6).log())
        relevance = relevance.masked_fill(
            scene['padding_mask'][:, None],
            torch.finfo(relevance.dtype).min)
        return relevance, normalized_distance2, delta

    def _select_gaussian_context(
            self, scene, reference_position, return_tokens=True):
        """Select corridor Gaussians and estimate evaluator-aligned risk.

        Risk is measured over samples covering the same axis-aligned ego
        rectangle used by the GaussianAD/VAD collision evaluator, including
        its +0.5 m longitudinal centre offset.  This is intentionally more
        conservative than testing only the trajectory centre.
        """
        relevance, normalized_distance2, _ = self._selection_geometry(
            scene, reference_position)
        topk = min(self.gaussian_topk, relevance.shape[-1])
        if topk < 1:
            raise RuntimeError('future Gaussian bank is empty')
        indices = relevance.topk(topk, dim=-1).indices

        selected_xy = self._gather_future(scene['future_xy'], indices)
        selected_scale = self._gather_future(scene['scale_xy'], indices)
        selected_rotation = self._gather_future(
            scene['rotation_xy'], indices)
        selected_obstacle = self._gather_future(
            scene['obstacle_importance'], indices)
        selected_padding = self._gather_future(
            scene['padding_mask'], indices)
        selected_obstacle = selected_obstacle.masked_fill(
            selected_padding[..., None], 0.0)

        offsets = self.ego_footprint_samples.to(
            device=reference_position.device,
            dtype=reference_position.dtype)
        footprint = (
            reference_position[..., None, None, :]
            + offsets.reshape(1, 1, 1, 1, -1, 2))
        footprint_delta = footprint - selected_xy[..., None, :]
        local_footprint_delta = torch.einsum(
            'bctkij,bctkpj->bctkpi',
            selected_rotation, footprint_delta)
        future_time = torch.arange(
            1, self.fut_ts + 1,
            device=reference_position.device,
            dtype=reference_position.dtype).reshape(1, 1, self.fut_ts, 1, 1, 1)
        footprint_radius = (
            selected_scale[..., None, :]
            + self.footprint_gaussian_margin
            + self.risk_uncertainty_growth * future_time)
        footprint_distance2 = (
            local_footprint_delta
            / footprint_radius.clamp_min(0.1)).square().sum(dim=-1)
        density = (
            torch.exp(-0.5 * footprint_distance2)
            * selected_obstacle[..., 0, None])
        # Max pooling mirrors the evaluator's "any occupied footprint cell"
        # event and avoids scene-density-dependent saturation from a sum.
        gaussian_risk = density.amax(dim=-1).amax(dim=-1).clamp(0.0, 1.0)

        if not return_tokens:
            return None, gaussian_risk

        selected_content = self._gather_future(scene['content'], indices)
        selected_opacity = self._gather_future(scene['opacity'], indices)
        selected_dynamic = self._gather_future(
            scene['dynamic_probability'], indices)
        selected_distance = normalized_distance2.gather(
            3, indices).sqrt()[..., None]
        relative = torch.cat([
            (selected_xy - reference_position[..., None, :]) / 30.0,
            selected_scale.clamp_min(1e-3).log(),
            selected_opacity,
            selected_dynamic,
            selected_distance / 10.0,
        ], dim=-1)
        tokens = (
            self.gaussian_context_proj(selected_content)
            + self.gaussian_relative_encoder(relative))
        return tokens, gaussian_risk

    def _apply_safety_guidance(self, candidate_position, scene):
        """Apply a bounded analytic repulsion to generated proposals only."""
        if self.safety_guidance_scale <= 0.0:
            return candidate_position
        relevance, normalized_distance2, delta = self._selection_geometry(
            scene, candidate_position)
        topk = min(self.guidance_gaussian_topk, relevance.shape[-1])
        indices = relevance.topk(topk, dim=-1).indices
        # ``delta`` already has a candidate axis, unlike ordinary scene
        # tensors handled by ``_gather_future``.
        gather_index = indices[..., None].expand(
            *indices.shape, delta.shape[-1])
        selected_delta = torch.gather(delta, 3, gather_index)
        selected_distance2 = normalized_distance2.gather(3, indices)
        selected_importance = self._gather_future(
            scene['obstacle_importance'], indices)[..., 0]
        selected_padding = self._gather_future(
            scene['padding_mask'], indices)
        selected_importance = selected_importance.masked_fill(
            selected_padding, 0.0)
        weight = torch.exp(-0.5 * selected_distance2) * selected_importance
        unit = selected_delta / torch.linalg.norm(
            selected_delta, dim=-1, keepdim=True).clamp_min(1e-3)
        direction = (weight[..., None] * unit).sum(dim=-2)
        direction_norm = torch.linalg.norm(
            direction, dim=-1, keepdim=True)

        origin = candidate_position.new_zeros(
            (*candidate_position.shape[:-2], 1, 2))
        velocity = torch.diff(
            torch.cat([origin, candidate_position], dim=-2), dim=-2)
        braking = -velocity / torch.linalg.norm(
            velocity, dim=-1, keepdim=True).clamp_min(1e-3)
        direction = torch.where(
            direction_norm > 1e-3,
            direction / direction_norm.clamp_min(1e-3),
            braking)
        risk = weight.max(dim=-1).values.clamp(0.0, 1.0)
        strength = (
            (risk - self.safety_guidance_activation)
            / max(1.0 - self.safety_guidance_activation, 1e-3)
        ).clamp(0.0, 1.0)
        adjustment = self.safety_guidance_scale * strength[..., None] * direction
        if adjustment.shape[-2] > 2:
            smoothed = adjustment.clone()
            smoothed[..., 1:-1, :] = (
                adjustment[..., :-2, :]
                + 2.0 * adjustment[..., 1:-1, :]
                + adjustment[..., 2:, :]) / 4.0
            adjustment = smoothed
        norm = torch.linalg.norm(adjustment, dim=-1, keepdim=True)
        adjustment = adjustment * (
            self.safety_guidance_clip
            / norm.clamp_min(self.safety_guidance_clip)).clamp(max=1.0)
        return candidate_position + adjustment

    def _sample_residual_proposals(
            self, baseline_displacement, baseline_position, scene):
        batch, modes = baseline_position.shape[:2]
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

        if self.training:
            noise = torch.randn_like(reference_position)
        else:
            noise = self.fixed_residual_noise.to(
                device=baseline_position.device,
                dtype=baseline_position.dtype)
            noise = noise[None, None].expand(
                batch, modes, -1, -1, -1).reshape_as(reference_position)
        start_t = reference_position.new_full(
            (batch, branches), self.truncated_start_t)
        _, start_sigma = self._diffusion_schedule(start_t)
        residual_state = (
            self.truncated_noise_scale
            * start_sigma[..., None, None]
            * noise)
        solver_times = torch.linspace(
            self.truncated_start_t, 0.0,
            self.diffusion_sample_steps + 1,
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
        flat_generated = generated_position.reshape(
            batch, modes * samples, self.fut_ts, 2)
        flat_generated = self._apply_safety_guidance(flat_generated, scene)
        return flat_generated.reshape(
            batch, modes, samples, self.fut_ts, 2)

    def _select_candidates(
            self, candidate_position, baseline_position,
            candidate_risk, quality_logits):
        """Risk-first selection with an exact candidate-zero fallback."""
        baseline_expanded = baseline_position[:, :, None].expand_as(
            candidate_position)
        costs, feasible, risk_mean, acceleration_max, jerk_max = (
            self._candidate_costs(
                candidate_position, baseline_expanded,
                candidate_risk, quality_logits))
        costs = torch.nan_to_num(
            costs, nan=1e6, posinf=1e6, neginf=-1e6)
        risk_max = torch.nan_to_num(
            candidate_risk, nan=1.0, posinf=1.0, neginf=0.0
        ).max(dim=-1).values
        baseline_risk = risk_max[..., :1]
        baseline_quality = quality_logits[..., :1]
        risk_better = (
            risk_max <= baseline_risk - self.selector_min_risk_improvement)
        risk_non_regressive = (
            risk_max <= baseline_risk + self.selector_risk_tolerance)
        quality_better = (
            quality_logits
            >= baseline_quality + self.selector_quality_improvement)

        scale = self.residual_scale.to(
            device=candidate_position.device,
            dtype=candidate_position.dtype).reshape(
                1, 1, 1, self.fut_ts, 2)
        residual_max = torch.linalg.norm(
            (candidate_position - baseline_expanded) / scale,
            dim=-1).max(dim=-1).values
        motion_ok = (
            (residual_max < self.selector_max_normalized_residual)
            & (acceleration_max < self.selector_max_acceleration)
            & (jerk_max < self.selector_max_jerk))
        baseline_unsafe = (
            (~feasible[..., :1])
            | (baseline_risk >= self.selector_risk_threshold))
        unsafe_replacement = baseline_unsafe & risk_better
        safe_replacement = (
            (~baseline_unsafe) & feasible
            & risk_non_regressive & quality_better)
        generated_eligible = motion_ok & (
            unsafe_replacement | safe_replacement)
        is_baseline = torch.zeros_like(generated_eligible)
        is_baseline[..., 0] = True
        eligible = generated_eligible | is_baseline

        safe_cost = costs + (~eligible).to(costs.dtype) * 1e6
        unsafe_cost = (
            100.0 * risk_max + 10.0 * risk_mean + costs
            + (~eligible).to(costs.dtype) * 1e6)
        selection_cost = torch.where(
            baseline_unsafe, unsafe_cost, safe_cost)
        selected = selection_cost.argmin(dim=-1)
        selected_risk_better = risk_better.gather(
            -1, selected[..., None]).squeeze(-1)
        selected_quality_better = quality_better.gather(
            -1, selected[..., None]).squeeze(-1)
        selected_non_regressive = risk_non_regressive.gather(
            -1, selected[..., None]).squeeze(-1)
        may_replace = (
            baseline_unsafe.squeeze(-1) & selected_risk_better) | (
                (~baseline_unsafe.squeeze(-1))
                & selected_non_regressive & selected_quality_better)
        selected = torch.where(
            may_replace, selected, selected.new_zeros(()))
        return {
            'selected': selected,
            'costs': selection_cost,
            'feasible': eligible,
            'risk_mean': risk_mean,
            'risk_max': risk_max,
            'risk_better': risk_better,
            'risk_non_regressive': risk_non_regressive,
            'quality_better': quality_better,
            'acceleration_max': acceleration_max,
            'jerk_max': jerk_max,
        }

    @staticmethod
    def _candidate_anchor(outputs, key, fallback):
        value = outputs.get(key)
        if value is None or value.shape != fallback.shape:
            return fallback
        return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)

    def forward(self, results):
        # ``planner_results`` contains only the segmentor's explicit prediction
        # whitelist.  In particular, no GT trajectory/box/occupancy is present.
        planner_results = self._planner_results(results)
        self._capture_future_content = True
        self._cached_future_content = None
        try:
            with torch.no_grad():
                baseline_outputs = (
                    VADHeadFutAttnGlobalResidualFutureGaussianIsolated.forward(
                        self, planner_results))
        finally:
            self._capture_future_content = False
        future_content = self._cached_future_content
        self._cached_future_content = None
        if future_content is None:
            raise RuntimeError('deterministic anchor did not build future OCC')

        outputs = {
            key: value.detach() if torch.is_tensor(value) else value
            for key, value in baseline_outputs.items()
        }
        baseline_displacement = torch.nan_to_num(
            outputs['ego_fut_preds'], nan=0.0, posinf=0.0, neginf=0.0)
        per_frame_displacement = self._candidate_anchor(
            outputs, 'ego_fut_per_frame_preds', baseline_displacement)
        global_displacement = self._candidate_anchor(
            outputs, 'ego_fut_aux_preds', baseline_displacement)
        baseline_position = baseline_displacement.cumsum(dim=-2)
        anchor_position = torch.stack([
            baseline_position,
            per_frame_displacement.cumsum(dim=-2),
            global_displacement.cumsum(dim=-2),
        ], dim=2)

        scene = self._build_gaussian_scene(
            planner_results, future_content=future_content)
        generated_position = self._sample_residual_proposals(
            baseline_displacement, baseline_position, scene)
        candidate_position = torch.cat(
            [anchor_position, generated_position], dim=2)
        batch, modes, candidate_count = candidate_position.shape[:3]
        flat_candidate = candidate_position.reshape(
            batch, modes * candidate_count, self.fut_ts, 2)
        _, flat_risk = self._select_gaussian_context(
            scene, flat_candidate, return_tokens=False)
        candidate_risk = flat_risk.reshape(
            batch, modes, candidate_count, self.fut_ts)
        baseline_for_candidates = baseline_position[:, :, None].expand_as(
            candidate_position)
        quality_logits = self._candidate_quality(
            candidate_position, baseline_for_candidates, candidate_risk)
        selection = self._select_candidates(
            candidate_position, baseline_position,
            candidate_risk, quality_logits)
        candidate_displacement = _positions_to_displacements(
            candidate_position)

        outputs.update({
            'ego_fut_base_preds': baseline_displacement,
            'ego_fut_candidates': candidate_displacement,
            'ego_fut_candidate_risk': candidate_risk,
            'ego_fut_candidate_quality_logits': quality_logits,
            'ego_fut_candidate_costs': selection['costs'],
            'ego_fut_candidate_feasible': selection['feasible'],
            'ego_fut_candidate_risk_better': selection['risk_better'],
            'ego_fut_candidate_risk_non_regressive': selection[
                'risk_non_regressive'],
            'ego_fut_selected_index': selection['selected'],
            'ego_fut_generated_risk': candidate_risk[
                :, :, self._ANCHOR_CANDIDATE_COUNT:],
            'ego_fut_ddim_nfe': baseline_displacement.new_full(
                (), self.diffusion_sample_steps),
            'ego_fut_anchor_candidate_count': baseline_displacement.new_full(
                (), self._ANCHOR_CANDIDATE_COUNT, dtype=torch.long),
        })
        if not self.training:
            selected_position = candidate_position.gather(
                2, selection['selected'][..., None, None, None].expand(
                    -1, -1, 1, self.fut_ts, 2)).squeeze(2)
            outputs['ego_fut_preds'] = _positions_to_displacements(
                selected_position)
        # During training the legacy main output remains the exact frozen
        # anchor.  New losses supervise candidates through the keys above.
        return outputs
