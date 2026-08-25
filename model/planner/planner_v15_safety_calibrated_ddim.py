"""Collision-guarded candidate selection for the v14 residual DDIM planner.

The original v15 selector allowed a weakly calibrated safety score to replace
the complete v14 score and, critically, dropped v14's Gaussian-risk hard
feasibility test.  This revision uses a conservative no-regression hierarchy:

1. retain every v14 residual/dynamics/Gaussian hard constraint;
2. a generated trajectory may replace candidate zero only when its Gaussian
   risk and learned safety score are no worse than the baseline;
3. preserve the v14 analytic/quality cost as the primary L2-aware ranker;
4. let the learned head override that result only for a high-confidence unsafe
   -> safe transition with a substantial score margin;
5. fall back exactly to the v14 selector whenever the safety head is not
   demonstrably informative.

Training calibrates safety and quality on the exact deterministic candidate
distribution used by baseline + K=4, 4-NFE DDIM inference.  The original
one-step teacher prediction is retained for the residual/DDIM regression
objective itself.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.registry import MODELS

from .planner_v14_residual_ddim import (
    VADHeadFutAttnResidualDDIM,
    _positions_to_displacements,
)


@MODELS.register_module()
class VADHeadFutAttnSafetyCalibratedResidualDDIM(
        VADHeadFutAttnResidualDDIM):
    """v14 residual DDIM plus metric-calibrated safety-first re-ranking."""

    def __init__(
            self,
            *args,
            safety_hidden_dims=96,
            safety_num_layers=2,
            safety_num_heads=4,
            safety_dropout=0.0,
            safety_probability_threshold=0.60,
            safety_safe_probability_threshold=0.30,
            safety_cvar_fraction=1.0 / 3.0,
            safety_max_weight=1.0,
            safety_cvar_weight=0.5,
            safety_tiebreak_weight=0.01,
            safety_baseline_margin=0.02,
            safety_override_margin=0.15,
            safety_min_informative_spread=0.05,
            safety_gaussian_risk_mean_tolerance=0.01,
            safety_gaussian_risk_max_tolerance=0.01,
            safety_training_candidate_count=5,
            **kwargs):
        self.safety_hidden_dims = int(safety_hidden_dims)
        self.safety_num_layers = int(safety_num_layers)
        self.safety_num_heads = int(safety_num_heads)
        self.safety_dropout = float(safety_dropout)
        self.safety_probability_threshold = float(
            safety_probability_threshold)
        self.safety_safe_probability_threshold = float(
            safety_safe_probability_threshold)
        self.safety_cvar_fraction = float(safety_cvar_fraction)
        self.safety_max_weight = float(safety_max_weight)
        self.safety_cvar_weight = float(safety_cvar_weight)
        self.safety_tiebreak_weight = float(safety_tiebreak_weight)
        self.safety_baseline_margin = float(safety_baseline_margin)
        self.safety_override_margin = float(safety_override_margin)
        self.safety_min_informative_spread = float(
            safety_min_informative_spread)
        self.safety_gaussian_risk_mean_tolerance = float(
            safety_gaussian_risk_mean_tolerance)
        self.safety_gaussian_risk_max_tolerance = float(
            safety_gaussian_risk_max_tolerance)
        self.safety_training_candidate_count = int(
            safety_training_candidate_count)

        if self.safety_hidden_dims % self.safety_num_heads:
            raise ValueError(
                'safety_hidden_dims must be divisible by safety_num_heads')
        if self.safety_num_layers < 1:
            raise ValueError('safety_num_layers must be positive')
        if not 0.0 < self.safety_probability_threshold <= 1.0:
            raise ValueError(
                'safety_probability_threshold must be in (0, 1]')
        if not 0.0 <= self.safety_safe_probability_threshold < (
                self.safety_probability_threshold):
            raise ValueError(
                'safety_safe_probability_threshold must be non-negative and '
                'strictly smaller than safety_probability_threshold')
        if not 0.0 < self.safety_cvar_fraction <= 1.0:
            raise ValueError('safety_cvar_fraction must be in (0, 1]')
        if self.safety_tiebreak_weight < 0.0:
            raise ValueError('safety_tiebreak_weight must be non-negative')
        if self.safety_baseline_margin < 0.0:
            raise ValueError('safety_baseline_margin must be non-negative')
        if self.safety_override_margin <= 0.0:
            raise ValueError('safety_override_margin must be positive')
        if self.safety_min_informative_spread <= 0.0:
            raise ValueError(
                'safety_min_informative_spread must be positive')
        if (self.safety_gaussian_risk_mean_tolerance < 0.0
                or self.safety_gaussian_risk_max_tolerance < 0.0):
            raise ValueError('Gaussian risk tolerances must be non-negative')
        if self.safety_training_candidate_count < 2:
            raise ValueError(
                'safety_training_candidate_count must be at least two')

        super().__init__(*args, **kwargs)
        expected_training_candidates = self.num_inference_samples + 1
        if self.safety_training_candidate_count != expected_training_candidates:
            raise ValueError(
                'safety training must use the exact inference candidate count '
                f'({expected_training_candidates})')

    @staticmethod
    def _residual_child_names():
        return VADHeadFutAttnResidualDDIM._residual_child_names() | {
            'candidate_safety_encoder',
            'candidate_gaussian_safety_proj',
            'candidate_safety_temporal',
            'candidate_safety_norm',
            'candidate_collision_head',
        }

    def _init_layers(self):
        super()._init_layers()
        # Creating extra nn.Linear/Transformer modules normally advances the
        # global RNG and would change initialization of modules constructed
        # later in the full model.  Restore the state after construction so a
        # same-seed v14/v15 run has byte-identical inherited/DDIM parameters;
        # ``init_weights`` below initializes the safety modules afterwards.
        inherited_rng_state = torch.random.get_rng_state()
        hidden = self.safety_hidden_dims
        # residual xy, absolute xy, velocity xy, acceleration, turn magnitude,
        # Gaussian risk, normalized horizon, and three command-mode indicators.
        safety_feature_dims = 13
        self.candidate_safety_encoder = nn.Sequential(
            nn.Linear(safety_feature_dims, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden))
        # Mean + max pooling preserves both scene context and the most critical
        # trajectory-aligned Gaussian token at each future timestep.
        self.candidate_gaussian_safety_proj = nn.Sequential(
            nn.Linear(self.residual_hidden_dims * 2, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=self.safety_num_heads,
            dim_feedforward=hidden * 2,
            dropout=self.safety_dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True)
        self.candidate_safety_temporal = nn.TransformerEncoder(
            encoder_layer, num_layers=self.safety_num_layers)
        self.candidate_safety_norm = nn.LayerNorm(hidden)
        self.candidate_collision_head = nn.Linear(hidden, 1)
        torch.random.set_rng_state(inherited_rng_state)

    def init_weights(self):
        super().init_weights()
        # Keep the post-initialization CPU RNG identical to v14.  Without this
        # guard, initializing the detached safety head changes dataloader and
        # diffusion-noise streams before the very first training iteration.
        inherited_rng_state = torch.random.get_rng_state()
        safety_modules = [
            self.candidate_safety_encoder,
            self.candidate_gaussian_safety_proj,
            self.candidate_safety_temporal,
            self.candidate_safety_norm,
            self.candidate_collision_head,
        ]
        for root in safety_modules:
            for module in root.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.LayerNorm):
                    if module.elementwise_affine:
                        nn.init.ones_(module.weight)
                        nn.init.zeros_(module.bias)
        # A zero logit means p(collision)=0.5 for every candidate.  Before the
        # safety head is trained, this constant cancels from re-ranking and the
        # original v14 cost remains the tie breaker.
        nn.init.zeros_(self.candidate_collision_head.weight)
        nn.init.zeros_(self.candidate_collision_head.bias)
        torch.random.set_rng_state(inherited_rng_state)

    def _candidate_safety_logits(
            self, candidate_position, baseline_position, candidate_risk,
            scene=None):
        """Predict a collision logit for every candidate and future step.

        The complete feature tensor is detached so calibration gradients train
        only the safety head.  This is important for preserving the already
        trained v14 denoising distribution and Gaussian/offset predictions.
        """
        baseline = baseline_position[:, :, None].expand_as(candidate_position)
        scale = self.residual_scale.to(
            device=candidate_position.device,
            dtype=candidate_position.dtype)
        scale = scale.reshape(1, 1, 1, self.fut_ts, 2)
        residual = (candidate_position - baseline) / scale

        displacement = _positions_to_displacements(candidate_position)
        velocity = displacement / self.time_interval
        speed = torch.linalg.norm(velocity, dim=-1)
        acceleration = torch.diff(speed, dim=-1) / self.time_interval
        acceleration = F.pad(acceleration, (1, 0))

        unit_heading = velocity / speed[..., None].clamp_min(1e-3)
        heading_delta = torch.linalg.norm(
            torch.diff(unit_heading, dim=-2), dim=-1)
        heading_delta = F.pad(heading_delta, (1, 0))

        batch, modes, candidates, timesteps = candidate_risk.shape
        horizon = torch.linspace(
            1.0 / timesteps, 1.0, timesteps,
            device=candidate_position.device,
            dtype=candidate_position.dtype)
        horizon = horizon.reshape(1, 1, 1, timesteps, 1).expand(
            batch, modes, candidates, -1, -1)
        mode = F.one_hot(
            torch.arange(modes, device=candidate_position.device),
            num_classes=modes).to(candidate_position.dtype)
        mode = mode.reshape(1, modes, 1, 1, modes).expand(
            batch, -1, candidates, timesteps, -1)

        risk = torch.nan_to_num(
            candidate_risk.to(candidate_position.dtype),
            nan=1.0, posinf=1.0, neginf=0.0)
        features = torch.cat([
            residual,
            candidate_position / 30.0,
            velocity / 15.0,
            (acceleration / 8.0)[..., None],
            heading_delta[..., None],
            risk[..., None],
            horizon,
            mode,
        ], dim=-1).detach()
        if features.shape[-1] != 13:
            raise RuntimeError(
                'safety feature size changed; expected 13, got '
                + str(features.shape[-1]))

        flat = features.reshape(-1, timesteps, features.shape[-1])
        encoded = self.candidate_safety_encoder(flat)
        if scene is not None:
            flat_position = candidate_position.reshape(
                batch, modes * candidates, timesteps, 2).detach()
            with torch.no_grad():
                gaussian_tokens, _ = self._select_gaussian_context(
                    scene, flat_position, return_tokens=True)
            gaussian_summary = torch.cat([
                gaussian_tokens.mean(dim=-2),
                gaussian_tokens.max(dim=-2).values,
            ], dim=-1).reshape(
                -1, timesteps, self.residual_hidden_dims * 2)
            # The safety objective calibrates only the new head.  In particular,
            # it cannot tune v14's Gaussian projection or residual DiT through
            # this richer sparse-world summary.
            encoded = encoded + self.candidate_gaussian_safety_proj(
                gaussian_summary)
        encoded = self.candidate_safety_temporal(encoded)
        logits = self.candidate_collision_head(
            self.candidate_safety_norm(encoded)).squeeze(-1)
        return logits.reshape(batch, modes, candidates, timesteps)

    def _safety_first_selection(
            self, candidate_position, baseline_position, candidate_risk,
            quality_logits, collision_logits):
        """Preserve v14 ranking while enforcing safety non-regression.

        Candidate zero is the deterministic baseline.  If it passes the v14
        hard guard, generated candidates must be non-inferior in both Gaussian
        risk and learned safety score.  The learned head can force an override
        only when the current result is confidently unsafe and another v14-
        feasible candidate is confidently safe by a configured margin.
        """
        baseline = baseline_position[:, :, None].expand_as(candidate_position)
        (legacy_cost, legacy_feasible, risk_mean,
         acceleration_max, jerk_max) = super()._candidate_costs(
             candidate_position, baseline, candidate_risk, quality_logits)

        probability = torch.nan_to_num(
            collision_logits.sigmoid(), nan=1.0, posinf=1.0, neginf=0.0)
        probability_max = probability.max(dim=-1).values
        tail_count = max(
            1, int(math.ceil(self.fut_ts * self.safety_cvar_fraction)))
        probability_cvar = probability.topk(
            min(tail_count, probability.shape[-1]), dim=-1).values.mean(dim=-1)
        safety_score = (
            self.safety_max_weight * probability_max
            + self.safety_cvar_weight * probability_cvar)

        # Reproduce the complete v14 selector first.  This remains the exact
        # fallback for zero-initialized, collapsed, or low-spread safety heads.
        finite_legacy = torch.nan_to_num(
            legacy_cost, nan=1e6, posinf=1e6, neginf=-1e6)
        legacy_selected = finite_legacy.argmin(dim=-1)
        legacy_selected_cost = finite_legacy.gather(
            -1, legacy_selected[..., None]).squeeze(-1)
        legacy_keep_baseline = (
            legacy_feasible[..., 0]
            & (finite_legacy[..., 0]
               <= legacy_selected_cost + self.selector_baseline_margin))
        legacy_selected = torch.where(
            legacy_keep_baseline,
            legacy_selected.new_zeros(()), legacy_selected)

        safety_spread = (
            safety_score.max(dim=-1).values
            - safety_score.min(dim=-1).values)
        safety_informative = (
            safety_spread >= self.safety_min_informative_spread)

        risk_max = candidate_risk.max(dim=-1).values
        baseline_is_feasible = legacy_feasible[..., :1]
        baseline_risk_mean = risk_mean[..., :1]
        baseline_risk_max = risk_max[..., :1]
        baseline_safety = safety_score[..., :1]
        risk_non_regressive = (
            (~baseline_is_feasible)
            | ((risk_mean <= baseline_risk_mean
                + self.safety_gaussian_risk_mean_tolerance)
               & (risk_max <= baseline_risk_max
                  + self.safety_gaussian_risk_max_tolerance)))
        learned_non_regressive = (
            (~baseline_is_feasible)
            | (safety_score <= baseline_safety + self.safety_baseline_margin))

        # Gaussian feasibility is never replaced by the learned head.  When
        # candidate zero is safe, generated candidates also have to satisfy a
        # relative no-regression contract against it.
        pareto_eligible = (
            legacy_feasible
            & risk_non_regressive
            & learned_non_regressive)
        has_pareto = pareto_eligible.any(dim=-1, keepdim=True)
        selection_cost = (
            finite_legacy
            + self.safety_tiebreak_weight * safety_score
            + (~pareto_eligible).to(finite_legacy.dtype) * 1e6)
        pareto_selected = selection_cost.argmin(dim=-1)
        pareto_selected_cost = finite_legacy.gather(
            -1, pareto_selected[..., None]).squeeze(-1)
        pareto_keep_baseline = (
            pareto_eligible[..., 0]
            & (finite_legacy[..., 0]
               <= pareto_selected_cost + self.selector_baseline_margin))
        pareto_selected = torch.where(
            pareto_keep_baseline,
            pareto_selected.new_zeros(()), pareto_selected)
        pareto_selected = torch.where(
            has_pareto.squeeze(-1), pareto_selected, legacy_selected)

        # A safety override is deliberately asymmetric: a high-risk current
        # choice may move to a low-risk alternative, but a tiny score advantage
        # can never reorder otherwise safe candidates.
        selected_safety = safety_score.gather(
            -1, pareto_selected[..., None]).squeeze(-1)
        selected_probability = probability_max.gather(
            -1, pareto_selected[..., None]).squeeze(-1)
        confidently_safe = (
            legacy_feasible
            & risk_non_regressive
            & (probability_max <= self.safety_safe_probability_threshold)
            & (safety_score
               <= selected_safety[..., None] - self.safety_override_margin))
        has_confidently_safe = confidently_safe.any(
            dim=-1, keepdim=True)
        override_cost = (
            finite_legacy
            + self.safety_tiebreak_weight * safety_score
            + (~confidently_safe).to(finite_legacy.dtype) * 1e6)
        override_selected = override_cost.argmin(dim=-1)
        safety_override = (
            safety_informative
            & (selected_probability >= self.safety_probability_threshold)
            & has_confidently_safe.squeeze(-1))
        guarded_selected = torch.where(
            safety_override, override_selected, pareto_selected)
        selected = torch.where(
            safety_informative, guarded_selected, legacy_selected)

        return {
            'selected': selected,
            'costs': selection_cost,
            'feasible': pareto_eligible,
            'legacy_costs': legacy_cost,
            'legacy_feasible': legacy_feasible,
            'safety_score': safety_score,
            'probability': probability,
            'probability_max': probability_max,
            'probability_cvar': probability_cvar,
            'safety_informative': safety_informative,
            'safety_override': safety_override,
            'risk_non_regressive': risk_non_regressive,
            'learned_non_regressive': learned_non_regressive,
            'legacy_selected': legacy_selected,
            'risk_mean': risk_mean,
            'risk_max': risk_max,
            'acceleration_max': acceleration_max,
            'jerk_max': jerk_max,
        }

    def _replace_teacher_safety_candidates(
            self, outputs, baseline_displacement, baseline_position, scene):
        """Use the exact eval-time DDIM proposals for safety calibration.

        The one-step stochastic teacher remains in ``ego_fut_ddim_preds`` and
        ``ego_fut_residual_preds`` for the unchanged v14 denoising objective.
        Candidate ranking/calibration instead receives baseline plus the same
        four fixed-noise, four-NFE proposals used at evaluation.  This removes
        the old train/test candidate-distribution mismatch.

        Residual modules are temporarily put in evaluation mode so dropout
        cannot make these proposals differ from inference.  Their training
        flags and all CPU/CUDA RNG states are restored before returning.
        """
        reference_displacement = baseline_displacement.detach()
        reference_position = baseline_position.detach()
        device = reference_position.device
        fork_devices = []
        if device.type == 'cuda':
            fork_devices = [
                device.index
                if device.index is not None else torch.cuda.current_device()]

        residual_modules = {}
        for name in VADHeadFutAttnResidualDDIM._residual_child_names():
            root = getattr(self, name, None)
            if root is not None:
                for module in root.modules():
                    residual_modules[id(module)] = module
        training_state = {
            module_id: module.training
            for module_id, module in residual_modules.items()
        }
        try:
            for module in residual_modules.values():
                module.training = False
            with torch.random.fork_rng(devices=fork_devices):
                with torch.no_grad():
                    sampled = VADHeadFutAttnResidualDDIM._ddim_sample(
                        self, reference_displacement, reference_position,
                        scene)
        finally:
            for module_id, module in residual_modules.items():
                module.training = training_state[module_id]

        candidate_displacement = sampled['ego_fut_candidates'].detach()
        candidate_risk = sampled['ego_fut_candidate_risk'].detach()
        if candidate_displacement.shape[2] != (
                self.safety_training_candidate_count):
            raise RuntimeError('inference/training candidate count drift')
        candidate_position = candidate_displacement.cumsum(dim=-2)
        baseline_for_candidates = reference_position[:, :, None].expand_as(
            candidate_position)
        # Recompute outside no_grad: features remain detached by the parent
        # method, but the inherited quality MLP must still receive ranking
        # gradients in order to preserve/improve L2 among safe candidates.
        quality_logits = self._candidate_quality(
            candidate_position, baseline_for_candidates, candidate_risk)

        expanded = dict(outputs)
        expanded.update({
            'ego_fut_candidates': candidate_displacement,
            'ego_fut_candidate_risk': candidate_risk,
            'ego_fut_candidate_quality_logits': quality_logits,
            # Preserve the differentiable v14 generator safety objective.
            'ego_fut_generated_risk': outputs['ego_fut_generated_risk'],
        })
        return expanded

    def _attach_safety_selection(
            self, outputs, baseline_position, replace_prediction, scene):
        candidate_displacement = outputs['ego_fut_candidates']
        candidate_position = candidate_displacement.cumsum(dim=-2)
        candidate_risk = outputs['ego_fut_candidate_risk']
        quality_logits = outputs['ego_fut_candidate_quality_logits']
        collision_logits = self._candidate_safety_logits(
            candidate_position, baseline_position, candidate_risk, scene=scene)
        selection = self._safety_first_selection(
            candidate_position, baseline_position, candidate_risk,
            quality_logits, collision_logits)

        result = dict(outputs)
        result.update({
            'ego_fut_candidate_collision_logits': collision_logits,
            'ego_fut_candidate_collision_probability': selection[
                'probability'],
            'ego_fut_candidate_safety_score': selection['safety_score'],
            'ego_fut_candidate_probability_max': selection[
                'probability_max'],
            'ego_fut_candidate_probability_cvar': selection[
                'probability_cvar'],
            'ego_fut_candidate_safety_informative': selection[
                'safety_informative'],
            'ego_fut_candidate_safety_override': selection[
                'safety_override'],
            'ego_fut_candidate_risk_non_regressive': selection[
                'risk_non_regressive'],
            'ego_fut_candidate_learned_non_regressive': selection[
                'learned_non_regressive'],
            'ego_fut_legacy_selected_index': selection['legacy_selected'],
            'ego_fut_candidate_legacy_costs': selection['legacy_costs'],
            'ego_fut_candidate_legacy_feasible': selection[
                'legacy_feasible'],
            'ego_fut_candidate_costs': selection['costs'],
            'ego_fut_candidate_feasible': selection['feasible'],
            'ego_fut_selected_index': selection['selected'],
        })
        if replace_prediction:
            selected_displacement = candidate_displacement.gather(
                2, selection['selected'][..., None, None, None].expand(
                    -1, -1, 1, self.fut_ts, 2)).squeeze(2)
            result['ego_fut_preds'] = selected_displacement
        return result

    def _teacher_forward(
            self, results, baseline_displacement, baseline_position, scene):
        outputs = super()._teacher_forward(
            results, baseline_displacement, baseline_position, scene)
        if outputs is None:
            return None
        outputs = self._replace_teacher_safety_candidates(
            outputs, baseline_displacement, baseline_position, scene)
        # Training keeps the parent's ego_fut_preds untouched; this selected
        # index is diagnostic and supplies labels to the calibration loss only.
        return self._attach_safety_selection(
            outputs, baseline_position.detach(), replace_prediction=False,
            scene=scene)

    def _ddim_sample(self, baseline_displacement, baseline_position, scene):
        outputs = super()._ddim_sample(
            baseline_displacement, baseline_position, scene)
        return self._attach_safety_selection(
            outputs, baseline_position, replace_prediction=True, scene=scene)
