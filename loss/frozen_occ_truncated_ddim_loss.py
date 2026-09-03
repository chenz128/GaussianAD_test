"""Losses for the frozen-OCC truncated residual diffusion planner."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import OPENOCC_LOSS
from .safety_calibrated_residual_ddim_loss import MetricAlignedVehicleSAT


@OPENOCC_LOSS.register_module()
class FrozenOccTruncatedDDIMPlanLoss(nn.Module):
    """Train proposal coverage, metric-aligned safety and candidate ranking.

    GT boxes are consumed only here.  The planner forward receives prediction
    tensors exclusively, so no label can leak into an inference feature.  The
    deterministic candidates are detached by the planner; consequently this
    loss updates only residual-DiT and candidate-quality parameters.
    """

    def __init__(
            self,
            weight=1.0,
            anchor_candidate_count=3,
            coverage_weight=1.0,
            fde_weight=0.5,
            sat_weight=0.5,
            occ_risk_weight=0.1,
            dynamics_weight=0.05,
            rank_weight=0.25,
            diversity_weight=0.01,
            trust_region_weight=0.05,
            beta=0.5,
            softmin_temperature=0.25,
            rank_temperature=0.5,
            oracle_collision_weight=8.0,
            oracle_fde_weight=0.5,
            sat_safety_margin=0.35,
            sat_temperature=0.20,
            time_interval=0.5,
            max_acceleration=8.0,
            max_jerk=15.0,
            max_anchor_deviation=2.0,
            minimum_endpoint_separation=0.30,
            timestep_weights=(0.5, 0.75, 1.0, 1.0, 1.25, 1.5)):
        super().__init__()
        self.weight = float(weight)
        self.anchor_candidate_count = int(anchor_candidate_count)
        self.coverage_weight = float(coverage_weight)
        self.fde_weight = float(fde_weight)
        self.sat_weight = float(sat_weight)
        self.occ_risk_weight = float(occ_risk_weight)
        self.dynamics_weight = float(dynamics_weight)
        self.rank_weight = float(rank_weight)
        self.diversity_weight = float(diversity_weight)
        self.trust_region_weight = float(trust_region_weight)
        self.beta = float(beta)
        self.softmin_temperature = float(softmin_temperature)
        self.rank_temperature = float(rank_temperature)
        self.oracle_collision_weight = float(oracle_collision_weight)
        self.oracle_fde_weight = float(oracle_fde_weight)
        self.sat_safety_margin = float(sat_safety_margin)
        self.sat_temperature = float(sat_temperature)
        self.time_interval = float(time_interval)
        self.max_acceleration = float(max_acceleration)
        self.max_jerk = float(max_jerk)
        self.max_anchor_deviation = float(max_anchor_deviation)
        self.minimum_endpoint_separation = float(
            minimum_endpoint_separation)
        self.register_buffer(
            'timestep_weights',
            torch.as_tensor(timestep_weights, dtype=torch.float32))
        self.metric_sat = MetricAlignedVehicleSAT(
            fut_ts=len(timestep_weights),
            safety_margin=sat_safety_margin,
            target_temperature=sat_temperature,
            collision_margin=0.0,
            gt_collision_margin=0.0)

        if self.anchor_candidate_count < 1:
            raise ValueError('anchor_candidate_count must be positive')
        if self.softmin_temperature <= 0.0:
            raise ValueError('softmin_temperature must be positive')
        if self.rank_temperature <= 0.0:
            raise ValueError('rank_temperature must be positive')
        if self.sat_temperature <= 0.0:
            raise ValueError('sat_temperature must be positive')

    @staticmethod
    def _squeeze_annotation(value, target_dims):
        while (torch.is_tensor(value) and value.dim() > target_dims
               and value.shape[1] == 1):
            value = value.squeeze(1)
        return value

    @staticmethod
    def _safe_divide(numerator, denominator):
        epsilon = torch.finfo(denominator.dtype).eps
        return numerator / denominator.clamp_min(epsilon)

    def _prepare(self, inputs, candidates):
        target = self._squeeze_annotation(inputs['ego_fut_gt'], 3).to(
            device=candidates.device, dtype=candidates.dtype)[..., :2]
        mask = self._squeeze_annotation(inputs['ego_fut_masks'], 2).to(
            device=candidates.device, dtype=candidates.dtype)
        command = self._squeeze_annotation(inputs['ego_fut_cmd'], 2).to(
            device=candidates.device)
        modes = candidates.shape[1]
        if command.dim() == 1 or command.shape[-1] != modes:
            command = F.one_hot(
                command.reshape(-1).long(), num_classes=modes)
        mode_index = command.argmax(dim=-1)
        batch_index = torch.arange(candidates.shape[0], device=candidates.device)
        selected_candidates = candidates[batch_index, mode_index]
        selected_quality = inputs['ego_fut_candidate_quality_logits'][
            batch_index, mode_index]
        selected_risk = inputs['ego_fut_candidate_risk'][
            batch_index, mode_index].to(candidates.dtype)
        return target, mask, selected_candidates, selected_quality, selected_risk

    def _metric_clearance(self, inputs, candidates, target, mask):
        batch, candidate_count, timesteps = candidates.shape[:3]
        clearance = candidates.new_full(
            (batch, candidate_count, timesteps), 50.0)
        valid = candidates.new_zeros((batch, timesteps))
        gt_collision = torch.zeros(
            (batch, timesteps), device=candidates.device, dtype=torch.bool)
        attr_labels = inputs.get('attr_labels_planner')
        gt_boxes = inputs.get('gt_boxes')
        fut_valid_flag = inputs.get('fut_valid_flag')
        missing = [
            key for key, value in (
                ('attr_labels_planner', attr_labels),
                ('gt_boxes', gt_boxes),
                ('fut_valid_flag', fut_valid_flag))
            if value is None]
        if missing:
            raise KeyError(
                'metric-aligned planner supervision is missing: '
                + ', '.join(missing))

        for batch_index in range(batch):
            if not self.metric_sat._sample_valid(
                    fut_valid_flag, batch_index):
                continue
            attr = self.metric_sat._sample_item(
                attr_labels, batch_index, candidates.device, candidates.dtype)
            boxes = self.metric_sat._sample_item(
                gt_boxes, batch_index, candidates.device, candidates.dtype)
            sample_clearance, gt_clearance = (
                self.metric_sat._sample_clearance(
                    candidates[batch_index], target[batch_index],
                    attr, boxes, timesteps))
            clearance[batch_index] = sample_clearance
            sample_gt_collision = gt_clearance <= 0.0
            gt_collision[batch_index] = sample_gt_collision
            valid[batch_index] = (
                (mask[batch_index, :timesteps] > 0.5)
                & (~sample_gt_collision)).to(valid.dtype)
        return clearance, valid, gt_collision

    def _trajectory_terms(self, candidates, target, mask):
        candidate_position = candidates.cumsum(dim=-2)
        target_position = target.cumsum(dim=-2)[:, None]
        distance = torch.linalg.norm(
            candidate_position - target_position, dim=-1)
        valid = mask[:, None]
        ade = self._safe_divide(
            (distance * valid).sum(dim=-1), valid.sum(dim=-1))

        valid_count = mask.sum(dim=-1)
        last_index = (
            mask * torch.arange(
                1, mask.shape[-1] + 1,
                device=mask.device, dtype=mask.dtype)[None]
        ).argmax(dim=-1)
        gather_candidate = last_index[:, None, None, None].expand(
            -1, candidates.shape[1], 1, 2)
        endpoint = candidate_position.gather(
            2, gather_candidate).squeeze(2)
        gather_target = last_index[:, None, None].expand(-1, 1, 2)
        target_endpoint = target_position[:, 0].gather(
            1, gather_target).squeeze(1)
        fde = torch.linalg.norm(endpoint - target_endpoint[:, None], dim=-1)
        sample_valid = (valid_count > 0).to(candidates.dtype)
        return candidate_position, ade, fde, sample_valid, endpoint

    def _proposal_coverage(
            self, candidate_position, target, mask, ade, fde, sample_valid):
        generated = candidate_position[:, self.anchor_candidate_count:]
        generated_ade = ade[:, self.anchor_candidate_count:]
        generated_fde = fde[:, self.anchor_candidate_count:]
        if generated.shape[1] < 1:
            raise ValueError('planner returned no generated candidates')
        soft_assignment = torch.softmax(
            -generated_ade / self.softmin_temperature, dim=-1).detach()
        target_position = target.cumsum(dim=-2)[:, None]
        element = F.smooth_l1_loss(
            generated, target_position.expand_as(generated),
            beta=self.beta, reduction='none').sum(dim=-1)
        timestep_weight = self.timestep_weights[:element.shape[-1]].to(
            device=element.device, dtype=element.dtype)
        valid = mask[:, None] * timestep_weight[None, None]
        per_candidate = self._safe_divide(
            (element * valid).sum(dim=-1), valid.sum(dim=-1))
        coverage = self._safe_divide(
            (soft_assignment * per_candidate * sample_valid[:, None]).sum(),
            sample_valid.sum())
        endpoint = self._safe_divide(
            (soft_assignment * generated_fde
             * sample_valid[:, None]).sum(), sample_valid.sum())
        return coverage, endpoint

    def _safety_terms(self, clearance, valid, selected_risk):
        valid_candidate = valid[:, None]
        soft_collision = torch.sigmoid(
            (self.sat_safety_margin - clearance)
            / self.sat_temperature)
        per_candidate_sat = self._safe_divide(
            (soft_collision * valid_candidate).sum(dim=-1),
            valid_candidate.sum(dim=-1))
        generated_sat = per_candidate_sat[:, self.anchor_candidate_count:]
        sat_assignment = torch.softmax(
            -generated_sat / self.softmin_temperature, dim=-1).detach()
        sample_valid = (valid.sum(dim=-1) > 0).to(clearance.dtype)
        sat_loss = self._safe_divide(
            (sat_assignment * generated_sat
             * sample_valid[:, None]).sum(), sample_valid.sum())

        risk = torch.nan_to_num(
            selected_risk, nan=1.0, posinf=1.0, neginf=0.0)
        per_candidate_risk = self._safe_divide(
            (risk * valid_candidate).sum(dim=-1),
            valid_candidate.sum(dim=-1))
        generated_risk = per_candidate_risk[:, self.anchor_candidate_count:]
        risk_assignment = torch.softmax(
            -generated_risk / self.softmin_temperature, dim=-1).detach()
        risk_loss = self._safe_divide(
            (risk_assignment * generated_risk
             * sample_valid[:, None]).sum(), sample_valid.sum())
        return (
            sat_loss, risk_loss, soft_collision,
            per_candidate_sat, per_candidate_risk)

    def _dynamics(self, candidates, mask):
        generated = candidates[:, self.anchor_candidate_count:]
        velocity = generated / self.time_interval
        if generated.shape[-2] < 2:
            return generated.new_zeros(())
        acceleration = torch.diff(velocity, dim=-2) / self.time_interval
        acceleration_penalty = F.relu(
            torch.linalg.norm(acceleration, dim=-1)
            - self.max_acceleration).square()
        acceleration_mask = mask[:, 1:] * mask[:, :-1]
        acceleration_loss = self._safe_divide(
            (acceleration_penalty * acceleration_mask[:, None]).sum(),
            acceleration_mask.sum() * generated.shape[1])
        if generated.shape[-2] < 3:
            return acceleration_loss
        jerk = torch.diff(acceleration, dim=-2) / self.time_interval
        jerk_penalty = F.relu(
            torch.linalg.norm(jerk, dim=-1) - self.max_jerk).square()
        jerk_mask = mask[:, 2:] * mask[:, 1:-1] * mask[:, :-2]
        jerk_loss = self._safe_divide(
            (jerk_penalty * jerk_mask[:, None]).sum(),
            jerk_mask.sum() * generated.shape[1])
        return acceleration_loss + 0.25 * jerk_loss

    def _diversity_and_trust(self, candidate_position, endpoint, mask):
        generated = candidate_position[:, self.anchor_candidate_count:]
        generated_endpoint = endpoint[:, self.anchor_candidate_count:]
        generated_count = generated.shape[1]
        sample_valid = (mask.sum(dim=-1) > 0).to(generated.dtype)
        if generated_count > 1:
            pair = torch.triu_indices(
                generated_count, generated_count, offset=1,
                device=generated.device)
            separation = torch.linalg.norm(
                generated_endpoint[:, pair[0]]
                - generated_endpoint[:, pair[1]], dim=-1)
            diversity = self._safe_divide(
                (F.relu(self.minimum_endpoint_separation - separation)
                 * sample_valid[:, None]).sum(),
                sample_valid.sum() * separation.shape[-1])
        else:
            diversity = generated.new_zeros(())

        baseline = candidate_position[:, :1]
        deviation = torch.linalg.norm(generated - baseline, dim=-1)
        trust_penalty = F.relu(
            deviation - self.max_anchor_deviation).square()
        trust = self._safe_divide(
            (trust_penalty * mask[:, None]).sum(),
            mask.sum() * generated_count)
        return diversity, trust

    def _ranking(
            self, quality_logits, ade, fde, soft_collision,
            valid, sample_valid):
        valid_candidate = valid[:, None]
        collision_event = soft_collision.masked_fill(
            valid_candidate <= 0, 0.0).max(dim=-1).values
        oracle_cost = (
            ade
            + self.oracle_fde_weight * fde
            + self.oracle_collision_weight * collision_event).detach()
        target_distribution = torch.softmax(
            -oracle_cost / self.rank_temperature, dim=-1)
        log_probability = F.log_softmax(quality_logits, dim=-1)
        per_sample = -(target_distribution * log_probability).sum(dim=-1)
        rank = self._safe_divide(
            (per_sample * sample_valid).sum(), sample_valid.sum())
        oracle_best = oracle_cost.argmin(dim=-1)
        return rank, oracle_best, collision_event

    def forward(self, inputs):
        candidates = inputs.get('ego_fut_candidates')
        quality_logits = inputs.get('ego_fut_candidate_quality_logits')
        candidate_risk = inputs.get('ego_fut_candidate_risk')
        if candidates is None or quality_logits is None or candidate_risk is None:
            reference = inputs.get('ego_fut_base_preds')
            if reference is None:
                raise KeyError('frozen-OCC diffusion outputs are missing')
            zero = reference.new_zeros(())
            return zero, {'loss_frozen_occ_truncated_ddim': 0.0}
        if candidates.shape[2] <= self.anchor_candidate_count:
            raise ValueError('candidate pool contains no residual proposal')

        target, mask, selected_candidates, selected_quality, selected_risk = (
            self._prepare(inputs, candidates))
        (candidate_position, ade, fde,
         sample_valid, endpoint) = self._trajectory_terms(
             selected_candidates, target, mask)
        coverage, endpoint_loss = self._proposal_coverage(
            candidate_position, target, mask, ade, fde, sample_valid)
        clearance, safety_valid, gt_collision = self._metric_clearance(
            inputs, selected_candidates, target, mask)
        (sat_loss, risk_loss, soft_collision,
         per_candidate_sat, per_candidate_risk) = self._safety_terms(
             clearance, safety_valid, selected_risk)
        dynamics = self._dynamics(selected_candidates, mask)
        diversity, trust = self._diversity_and_trust(
            candidate_position, endpoint, mask)
        rank, oracle_best, collision_event = self._ranking(
            selected_quality, ade, fde, soft_collision,
            safety_valid, sample_valid)

        total = self.weight * (
            self.coverage_weight * coverage
            + self.fde_weight * endpoint_loss
            + self.sat_weight * sat_loss
            + self.occ_risk_weight * risk_loss
            + self.dynamics_weight * dynamics
            + self.rank_weight * rank
            + self.diversity_weight * diversity
            + self.trust_region_weight * trust)
        total = torch.nan_to_num(total)
        generated_best = (
            oracle_best >= self.anchor_candidate_count).to(total.dtype)
        log_values = {
            'loss_frozen_occ_truncated_ddim': total.detach().item(),
            'loss_v16_coverage': coverage.detach().item(),
            'loss_v16_fde': endpoint_loss.detach().item(),
            'loss_v16_sat': sat_loss.detach().item(),
            'loss_v16_occ_risk': risk_loss.detach().item(),
            'loss_v16_dynamics': dynamics.detach().item(),
            'loss_v16_rank': rank.detach().item(),
            'loss_v16_diversity': diversity.detach().item(),
            'loss_v16_trust': trust.detach().item(),
            'v16_oracle_generated_best_rate': generated_best.mean().item(),
            'v16_oracle_collision_event': collision_event.detach().mean().item(),
            'v16_gt_collision_mask_rate': gt_collision.float().mean().item(),
            'v16_sat_candidate_mean': per_candidate_sat.detach().mean().item(),
            'v16_occ_candidate_mean': per_candidate_risk.detach().mean().item(),
        }
        selected = inputs.get('ego_fut_selected_index')
        if selected is not None:
            log_values['v16_selector_anchor_rate'] = (
                (selected < self.anchor_candidate_count)
                .float().mean().detach().item())
            log_values['v16_selector_exact_baseline_rate'] = (
                (selected == 0).float().mean().detach().item())
        return total, log_values
