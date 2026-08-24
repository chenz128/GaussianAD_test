"""Metric-aligned safety calibration for the v15 residual DDIM planner."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import OPENOCC_LOSS
from .residual_ddim_plan_loss import ResidualDDIMPlanLoss


class MetricAlignedVehicleSAT(nn.Module):
    """Create candidate collision labels with the planning metric's geometry.

    Important differences from the legacy training SAT loss are deliberate:

    * only nuScenes vehicle categories are included, matching ``box_col``;
    * the fixed ego centre offset ``(+0.5, 0)`` is applied;
    * timesteps where the GT ego trajectory already collides are excluded, as
      done by ``PlanningMetric.evaluate_coll``;
    * labels are detached targets for a safety scorer, not an auxiliary force
      that biases the v14 diffusion score.
    """

    def __init__(
            self,
            fut_ts=6,
            ego_width=1.85,
            ego_length=4.084,
            safety_margin=0.5,
            target_temperature=0.25,
            collision_margin=0.0,
            vehicle_category_ids=(14, 15, 16, 17, 18,
                                  19, 20, 21, 22, 23)):
        super().__init__()
        self.fut_ts = int(fut_ts)
        self.ego_half_width = 0.5 * float(ego_width)
        self.ego_half_length = 0.5 * float(ego_length)
        self.safety_margin = float(safety_margin)
        self.target_temperature = float(target_temperature)
        self.collision_margin = float(collision_margin)
        self.register_buffer(
            'vehicle_category_ids',
            torch.as_tensor(vehicle_category_ids, dtype=torch.long))

    @staticmethod
    def _sample_valid(value, batch_index):
        if value is None:
            return True
        if isinstance(value, (list, tuple)):
            value = value[batch_index] if batch_index < len(value) else value[0]
        elif torch.is_tensor(value) and value.dim() >= 1:
            if value.shape[0] > batch_index:
                value = value[batch_index]
        if torch.is_tensor(value):
            value = value.reshape(-1)
            return bool(value[0].item()) if value.numel() else True
        return bool(value)

    @staticmethod
    def _sample_item(value, batch_index, device, dtype):
        if value is None:
            return None
        # Handle common DataContainer/list/LiDARInstance3DBoxes forms without
        # importing dataset-specific classes into the loss module.
        if hasattr(value, 'data') and not torch.is_tensor(value):
            value = value.data
        if isinstance(value, (list, tuple)):
            if not value:
                return None
            value = value[batch_index] if batch_index < len(value) else value[0]
            if isinstance(value, (list, tuple)) and len(value) == 1:
                value = value[0]
        if hasattr(value, 'tensor'):
            value = value.tensor
        if not torch.is_tensor(value):
            try:
                value = torch.as_tensor(value)
            except (TypeError, ValueError):
                return None
        if value.dim() >= 3:
            if value.shape[0] <= batch_index:
                return None
            value = value[batch_index]
        if value.dim() != 2:
            return None
        return value.to(device=device, dtype=dtype)

    def _sample_clearance(
            self, candidate_displacement, target_displacement,
            attr_value, boxes_value, timesteps):
        """Return candidate clearance and whether the GT ego collides."""
        device = candidate_displacement.device
        dtype = candidate_displacement.dtype
        candidate_count = candidate_displacement.shape[0]
        far = candidate_displacement.new_full(
            (candidate_count, timesteps), 50.0)
        gt_far = candidate_displacement.new_full((timesteps,), 50.0)
        required_width = self.fut_ts * 4 + 10
        if attr_value is None:
            raise KeyError('attr_labels_planner is required for v15 SAT labels')
        if attr_value.shape[-1] < required_width:
            raise ValueError(
                'attr_labels_planner has an incompatible layout: expected '
                f'at least {required_width} values, got {attr_value.shape[-1]}')

        t2, t3 = self.fut_ts * 2, self.fut_ts * 3
        future_traj = attr_value[:, :t2].reshape(
            -1, self.fut_ts, 2)[:, :timesteps]
        future_mask = attr_value[:, t2:t3][:, :timesteps] > 0.5
        lcf = attr_value[:, t3 + 1:t3 + 10]
        future_yaw_delta = attr_value[
            :, t3 + 10:t3 + 10 + self.fut_ts][:, :timesteps]

        category = attr_value[:, t3 + 9].round().long()
        vehicle_ids = self.vehicle_category_ids.to(device=device)
        vehicle = (category[:, None] == vehicle_ids[None]).any(dim=-1)
        future_mask = future_mask & vehicle[:, None]
        if not future_mask.any():
            return far, gt_far <= self.collision_margin

        boxes = boxes_value
        if boxes is not None and boxes.shape[-1] >= 7:
            agent_count = min(attr_value.shape[0], boxes.shape[0])
            future_traj = future_traj[:agent_count]
            future_mask = future_mask[:agent_count]
            future_yaw_delta = future_yaw_delta[:agent_count]
            boxes = boxes[:agent_count]
            agent_xy = boxes[:, 0:2]
            agent_width = boxes[:, 3].clamp_min(0.0)
            agent_length = boxes[:, 4].clamp_min(0.0)
            agent_yaw = -(boxes[:, 6] + math.pi / 2.0)
        else:
            agent_xy = lcf[:, 0:2]
            agent_width = lcf[:, 5].clamp_min(0.0)
            agent_length = lcf[:, 6].clamp_min(0.0)
            agent_yaw = -(lcf[:, 2] + math.pi / 2.0)

        agent_future = agent_xy[:, None] + future_traj.cumsum(dim=1)
        future_yaw = agent_yaw[:, None] + future_yaw_delta.cumsum(dim=1)
        cosine = torch.cos(future_yaw)
        sine = torch.sin(future_yaw)

        # Evaluate all candidates and the GT ego trajectory in one SAT pass.
        candidate_position = candidate_displacement[
            :, :timesteps].cumsum(dim=-2)
        target_position = target_displacement[:timesteps].cumsum(dim=-2)
        ego_position = torch.cat([
            candidate_position,
            target_position[None],
        ], dim=0)
        ego_position = ego_position + ego_position.new_tensor([0.5, 0.0])

        dx = (
            ego_position[:, None, :, 0]
            - agent_future[None, :, :, 0])
        dy = (
            ego_position[:, None, :, 1]
            - agent_future[None, :, :, 1])
        cosine = cosine[None]
        sine = sine[None]
        local_x = cosine * dx + sine * dy
        local_y = -sine * dx + cosine * dy

        agent_half_x = 0.5 * agent_length[None, :, None]
        agent_half_y = 0.5 * agent_width[None, :, None]
        projected_ego_x = (
            self.ego_half_length * cosine.abs()
            + self.ego_half_width * sine.abs())
        projected_ego_y = (
            self.ego_half_length * sine.abs()
            + self.ego_half_width * cosine.abs())
        separation_agent_x = (
            local_x.abs() - (agent_half_x + projected_ego_x))
        separation_agent_y = (
            local_y.abs() - (agent_half_y + projected_ego_y))

        projected_agent_x = (
            agent_half_x * cosine.abs() + agent_half_y * sine.abs())
        projected_agent_y = (
            agent_half_x * sine.abs() + agent_half_y * cosine.abs())
        separation_world_x = (
            dx.abs() - (self.ego_half_length + projected_agent_x))
        separation_world_y = (
            dy.abs() - (self.ego_half_width + projected_agent_y))

        signed_separation = torch.stack([
            separation_agent_x,
            separation_agent_y,
            separation_world_x,
            separation_world_y,
        ], dim=-1).max(dim=-1).values
        valid_agent = future_mask[None].expand(
            ego_position.shape[0], -1, -1)
        signed_separation = signed_separation.masked_fill(
            ~valid_agent, float('inf'))
        clearance = signed_separation.min(dim=1).values
        has_vehicle = future_mask.any(dim=0)[None]
        clearance = torch.where(
            has_vehicle, clearance,
            clearance.new_full((), 50.0))
        return clearance[:-1], clearance[-1] <= self.collision_margin

    @torch.no_grad()
    def forward(
            self, candidate_displacement, target_displacement,
            ego_mask, attr_labels, gt_boxes=None, fut_valid_flag=None):
        batch, candidate_count, timesteps = candidate_displacement.shape[:3]
        timesteps = min(timesteps, self.fut_ts)
        soft_target = candidate_displacement.new_zeros(
            (batch, candidate_count, timesteps))
        hard_target = torch.zeros(
            (batch, candidate_count, timesteps),
            device=candidate_displacement.device, dtype=torch.bool)
        valid = candidate_displacement.new_zeros(
            (batch, candidate_count, timesteps))
        gt_collision = torch.zeros(
            (batch, timesteps),
            device=candidate_displacement.device, dtype=torch.bool)

        for batch_index in range(batch):
            if not self._sample_valid(fut_valid_flag, batch_index):
                continue
            attr = self._sample_item(
                attr_labels, batch_index,
                candidate_displacement.device, candidate_displacement.dtype)
            boxes = self._sample_item(
                gt_boxes, batch_index,
                candidate_displacement.device, candidate_displacement.dtype)
            clearance, gt_coll = self._sample_clearance(
                candidate_displacement[batch_index],
                target_displacement[batch_index],
                attr, boxes, timesteps)
            gt_collision[batch_index] = gt_coll
            hard = clearance <= self.collision_margin
            soft = torch.sigmoid(
                (self.safety_margin - clearance)
                / max(self.target_temperature, 1e-4))
            # With no valid vehicle the clearance sentinel represents an
            # unambiguously safe target and naturally maps to probability zero.
            hard_target[batch_index] = hard
            soft_target[batch_index] = soft
            metric_valid = (
                ego_mask[batch_index, :timesteps] > 0.5) & (~gt_coll)
            valid[batch_index] = metric_valid[None].expand(
                candidate_count, -1).to(valid.dtype)

        return {
            'soft_target': soft_target,
            'hard_target': hard_target,
            'valid': valid,
            'gt_collision': gt_collision,
        }


@OPENOCC_LOSS.register_module()
class SafetyCalibratedResidualDDIMPlanLoss(ResidualDDIMPlanLoss):
    """The original v14 objective plus detached SAT safety calibration."""

    def __init__(
            self,
            *args,
            safety_calibration_weight=0.25,
            safety_brier_weight=0.25,
            safety_rank_weight=0.1,
            safety_positive_weight=4.0,
            sat_safety_margin=0.5,
            sat_target_temperature=0.25,
            sat_collision_margin=0.0,
            **kwargs):
        super().__init__(*args, **kwargs)
        self.safety_calibration_weight = float(safety_calibration_weight)
        self.safety_brier_weight = float(safety_brier_weight)
        self.safety_rank_weight = float(safety_rank_weight)
        self.safety_positive_weight = float(safety_positive_weight)
        self.metric_sat = MetricAlignedVehicleSAT(
            fut_ts=len(self.timestep_weights),
            safety_margin=sat_safety_margin,
            target_temperature=sat_target_temperature,
            collision_margin=sat_collision_margin)

    def _calibration_loss(self, inputs, target, mask, command):
        logits = inputs.get('ego_fut_candidate_collision_logits')
        candidates = inputs.get('ego_fut_candidates')
        if logits is None or candidates is None:
            raise KeyError(
                'v15 safety calibration requires candidate collision logits '
                'and candidate trajectories')
        missing_annotations = [
            key for key in (
                'attr_labels_planner', 'gt_boxes', 'fut_valid_flag')
            if inputs.get(key) is None]
        if missing_annotations:
            raise KeyError(
                'v15 metric-aligned SAT annotations are missing: '
                + ', '.join(missing_annotations))
        batch = logits.shape[0]
        mode_index = command.argmax(dim=-1)
        batch_index = torch.arange(batch, device=logits.device)
        selected_logits = logits[batch_index, mode_index]
        selected_candidates = candidates[batch_index, mode_index]

        sat = self.metric_sat(
            selected_candidates,
            target,
            mask,
            inputs['attr_labels_planner'],
            gt_boxes=inputs['gt_boxes'],
            fut_valid_flag=inputs['fut_valid_flag'])
        soft_target = sat['soft_target'].to(selected_logits.dtype)
        hard_target = sat['hard_target']
        valid = sat['valid'].to(selected_logits.dtype)

        timestep_weight = self.timestep_weights[
            :selected_logits.shape[-1]].to(
                device=selected_logits.device, dtype=selected_logits.dtype)
        base_weight = valid * timestep_weight[None, None]
        positive_weight = (
            1.0
            + (self.safety_positive_weight - 1.0) * soft_target)
        bce_weight = base_weight * positive_weight
        element_bce = F.binary_cross_entropy_with_logits(
            selected_logits, soft_target, reduction='none')
        bce = self._safe_divide(
            (element_bce * bce_weight).sum(), bce_weight.sum())

        probability = selected_logits.sigmoid()
        brier = self._safe_divide(
            ((probability - soft_target).square() * base_weight).sum(),
            base_weight.sum())

        valid_bool = valid > 0
        target_for_max = soft_target.masked_fill(~valid_bool, 0.0)
        probability_for_max = probability.masked_fill(~valid_bool, 0.0)
        target_risk = target_for_max.max(dim=-1).values
        predicted_risk = probability_for_max.max(dim=-1).values
        best_candidate = target_risk.argmin(dim=-1)
        valid_sample = (valid.sum(dim=(-1, -2)) > 0).to(logits.dtype)
        rank_element = F.cross_entropy(
            -5.0 * predicted_risk,
            best_candidate,
            reduction='none')
        rank = self._safe_divide(
            (rank_element * valid_sample).sum(), valid_sample.sum())

        calibration = bce + self.safety_brier_weight * brier
        extra = (
            self.safety_calibration_weight * calibration
            + self.safety_rank_weight * rank)
        return extra, {
            'bce': bce,
            'brier': brier,
            'rank': rank,
            'probability': probability,
            'hard_target': hard_target,
            'valid': valid,
            'soft_target': soft_target,
            'best_candidate': best_candidate,
        }

    def forward(self, inputs):
        base_total, log_values = super().forward(inputs)
        diffusion_prediction = inputs.get('ego_fut_ddim_preds')
        if diffusion_prediction is None:
            return base_total, log_values
        target, mask, command = self._prepare_annotations(
            inputs, diffusion_prediction)
        extra, details = self._calibration_loss(
            inputs, target, mask, command)
        total = torch.nan_to_num(base_total + extra)

        valid = details['valid']
        hard = details['hard_target'].to(valid.dtype)
        probability = details['probability']
        denominator = valid.sum()
        hard_rate = self._safe_divide((hard * valid).sum(), denominator)
        probability_mean = self._safe_divide(
            (probability * valid).sum(), denominator)
        baseline_hard_rate = self._safe_divide(
            (hard[:, 0] * valid[:, 0]).sum(), valid[:, 0].sum())
        if hard.shape[1] > 1:
            generated_hard_rate = self._safe_divide(
                (hard[:, 1:] * valid[:, 1:]).sum(), valid[:, 1:].sum())
        else:
            generated_hard_rate = hard_rate.new_zeros(())

        trajectory_collision = (
            (details['hard_target'] & (valid > 0)).any(dim=-1))
        sample_valid = valid.sum(dim=(-1, -2)) > 0
        oracle_collision = trajectory_collision.all(dim=-1).to(valid.dtype)
        oracle_collision_rate = self._safe_divide(
            (oracle_collision * sample_valid.to(valid.dtype)).sum(),
            sample_valid.to(valid.dtype).sum())

        selected_collision_rate = hard_rate.new_zeros(())
        command_baseline_rate = hard_rate.new_zeros(())
        all_infeasible_rate = hard_rate.new_zeros(())
        selected_histogram = []
        selected = inputs.get('ego_fut_selected_index')
        if selected is not None:
            batch = selected.shape[0]
            mode_index = command.argmax(dim=-1)
            selected_for_command = selected[
                torch.arange(batch, device=selected.device), mode_index]
            selected_for_command = selected_for_command.clamp(
                0, trajectory_collision.shape[1] - 1)
            selected_collision = trajectory_collision.gather(
                1, selected_for_command[:, None]).squeeze(1).to(valid.dtype)
            selected_collision_rate = self._safe_divide(
                (selected_collision * sample_valid.to(valid.dtype)).sum(),
                sample_valid.to(valid.dtype).sum())
            command_baseline_rate = self._safe_divide(
                ((selected_for_command == 0).to(valid.dtype)
                 * sample_valid.to(valid.dtype)).sum(),
                sample_valid.to(valid.dtype).sum())
            for candidate_index in range(trajectory_collision.shape[1]):
                selected_histogram.append(self._safe_divide(
                    ((selected_for_command == candidate_index).to(valid.dtype)
                     * sample_valid.to(valid.dtype)).sum(),
                    sample_valid.to(valid.dtype).sum()))

            feasible = inputs.get('ego_fut_candidate_feasible')
            if feasible is not None:
                command_feasible = feasible[
                    torch.arange(batch, device=feasible.device), mode_index]
                all_infeasible = (~command_feasible.any(dim=-1)).to(valid.dtype)
                all_infeasible_rate = self._safe_divide(
                    (all_infeasible * sample_valid.to(valid.dtype)).sum(),
                    sample_valid.to(valid.dtype).sum())

        log_values.update({
            'loss_plan_safety_calibrated_ddim': total.detach().item(),
            'loss_safety_calibration_bce': details['bce'].detach().item(),
            'loss_safety_calibration_brier': details['brier'].detach().item(),
            'loss_safety_candidate_rank': details['rank'].detach().item(),
            'sat_candidate_collision_rate': hard_rate.detach().item(),
            'sat_baseline_collision_rate': baseline_hard_rate.detach().item(),
            'sat_generated_collision_rate': generated_hard_rate.detach().item(),
            'sat_oracle_all_candidates_collision_rate': (
                oracle_collision_rate.detach().item()),
            'sat_selected_trajectory_collision_rate': (
                selected_collision_rate.detach().item()),
            'safety_probability_mean': probability_mean.detach().item(),
            'safety_command_baseline_selected_rate': (
                command_baseline_rate.detach().item()),
            'safety_all_infeasible_rate': all_infeasible_rate.detach().item(),
        })
        for candidate_index, rate in enumerate(selected_histogram):
            log_values[
                f'safety_selected_candidate_{candidate_index}_rate'] = (
                    rate.detach().item())
        return total, log_values
