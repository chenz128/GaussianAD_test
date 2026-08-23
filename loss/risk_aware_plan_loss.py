"""Losses used only by the isolated v13 risk-aware planner config."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import OPENOCC_LOSS
from .plan_loss import PlanAgentSATCollisionLoss, PlanLoss


class HardNegativePlanAgentSATCollisionLoss(PlanAgentSATCollisionLoss):
    """SAT collision loss that focuses on the closest valid agents.

    The legacy SAT loss averages over every valid agent/timestep pair.  With a
    large number of safe agents, the few safety-critical gradients are diluted.
    Here a normalized log-sum-exp is applied over agents at every timestep.  It
    is zero when every penalty is zero and smoothly approaches the hardest
    collision as ``temperature`` decreases.
    """

    def __init__(
            self,
            *args,
            temperature=0.2,
            timestep_weights=(0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
            **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = float(temperature)
        self.register_buffer(
            'timestep_weights',
            torch.as_tensor(timestep_weights, dtype=torch.float32))

    def forward(self, ego_fut_preds, attr_labels, fut_valid_flag,
                ego_fut_masks, agent_boxes=None):
        if (ego_fut_preds is None or not torch.is_tensor(ego_fut_preds)
                or ego_fut_preds.numel() == 0):
            return torch.zeros(
                (), device=getattr(ego_fut_preds, 'device', None))
        if attr_labels is None or not torch.is_tensor(attr_labels):
            return ego_fut_preds.new_zeros(())

        device = ego_fut_preds.device
        attr = attr_labels.to(device).float()
        if attr.dim() == 2:
            attr = attr[None]
        batch = ego_fut_preds.shape[0]
        timesteps = min(self.fut_ts, ego_fut_preds.shape[1])
        if self.timestep_weights.numel() < timesteps:
            raise ValueError(
                'timestep_weights must cover every predicted future step')
        t2, t3 = self.fut_ts * 2, self.fut_ts * 3

        total = ego_fut_preds.new_zeros(())
        sample_count = 0
        for batch_index in range(min(batch, attr.shape[0])):
            if not self._sample_valid(fut_valid_flag, batch_index):
                continue
            attr_b = attr[batch_index]
            if attr_b.shape[-1] < t3 + 10 + timesteps:
                continue

            future_trajs = attr_b[:, :t2].reshape(
                -1, self.fut_ts, 2)[:, :timesteps]
            future_mask = attr_b[:, t2:t3][:, :timesteps]
            lcf = attr_b[:, t3 + 1:t3 + 10]
            future_yaw_delta = attr_b[
                :, t3 + 10:t3 + 10 + timesteps]

            boxes_b = None
            if torch.is_tensor(agent_boxes):
                boxes_b = agent_boxes.to(device).float()
                if boxes_b.dim() >= 3:
                    boxes_b = boxes_b[batch_index]
                if boxes_b.dim() != 2 or boxes_b.shape[-1] < 7:
                    boxes_b = None

            if boxes_b is not None:
                agent_count = min(attr_b.shape[0], boxes_b.shape[0])
                future_trajs = future_trajs[:agent_count]
                future_mask = future_mask[:agent_count]
                lcf = lcf[:agent_count]
                future_yaw_delta = future_yaw_delta[:agent_count]
                boxes_b = boxes_b[:agent_count]
                agent_xy = boxes_b[:, 0:2]
                agent_width = boxes_b[:, 3].clamp_min(0.0)
                agent_length = boxes_b[:, 4].clamp_min(0.0)
                agent_yaw = -(boxes_b[:, 6] + math.pi / 2)
            else:
                agent_xy = lcf[:, 0:2]
                agent_width = lcf[:, 5].clamp_min(0.0)
                agent_length = lcf[:, 6].clamp_min(0.0)
                agent_yaw = -(lcf[:, 2] + math.pi / 2)

            if future_mask.sum() == 0:
                continue

            agent_future = (
                agent_xy[:, None, :] + future_trajs.cumsum(dim=1))
            ego_future = ego_fut_preds[
                batch_index, :timesteps].cumsum(dim=0)
            # Match PlanningMetric's fixed ego centre offset.
            ego_future = ego_future + ego_future.new_tensor([0.5, 0.0])

            agent_hx = 0.5 * agent_length[:, None, None]
            agent_hy = 0.5 * agent_width[:, None, None]
            future_yaw = (
                agent_yaw[:, None] + future_yaw_delta.cumsum(dim=1))
            cosine = torch.cos(future_yaw)[..., None]
            sine = torch.sin(future_yaw)[..., None]

            dx = (
                ego_future[:, 0][None, :, None]
                - agent_future[..., 0][..., None])
            dy = (
                ego_future[:, 1][None, :, None]
                - agent_future[..., 1][..., None])
            local_x = cosine * dx + sine * dy
            local_y = -sine * dx + cosine * dy

            projected_ego_x = (
                self.ego_hl * torch.abs(cosine)
                + self.ego_hw * torch.abs(sine))
            projected_ego_y = (
                self.ego_hl * torch.abs(sine)
                + self.ego_hw * torch.abs(cosine))
            separation_agent_x = (
                torch.abs(local_x) - (agent_hx + projected_ego_x))
            separation_agent_y = (
                torch.abs(local_y) - (agent_hy + projected_ego_y))

            projected_agent_x = (
                agent_hx * torch.abs(cosine)
                + agent_hy * torch.abs(sine))
            projected_agent_y = (
                agent_hx * torch.abs(sine)
                + agent_hy * torch.abs(cosine))
            separation_world_x = (
                torch.abs(dx) - (self.ego_hl + projected_agent_x))
            separation_world_y = (
                torch.abs(dy) - (self.ego_hw + projected_agent_y))

            separations = torch.stack([
                separation_agent_x.squeeze(-1),
                separation_agent_y.squeeze(-1),
                separation_world_x.squeeze(-1),
                separation_world_y.squeeze(-1),
            ], dim=-1)
            max_separation = separations.max(dim=-1).values
            penalty = torch.relu(-max_separation + self.safe_margin)

            mask = future_mask.to(penalty.dtype)
            if ego_fut_masks is not None and torch.is_tensor(ego_fut_masks):
                ego_mask = ego_fut_masks.to(device).float()
                if ego_mask.dim() >= 2:
                    ego_mask = ego_mask[batch_index]
                mask = mask * ego_mask[:timesteps][None, :]

            valid_agents = mask.sum(dim=0)
            temperature = max(self.temperature, 1e-4)
            scaled_penalty = penalty / temperature
            scaled_penalty = scaled_penalty.masked_fill(mask <= 0, -1e4)
            hard_penalty = temperature * (
                torch.logsumexp(scaled_penalty, dim=0)
                - valid_agents.clamp_min(1.0).log())
            hard_penalty = torch.where(
                valid_agents > 0,
                hard_penalty.clamp_min(0.0),
                hard_penalty.new_zeros(()))

            time_weight = self.timestep_weights[
                :timesteps].to(device=device, dtype=hard_penalty.dtype)
            valid_time = (valid_agents > 0).to(hard_penalty.dtype)
            weighted_time = time_weight * valid_time
            total = total + (
                hard_penalty * weighted_time).sum() / weighted_time.sum(
                    ).clamp_min(1.0)
            sample_count += 1

        if sample_count > 0:
            total = total / sample_count
        return self.loss_weight * total


@OPENOCC_LOSS.register_module()
class RiskAwarePlanLoss(PlanLoss):
    """Original PlanLoss with a hard-negative SAT collision guard."""

    def __init__(
            self,
            *args,
            col_temperature=0.2,
            col_timestep_weights=(0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
            **kwargs):
        super().__init__(*args, **kwargs)
        if self.plan_col_loss is not None and self.col_sat:
            self.plan_col_loss = HardNegativePlanAgentSATCollisionLoss(
                loss_weight=self.col_loss_weight,
                safe_margin=kwargs.get('col_safe_margin', 0.5),
                temperature=col_temperature,
                timestep_weights=col_timestep_weights)


@OPENOCC_LOSS.register_module()
class RiskAwareGateLoss(nn.Module):
    """Teach each gate to select the lower-error, lower-risk expert."""

    def __init__(self, weight=0.1, risk_weight=1.0):
        super().__init__()
        self.weight = float(weight)
        self.risk_weight = float(risk_weight)

    @staticmethod
    def _squeeze_annotations(value, target_dims):
        while value.dim() > target_dims and value.shape[1] == 1:
            value = value.squeeze(1)
        return value

    def forward(self, inputs):
        gate_logits = inputs['ego_fut_gate_logits']
        gate = inputs['ego_fut_gate']
        global_prediction = inputs['ego_fut_aux_preds']
        per_frame_prediction = inputs['ego_fut_per_frame_preds']
        global_risk = inputs['ego_fut_global_risk']
        per_frame_risk = inputs['ego_fut_per_frame_risk']

        target = self._squeeze_annotations(inputs['ego_fut_gt'], 3)
        valid_mask = self._squeeze_annotations(
            inputs['ego_fut_masks'], 2).to(gate_logits.dtype)
        command = self._squeeze_annotations(
            inputs['ego_fut_cmd'], 2).to(gate_logits.dtype)

        target_position = target[:, None, :, :2].cumsum(dim=2)
        global_error = torch.linalg.norm(
            global_prediction.cumsum(dim=2) - target_position, dim=-1)
        per_frame_error = torch.linalg.norm(
            per_frame_prediction.cumsum(dim=2) - target_position, dim=-1)

        global_score = global_error + self.risk_weight * global_risk
        per_frame_score = per_frame_error + self.risk_weight * per_frame_risk
        prefer_global = (
            global_score.detach() < per_frame_score.detach()).to(
                gate_logits.dtype)

        weight = command[:, :, None] * valid_mask[:, None, :]
        element_loss = F.binary_cross_entropy_with_logits(
            gate_logits, prefer_global, reduction='none')
        gate_loss = (
            (element_loss * weight).sum() / weight.sum().clamp_min(1.0))
        total = self.weight * torch.nan_to_num(gate_loss)

        selected_target_rate = (
            (prefer_global * weight).sum() / weight.sum().clamp_min(1.0))
        denominator = weight.sum().clamp_min(1.0)

        def weighted_mean(value):
            return (value * weight).sum() / denominator

        selected_gate_mean = weighted_mean(gate)
        selected_gate_abs_mean = weighted_mean(gate.abs())
        gate_positive_rate = weighted_mean((gate > 0.1).to(gate.dtype))
        gate_saturated_rate = weighted_mean((gate.abs() > 0.8).to(gate.dtype))
        global_risk_mean = weighted_mean(global_risk)
        per_frame_risk_mean = weighted_mean(per_frame_risk)
        risk_delta = global_risk - per_frame_risk
        risk_delta_mean = weighted_mean(risk_delta)
        risk_delta_abs_mean = weighted_mean(risk_delta.abs())
        risk_delta_variance = weighted_mean(
            (risk_delta - risk_delta_mean).square())

        log_values = {
            'loss_plan_gate_rank': total.detach().item(),
            'gate_prefer_global_rate': selected_target_rate.detach().item(),
            'gate_selected_mean': selected_gate_mean.detach().item(),
            'gate_selected_abs_mean': selected_gate_abs_mean.detach().item(),
            'gate_positive_rate_0p1': gate_positive_rate.detach().item(),
            'gate_saturated_rate_0p8': gate_saturated_rate.detach().item(),
            'risk_global_mean': global_risk_mean.detach().item(),
            'risk_per_frame_mean': per_frame_risk_mean.detach().item(),
            'risk_delta_mean': risk_delta_mean.detach().item(),
            'risk_delta_abs_mean': risk_delta_abs_mean.detach().item(),
            'risk_delta_std': risk_delta_variance.sqrt().detach().item(),
        }
        for mode_index in range(gate.shape[1]):
            mode_weight = weight[:, mode_index]
            mode_denominator = mode_weight.sum().clamp_min(1.0)
            mode_gate = gate[:, mode_index]
            mode_mean = (mode_gate * mode_weight).sum() / mode_denominator
            log_values[
                'gate_mode_{}_mean'.format(mode_index)] = (
                    mode_mean.detach().item())
        return total, log_values
