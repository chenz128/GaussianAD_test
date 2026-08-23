"""Additional losses for the v14 baseline-conditioned residual DDIM planner."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import OPENOCC_LOSS


@OPENOCC_LOSS.register_module()
class ResidualDDIMPlanLoss(nn.Module):
    """Masked residual, position, safety, dynamics and ranking objectives.

    Only the navigation-command mode selected by ``ego_fut_cmd`` contributes
    to any supervised term.  Candidate ranking labels are generated from the
    detached future-position error and Gaussian risk, so the selector cannot
    reduce its loss by changing the label cost itself.
    """

    def __init__(
            self,
            weight=1.0,
            diffusion_weight=1.0,
            position_weight=0.5,
            fde_weight=0.25,
            safety_weight=0.1,
            dynamics_weight=0.05,
            rank_weight=0.1,
            rank_risk_weight=2.0,
            beta=0.5,
            time_interval=0.5,
            max_acceleration=8.0,
            max_jerk=15.0,
            timestep_weights=(0.5, 0.75, 1.0, 1.0, 1.25, 1.5)):
        super().__init__()
        self.weight = float(weight)
        self.diffusion_weight = float(diffusion_weight)
        self.position_weight = float(position_weight)
        self.fde_weight = float(fde_weight)
        self.safety_weight = float(safety_weight)
        self.dynamics_weight = float(dynamics_weight)
        self.rank_weight = float(rank_weight)
        self.rank_risk_weight = float(rank_risk_weight)
        self.beta = float(beta)
        self.time_interval = float(time_interval)
        self.max_acceleration = float(max_acceleration)
        self.max_jerk = float(max_jerk)
        self.register_buffer(
            'timestep_weights',
            torch.as_tensor(timestep_weights, dtype=torch.float32))

    @staticmethod
    def _squeeze_annotation(value, target_dims):
        while value.dim() > target_dims and value.shape[1] == 1:
            value = value.squeeze(1)
        return value

    @staticmethod
    def _safe_divide(numerator, denominator):
        return numerator / denominator.clamp_min(1.0)

    def _prepare_annotations(self, inputs, prediction):
        target = self._squeeze_annotation(inputs['ego_fut_gt'], 3).to(
            device=prediction.device, dtype=prediction.dtype)[..., :2]
        mask = self._squeeze_annotation(inputs['ego_fut_masks'], 2).to(
            device=prediction.device, dtype=prediction.dtype)
        command = self._squeeze_annotation(inputs['ego_fut_cmd'], 2).to(
            device=prediction.device)
        modes = prediction.shape[1]
        if command.dim() == 1 or command.shape[-1] != modes:
            command = F.one_hot(
                command.reshape(-1).long(), num_classes=modes)
        command = command.to(dtype=prediction.dtype)
        if mask.dim() == 1:
            mask = mask[None]
        return target, mask, command

    def _masked_mean(self, value, command, mask):
        weight = command[:, :, None] * mask[:, None, :]
        return self._safe_divide(
            (value * weight).sum(), weight.sum())

    def _diffusion_loss(self, inputs, command, mask):
        prediction = inputs['ego_fut_ddim_preds']
        target = inputs['ego_fut_ddim_targets'].to(prediction.dtype)
        element = F.smooth_l1_loss(
            prediction, target, beta=self.beta, reduction='none').sum(dim=-1)
        return self._masked_mean(element, command, mask)

    def _trajectory_loss(self, prediction, target, command, mask):
        prediction_position = prediction.cumsum(dim=-2)
        target_position = target.cumsum(dim=-2)[:, None]
        element = F.smooth_l1_loss(
            prediction_position, target_position,
            beta=self.beta, reduction='none').sum(dim=-1)
        timestep_weight = self.timestep_weights[:prediction.shape[-2]].to(
            device=prediction.device, dtype=prediction.dtype)
        weighted_mask = mask * timestep_weight[None]
        position_loss = self._masked_mean(
            element, command, weighted_mask)

        valid_count = mask.sum(dim=-1)
        last_index = (
            mask * torch.arange(
                1, mask.shape[-1] + 1,
                device=mask.device, dtype=mask.dtype)[None]
        ).argmax(dim=-1)
        gather_prediction = last_index[:, None, None, None].expand(
            -1, prediction.shape[1], 1, 2)
        prediction_endpoint = prediction_position.gather(
            2, gather_prediction).squeeze(2)
        gather_target = last_index[:, None, None].expand(-1, 1, 2)
        target_endpoint = target_position[:, 0].gather(
            1, gather_target).squeeze(1)
        endpoint_error = F.smooth_l1_loss(
            prediction_endpoint, target_endpoint[:, None],
            beta=self.beta, reduction='none').sum(dim=-1)
        endpoint_weight = command * (valid_count > 0).to(command.dtype)[:, None]
        endpoint_loss = self._safe_divide(
            (endpoint_error * endpoint_weight).sum(), endpoint_weight.sum())
        return position_loss, endpoint_loss

    def _safety_loss(self, inputs, command, mask):
        risk = inputs.get('ego_fut_generated_risk')
        if risk is None:
            return command.new_zeros(())
        risk = torch.nan_to_num(
            risk.to(command.dtype), nan=1.0, posinf=1.0, neginf=0.0)
        timestep_weight = self.timestep_weights[:risk.shape[-1]].to(
            device=risk.device, dtype=risk.dtype)
        return self._masked_mean(
            risk, command, mask * timestep_weight[None])

    def _dynamics_loss(self, prediction, command, mask):
        velocity = prediction / self.time_interval
        if prediction.shape[-2] < 2:
            return prediction.new_zeros(())
        acceleration = torch.diff(velocity, dim=-2) / self.time_interval
        acceleration_norm = torch.linalg.norm(acceleration, dim=-1)
        acceleration_penalty = F.relu(
            acceleration_norm - self.max_acceleration).square()
        acceleration_mask = mask[:, 1:] * mask[:, :-1]
        acceleration_loss = self._masked_mean(
            acceleration_penalty, command, acceleration_mask)

        if prediction.shape[-2] < 3:
            return acceleration_loss
        jerk = torch.diff(acceleration, dim=-2) / self.time_interval
        jerk_norm = torch.linalg.norm(jerk, dim=-1)
        jerk_penalty = F.relu(jerk_norm - self.max_jerk).square()
        jerk_mask = mask[:, 2:] * mask[:, 1:-1] * mask[:, :-2]
        jerk_loss = self._masked_mean(jerk_penalty, command, jerk_mask)
        return acceleration_loss + 0.25 * jerk_loss

    def _ranking_loss(self, inputs, target, command, mask):
        candidates = inputs.get('ego_fut_candidates')
        quality_logits = inputs.get('ego_fut_candidate_quality_logits')
        if candidates is None or quality_logits is None:
            return command.new_zeros(()), None
        candidate_count = candidates.shape[2]
        if candidate_count < 2:
            return command.new_zeros(()), None

        candidate_position = candidates.cumsum(dim=-2)
        target_position = target.cumsum(dim=-2)[:, None, None]
        distance = torch.linalg.norm(
            candidate_position - target_position, dim=-1)
        valid = mask[:, None, None]
        ade = self._safe_divide(
            (distance * valid).sum(dim=-1), valid.sum(dim=-1))
        risk = inputs.get('ego_fut_candidate_risk')
        if risk is None:
            risk_cost = ade.new_zeros(ade.shape)
        else:
            risk = risk.to(ade.dtype)
            risk_cost = self._safe_divide(
                (risk * valid).sum(dim=-1), valid.sum(dim=-1))
        target_cost = (ade + self.rank_risk_weight * risk_cost).detach()
        best_candidate = target_cost.argmin(dim=-1)
        cross_entropy = F.cross_entropy(
            quality_logits.reshape(-1, candidate_count),
            best_candidate.reshape(-1), reduction='none').reshape(
                quality_logits.shape[:2])
        valid_mode = command * (mask.sum(dim=-1) > 0).to(
            command.dtype)[:, None]
        ranking_loss = self._safe_divide(
            (cross_entropy * valid_mode).sum(), valid_mode.sum())
        generated_is_best = (
            (best_candidate > 0).to(command.dtype) * valid_mode).sum()
        generated_is_best = self._safe_divide(
            generated_is_best, valid_mode.sum())
        return ranking_loss, generated_is_best

    def forward(self, inputs):
        prediction = inputs.get(
            'ego_fut_residual_preds', inputs.get('ego_fut_preds'))
        diffusion_prediction = inputs.get('ego_fut_ddim_preds')
        if prediction is None or diffusion_prediction is None:
            reference = prediction
            if reference is None:
                reference = inputs.get('ego_fut_base_preds')
            if reference is None:
                raise KeyError('ResidualDDIMPlanLoss received no planner output')
            zero = reference.new_zeros(())
            return zero, {'loss_plan_residual_ddim': 0.0}

        target, mask, command = self._prepare_annotations(
            inputs, diffusion_prediction)
        diffusion_loss = self._diffusion_loss(inputs, command, mask)
        position_loss, endpoint_loss = self._trajectory_loss(
            prediction, target, command, mask)
        safety_loss = self._safety_loss(inputs, command, mask)
        dynamics_loss = self._dynamics_loss(prediction, command, mask)
        ranking_loss, generated_is_best = self._ranking_loss(
            inputs, target, command, mask)

        total = self.weight * (
            self.diffusion_weight * diffusion_loss
            + self.position_weight * position_loss
            + self.fde_weight * endpoint_loss
            + self.safety_weight * safety_loss
            + self.dynamics_weight * dynamics_loss
            + self.rank_weight * ranking_loss)
        total = torch.nan_to_num(total)
        log_values = {
            'loss_plan_residual_ddim': total.detach().item(),
            'loss_residual_x0': diffusion_loss.detach().item(),
            'loss_residual_position': position_loss.detach().item(),
            'loss_residual_fde': endpoint_loss.detach().item(),
            'loss_residual_gaussian_safe': safety_loss.detach().item(),
            'loss_residual_dynamics': dynamics_loss.detach().item(),
            'loss_residual_candidate_rank': ranking_loss.detach().item(),
            'residual_abs_mean': diffusion_prediction.detach().abs().mean().item(),
        }
        if generated_is_best is not None:
            log_values['candidate_generated_is_best_rate'] = (
                generated_is_best.detach().item())
        selected = inputs.get('ego_fut_selected_index')
        if selected is not None:
            log_values['candidate_baseline_selected_rate'] = (
                (selected == 0).to(torch.float32).mean().detach().item())
        return total, log_values
