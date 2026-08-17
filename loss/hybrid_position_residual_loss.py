import torch
import torch.nn as nn
import torch.nn.functional as F

from . import OPENOCC_LOSS


@OPENOCC_LOSS.register_module()
class HybridPositionResidualLoss(nn.Module):
    """Supervise the detached-anchor auxiliary trajectory and trust region."""

    def __init__(self, weight=0.5, position_weight=1.0,
                 trust_region_weight=0.05, beta=0.5):
        super().__init__()
        self.weight = weight
        self.position_weight = position_weight
        self.trust_region_weight = trust_region_weight
        self.beta = beta

    @staticmethod
    def _squeeze_annotations(value, target_dims):
        while value.dim() > target_dims and value.shape[1] == 1:
            value = value.squeeze(1)
        return value

    @staticmethod
    def _masked_mean(value, weight):
        expanded_weight = weight.expand_as(value)
        return ((value * expanded_weight).sum()
                / expanded_weight.sum().clamp_min(1.0))

    def forward(self, inputs):
        prediction = inputs['ego_fut_position_aux_preds']
        target = self._squeeze_annotations(inputs['ego_fut_gt'], 3)
        valid_mask = self._squeeze_annotations(inputs['ego_fut_masks'], 2)
        command = self._squeeze_annotations(inputs['ego_fut_cmd'], 2)

        target = target[:, None, :prediction.shape[2], :2].expand_as(prediction)
        valid_mask = valid_mask[:, :prediction.shape[2]].to(prediction.dtype)
        command = command[:, :prediction.shape[1]].to(prediction.dtype)
        sample_weight = command[:, :, None, None] * valid_mask[:, None, :, None]

        step_loss = F.smooth_l1_loss(
            prediction, target, reduction='none', beta=self.beta)
        step_loss = self._masked_mean(step_loss, sample_weight)

        position_loss = F.smooth_l1_loss(
            prediction.cumsum(dim=2), target.cumsum(dim=2),
            reduction='none', beta=self.beta)
        position_loss = self._masked_mean(position_loss, sample_weight)

        applied_residual = inputs['ego_fut_applied_residual_normalized']
        trust_loss = F.smooth_l1_loss(
            applied_residual, torch.zeros_like(applied_residual),
            reduction='none', beta=self.beta)
        trust_loss = self._masked_mean(trust_loss, sample_weight)

        supervised = self.weight * (
            step_loss + self.position_weight * position_loss)
        trust = self.trust_region_weight * trust_loss
        total = torch.nan_to_num(supervised + trust)
        gate = inputs.get('ego_fut_position_gate')
        gate_mean = (gate.detach().mean().item()
                     if torch.is_tensor(gate) else 0.0)
        applied_mean = applied_residual.detach().abs().mean().item()
        return total, {
            'plan_hybrid_gate_mean': gate_mean,
            'plan_hybrid_applied_residual_mean': applied_mean,
            'loss_plan_hybrid_residual_step': (
                self.weight * step_loss).detach().item(),
            'loss_plan_hybrid_residual_position': (
                self.weight * self.position_weight * position_loss
            ).detach().item(),
            'loss_plan_hybrid_residual_trust': trust.detach().item(),
        }
