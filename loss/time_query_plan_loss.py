import torch.nn as nn
import torch.nn.functional as F

from . import OPENOCC_LOSS


@OPENOCC_LOSS.register_module()
class TimeQueryPlanLoss(nn.Module):
    """Explicitly supervise per-timestep planner queries."""

    def __init__(self, weight=1.0, position_weight=1.0, beta=0.5):
        super().__init__()
        self.weight = weight
        self.position_weight = position_weight
        self.beta = beta

    @staticmethod
    def _masked_mean(loss, weight):
        expanded_weight = weight.expand_as(loss)
        return (loss * expanded_weight).sum() / expanded_weight.sum().clamp_min(1.0)

    def forward(self, inputs):
        prediction = inputs['ego_fut_aux_preds']
        target = inputs['ego_fut_gt'].squeeze(1)
        valid_mask = inputs['ego_fut_masks'].squeeze(1).squeeze(1)
        command = inputs['ego_fut_cmd'].squeeze(1).squeeze(1)

        target = target[:, None, :, :].expand_as(prediction)
        weight = command[..., None, None].to(prediction.dtype)
        weight = weight * valid_mask[:, None, :, None].to(prediction.dtype)

        step_loss = F.smooth_l1_loss(
            prediction, target, reduction='none', beta=self.beta)
        step_loss = self._masked_mean(step_loss, weight)

        position_prediction = prediction.cumsum(dim=2)
        position_target = target.cumsum(dim=2)
        position_loss = F.smooth_l1_loss(
            position_prediction, position_target,
            reduction='none', beta=self.beta)
        position_loss = self._masked_mean(position_loss, weight)

        total = self.weight * (
            step_loss + self.position_weight * position_loss)
        return total, {
            'loss_plan_time_query_step': (self.weight * step_loss).detach().item(),
            'loss_plan_time_query_position': (
                self.weight * self.position_weight * position_loss).detach().item(),
        }
