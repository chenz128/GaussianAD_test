import torch
import torch.nn as nn
import torch.nn.functional as F

from . import OPENOCC_LOSS


@OPENOCC_LOSS.register_module()
class ResidualDiffusionPlanLoss(nn.Module):
    """Noise-prediction objective for the command-selected trajectory mode."""

    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    @staticmethod
    def _squeeze_annotations(value, target_dims):
        while value.dim() > target_dims and value.shape[1] == 1:
            value = value.squeeze(1)
        return value

    def forward(self, inputs):
        prediction = inputs.get('residual_diffusion_noise_pred')
        target = inputs.get('residual_diffusion_noise_target')
        if prediction is None or target is None or not torch.is_tensor(prediction):
            device = prediction.device if torch.is_tensor(prediction) else None
            return torch.zeros((), device=device), {}

        target = target.detach()
        valid_mask = self._squeeze_annotations(
            inputs['ego_fut_masks'], target_dims=2)
        command = self._squeeze_annotations(
            inputs['ego_fut_cmd'], target_dims=2)

        valid_mask = valid_mask[:, :prediction.shape[2]].to(prediction.dtype)
        command = command[:, :prediction.shape[1]].to(prediction.dtype)
        weight = command[:, :, None, None] * valid_mask[:, None, :, None]
        weight = weight.expand_as(prediction)

        element_loss = F.mse_loss(prediction, target, reduction='none')
        noise_loss = (element_loss * weight).sum() / weight.sum().clamp_min(1.0)
        noise_loss = torch.nan_to_num(noise_loss)
        total = self.weight * noise_loss
        return total, {
            'loss_plan_residual_diffusion': total.detach().item(),
        }
