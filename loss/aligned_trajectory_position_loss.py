import torch
import torch.nn as nn
import torch.nn.functional as F

from . import OPENOCC_LOSS


@OPENOCC_LOSS.register_module()
class AlignedTrajectoryPositionLoss(nn.Module):
    """Apply low-weight cumulative-position supervision to the final path."""

    def __init__(self, weight=0.5, beta=0.5,
                 timestep_weights=(0.5, 0.75, 1.0, 1.0, 1.25, 1.5),
                 pred_key='ego_fut_preds'):
        super().__init__()
        self.weight = weight
        self.beta = beta
        # 监督哪条轨迹：默认 fused main('ego_fut_preds')，也可指向
        # 'ego_fut_aux_preds'(全局分支) / 'ego_fut_per_frame_preds'(逐帧 base)。
        self.pred_key = pred_key
        self.register_buffer(
            'timestep_weights',
            torch.as_tensor(timestep_weights, dtype=torch.float32))

    @staticmethod
    def _squeeze_annotations(value, target_dims):
        while value.dim() > target_dims and value.shape[1] == 1:
            value = value.squeeze(1)
        return value

    def forward(self, inputs):
        prediction = inputs[self.pred_key]
        target = self._squeeze_annotations(inputs['ego_fut_gt'], 3)
        valid_mask = self._squeeze_annotations(inputs['ego_fut_masks'], 2)
        command = self._squeeze_annotations(inputs['ego_fut_cmd'], 2)

        timesteps = prediction.shape[2]
        if self.timestep_weights.numel() != timesteps:
            raise ValueError(
                'timestep_weights must contain one value per prediction step')
        target = target[:, None, :timesteps, :2].expand_as(prediction)
        valid_mask = valid_mask[:, :timesteps].to(prediction.dtype)
        command = command[:, :prediction.shape[1]].to(prediction.dtype)
        time_weight = self.timestep_weights.to(prediction.dtype)
        sample_weight = (
            command[:, :, None, None]
            * valid_mask[:, None, :, None]
            * time_weight[None, None, :, None])

        element_loss = F.smooth_l1_loss(
            prediction.cumsum(dim=2), target.cumsum(dim=2),
            reduction='none', beta=self.beta)
        expanded_weight = sample_weight.expand_as(element_loss)
        position_loss = (
            (element_loss * expanded_weight).sum()
            / expanded_weight.sum().clamp_min(1.0))
        total = torch.nan_to_num(self.weight * position_loss)
        log_key = 'loss_plan_aligned_position'
        if self.pred_key != 'ego_fut_preds':
            log_key = 'loss_plan_aligned_position_' + self.pred_key
        return total, {
            log_key: total.detach().item(),
        }
