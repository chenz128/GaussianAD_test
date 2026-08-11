import torch
import torch.nn as nn
import torch.nn.functional as F

from . import OPENOCC_LOSS


@OPENOCC_LOSS.register_module()
class FlowMatchingLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, inputs):
        loss = inputs.get('flow_matching_loss')
        if loss is None:
            reference = inputs['pred_occ'][-1][0]
            return reference.sum() * 0.0
        return loss * self.weight


@OPENOCC_LOSS.register_module()
class InnovationOccupancyLoss(nn.Module):
    def __init__(self, weight=3.0, dynamic_multiplier=5.0,
                 empty_label=17, num_classes=18):
        super().__init__()
        self.weight = weight
        self.dynamic_multiplier = dynamic_multiplier
        self.empty_label = empty_label
        self.num_classes = num_classes
        class_weight = torch.ones(num_classes, dtype=torch.float32)
        class_weight[[2, 3, 4, 5, 6, 7, 9, 10]] = dynamic_multiplier
        self.register_buffer('class_weight', class_weight)

    def forward(self, inputs):
        losses = []
        for future in inputs['occ_flow']:
            prediction = future[0]
            mask = prediction.get('innovation_mask')
            if mask is None or not bool(prediction['flow_valid_flag']):
                continue
            logits = prediction['pred_flow']
            target = prediction['sampled_label']
            if not torch.is_tensor(target):
                target = torch.as_tensor(target)
            target = target.to(device=logits.device, dtype=torch.long).reshape(-1)
            mask = mask.to(device=logits.device, dtype=torch.bool).reshape(-1)
            if not mask.any():
                continue
            losses.append(F.cross_entropy(
                logits[0, :, mask].transpose(0, 1).float(),
                target[mask], weight=self.class_weight.float()))
        if not losses:
            reference = inputs['pred_occ'][-1][0]
            return reference.sum() * 0.0
        return torch.stack(losses).mean() * self.weight