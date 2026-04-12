import torch, torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmseg.models.losses import DiceLoss

from . import OPENOCC_LOSS
from .base_loss import BaseLoss
from .utils.lovasz_softmax import lovasz_softmax
from model.utils import loss_utils
from model.dense_heads.voxelnext_head import VoxelNeXtHead


@OPENOCC_LOSS.register_module()
class DetectionLoss(BaseLoss):
    def __init__(
        self,
        weight=1,
        loss_weights=None,
        head_order=None,
        input_dict=None):
        super().__init__()

        self.weight = weight
        self.loss_weights = loss_weights
        self.input_dict = input_dict
        self.head_order = head_order
        self.build_losses()

    def build_losses(self):
        self.add_module('hm_loss_func', loss_utils.FocalLossSparse())
        self.add_module('reg_loss_func', loss_utils.RegLossSparse())

    def sigmoid(self, x):
        y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
        return y

    def get_loss(self, forward_ret_dict):
        pred_dicts = forward_ret_dict['pred_dicts']
        target_dicts = forward_ret_dict['target_dicts']
        batch_index = forward_ret_dict['batch_index']

        tb_dict = {}
        loss = 0
        batch_indices = forward_ret_dict['voxel_indices'][:, 0]
        spatial_indices = forward_ret_dict['voxel_indices'][:, 1:]

        for idx, pred_dict in enumerate(pred_dicts):
            pred_dict['hm'] = self.sigmoid(pred_dict['hm'])
            hm_loss = self.hm_loss_func(pred_dict['hm'], target_dicts['heatmaps'][idx])
            hm_loss *= self.loss_weights['cls_weight']

            target_boxes = target_dicts['target_boxes'][idx]
            pred_boxes = torch.cat([pred_dict[head_name] for head_name in self.head_order], dim=1)

            reg_loss = self.reg_loss_func(
                pred_boxes, target_dicts['masks'][idx], target_dicts['inds'][idx], target_boxes, batch_index
            )
            loc_loss = (reg_loss * reg_loss.new_tensor(self.loss_weights['code_weights'])).sum()
            loc_loss = loc_loss * self.loss_weights['loc_weight']

            tb_dict['hm_loss_head_%d' % idx] = hm_loss.item()
            tb_dict['loc_loss_head_%d' % idx] = loc_loss.item()

            loss += hm_loss + loc_loss

        tb_dict['rpn_loss'] = loss.item()
        return loss, tb_dict

    def forward(self, inputs):
        loss, tb_dict = self.get_loss(inputs)
        return loss

