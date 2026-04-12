# Copyright (c) OpenMMLab. All rights reserved.
import copy
import torch
from torch import nn as nn
from torch.nn.functional import l1_loss, mse_loss, smooth_l1_loss

from . import MAPTR_LOSS
from . import MAPTR_COST
from . import OPENOCC_LOSS
from .base_loss import BaseLoss
from mmdet.models import weighted_loss
from mmdet.structures.bbox.bbox_overlaps import bbox_overlaps
from mmdet.structures.bbox.transforms import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from mmdet.registry import MODELS as MMDET_MODEL
import mmcv
import torch.nn.functional as F
import functools
from mmdet.models.task_modules import build_assigner, build_sampler
from mmdet.models.utils import multi_apply
from mmdet.utils import reduce_mean
from mmengine.utils.dl_utils import TORCH_VERSION
from mmengine.utils import digit_version


def denormalize_3d_pts(pts, pc_range):
    new_pts = pts.clone()
    new_pts[...,0:1] = (pts[..., 0:1]*(pc_range[3] -
                            pc_range[0]) + pc_range[0])
    new_pts[...,1:2] = (pts[...,1:2]*(pc_range[4] -
                            pc_range[1]) + pc_range[1])
    new_pts[...,2:3] = (pts[...,2:3]*(pc_range[5] -
                            pc_range[2]) + pc_range[2])
    return new_pts

def normalize_3d_pts(pts, pc_range):
    patch_h = pc_range[4]-pc_range[1]
    patch_w = pc_range[3]-pc_range[0]
    patch_z = pc_range[5]-pc_range[2]
    new_pts = pts.clone()
    new_pts[...,0:1] = pts[..., 0:1] - pc_range[0]
    new_pts[...,1:2] = pts[...,1:2] - pc_range[1]
    new_pts[...,2:3] = pts[...,2:3] - pc_range[2]
    factor = pts.new_tensor([patch_w, patch_h,patch_z])
    normalized_pts = new_pts / factor
    return normalized_pts

def normalize_2d_bbox(bboxes, pc_range):

    patch_h = pc_range[4]-pc_range[1]
    patch_w = pc_range[3]-pc_range[0]
    cxcywh_bboxes = bbox_xyxy_to_cxcywh(bboxes)
    cxcywh_bboxes[...,0:1] = cxcywh_bboxes[..., 0:1] - pc_range[0]
    cxcywh_bboxes[...,1:2] = cxcywh_bboxes[...,1:2] - pc_range[1]
    factor = bboxes.new_tensor([patch_w, patch_h,patch_w,patch_h])

    normalized_bboxes = cxcywh_bboxes / factor
    return normalized_bboxes

def normalize_2d_pts(pts, pc_range):
    patch_h = pc_range[4]-pc_range[1]
    patch_w = pc_range[3]-pc_range[0]
    new_pts = pts.clone()
    new_pts[...,0:1] = pts[..., 0:1] - pc_range[0]
    new_pts[...,1:2] = pts[...,1:2] - pc_range[1]
    factor = pts.new_tensor([patch_w, patch_h])
    normalized_pts = new_pts / factor
    return normalized_pts

def denormalize_2d_bbox(bboxes, pc_range):

    bboxes = bbox_cxcywh_to_xyxy(bboxes)
    bboxes[..., 0::2] = (bboxes[..., 0::2]*(pc_range[3] -
                            pc_range[0]) + pc_range[0])
    bboxes[..., 1::2] = (bboxes[..., 1::2]*(pc_range[4] -
                            pc_range[1]) + pc_range[1])

    return bboxes
def denormalize_2d_pts(pts, pc_range):
    new_pts = pts.clone()
    new_pts[...,0:1] = (pts[..., 0:1]*(pc_range[3] -
                            pc_range[0]) + pc_range[0])
    new_pts[...,1:2] = (pts[...,1:2]*(pc_range[4] -
                            pc_range[1]) + pc_range[1])
    return new_pts

@OPENOCC_LOSS.register_module()
class MapLoss(BaseLoss):

    def __init__(self,
                 *args,
                 loss_pts=dict(type='ChamferDistance',
                             loss_src_weight=1.0,
                             loss_dst_weight=1.0),
                 loss_seg=dict(type='SimpleLoss',
                              pos_weight=2.13,
                              loss_weight=1.0),
                 loss_pv_seg=dict(type='SimpleLoss',
                              pos_weight=2.13,
                              loss_weight=1.0),
                 loss_dir=dict(type='PtsDirCosLoss', loss_weight=2.0),
                 loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=0.0),
                 loss_iou=dict(type='GIoULoss', loss_weight=0.0),
                 num_classes=3,
                 assigner=dict(
                    type='HungarianAssigner',
                    cls_cost=dict(type='ClassificationCost', weight=1.),
                    reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                    iou_cost=dict(
                    type='IoUCost', iou_mode='giou', weight=2.0)),
                 sync_cls_avg_factor=False,
                 gt_shift_pts_pattern='v0',
                 pc_range=None,
                 code_weights=None,
                 aux_seg = dict(
                    use_aux_seg=False,
                    bev_seg=False,
                    pv_seg=False,
                    seg_classes=1,
                    feat_down_sample=32,
                 ),
                 z_cfg = dict(
                    pred_z_flag=False,
                    gt_z_flag=False,
                 ),
                 num_pts_per_vec=2,
                 num_pts_per_gt_vec=2,
                 dir_interval=1,
                 **kwargs):

        super().__init__()
        self.loss_pts = MAPTR_LOSS.build(loss_pts)
        self.loss_dir = MAPTR_LOSS.build(loss_dir)
        self.loss_cls = MMDET_MODEL.build(loss_cls)
        self.loss_bbox = MMDET_MODEL.build(loss_bbox)
        self.loss_iou = MMDET_MODEL.build(loss_iou)

        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor

        assert loss_cls['loss_weight'] == assigner['cls_cost']['weight'], \
            'The classification weight for loss and matcher should be' \
            'exactly the same.'
        assert loss_bbox['loss_weight'] == assigner['reg_cost'][
            'weight'], 'The regression L1 weight for loss and matcher ' \
            'should be exactly the same.'
        assert loss_iou['loss_weight'] == assigner['iou_cost']['weight'], \
            'The regression iou weight for loss and matcher should be' \
            'exactly the same.'
        self.assigner = build_assigner(assigner)
        # DETR sampling=False, so use PseudoSampler
        sampler_cfg = dict(type='MapTRPseudoSampler')
        self.sampler = build_sampler(sampler_cfg, context=self)
        self.num_classes = num_classes
        self.gt_shift_pts_pattern = gt_shift_pts_pattern
        self.cls_out_channels = num_classes
        self.pc_range = pc_range
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)
        self.aux_seg = aux_seg
        self.z_cfg = z_cfg
        self.num_pts_per_vec = num_pts_per_vec
        self.num_pts_per_gt_vec = num_pts_per_gt_vec
        self.dir_interval = dir_interval

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           pts_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_shifts_pts,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """
        # import pdb;pdb.set_trace()
        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        gt_c = gt_bboxes.shape[-1]
        assign_result, order_index = self.assigner.assign(bbox_pred, cls_score, pts_pred,
                                             gt_bboxes, gt_labels, gt_shifts_pts,
                                             gt_bboxes_ignore)

        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        # pts_sampling_result = self.sampler.sample(assign_result, pts_pred,
        #                                       gt_pts)


        # import pdb;pdb.set_trace()
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds].long()
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # pts targets
        # import pdb;pdb.set_trace()
        # pts_targets = torch.zeros_like(pts_pred)
        # num_query, num_order, num_points, num_coords
        if order_index is None:
            # import pdb;pdb.set_trace()
            assigned_shift = gt_labels[sampling_result.pos_assigned_gt_inds]
        else:
            assigned_shift = order_index[sampling_result.pos_inds, sampling_result.pos_assigned_gt_inds]
        pts_targets = pts_pred.new_zeros((pts_pred.size(0),
                        pts_pred.size(1), pts_pred.size(2)))
        pts_weights = torch.zeros_like(pts_targets)
        pts_weights[pos_inds] = 1.0

        # DETR
        if pos_inds.numel() > 0:
            bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
            pts_targets[pos_inds] = gt_shifts_pts[sampling_result.pos_assigned_gt_inds,assigned_shift,:,:]
        return (labels, label_weights, bbox_targets, bbox_weights,
                pts_targets, pts_weights,
                pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    pts_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_shifts_pts_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pts_targets_list, pts_weights_list,
         pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,pts_preds_list,
            gt_labels_list, gt_bboxes_list, gt_shifts_pts_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, pts_targets_list, pts_weights_list,
                num_total_pos, num_total_neg)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    pts_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_shifts_pts_list,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_pts_list (list[Tensor]): Ground truth pts for each image
                with shape (num_gts, fixed_num, 2) in [x,y] format.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        pts_preds_list = [pts_preds[i] for i in range(num_imgs)]
        # import pdb;pdb.set_trace()
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,pts_preds_list,
                                           gt_bboxes_list, gt_labels_list,gt_shifts_pts_list,
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         pts_targets_list, pts_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        # import pdb;pdb.set_trace()
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        pts_targets = torch.cat(pts_targets_list, 0)
        pts_weights = torch.cat(pts_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp((num_total_pos).mean(), min=1).item()

        # import pdb;pdb.set_trace()
        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_2d_bbox(bbox_targets, self.pc_range) if bbox_targets.numel() > 0 else bbox_targets
        # normalized_bbox_targets = bbox_targets
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :4], normalized_bbox_targets[isnotnan,
                                                               :4], bbox_weights[isnotnan, :4],
            avg_factor=num_total_pos)

        # regression pts CD loss
        # pts_preds = pts_preds
        # import pdb;pdb.set_trace()

        # num_samples, num_order, num_pts, num_coords
        if pts_targets.numel() > 0:
            normalized_pts_targets = normalize_2d_pts(pts_targets, self.pc_range) if not self.z_cfg['gt_z_flag'] \
                                else normalize_3d_pts(pts_targets, self.pc_range)
        else:
            normalized_pts_targets = pts_targets

        # num_samples, num_pts, num_coords
        pts_preds = pts_preds.reshape(-1, pts_preds.size(-2),pts_preds.size(-1))
        if self.num_pts_per_vec != self.num_pts_per_gt_vec:
            pts_preds = pts_preds.permute(0,2,1)
            pts_preds = F.interpolate(pts_preds, size=(self.num_pts_per_gt_vec), mode='linear',
                                    align_corners=True)
            pts_preds = pts_preds.permute(0,2,1).contiguous()

        # import pdb;pdb.set_trace()
        loss_pts = self.loss_pts(
            pts_preds[isnotnan,:,:], normalized_pts_targets[isnotnan,
                                                            :,:],
            pts_weights[isnotnan,:,:],
            avg_factor=num_total_pos)
        dir_weights = pts_weights[:, :-self.dir_interval,0]
        denormed_pts_preds = denormalize_2d_pts(pts_preds, self.pc_range) if not self.z_cfg['gt_z_flag'] \
                                else denormalize_3d_pts(pts_preds, self.pc_range)
        denormed_pts_preds_dir = denormed_pts_preds[:,self.dir_interval:,:] - denormed_pts_preds[:,:-self.dir_interval,:]
        pts_targets_dir = pts_targets[:, self.dir_interval:,:] - pts_targets[:,:-self.dir_interval,:]
        # dir_weights = pts_weights[:, indice,:-1,0]
        # import pdb;pdb.set_trace()
        loss_dir = self.loss_dir(
            denormed_pts_preds_dir[isnotnan,:,:], pts_targets_dir[isnotnan,
                                                                          :,:],
            dir_weights[isnotnan,:],
            avg_factor=num_total_pos)

        bboxes = denormalize_2d_bbox(bbox_preds, self.pc_range)
        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes[isnotnan, :4], bbox_targets[isnotnan, :4], bbox_weights[isnotnan, :4],
            avg_factor=num_total_pos)

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
            loss_iou = torch.nan_to_num(loss_iou)
            loss_pts = torch.nan_to_num(loss_pts)
            loss_dir = torch.nan_to_num(loss_dir)
        return loss_cls, loss_bbox, loss_iou, loss_pts, loss_dir

    # @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             gt_seg_mask,
             gt_pv_seg_mask,
             preds_dicts,
             gt_bboxes_ignore=None,
             img_metas=None):
        """"Loss function.
        Args:

            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'
        gt_vecs_list = copy.deepcopy(gt_bboxes_list)
        # import pdb;pdb.set_trace()
        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        all_pts_preds  = preds_dicts['all_pts_preds']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        # gt_bboxes_list = [torch.cat(
        #     (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
        #     dim=1).to(device) for gt_bboxes in gt_bboxes_list]
        # import pdb;pdb.set_trace()
        # gt_bboxes_list = [
        #     gt_bboxes.to(device) for gt_bboxes in gt_bboxes_list]
        gt_bboxes_list = [
            gt_bboxes.bbox.to(device) for gt_bboxes in gt_vecs_list]
        gt_pts_list = [
            gt_bboxes.fixed_num_sampled_points.to(device) for gt_bboxes in gt_vecs_list]
        if self.gt_shift_pts_pattern == 'v0':
            gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points.to(device) for gt_bboxes in gt_vecs_list]
        elif self.gt_shift_pts_pattern == 'v1':
            gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v1.to(device) for gt_bboxes in gt_vecs_list]
        elif self.gt_shift_pts_pattern == 'v2':
            gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v2.to(device) for gt_bboxes in gt_vecs_list]
        elif self.gt_shift_pts_pattern == 'v3':
            gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v3.to(device) for gt_bboxes in gt_vecs_list]
        elif self.gt_shift_pts_pattern == 'v4':
            gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v4.to(device) for gt_bboxes in gt_vecs_list]
        else:
            raise NotImplementedError
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_pts_list = [gt_pts_list for _ in range(num_dec_layers)]
        all_gt_shifts_pts_list = [gt_shifts_pts_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]
        # import pdb;pdb.set_trace()
        losses_cls, losses_bbox, losses_iou, losses_pts, losses_dir = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,all_pts_preds,
            all_gt_bboxes_list, all_gt_labels_list,all_gt_shifts_pts_list,
            all_gt_bboxes_ignore_list)

        loss_dict = dict()
        if self.aux_seg['use_aux_seg']:
            # import ipdb;ipdb.set_trace()
            if self.aux_seg['bev_seg']:
                if preds_dicts['seg'] is not None:
                    seg_output = preds_dicts['seg']
                    num_imgs = seg_output.size(0)
                    seg_gt = torch.stack([gt_seg_mask[i] for i in range(num_imgs)],dim=0)
                    loss_seg = self.loss_seg(seg_output, seg_gt.float())
                    loss_dict['loss_seg'] = loss_seg
            if self.aux_seg['pv_seg']:
                # import ipdb;ipdb.set_trace()
                if preds_dicts['pv_seg'] is not None:
                    pv_seg_output = preds_dicts['pv_seg']
                    num_imgs = pv_seg_output.size(0)
                    pv_seg_gt = torch.stack([gt_pv_seg_mask[i] for i in range(num_imgs)],dim=0)
                    loss_pv_seg = self.loss_pv_seg(pv_seg_output, pv_seg_gt.float())
                    loss_dict['loss_pv_seg'] = loss_pv_seg

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]
        loss_dict['loss_pts'] = losses_pts[-1]
        loss_dict['loss_dir'] = losses_dir[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i, loss_pts_i, loss_dir_i in zip(losses_cls[:-1],
                                           losses_bbox[:-1],
                                           losses_iou[:-1],
                                           losses_pts[:-1],
                                           losses_dir[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            loss_dict[f'd{num_dec_layer}.loss_pts'] = loss_pts_i
            loss_dict[f'd{num_dec_layer}.loss_dir'] = loss_dir_i
            num_dec_layer += 1
        return loss_dict

    def forward(self, results):
        loss_inputs = [results['metas']['gt_bboxes_3d'], results['metas']['gt_labels_3d'], None, None, results]
        losses_pts = self.loss(*loss_inputs, img_metas=None)
        loss = sum([v for v in losses_pts.values()])
        return loss if loss.dim()==0 else loss[0]


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()

@mmcv.utils.parrots_jit.jit(derivate=True, coderize=True)
def custom_weight_dir_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): num_sample, num_dir
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Average factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        raise ValueError('avg_factor should not be none for OrderedPtsL1Loss')
        # loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            # import pdb;pdb.set_trace()
            # loss = loss.permute(1,0,2,3).contiguous()
            loss = loss.sum()
            loss = loss / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss

@mmcv.utils.parrots_jit.jit(derivate=True, coderize=True)
def custom_weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): num_sample, num_order, num_pts, num_coords
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Average factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        raise ValueError('avg_factor should not be none for OrderedPtsL1Loss')
        # loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            # import pdb;pdb.set_trace()
            loss = loss.permute(1,0,2,3).contiguous()
            loss = loss.sum((1,2,3))
            loss = loss / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss

def custom_weighted_loss(loss_func):
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    avg_factor=None, **kwargs)`.

    :Example:

    >>> import torch
    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, avg_factor=2)
    tensor(1.5000)
    """

    @functools.wraps(loss_func)
    def wrapper(pred,
                target,
                weight=None,
                reduction='mean',
                avg_factor=None,
                **kwargs):
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = custom_weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper


def custom_weighted_dir_loss(loss_func):
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    avg_factor=None, **kwargs)`.

    :Example:

    >>> import torch
    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, avg_factor=2)
    tensor(1.5000)
    """

    @functools.wraps(loss_func)
    def wrapper(pred,
                target,
                weight=None,
                reduction='mean',
                avg_factor=None,
                **kwargs):
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = custom_weight_dir_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper

@mmcv.utils.parrots_jit.jit(derivate=True, coderize=True)
@custom_weighted_loss
def ordered_pts_smooth_l1_loss(pred, target):
    """L1 loss.

    Args:
        pred (torch.Tensor): shape [num_samples, num_pts, num_coords]
        target (torch.Tensor): shape [num_samples, num_order, num_pts, num_coords]

    Returns:
        torch.Tensor: Calculated loss
    """
    if target.numel() == 0:
        return pred.sum() * 0
    pred = pred.unsqueeze(1).repeat(1, target.size(1),1,1)
    assert pred.size() == target.size()
    loss =smooth_l1_loss(pred,target, reduction='none')
    # import pdb;pdb.set_trace()
    return loss

@mmcv.utils.parrots_jit.jit(derivate=True, coderize=True)
@weighted_loss
def pts_l1_loss(pred, target):
    """L1 loss.

    Args:
        pred (torch.Tensor): shape [num_samples, num_pts, num_coords]
        target (torch.Tensor): shape [num_samples, num_pts, num_coords]

    Returns:
        torch.Tensor: Calculated loss
    """
    if target.numel() == 0:
        return pred.sum() * 0
    assert pred.size() == target.size()
    loss = torch.abs(pred - target)
    return loss

@mmcv.utils.parrots_jit.jit(derivate=True, coderize=True)
@custom_weighted_loss
def ordered_pts_l1_loss(pred, target):
    """L1 loss.

    Args:
        pred (torch.Tensor): shape [num_samples, num_pts, num_coords]
        target (torch.Tensor): shape [num_samples, num_order, num_pts, num_coords]

    Returns:
        torch.Tensor: Calculated loss
    """
    if target.numel() == 0:
        return pred.sum() * 0
    pred = pred.unsqueeze(1).repeat(1, target.size(1),1,1)
    assert pred.size() == target.size()
    loss = torch.abs(pred - target)
    return loss

@mmcv.utils.parrots_jit.jit(derivate=True, coderize=True)
@custom_weighted_dir_loss
def pts_dir_cos_loss(pred, target):
    """ Dir cosine similiarity loss
    pred (torch.Tensor): shape [num_samples, num_dir, num_coords]
    target (torch.Tensor): shape [num_samples, num_dir, num_coords]

    """
    if target.numel() == 0:
        return pred.sum() * 0
    # import pdb;pdb.set_trace()
    num_samples, num_dir, num_coords = pred.shape
    loss_func = torch.nn.CosineEmbeddingLoss(reduction='none')
    tgt_param = target.new_ones((num_samples, num_dir))
    tgt_param = tgt_param.flatten(0)
    loss = loss_func(pred.flatten(0,1), target.flatten(0,1), tgt_param)
    loss = loss.view(num_samples, num_dir)
    return loss

@MAPTR_LOSS.register_module()
class OrderedPtsSmoothL1Loss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(OrderedPtsSmoothL1Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        # import pdb;pdb.set_trace()
        loss_bbox = self.loss_weight * ordered_pts_smooth_l1_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox


@MAPTR_LOSS.register_module()
class PtsDirCosLoss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(PtsDirCosLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        # import pdb;pdb.set_trace()
        loss_dir = self.loss_weight * pts_dir_cos_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_dir



@MAPTR_LOSS.register_module()
class PtsL1Loss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(PtsL1Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        # import pdb;pdb.set_trace()
        loss_bbox = self.loss_weight * pts_l1_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox

@MAPTR_LOSS.register_module()
class OrderedPtsL1Loss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(OrderedPtsL1Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        # import pdb;pdb.set_trace()
        loss_bbox = self.loss_weight * ordered_pts_l1_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox




@MAPTR_COST.register_module()
class OrderedPtsSmoothL1Cost(object):
    """OrderedPtsL1Cost.
     Args:
         weight (int | float, optional): loss_weight
    """

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, bbox_pred, gt_bboxes):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (x, y), which are all in range [0, 1]. Shape
                [num_query, num_pts, 2].
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x,y).
                Shape [num_gt, num_ordered, num_pts, 2].
        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        num_gts, num_orders, num_pts, num_coords = gt_bboxes.shape
        # import pdb;pdb.set_trace()
        bbox_pred = bbox_pred.view(bbox_pred.size(0),-1).unsqueeze(1).repeat(1,num_gts*num_orders,1)
        gt_bboxes = gt_bboxes.flatten(2).view(num_gts*num_orders,-1).unsqueeze(0).repeat(bbox_pred.size(0),1,1)
        # import pdb;pdb.set_trace()
        bbox_cost = smooth_l1_loss(bbox_pred, gt_bboxes, reduction='none').sum(-1)
        # bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight

@MAPTR_COST.register_module()
class PtsL1Cost(object):
    """OrderedPtsL1Cost.
     Args:
         weight (int | float, optional): loss_weight
    """

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, bbox_pred, gt_bboxes):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (x, y), which are all in range [0, 1]. Shape
                [num_query, num_pts, 2].
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x,y).
                Shape [num_gt, num_ordered, num_pts, 2].
        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        num_gts, num_pts, num_coords = gt_bboxes.shape
        # import pdb;pdb.set_trace()
        bbox_pred = bbox_pred.view(bbox_pred.size(0),-1)
        gt_bboxes = gt_bboxes.view(num_gts,-1)
        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight

@MAPTR_COST.register_module()
class OrderedPtsL1Cost(object):
    """OrderedPtsL1Cost.
     Args:
         weight (int | float, optional): loss_weight
    """

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, bbox_pred, gt_bboxes):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (x, y), which are all in range [0, 1]. Shape
                [num_query, num_pts, 2].
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x,y).
                Shape [num_gt, num_ordered, num_pts, 2].
        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        num_gts, num_orders, num_pts, num_coords = gt_bboxes.shape
        # import pdb;pdb.set_trace()
        bbox_pred = bbox_pred.view(bbox_pred.size(0),-1)
        gt_bboxes = gt_bboxes.flatten(2).view(num_gts*num_orders,-1)
        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight

@MAPTR_COST.register_module()
class MyChamferDistanceCost:
    def __init__(self, loss_src_weight=1., loss_dst_weight=1.):
        # assert mode in ['smooth_l1', 'l1', 'l2']
        # self.mode = mode
        self.loss_src_weight = loss_src_weight
        self.loss_dst_weight = loss_dst_weight

    def __call__(self, src, dst,src_weight=1.0,dst_weight=1.0,):
        """
        pred_pts (Tensor): normed coordinate(x,y), shape (num_q, num_pts_M, 2)
        gt_pts (Tensor): normed coordinate(x,y), shape (num_gt, num_pts_N, 2)
        """
        # criterion_mode = self.mode
        # if criterion_mode == 'smooth_l1':
        #     criterion = smooth_l1_loss
        # elif criterion_mode == 'l1':
        #     criterion = l1_loss
        # elif criterion_mode == 'l2':
        #     criterion = mse_loss
        # else:
        #     raise NotImplementedError
        # import pdb;pdb.set_trace()
        src_expand = src.unsqueeze(1).repeat(1,dst.shape[0],1,1)
        dst_expand = dst.unsqueeze(0).repeat(src.shape[0],1,1,1)
        # src_expand = src.unsqueeze(2).unsqueeze(1).repeat(1,dst.shape[0], 1, dst.shape[1], 1)
        # dst_expand = dst.unsqueeze(1).unsqueeze(0).repeat(src.shape[0],1, src.shape[1], 1, 1)
        distance = torch.cdist(src_expand, dst_expand)
        src2dst_distance = torch.min(distance, dim=3)[0]  # (num_q, num_gt, num_pts_N)
        dst2src_distance = torch.min(distance, dim=2)[0]  # (num_q, num_gt, num_pts_M)
        loss_src = (src2dst_distance * src_weight).mean(-1)
        loss_dst = (dst2src_distance * dst_weight).mean(-1)
        loss = loss_src*self.loss_src_weight + loss_dst * self.loss_dst_weight
        return loss

@MAPTR_COST.register_module()
class BBoxL1Cost:
    """BBoxL1Cost.

     Args:
         weight (int | float, optional): loss_weight
         box_format (str, optional): 'xyxy' for DETR, 'xywh' for Sparse_RCNN

     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import BBoxL1Cost
         >>> import torch
         >>> self = BBoxL1Cost()
         >>> bbox_pred = torch.rand(1, 4)
         >>> gt_bboxes= torch.FloatTensor([[0, 0, 2, 4], [1, 2, 3, 4]])
         >>> factor = torch.tensor([10, 8, 10, 8])
         >>> self(bbox_pred, gt_bboxes, factor)
         tensor([[1.6172, 1.6422]])
    """

    def __init__(self, weight=1., box_format='xyxy'):
        self.weight = weight
        assert box_format in ['xyxy', 'xywh']
        self.box_format = box_format

    def __call__(self, bbox_pred, gt_bboxes):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].

        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        if self.box_format == 'xywh':
            gt_bboxes = bbox_xyxy_to_cxcywh(gt_bboxes)
        elif self.box_format == 'xyxy':
            bbox_pred = bbox_cxcywh_to_xyxy(bbox_pred)
        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight

@MAPTR_COST.register_module()
class FocalLossCost:
    """FocalLossCost.

     Args:
         weight (int | float, optional): loss_weight
         alpha (int | float, optional): focal_loss alpha
         gamma (int | float, optional): focal_loss gamma
         eps (float, optional): default 1e-12

     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import FocalLossCost
         >>> import torch
         >>> self = FocalLossCost()
         >>> cls_pred = torch.rand(4, 3)
         >>> gt_labels = torch.tensor([0, 1, 2])
         >>> factor = torch.tensor([10, 8, 10, 8])
         >>> self(cls_pred, gt_labels)
         tensor([[-0.3236, -0.3364, -0.2699],
                [-0.3439, -0.3209, -0.4807],
                [-0.4099, -0.3795, -0.2929],
                [-0.1950, -0.1207, -0.2626]])
    """

    def __init__(self, weight=1., alpha=0.25, gamma=2, eps=1e-12):
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def __call__(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).

        Returns:
            torch.Tensor: cls_cost value with weight
        """
        cls_pred = cls_pred.sigmoid()
        neg_cost = -(1 - cls_pred + self.eps).log() * (
            1 - self.alpha) * cls_pred.pow(self.gamma)
        pos_cost = -(cls_pred + self.eps).log() * self.alpha * (
            1 - cls_pred).pow(self.gamma)
        cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
        return cls_cost * self.weight

@MAPTR_COST.register_module()
class IoUCost:
    """IoUCost.

     Args:
         iou_mode (str, optional): iou mode such as 'iou' | 'giou'
         weight (int | float, optional): loss weight

     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import IoUCost
         >>> import torch
         >>> self = IoUCost()
         >>> bboxes = torch.FloatTensor([[1,1, 2, 2], [2, 2, 3, 4]])
         >>> gt_bboxes = torch.FloatTensor([[0, 0, 2, 4], [1, 2, 3, 4]])
         >>> self(bboxes, gt_bboxes)
         tensor([[-0.1250,  0.1667],
                [ 0.1667, -0.5000]])
    """

    def __init__(self, iou_mode='giou', weight=1.):
        self.weight = weight
        self.iou_mode = iou_mode

    def __call__(self, bboxes, gt_bboxes):
        """
        Args:
            bboxes (Tensor): Predicted boxes with unnormalized coordinates
                (x1, y1, x2, y2). Shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].

        Returns:
            torch.Tensor: iou_cost value with weight
        """
        # overlaps: [num_bboxes, num_gt]
        overlaps = bbox_overlaps(
            bboxes, gt_bboxes, mode=self.iou_mode, is_aligned=False)
        # The 1 is a constant that doesn't change the matching, so omitted.
        iou_cost = -overlaps
        return iou_cost * self.weight

@MAPTR_LOSS.register_module()
class SimpleLoss(torch.nn.Module):
    def __init__(self, pos_weight, loss_weight):
        super(SimpleLoss, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]))
        self.loss_weight = loss_weight

    def forward(self, ypred, ytgt):
        # import ipdb;ipdb.set_trace()
        loss = self.loss_fn(ypred, ytgt)
        return loss*self.loss_weight

@mmcv.utils.parrots_jit.jit(derivate=True, coderize=True)
def chamfer_distance(src,
                     dst,
                     src_weight=1.0,
                     dst_weight=1.0,
                    #  criterion_mode='l1',
                     reduction='mean',
                     avg_factor=None):
    """Calculate Chamfer Distance of two sets.

    Args:
        src (torch.Tensor): Source set with shape [B, N, C] to
            calculate Chamfer Distance.
        dst (torch.Tensor): Destination set with shape [B, M, C] to
            calculate Chamfer Distance.
        src_weight (torch.Tensor or float): Weight of source loss.
        dst_weight (torch.Tensor or float): Weight of destination loss.
        criterion_mode (str): Criterion mode to calculate distance.
            The valid modes are smooth_l1, l1 or l2.
        reduction (str): Method to reduce losses.
            The valid reduction method are 'none', 'sum' or 'mean'.

    Returns:
        tuple: Source and Destination loss with the corresponding indices.

            - loss_src (torch.Tensor): The min distance \
                from source to destination.
            - loss_dst (torch.Tensor): The min distance \
                from destination to source.
            - indices1 (torch.Tensor): Index the min distance point \
                for each point in source to destination.
            - indices2 (torch.Tensor): Index the min distance point \
                for each point in destination to source.
    """

    # if criterion_mode == 'smooth_l1':
    #     criterion = smooth_l1_loss
    # elif criterion_mode == 'l1':
    #     criterion = l1_loss
    # elif criterion_mode == 'l2':
    #     criterion = mse_loss
    # else:
    #     raise NotImplementedError

    # src_expand = src.unsqueeze(2).repeat(1, 1, dst.shape[1], 1)
    # dst_expand = dst.unsqueeze(1).repeat(1, src.shape[1], 1, 1)
    # import pdb;pdb.set_trace()
    distance = torch.cdist(src, dst)
    src2dst_distance, indices1 = torch.min(distance, dim=2)  # (B,N)
    dst2src_distance, indices2 = torch.min(distance, dim=1)  # (B,M)
    # import pdb;pdb.set_trace()
    #TODO this may be wrong for misaligned src_weight, now[N,fixed_num]
    # should be [N], then view
    loss_src = (src2dst_distance * src_weight)
    loss_dst = (dst2src_distance * dst_weight)
    if avg_factor is None:
        reduction_enum = F._Reduction.get_enum(reduction)
        if reduction_enum == 0:
            raise ValueError('MyCDLoss can not be used with reduction=`none`')
        elif reduction_enum == 1:
            loss_src = loss_src.mean(-1).mean()
            loss_dst = loss_dst.mean(-1).mean()
        elif reduction_enum == 2:
            loss_src = loss_src.mean(-1).sum()
            loss_dst = loss_dst.mean(-1).sum()
        else:
            raise NotImplementedError
    else:
        if reduction == 'mean':
            eps = torch.finfo(torch.float32).eps
            loss_src = loss_src.mean(-1).sum() / (avg_factor + eps)
            loss_dst = loss_dst.mean(-1).sum() / (avg_factor + eps)
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')

    return loss_src, loss_dst, indices1, indices2


@MAPTR_LOSS.register_module()
class MyChamferDistance(nn.Module):
    """Calculate Chamfer Distance of two sets.

    Args:
        mode (str): Criterion mode to calculate distance.
            The valid modes are smooth_l1, l1 or l2.
        reduction (str): Method to reduce losses.
            The valid reduction method are none, sum or mean.
        loss_src_weight (float): Weight of loss_source.
        loss_dst_weight (float): Weight of loss_target.
    """

    def __init__(self,
                #  mode='l1',
                 reduction='mean',
                 loss_src_weight=1.0,
                 loss_dst_weight=1.0):
        super(MyChamferDistance, self).__init__()

        # assert mode in ['smooth_l1', 'l1', 'l2']
        assert reduction in ['none', 'sum', 'mean']
        # self.mode = mode
        self.reduction = reduction
        self.loss_src_weight = loss_src_weight
        self.loss_dst_weight = loss_dst_weight

    def forward(self,
                source,
                target,
                src_weight=1.0,
                dst_weight=1.0,
                avg_factor=None,
                reduction_override=None,
                return_indices=False,
                **kwargs):
        """Forward function of loss calculation.

        Args:
            source (torch.Tensor): Source set with shape [B, N, C] to
                calculate Chamfer Distance.
            target (torch.Tensor): Destination set with shape [B, M, C] to
                calculate Chamfer Distance.
            src_weight (torch.Tensor | float, optional):
                Weight of source loss. Defaults to 1.0.
            dst_weight (torch.Tensor | float, optional):
                Weight of destination loss. Defaults to 1.0.
            reduction_override (str, optional): Method to reduce losses.
                The valid reduction method are 'none', 'sum' or 'mean'.
                Defaults to None.
            return_indices (bool, optional): Whether to return indices.
                Defaults to False.

        Returns:
            tuple[torch.Tensor]: If ``return_indices=True``, return losses of \
                source and target with their corresponding indices in the \
                order of ``(loss_source, loss_target, indices1, indices2)``. \
                If ``return_indices=False``, return \
                ``(loss_source, loss_target)``.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        loss_source, loss_target, indices1, indices2 = chamfer_distance(
            source, target, src_weight, dst_weight, reduction,
            avg_factor=avg_factor)

        loss_source *= self.loss_src_weight
        loss_target *= self.loss_dst_weight

        loss_pts = loss_source + loss_target

        if return_indices:
            return loss_pts, indices1, indices2
        else:
            return loss_pts
