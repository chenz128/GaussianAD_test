"""Full-data Stage-1 training for V3-SE3 perception and occupancy.

This isolated configuration keeps the V3-SE3 current/future OCC and
detection paths, trains the inherited map decoder for later planning use, and
removes trajectory planning completely.  It follows the full-data reference:
all nuScenes train/val keyframes, 20 epochs and evaluation every four epochs.
Only the generic R101 detector pretraining is loaded.
"""

from copy import deepcopy
import os


_base_ = [
    '../nuscenes_gs25600_v3_se3/nuscenes_gs25600_v3_se3.py'
]

custom_imports = deepcopy(_base_.custom_imports)
custom_imports['imports'] = list(custom_imports['imports']) + [
    'model.segmentor.bev_segmentor_v3_se3_full_no_planner_isolated',
]


# Full nuScenes train/val and the same 20-epoch policy as the joint full run.
max_epochs = 20
eval_every_epochs = 4
train_dataset_config = dict(num_samples=0)
val_dataset_config = dict(num_samples=0)

# V3-SE3 currently requires one sample per DDP worker.  Eight workers give a
# global batch size of eight, matching the full-data reference experiment.
train_loader = dict(batch_size=1)
val_loader = dict(batch_size=1)


# Stage 1 starts from the same generic R101 initialization as the joint run.
load_from = 'ckpts/r101_dcn_fcos3d_pretrain.pth'

lr = float(os.environ.get('LR', 2e-4))
optimizer = dict(
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    paramwise_cfg=dict(custom_keys={'img_backbone': dict(lr_mult=0.1)}),
)

# There is no planner module.  The map branch remains trainable so a later
# frozen-front-end Planner receives a learned map representation.
frozen_modules = []
find_unused_parameters = True
static_graph = True

model = dict(
    type='BEVSegmentorV3SE3FullNoPlannerIsolated',
    head=dict(type='GaussianHeadFrontierV3Isolated'),
    planner_head=None,
)


# Keep every V3-SE3 perception/OCC loss and the map supervision used by the
# joint full experiment.  No PlanLoss or planner auxiliary loss is present.
num_map_classes = len(_base_.map_classes)
pc_range = [-30.0, -30.0, -2.0, 30.0, 30.0, 2.0]
fixed_ptsnum_per_gt_line = 20
fixed_ptsnum_per_pred_line = 20

_map_loss_cfg = dict(
    type='MapLoss',
    loss_cls=dict(
        type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25,
        loss_weight=2.0),
    loss_bbox=dict(type='L1Loss', loss_weight=0.0),
    loss_iou=dict(type='GIoULoss', loss_weight=0.0),
    loss_pts=dict(type='PtsL1Loss', loss_weight=5.0),
    loss_dir=dict(type='PtsDirCosLoss', loss_weight=0.005),
    loss_seg=dict(type='SimpleLoss', pos_weight=4.0, loss_weight=1.0),
    loss_pv_seg=dict(type='SimpleLoss', pos_weight=1.0, loss_weight=2.0),
    assigner=dict(
        type='MapTRAssigner',
        cls_cost=dict(type='FocalLossCost', weight=2.0),
        reg_cost=dict(type='BBoxL1Cost', weight=0.0, box_format='xywh'),
        iou_cost=dict(type='IoUCost', iou_mode='giou', weight=0.0),
        pts_cost=dict(type='OrderedPtsL1Cost', weight=5),
        pc_range=pc_range,
    ),
    sync_cls_avg_factor=True,
    num_classes=num_map_classes,
    gt_shift_pts_pattern='v2',
    pc_range=pc_range,
    code_weights=[1.0, 1.0, 1.0, 1.0],
    aux_seg=_base_.aux_seg_cfg,
    num_pts_per_vec=fixed_ptsnum_per_pred_line,
    num_pts_per_gt_vec=fixed_ptsnum_per_gt_line,
    dir_interval=1,
)

loss = dict(
    type='MultiLoss',
    loss_cfgs=deepcopy(_base_.loss['loss_cfgs']) + [_map_loss_cfg],
)

# Preserve every V3-SE3 mapping explicitly, then add map supervision inputs.
# This avoids depending on implicit config-dictionary merge behaviour.
loss_input_convertion = deepcopy(_base_.loss_input_convertion)
loss_input_convertion.update(
    all_cls_scores='all_cls_scores',
    all_bbox_preds='all_bbox_preds',
    all_pts_preds='all_pts_preds',
)
