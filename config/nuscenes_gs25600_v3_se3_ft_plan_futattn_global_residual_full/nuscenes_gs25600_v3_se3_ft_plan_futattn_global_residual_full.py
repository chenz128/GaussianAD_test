"""Full-data joint training: V3-SE3 OCC frontend + best planner backend.

This configuration is intentionally isolated.  It inherits the isolated
V3-SE3 port, preserves its OCC/flow/detection path and replaces only the
planner backend.  Initial weights are assembled as follows:

* V3-SE3 epoch 15: perception, detection and V3-SE3 OCC/future generator.
* FutAttn-Global-Residual epoch 15: map_decoder and planner_head.

The full-data schedule follows ``chenz/nuscenes_gs25600_v12_full``:
all train/val keyframes, 20 epochs, validation every 4 epochs.
"""

from copy import deepcopy
import os


_base_ = [
    '../nuscenes_gs25600_v3_se3/nuscenes_gs25600_v3_se3.py'
]


# -------------------------------------------------------------------------
# Full-data schedule (same policy as nuscenes_gs25600_v12_full).
# -------------------------------------------------------------------------
max_epochs = 20
eval_every_epochs = 4
train_dataset_config = dict(num_samples=0)
val_dataset_config = dict(num_samples=0)


# -------------------------------------------------------------------------
# Stage initialization and joint optimization.
# load_from starts a fresh 20-epoch schedule; it does not resume either
# source optimizer/scheduler.
# -------------------------------------------------------------------------
load_from = (
    'exp/nuscenes_gs25600_v3_se3_ft_plan_futattn_global_residual_full/'
    'init/v3_se3_occ_plus_futattn_global_residual_planner_epoch15.pth'
)

lr = float(os.environ.get('LR', 2e-4))
optimizer = dict(
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    paramwise_cfg=dict(custom_keys={'img_backbone': dict(lr_mult=0.1)}),
)

# All frontend/backend modules take part in the full joint fine-tuning.
frozen_modules = []
# Preserve the proven V3-SE3 DDP policy.  V3 direct-future generation can
# leave conditional parameters unused for a batch; static_graph was used by
# the source V3-SE3 experiment.
find_unused_parameters = True
static_graph = True


# -------------------------------------------------------------------------
# Best Planner backend.  Base VAD fields (embed_dims/fut_ts/modes and the
# current agent/map/Gaussian decoders) are inherited from V3-SE3; only the
# exact FutAttn-Global-Residual delta is specified here.
# -------------------------------------------------------------------------
_attn = dict(
    type='MultiheadAttention', embed_dims=128, num_heads=8, dropout=0.1)
_self_layer = dict(
    type='MyCustomBaseTransformerLayer',
    attn_cfgs=[_attn],
    feedforward_channels=256,
    ffn_dropout=0.1,
    batch_first=False,
    operation_order=('self_attn', 'norm', 'ffn', 'norm'),
)
_cross_layer = dict(
    type='MyCustomBaseTransformerLayer',
    attn_cfgs=[_attn],
    feedforward_channels=256,
    ffn_dropout=0.1,
    batch_first=False,
    operation_order=('cross_attn', 'norm', 'ffn', 'norm'),
)

model = dict(
    planner_head=dict(
        type='VADHeadFutAttnGlobalResidual',
        time_interval=0.5,
        num_fourier_bands=8,
        fut_self_decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=_self_layer,
        ),
        ego_fut_gaussian_decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=_cross_layer,
        ),
        global_fut_gaussian_decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=_cross_layer,
        ),
    ),
)


# -------------------------------------------------------------------------
# Joint loss: keep all six isolated V3 losses, then add the exact map/planner
# supervision used by the best planner experiment.
# -------------------------------------------------------------------------
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

_col_w = float(os.environ.get('COL_W', 0.1))
_plan_loss_cfg = dict(type='PlanLoss', weight=10.0)
if _col_w > 0:
    _plan_loss_cfg.update(
        col_loss_weight=_col_w,
        col_sat=True,
        col_safe_margin=0.5,
    )

_aux_w = float(os.environ.get('AUX_W', 2.0))
_aux_time_query_loss = dict(
    type='TimeQueryPlanLoss',
    weight=_aux_w,
    position_weight=1.0,
    beta=0.5,
)
_aligned_pos_loss = dict(
    type='AlignedTrajectoryPositionLoss',
    weight=0.5,
    beta=0.5,
    timestep_weights=(0.5, 0.75, 1.0, 1.0, 1.25, 1.5),
)
_aligned_pos_aux = dict(
    type='AlignedTrajectoryPositionLoss',
    weight=0.3,
    beta=0.5,
    pred_key='ego_fut_aux_preds',
    timestep_weights=(0.5, 0.75, 1.0, 1.0, 1.25, 1.5),
)
_aligned_pos_per_frame = dict(
    type='AlignedTrajectoryPositionLoss',
    weight=0.2,
    beta=0.5,
    pred_key='ego_fut_per_frame_preds',
    timestep_weights=(0.5, 0.75, 1.0, 1.0, 1.25, 1.5),
)

_v3_loss_cfgs = deepcopy(_base_.loss['loss_cfgs'])
loss = dict(
    type='MultiLoss',
    loss_cfgs=_v3_loss_cfgs + [
        _map_loss_cfg,
        _plan_loss_cfg,
        _aux_time_query_loss,
        _aligned_pos_loss,
        _aligned_pos_aux,
        _aligned_pos_per_frame,
    ],
)


# Add the map/planner keys to the inherited V3 loss plumbing.
loss_input_convertion = dict(
    ego_fut_preds='ego_fut_preds',
    ego_fut_gt='ego_fut_trajs',
    ego_fut_masks='ego_fut_masks',
    ego_fut_cmd='ego_fut_cmd',
    all_cls_scores='all_cls_scores',
    all_bbox_preds='all_bbox_preds',
    all_pts_preds='all_pts_preds',
    ego_fut_aux_preds='ego_fut_aux_preds',
    ego_fut_per_frame_preds='ego_fut_per_frame_preds',
)
