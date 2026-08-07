"""
nuscenes_gs25600_gtbox_oracle_v12_ft_plan —— v12_fixempty 续训：接入 planner 轨迹

在 exp/nuscenes_gs25600_v12_fixempty/checkpoints/epoch_15.pth 基础上再训 15 epoch，
本配置只做“接 planner + 解冻 map/plan”这一件事，不改动其它 oracle 配置。

【改动 1：occ_flow 的 ego 补偿改用 planner 预测轨迹】
  model.head.use_plan_ego=True 时，forward_flow 里未来帧扣除自车运动所用的轨迹
  由 GT ego（metas['ego_fut_trajs']）切换为 planner_head 输出（ego_fut_preds），
  按 ego_fut_cmd 选中对应模态。这样 planner 才能收到 occ_flow 的一致性监督。
  plan_ego_warmup_epochs=2：前 2 轮仍用 GT ego，等 planner 被 PlanLoss 稳住后再切换。

【改动 2：解冻 map_decoder / planner_head 并重新启用 MapLoss + PlanLoss】
  v8 里 frozen_modules=['map_decoder','planner_head'] 且 loss 中没有 Map/Plan；
  这里 frozen_modules=[] 放开二者，并在原有 6 个 loss 之后追加 MapLoss / PlanLoss。

【改动 3：Gaussian 部分默认不冻结】
  继续用 occ/flow/det 微调当前帧高斯；若要冻结，把
  'lifter','encoder','temporal_encoder','decoder','head' 加进 frozen_modules 即可。

flow_grad_scale 沿用 v12 的 0.0（未来帧只训 offset，不回流当前帧高斯）。
续训用 load_from（从 epoch 0 计数训 15 轮）；若要连续 epoch/optimizer，改用
命令行 --resume-from。
"""

_base_ = ['../nuscenes_gs25600_gtbox_oracle_v12.py']
import os

# ============ 续训权重 / 轮数 ============
load_from = 'exp/nuscenes_gs25600_v12_fixempty/checkpoints/epoch_15.pth'
max_epochs = 15   # load_from 下 epoch 从 0 计数，训 15 轮（stage2）

# ============ 学习率：续训用更小 lr，避免打坏已收敛的 encoder ============
lr = float(os.environ.get("LR", 2e-4))
optimizer = dict(
    optimizer=dict(type="AdamW", lr=lr, weight_decay=0.01),
    paramwise_cfg=dict(custom_keys={'img_backbone': dict(lr_mult=0.1)}),
)

# ============ 解冻 map / plan（覆盖 v8 的 ['map_decoder','planner_head']）============
# 默认不冻结 Gaussian 部分；如需冻结把对应模块名加入下表。
frozen_modules = []
find_unused_parameters = False  # 对齐 base_plan；with_cp=True 与 find_unused_parameters=True 冲突

# ============ 用 planner 预测的 ego 轨迹替换 GT 做 occ_flow 补偿 ============
model = dict(
    head=dict(
        use_plan_ego=True,
        plan_ego_warmup_epochs=2,   # 前 2 轮用 GT ego，之后切 planner 预测
        plan_ego_detach=False,      # False: occ_flow 一致性梯度可回传 planner
    ),
)

# ============ map / plan loss 需要的维度 ============
num_map_classes = len(_base_.map_classes)
pc_range = [-30.0, -30.0, -2.0, 30.0, 30.0, 2.0]
fixed_ptsnum_per_gt_line = 20
fixed_ptsnum_per_pred_line = 20

# ============ 重新启用 MapLoss + PlanLoss（追加到 v12 原有 6 个 loss 之后）============
_map_loss_cfg = dict(
    type='MapLoss',
    loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0),
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
        pc_range=pc_range),
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
_plan_loss_cfg = dict(
    type='PlanLoss',
    weight=10.0,  # 对齐 base_plan，planner 需要足够梯度信号
)

# 复用 v12（=v8）已定义的 6 个 loss，再追加 Map / Plan 两项。
loss = dict(
    type='MultiLoss',
    loss_cfgs=list(_base_.loss['loss_cfgs']) + [_map_loss_cfg, _plan_loss_cfg],
)

# ============ 补充 loss_input_convertion（dict 深合并，追加 map/plan 输入键）============
loss_input_convertion = dict(
    # PlanLoss 需要
    ego_fut_preds='ego_fut_preds',
    ego_fut_gt='ego_fut_trajs',
    ego_fut_masks='ego_fut_masks',
    ego_fut_cmd='ego_fut_cmd',
    # MapLoss + PlanLoss 需要 map decoder 输出
    all_cls_scores='all_cls_scores',
    all_bbox_preds='all_bbox_preds',
    all_pts_preds='all_pts_preds',
)
