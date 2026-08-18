"""Exact v7 dual-time anchor plus hybrid-time bounded position residuals.

This stage warm-starts from the v7 epoch-15 checkpoint. The complete v7 dual-time anchor
is unchanged; only the new hybrid-time residual branch receives detached-anchor
auxiliary supervision. No collision loss is added.
"""

_base_ = [
    '../nuscenes_gs25600_gtbox_oracle_v12_ft_plan_dualtime_residual/'
    'nuscenes_gs25600_gtbox_oracle_v12_ft_plan_dualtime_residual.py'
]

import os

load_from = (
    'exp/nuscenes_gs25600_v12_fixempty_ft_plan_dualtime_residual/'
    'checkpoints/epoch_15.pth')
max_epochs = 15

lr = float(os.environ.get('LR', 2e-4))
optimizer = dict(
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    paramwise_cfg=dict(custom_keys={'img_backbone': dict(lr_mult=0.1)}),
)

model = dict(
    planner_head=dict(
        type='VADHeadHybridTimePositionResidual',
        time_interval=0.5,
        num_fourier_bands=8,
        position_residual_scale=(0.35, 0.65, 0.95, 1.25, 1.60, 2.00),
        initial_gate=0.2,
        anchor_gradient_scale=0.25,
        hybrid_feature_gradient_scale=0.25,
    ),
)

_hybrid_residual_loss = dict(
    type='HybridPositionResidualLoss',
    weight=0.5,
    position_weight=1.0,
    trust_region_weight=0.05,
    beta=0.5,
)

loss = dict(
    type='MultiLoss',
    loss_cfgs=list(_base_.loss['loss_cfgs']) + [_hybrid_residual_loss],
)

loss_input_convertion = dict(
    ego_fut_position_aux_preds='ego_fut_position_aux_preds',
    ego_fut_applied_residual_normalized=(
        'ego_fut_applied_residual_normalized'),
    ego_fut_position_gate='ego_fut_position_gate',
)
