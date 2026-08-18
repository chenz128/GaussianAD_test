"""Single-head time-aligned ego-to-future-Gaussian planning.

The planner starts from the trained same-frame FutAttn model. Learned time is
shared by ego queries and future Gaussian K/V through zero-initialized Gaussian
and continuous-time scales. A fixed locality prior protects same-frame
attention while still allowing adjacent-frame context. No trajectory residual
or collision loss is used.
"""

_base_ = [
    '../nuscenes_gs25600_gtbox_oracle_v12_ft_plan_futattn/'
    'nuscenes_gs25600_gtbox_oracle_v12_ft_plan_futattn.py'
]

import os

load_from = (
    'exp/nuscenes_gs25600_v12_fixempty_ft_plan_futattn/'
    'checkpoints/epoch_15.pth')
max_epochs = 15
eval_every_epochs = 9999  # 关闭每轮 EVAL（0 会除零，用大数跳过）

lr = float(os.environ.get('LR', 2e-4))
optimizer = dict(
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    paramwise_cfg=dict(custom_keys={'img_backbone': dict(lr_mult=0.1)}),
)

# 全量训练：不冻结任何模块
frozen_modules = []
find_unused_parameters = True

_decoder_layer = dict(
    type='MyCustomBaseTransformerLayer',
    attn_cfgs=[dict(
        type='MultiheadAttention',
        embed_dims=128,
        num_heads=8,
        dropout=0.1)],
    feedforward_channels=256,
    ffn_dropout=0.1,
    batch_first=False,
)

model = dict(
    planner_head=dict(
        type='VADHeadTimeAlignedGaussian',
        time_interval=0.5,
        num_fourier_bands=4,
        fut_self_decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=dict(
                **_decoder_layer,
                operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
        ego_fut_gaussian_decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=dict(
                **_decoder_layer,
                operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
    ),
)

_aligned_position_loss = dict(
    type='AlignedTrajectoryPositionLoss',
    weight=0.5,
    beta=0.5,
    timestep_weights=(0.5, 0.75, 1.0, 1.0, 1.25, 1.5),
)
loss = dict(
    type='MultiLoss',
    loss_cfgs=list(_base_.loss['loss_cfgs']) + [_aligned_position_loss],
)
