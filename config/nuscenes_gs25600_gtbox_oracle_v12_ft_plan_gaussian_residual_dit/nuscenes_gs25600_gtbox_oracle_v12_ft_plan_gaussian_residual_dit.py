"""Gaussian-conditioned residual DiT planner with adaLN-Zero DDIM.

The v7 dual-time planner is retained as a deterministic Gaussian-aware chain
proposal and warm-started from the trained dualtime_residual checkpoint. A
deeper DiT (facebookresearch style: adaLN-Zero modulation, gated attention and
MLP sub-layers, gaussian-conditioned embeddings) refines cumulative-position
residuals along a multi-step DDIM trajectory. The DiT output gate is zero
initialized so the first iterations reproduce the pre-trained chain exactly.
No collision loss is enabled, so this experiment isolates the planner
architecture.
"""

_base_ = [
    '../nuscenes_gs25600_gtbox_oracle_v12_ft_plan_dualtime_residual/'
    'nuscenes_gs25600_gtbox_oracle_v12_ft_plan_dualtime_residual.py'
]

import os

# Warm-start from the trained dualtime chain: all deterministic chain modules
# (ego_fut_decoder, ego_to_fut, fut/Gaussian decoders, residual/gate MLPs)
# match exactly. Only the new DiT head initializes from scratch.
load_from = (
    'exp/nuscenes_gs25600_v12_fixempty_ft_plan_dualtime_residual/'
    'checkpoints/epoch_15.pth')
max_epochs = 15

# 关闭训练阶段 EVAL：epoch % eval_every_epochs != 0 时跳过验证，
# 设为 100（> max_epochs）则整个训练期间都不跑 EVAL，避免浪费时间。
eval_every_epochs = 100

lr = float(os.environ.get('LR', 2e-4))
optimizer = dict(
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    paramwise_cfg=dict(custom_keys={'img_backbone': dict(lr_mult=0.1)}),
)

model = dict(
    planner_head=dict(
        type='VADHeadGaussianResidualDiT',
        dit_num_layers=4,
        dit_num_heads=8,
        dit_mlp_ratio=4.0,
        dit_dropout=0.1,
        local_gaussian_topk=64,
        num_diffusion_timesteps=100,
        diffusion_truncation_step=50,
        num_inference_steps=8,
        num_train_steps=8,
        beta_start=1e-4,
        beta_end=2e-2,
        residual_scale=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
        use_gated_output=True,
    ),
)

_diffusion_loss_cfg = dict(
    type='ResidualDiffusionPlanLoss',
    weight=1.0,
)

loss = dict(
    type='MultiLoss',
    loss_cfgs=list(_base_.loss['loss_cfgs']) + [_diffusion_loss_cfg],
)

loss_input_convertion = dict(
    residual_diffusion_noise_pred='residual_diffusion_noise_pred',
    residual_diffusion_noise_target='residual_diffusion_noise_target',
)
