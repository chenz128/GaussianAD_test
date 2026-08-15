"""Gaussian-conditioned residual DiT planner.

The v7 dual-time planner is retained as a deterministic Gaussian-aware chain
proposal. A new DiT refines cumulative-position residuals with two truncated
DDIM steps and same-frame nearest-Gaussian cross-attention. No collision loss
is enabled, so this experiment isolates the planner architecture.
"""

_base_ = [
    '../nuscenes_gs25600_gtbox_oracle_v12_ft_plan_dualtime_residual/'
    'nuscenes_gs25600_gtbox_oracle_v12_ft_plan_dualtime_residual.py'
]

model = dict(
    planner_head=dict(
        type='VADHeadGaussianResidualDiT',
        dit_num_layers=2,
        dit_num_heads=8,
        dit_feedforward_channels=512,
        dit_dropout=0.1,
        local_gaussian_topk=128,
        num_diffusion_timesteps=100,
        diffusion_truncation_step=20,
        num_inference_steps=2,
        residual_scale=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
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
