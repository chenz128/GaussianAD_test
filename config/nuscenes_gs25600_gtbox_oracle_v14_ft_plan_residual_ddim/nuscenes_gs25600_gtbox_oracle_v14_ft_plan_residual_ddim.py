"""v14: exact v12 global-residual baseline + residual DiT/DDIM.

The resolved configuration is deliberately a strict extension of
``v12_ft_plan_futattn_global_residual``. Dataset, augmentation, model modules
outside ``planner_head``, legacy losses, optimizer, learning-rate schedule,
epoch count and evaluation settings are inherited without modification. The
only changes are the planner class/new residual-DDIM parameters, one appended
loss, its input mappings, and an explicit weights-only checkpoint source.
"""

import os


custom_imports = dict(
    imports=[
        'model.planner.planner_v14_residual_ddim',
        'loss.residual_ddim_plan_loss',
    ],
    allow_failed_imports=False)

# This must be the exact configuration that produced the audited continuation
# checkpoint. Do not substitute the earlier plain v12 ft_plan configuration.
_base_ = [
    '../nuscenes_gs25600_gtbox_oracle_v12_ft_plan_'
    'futattn_global_residual/'
    'nuscenes_gs25600_gtbox_oracle_v12_ft_plan_'
    'futattn_global_residual.py'
]

_verified_v12_checkpoint = os.environ.get('VERIFIED_V12_CHECKPOINT', '')
if not _verified_v12_checkpoint:
    raise RuntimeError(
        'VERIFIED_V12_CHECKPOINT must point to the audited checkpoint from '
        'the v12_ft_plan_futattn_global_residual experiment.')
if 'gaussian_residual_dit' in _verified_v12_checkpoint.replace('\\', '/').lower():
    raise RuntimeError(
        'A gaussian_residual_dit checkpoint is invalid for v14 initialization.')

# ``load_from`` initializes weights only. ``resume_from`` remains empty so no
# optimizer, scheduler, scaler or epoch state can leak in from an old run.
load_from = _verified_v12_checkpoint
resume_from = ''

_residual_scale_text = os.environ.get('RESIDUAL_SCALE', '')
if _residual_scale_text:
    _residual_scale_values = tuple(
        float(item.strip()) for item in _residual_scale_text.split(','))
    if len(_residual_scale_values) != 12:
        raise ValueError('RESIDUAL_SCALE must contain 12 comma-separated values')
    _residual_scale = tuple(
        _residual_scale_values[index:index + 2]
        for index in range(0, 12, 2))
else:
    _residual_scale = ((1.0, 1.0),) * 6

model = dict(
    planner_head=dict(
        type='VADHeadFutAttnResidualDDIM',
        # Six-token micro-DiT in normalized cumulative-position residual space.
        residual_hidden_dims=192,
        residual_num_layers=4,
        residual_num_heads=8,
        residual_dropout=0.1,
        residual_scale=_residual_scale,
        residual_clip=8.0,
        # Truncated VP corruption and deterministic two-NFE DDIM; no CFG.
        diffusion_sigma_max=float(os.environ.get('SIGMA_MAX', 0.5)),
        diffusion_train_t_min=0.02,
        diffusion_sample_steps=2,
        num_inference_samples=4,
        fixed_noise_seed=3407,
        # Per-horizon Gaussian corridor selection and differentiable risk.
        gaussian_topk=int(os.environ.get('GAUSSIAN_TOPK', 128)),
        gaussian_corridor_radius=0.75,
        gaussian_importance_floor=0.1,
        dynamic_semantic_dims=10,
        risk_margin=0.5,
        risk_uncertainty_growth=0.15,
        # Keep the inherited v12 branch trainable exactly as in its baseline.
        # Only the new branch sees detached reference/context tensors, so its
        # additional loss cannot back-propagate into v12 or perception modules.
        detach_baseline=False,
        detach_residual_reference=True,
        detach_gaussian_context=True,
        keep_baseline_eval=False,
        # Conservative hard selection with baseline candidate zero.
        selector_risk_weight=4.0,
        selector_residual_weight=0.05,
        selector_dynamics_weight=0.02,
        selector_learned_weight=0.25,
        selector_baseline_margin=0.05,
        selector_risk_threshold=0.45,
        selector_max_normalized_residual=6.0,
        selector_max_acceleration=8.0,
        selector_max_jerk=15.0,
    ),
)

_residual_ddim_loss = dict(
    type='ResidualDDIMPlanLoss',
    weight=1.0,
    diffusion_weight=1.0,
    position_weight=0.5,
    fde_weight=0.25,
    safety_weight=0.1,
    dynamics_weight=0.05,
    rank_weight=0.1,
    rank_risk_weight=2.0,
    beta=0.5,
    time_interval=0.5,
    max_acceleration=8.0,
    max_jerk=15.0,
    timestep_weights=(0.5, 0.75, 1.0, 1.0, 1.25, 1.5))

# Preserve every baseline loss in its original order and append one new loss.
loss = dict(
    loss_cfgs=list(_base_.loss['loss_cfgs']) + [_residual_ddim_loss],
)

# Config dictionaries deep-merge, so every baseline mapping remains intact.
loss_input_convertion = dict(
    ego_fut_base_preds='ego_fut_base_preds',
    ego_fut_residual_preds='ego_fut_residual_preds',
    ego_fut_ddim_preds='ego_fut_ddim_preds',
    ego_fut_ddim_targets='ego_fut_ddim_targets',
    ego_fut_candidates='ego_fut_candidates',
    ego_fut_candidate_risk='ego_fut_candidate_risk',
    ego_fut_candidate_quality_logits='ego_fut_candidate_quality_logits',
    ego_fut_selected_index='ego_fut_selected_index',
    ego_fut_generated_risk='ego_fut_generated_risk',
)
