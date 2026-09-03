"""Planner-only v16 on the audited strong V3-SE3 OCC checkpoint.

The resolved configuration is a strict extension of
``nuscenes_gs25600_v3_se3_ft_plan_futattn_global_residual_full``.  Dataset,
augmentation, frontend architecture, OCC/map/detection heads, legacy losses,
optimizer, scheduler, epoch count, batch size and evaluation settings are
inherited unchanged.  The allowed differences are:

1. a new planner class and its new parameters;
2. one appended planner-only loss and its input mappings;
3. weights-only initialization from the audited strong-OCC epoch-16 model;
4. freezing every frontend module and the inherited deterministic planner.
"""

from copy import deepcopy
import os


_base_ = [
    '../nuscenes_gs25600_v3_se3_ft_plan_futattn_global_residual_full/'
    'nuscenes_gs25600_v3_se3_ft_plan_futattn_global_residual_full.py'
]

custom_imports = deepcopy(_base_.custom_imports)
custom_imports['imports'] = list(custom_imports['imports']) + [
    'model.segmentor.bev_segmentor_v3_future_plan_frozen_frontend',
    'model.planner.planner_v14_residual_ddim',
    'model.planner.planner_v16_frozen_occ_truncated_ddim',
    'loss.residual_ddim_plan_loss',
    'loss.safety_calibrated_residual_ddim_loss',
    'loss.frozen_occ_truncated_ddim_loss',
]

_default_strong_occ_checkpoint = (
    'exp/nuscenes_gs25600_v3_se3_ft_plan_futattn_global_residual_full/'
    'checkpoints/epoch_16.pth')
_strong_occ_checkpoint = os.environ.get(
    'STRONG_OCC_CHECKPOINT', _default_strong_occ_checkpoint)
if not _strong_occ_checkpoint:
    raise RuntimeError('STRONG_OCC_CHECKPOINT must not be empty')
if 'v3_se3_ft_plan_futattn_global_residual_full' not in (
        _strong_occ_checkpoint.replace('\\', '/')):
    raise RuntimeError(
        'v16 must initialize from the audited strong V3-SE3 OCC experiment')

# Weights only.  ``train.py`` can still auto-resume from work_dir/latest.pth;
# the launch script therefore refuses a non-empty reused work directory.
load_from = _strong_occ_checkpoint
resume_from = ''

# The frontend must be numerically immutable.  The new planner additionally
# freezes all inherited deterministic-anchor children inside its constructor.
frozen_modules = [
    'img_backbone',
    'img_neck',
    'lifter',
    'encoder',
    'temporal_encoder',
    'decoder',
    'map_decoder',
    'head',
]

_residual_scale_text = os.environ.get('RESIDUAL_SCALE', '')
if _residual_scale_text:
    _scale_values = tuple(
        float(item.strip()) for item in _residual_scale_text.split(','))
    if len(_scale_values) != 12:
        raise ValueError('RESIDUAL_SCALE must contain 12 comma-separated values')
    _residual_scale = tuple(
        _scale_values[index:index + 2]
        for index in range(0, 12, 2))
else:
    _residual_scale = ((0.75, 0.75), (0.85, 0.85), (1.0, 1.0),
                       (1.1, 1.1), (1.25, 1.25), (1.4, 1.4))

model = dict(
    # Parameter-free guard: inherited forward is unchanged, while frozen
    # frontend BN/dropout modules remain in eval mode during planner training.
    type='BEVSegmentorV3FuturePlanFrozenFrontend',
    planner_head=dict(
        type='VADHeadFrozenOccTruncatedResidualDDIM',
        # Small six-token DiT; diffusion refines residuals around the frozen
        # strong planner rather than generating a complete path from noise.
        residual_hidden_dims=192,
        residual_num_layers=3,
        residual_num_heads=8,
        residual_dropout=0.1,
        residual_scale=_residual_scale,
        residual_clip=4.0,
        diffusion_sample_steps=int(os.environ.get('DDIM_STEPS', 2)),
        num_inference_samples=int(os.environ.get('DDIM_SAMPLES', 4)),
        fixed_noise_seed=3407,
        truncated_start_t=float(os.environ.get('DDIM_START_T', 0.25)),
        truncated_noise_scale=1.0,
        # Exact strong-OCC bank, route-corridor Top-K and full ego footprint.
        gaussian_topk=int(os.environ.get('GAUSSIAN_TOPK', 128)),
        gaussian_corridor_radius=0.75,
        gaussian_importance_floor=0.05,
        obstacle_semantic_indices=(2, 3, 4, 5, 6, 7, 9, 10),
        footprint_longitudinal_samples=5,
        footprint_lateral_samples=3,
        footprint_gaussian_margin=0.35,
        risk_margin=0.35,
        risk_uncertainty_growth=0.12,
        # Bounded analytic guidance affects generated proposals only.  The
        # exact deterministic baseline remains candidate zero.
        safety_guidance_scale=0.20,
        safety_guidance_clip=0.25,
        safety_guidance_activation=0.15,
        guidance_gaussian_topk=32,
        # Conservative risk-first no-regression selector.
        selector_risk_weight=6.0,
        selector_residual_weight=0.10,
        selector_dynamics_weight=0.05,
        selector_learned_weight=0.25,
        selector_baseline_margin=0.05,
        selector_risk_threshold=0.35,
        selector_max_normalized_residual=2.5,
        selector_max_acceleration=8.0,
        selector_max_jerk=15.0,
        selector_min_risk_improvement=0.03,
        selector_risk_tolerance=0.01,
        selector_quality_improvement=0.05,
        detach_baseline=True,
        detach_residual_reference=True,
        detach_gaussian_context=True,
        keep_baseline_eval=True,
        freeze_deterministic_anchor=True,
    ),
)

_v16_loss = dict(
    type='FrozenOccTruncatedDDIMPlanLoss',
    weight=1.0,
    anchor_candidate_count=3,
    coverage_weight=1.0,
    fde_weight=0.5,
    sat_weight=0.5,
    occ_risk_weight=0.1,
    dynamics_weight=0.05,
    rank_weight=0.25,
    diversity_weight=0.01,
    trust_region_weight=0.05,
    beta=0.5,
    softmin_temperature=0.25,
    rank_temperature=0.5,
    oracle_collision_weight=8.0,
    oracle_fde_weight=0.5,
    sat_safety_margin=0.35,
    sat_temperature=0.20,
    time_interval=0.5,
    max_acceleration=8.0,
    max_jerk=15.0,
    max_anchor_deviation=2.0,
    minimum_endpoint_separation=0.30,
    timestep_weights=(0.5, 0.75, 1.0, 1.0, 1.25, 1.5),
)

# Preserve every strong-baseline loss and append one planner-only objective.
loss = dict(loss_cfgs=list(_base_.loss['loss_cfgs']) + [_v16_loss])

# Deep merge preserves every inherited mapping.
loss_input_convertion = dict(
    ego_fut_base_preds='ego_fut_base_preds',
    ego_fut_candidates='ego_fut_candidates',
    ego_fut_candidate_risk='ego_fut_candidate_risk',
    ego_fut_candidate_quality_logits='ego_fut_candidate_quality_logits',
    ego_fut_selected_index='ego_fut_selected_index',
    ego_fut_generated_risk='ego_fut_generated_risk',
    attr_labels_planner='attr_labels_planner',
    gt_boxes='gt_boxes',
    fut_valid_flag='fut_valid_flag',
)
