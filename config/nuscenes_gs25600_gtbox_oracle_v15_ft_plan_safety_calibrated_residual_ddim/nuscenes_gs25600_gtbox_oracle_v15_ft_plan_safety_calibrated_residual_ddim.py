"""v15: v14 residual DDIM + metric-aligned safety calibration/re-ranking.

This is a strict extension of the audited v12 global-residual baseline.  Every
dataset/model setting outside ``planner_head``, every legacy loss, optimizer,
LR schedule, epoch count, evaluation option and freeze setting is inherited
without modification.  For a fair controlled comparison with v14, weights are
initialized from the same audited v12-fixempty epoch-15 baseline.  Both the
residual-DDIM modules and the v15 safety modules are new parameters.
"""

import os


custom_imports = dict(
    imports=[
        'model.planner.planner_v15_safety_calibrated_ddim',
        'loss.safety_calibrated_residual_ddim_loss',
    ],
    allow_failed_imports=False)

_base_ = [
    '../nuscenes_gs25600_gtbox_oracle_v12_ft_plan_'
    'futattn_global_residual/'
    'nuscenes_gs25600_gtbox_oracle_v12_ft_plan_'
    'futattn_global_residual.py'
]

_verified_v12_checkpoint = os.environ.get('VERIFIED_V12_CHECKPOINT', '')
if not _verified_v12_checkpoint:
    raise RuntimeError(
        'VERIFIED_V12_CHECKPOINT must point to the audited epoch-15 '
        'checkpoint from '
        'nuscenes_gs25600_v12_fixempty_ft_plan_futattn_global_residual.')
_checkpoint_lower = _verified_v12_checkpoint.replace('\\', '/').lower()
_required_checkpoint_suffix = (
    '/exp/nuscenes_gs25600_v12_fixempty_ft_plan_futattn_global_residual/'
    'checkpoints/epoch_15.pth')
if not _checkpoint_lower.endswith(_required_checkpoint_suffix):
    raise RuntimeError(
        'v15 must use the exact v12-fixempty epoch-15 baseline checkpoint; '
        f'got {_verified_v12_checkpoint!r}')

# Weights-only continuation.  Optimizer/scheduler/scaler/epoch state is never
# restored, and the baseline's original 15-epoch schedule starts at epoch 0.
load_from = _verified_v12_checkpoint
resume_from = ''

# Keep the exact v14 residual-DDIM architecture for a controlled architecture
# comparison.  These modules are initialized fresh on top of v12-fixempty.
_residual_scale = ((1.0, 1.0),) * 6

model = dict(
    planner_head=dict(
        type='VADHeadFutAttnSafetyCalibratedResidualDDIM',
        # Exact v14 generator architecture/hyperparameters (fresh parameters).
        residual_hidden_dims=192,
        residual_num_layers=4,
        residual_num_heads=8,
        residual_dropout=0.1,
        residual_scale=_residual_scale,
        residual_clip=8.0,
        diffusion_train_t_min=0.02,
        diffusion_sample_steps=int(os.environ.get('DDIM_STEPS', 4)),
        num_inference_samples=4,
        fixed_noise_seed=3407,
        gaussian_topk=int(os.environ.get('GAUSSIAN_TOPK', 128)),
        gaussian_corridor_radius=0.75,
        gaussian_importance_floor=0.1,
        dynamic_semantic_dims=10,
        risk_margin=0.5,
        risk_uncertainty_growth=0.15,
        detach_baseline=False,
        detach_residual_reference=True,
        detach_gaussian_context=True,
        keep_baseline_eval=False,
        selector_risk_weight=4.0,
        selector_residual_weight=0.05,
        selector_dynamics_weight=0.02,
        selector_learned_weight=0.25,
        selector_baseline_margin=0.05,
        selector_risk_threshold=0.45,
        selector_max_normalized_residual=6.0,
        selector_max_acceleration=8.0,
        selector_max_jerk=15.0,
        # New v15 candidate-count-invariant temporal safety head.
        safety_hidden_dims=96,
        safety_num_layers=2,
        safety_num_heads=4,
        safety_dropout=0.1,
        safety_probability_threshold=float(
            os.environ.get('SAFETY_PROB_THRESHOLD', 0.5)),
        safety_cvar_fraction=1.0 / 3.0,
        safety_max_weight=1.0,
        safety_cvar_weight=0.5,
        safety_priority_weight=20.0,
        safety_tiebreak_weight=0.1,
        safety_baseline_margin=0.02,
    ),
)

_safety_calibrated_loss = dict(
    type='SafetyCalibratedResidualDDIMPlanLoss',
    # Exact v14 residual-DDIM objective.
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
    timestep_weights=(0.5, 0.75, 1.0, 1.0, 1.25, 1.5),
    # New detached metric-aligned calibration objective.
    safety_calibration_weight=0.25,
    safety_brier_weight=0.25,
    safety_rank_weight=0.1,
    safety_positive_weight=4.0,
    sat_safety_margin=0.5,
    sat_target_temperature=0.25,
    sat_collision_margin=0.0)

# Preserve every baseline loss and append one self-contained v15 loss.  The
# latter includes the exact v14 residual objective plus safety calibration.
loss = dict(
    loss_cfgs=list(_base_.loss['loss_cfgs']) + [_safety_calibrated_loss],
)

loss_input_convertion = dict(
    ego_fut_base_preds='ego_fut_base_preds',
    ego_fut_residual_preds='ego_fut_residual_preds',
    ego_fut_ddim_preds='ego_fut_ddim_preds',
    ego_fut_ddim_targets='ego_fut_ddim_targets',
    ego_fut_candidates='ego_fut_candidates',
    ego_fut_candidate_risk='ego_fut_candidate_risk',
    ego_fut_candidate_quality_logits='ego_fut_candidate_quality_logits',
    ego_fut_candidate_collision_logits=(
        'ego_fut_candidate_collision_logits'),
    ego_fut_candidate_feasible='ego_fut_candidate_feasible',
    ego_fut_selected_index='ego_fut_selected_index',
    ego_fut_generated_risk='ego_fut_generated_risk',
    # PlanLoss historically reads these annotations from ``metas`` itself.
    # The standalone v15 calibration loss needs explicit top-level mappings;
    # otherwise a missing annotation could be mistaken for an empty scene.
    attr_labels_planner='attr_labels_planner',
    gt_boxes='gt_boxes',
    fut_valid_flag='fut_valid_flag',
)
