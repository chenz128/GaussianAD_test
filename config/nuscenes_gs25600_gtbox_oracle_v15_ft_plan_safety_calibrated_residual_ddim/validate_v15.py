"""Strict audit for the v15 safety-calibrated residual DDIM continuation.

Usage::

    VERIFIED_V12_CHECKPOINT=/absolute/path/to/fixempty/epoch_15.pth python \
      config/nuscenes_gs25600_gtbox_oracle_v15_ft_plan_safety_calibrated_residual_ddim/validate_v15.py
"""

import math
import os
from pathlib import Path
import sys

import torch
from mmengine.config import Config
from mmengine.registry import MODELS


config_path = Path(__file__).with_name(
    'nuscenes_gs25600_gtbox_oracle_v15_ft_plan_'
    'safety_calibrated_residual_ddim.py')
repo_root = config_path.parents[2]
baseline_path = repo_root / (
    'config/nuscenes_gs25600_gtbox_oracle_v12_ft_plan_'
    'futattn_global_residual/'
    'nuscenes_gs25600_gtbox_oracle_v12_ft_plan_'
    'futattn_global_residual.py')
v14_path = repo_root / (
    'config/nuscenes_gs25600_gtbox_oracle_v14_ft_plan_residual_ddim/'
    'nuscenes_gs25600_gtbox_oracle_v14_ft_plan_residual_ddim.py')
sys.path.insert(0, str(repo_root))

checkpoint_text = os.environ.get('VERIFIED_V12_CHECKPOINT', '')
if not checkpoint_text:
    raise RuntimeError('VERIFIED_V12_CHECKPOINT is required')
checkpoint_path = Path(checkpoint_text)
if not checkpoint_path.is_absolute():
    raise ValueError('VERIFIED_V12_CHECKPOINT must be an absolute path')
checkpoint_lower = checkpoint_text.replace('\\', '/').lower()
required_checkpoint_suffix = (
    '/exp/nuscenes_gs25600_v12_fixempty_ft_plan_futattn_global_residual/'
    'checkpoints/epoch_15.pth')
if not checkpoint_lower.endswith(required_checkpoint_suffix):
    raise ValueError(
        'the exact v12-fixempty epoch-15 checkpoint is required')
if not checkpoint_path.is_file():
    raise FileNotFoundError(checkpoint_path)
if not baseline_path.is_file() or not v14_path.is_file():
    raise FileNotFoundError('the audited v12/v14 configs must both exist')

config = Config.fromfile(config_path)
baseline = Config.fromfile(baseline_path)
v14_config = Config.fromfile(v14_path)

# train.py relies on Config.fromfile(custom_imports) before it imports the
# model package.  Check that exact startup path here; later explicit imports
# must not be allowed to hide a broken registration configuration.
if MODELS.get('VADHeadFutAttnSafetyCalibratedResidualDDIM') is None:
    raise AssertionError(
        'v15 planner was not registered by config custom_imports')

from loss import OPENOCC_LOSS  # noqa: E402
from loss.safety_calibrated_residual_ddim_loss import (  # noqa: E402
    MetricAlignedVehicleSAT,
    SafetyCalibratedResidualDDIMPlanLoss,
)
from model.planner.planner_v15_safety_calibrated_ddim import (  # noqa: E402,F401
    VADHeadFutAttnSafetyCalibratedResidualDDIM,
)
from model.planner.planner_v14_residual_ddim import (  # noqa: E402
    VADHeadFutAttnResidualDDIM,
)


def _assert_equal(actual, expected, label):
    if actual != expected:
        raise AssertionError(
            f'{label} changed:\n  expected={expected!r}\n  actual={actual!r}')


# -------------------------------------------------------------------------
# 1. Config parity: only the documented v15 extension may differ from v12.
# -------------------------------------------------------------------------
intentional_top_level = {
    'custom_imports',
    'model',
    'loss',
    'loss_input_convertion',
    'load_from',
    'resume_from',
}
resolved_keys = {
    key for key in set(config.keys()) | set(baseline.keys())
    if not key.startswith('_')
}
for key in sorted(resolved_keys - intentional_top_level):
    if key not in config or key not in baseline:
        raise AssertionError(f'unexpected top-level config key: {key}')
    _assert_equal(config[key], baseline[key], f'top-level {key}')

for key in sorted(set(config.model.keys()) | set(baseline.model.keys())):
    if key == 'planner_head':
        continue
    if key not in config.model or key not in baseline.model:
        raise AssertionError(f'unexpected non-planner model key: {key}')
    _assert_equal(config.model[key], baseline.model[key], f'model.{key}')

baseline_head = baseline.model.planner_head
v14_head = v14_config.model.planner_head
v15_head = config.model.planner_head
assert baseline_head.type == 'VADHeadFutAttnGlobalResidual'
assert v14_head.type == 'VADHeadFutAttnResidualDDIM'
assert v15_head.type == 'VADHeadFutAttnSafetyCalibratedResidualDDIM'
for key in sorted(baseline_head.keys()):
    if key != 'type':
        _assert_equal(v15_head[key], baseline_head[key], f'planner_head.{key}')

v15_only_head_keys = {
    'safety_hidden_dims',
    'safety_num_layers',
    'safety_num_heads',
    'safety_dropout',
    'safety_probability_threshold',
    'safety_cvar_fraction',
    'safety_max_weight',
    'safety_cvar_weight',
    'safety_priority_weight',
    'safety_tiebreak_weight',
    'safety_baseline_margin',
}
for key in sorted(v14_head.keys()):
    if key != 'type':
        _assert_equal(v15_head[key], v14_head[key], f'v14 planner setting {key}')
unexpected_v15_head = (
    set(v15_head.keys()) - set(v14_head.keys()) - v15_only_head_keys)
if unexpected_v15_head:
    raise AssertionError(
        'unapproved v15 planner keys: ' + repr(sorted(unexpected_v15_head)))

baseline_losses = list(baseline.loss.loss_cfgs)
v14_residual_loss = dict(v14_config.loss.loss_cfgs[-1])
v15_losses = list(config.loss.loss_cfgs)
_assert_equal(v15_losses[:-1], baseline_losses, 'legacy loss list/order')
assert len(v15_losses) == len(baseline_losses) + 1
assert v15_losses[-1].type == 'SafetyCalibratedResidualDDIMPlanLoss'
for key, value in v14_residual_loss.items():
    if key != 'type':
        _assert_equal(v15_losses[-1][key], value, f'v14 residual loss {key}')
allowed_v15_loss_keys = {
    'safety_calibration_weight',
    'safety_brier_weight',
    'safety_rank_weight',
    'safety_positive_weight',
    'sat_safety_margin',
    'sat_target_temperature',
    'sat_collision_margin',
}
unexpected_v15_loss = (
    set(v15_losses[-1].keys())
    - set(v14_residual_loss.keys())
    - allowed_v15_loss_keys)
if unexpected_v15_loss:
    raise AssertionError(
        'unapproved v15 loss keys: ' + repr(sorted(unexpected_v15_loss)))

for key, value in baseline.loss_input_convertion.items():
    _assert_equal(
        config.loss_input_convertion[key], value,
        f'loss_input_convertion.{key}')
for key, value in v14_config.loss_input_convertion.items():
    _assert_equal(
        config.loss_input_convertion[key], value,
        f'v14 loss input {key}')
new_loss_inputs = (
    set(config.loss_input_convertion.keys())
    - set(v14_config.loss_input_convertion.keys()))
_assert_equal(
    new_loss_inputs,
    {
        'ego_fut_candidate_collision_logits',
        'ego_fut_candidate_feasible',
        'fut_valid_flag',
    },
    'loss inputs added beyond v14')

assert config.load_from == checkpoint_text
assert config.resume_from == ''
assert MODELS.get('VADHeadFutAttnSafetyCalibratedResidualDDIM') is not None
assert OPENOCC_LOSS.get('SafetyCalibratedResidualDDIMPlanLoss') is not None
for annotation_key in ('attr_labels_planner', 'gt_boxes', 'fut_valid_flag'):
    assert config.loss_input_convertion[annotation_key] == annotation_key


# -------------------------------------------------------------------------
# 2. Checkpoint lineage: exact v12-fixempty, with only v14/v15 modules missing.
# -------------------------------------------------------------------------
checkpoint = torch.load(checkpoint_path, map_location='cpu')
if checkpoint.get('epoch') != 15:
    raise AssertionError(
        f'expected checkpoint epoch=15, got {checkpoint.get("epoch")!r}')
state_dict = checkpoint.get('state_dict', checkpoint)
planner_state = {}
for key, value in state_dict.items():
    normalized = key[len('module.'):] if key.startswith('module.') else key
    marker = 'planner_head.'
    if marker in normalized:
        planner_state[normalized.split(marker, 1)[1]] = value
if not planner_state:
    raise AssertionError('checkpoint has no planner_head.* state')
if any((key.startswith('residual_output.')
        or key.startswith('candidate_safety_')
        or key.startswith('candidate_gaussian_safety_'))
       for key in planner_state):
    raise AssertionError(
        'continuation checkpoint contains v14/v15 parameters; expected the '
        'clean v12-fixempty baseline')

baseline_planner = MODELS.build(baseline_head)
baseline_planner.init_weights()
baseline_load = baseline_planner.load_state_dict(planner_state, strict=False)
if baseline_load.missing_keys or baseline_load.unexpected_keys:
    raise AssertionError(
        'checkpoint is not the exact v12 baseline planner:\n'
        f'  missing={baseline_load.missing_keys!r}\n'
        f'  unexpected={baseline_load.unexpected_keys!r}')
baseline_state = baseline_planner.state_dict()

initialization_seed = 3407
torch.manual_seed(initialization_seed)
v14_planner = MODELS.build(v14_head)
v14_planner.init_weights()
v14_load = v14_planner.load_state_dict(planner_state, strict=False)
v14_state = v14_planner.state_dict()
v14_new_keys = set(v14_state) - set(baseline_state)
if set(v14_load.missing_keys) != v14_new_keys or v14_load.unexpected_keys:
    raise AssertionError(
        'unexpected v12 -> v14 checkpoint load result:\n'
        f'  missing={v14_load.missing_keys!r}\n'
        f'  unexpected={v14_load.unexpected_keys!r}')
v14_allowed_prefixes = tuple(
    name + '.'
    for name in VADHeadFutAttnResidualDDIM._residual_child_names()) + (
        'residual_scale', 'fixed_residual_noise')
illegal_v14_new = [
    key for key in v14_new_keys
    if not key.startswith(v14_allowed_prefixes)]
if illegal_v14_new:
    raise AssertionError(
        'unapproved fresh v14 keys: ' + repr(sorted(illegal_v14_new)))

torch.manual_seed(initialization_seed)
planner = MODELS.build(v15_head)
planner.init_weights()
v15_load = planner.load_state_dict(planner_state, strict=False)
v15_state = planner.state_dict()
v15_new_keys = set(v15_state) - set(baseline_state)
if set(v15_load.missing_keys) != v15_new_keys:
    raise AssertionError(
        'unexpected v12 -> v15 missing keys:\n'
        f'  expected={sorted(v15_new_keys)!r}\n'
        f'  actual={v15_load.missing_keys!r}')
safety_prefixes = (
    'candidate_safety_encoder.',
    'candidate_gaussian_safety_proj.',
    'candidate_safety_temporal.',
    'candidate_safety_norm.',
    'candidate_collision_head.',
)
allowed_v15_prefixes = v14_allowed_prefixes + safety_prefixes
illegal_v15_new = [
    key for key in v15_new_keys
    if not key.startswith(allowed_v15_prefixes)]
if illegal_v15_new:
    raise AssertionError(
        'unapproved fresh v15 keys: ' + repr(sorted(illegal_v15_new)))
if v15_load.unexpected_keys:
    raise AssertionError('unexpected v12 keys: ' + repr(v15_load.unexpected_keys))

for key, value in baseline_state.items():
    if (not torch.equal(v14_state[key], value)
            or not torch.equal(v15_state[key], value)):
        raise AssertionError(
            f'inherited v12-fixempty tensor differs after load: {key}')
for key, value in v14_state.items():
    if key not in v15_state or not torch.equal(v15_state[key], value):
        raise AssertionError(
            f'v15 DDIM initialization differs from same-seed v14: {key}')
assert torch.count_nonzero(planner.residual_output.weight) == 0
assert torch.count_nonzero(planner.residual_output.bias) == 0
assert torch.count_nonzero(planner.candidate_quality_mlp[-1].weight) == 0
assert torch.count_nonzero(planner.candidate_quality_mlp[-1].bias) == 0
assert torch.count_nonzero(planner.candidate_collision_head.weight) == 0
assert torch.count_nonzero(planner.candidate_collision_head.bias) == 0
assert torch.equal(
    planner.residual_scale,
    torch.ones_like(planner.residual_scale))

schedule_t = torch.tensor([0.0, 0.5, 1.0])
schedule_alpha, schedule_sigma = planner._diffusion_schedule(schedule_t)
assert torch.allclose(schedule_alpha.square() + schedule_sigma.square(),
                      torch.ones_like(schedule_alpha), atol=1e-6)
assert schedule_alpha[0] == 1.0 and schedule_sigma[0] == 0.0
assert schedule_alpha[-1] == 0.0 and schedule_sigma[-1] == 1.0


# -------------------------------------------------------------------------
# 3. At zero safety logits, v15 re-ranking retains the v14 selection.
# -------------------------------------------------------------------------
batch, modes, candidates, timesteps = 2, 3, 5, 6
baseline_displacement = 0.05 * torch.randn(batch, modes, timesteps, 2)
baseline_position = baseline_displacement.cumsum(dim=-2)
candidate_position = baseline_position[:, :, None] + (
    0.05 * torch.randn(batch, modes, candidates, timesteps, 2))
candidate_position[:, :, 0] = baseline_position
# Include Gaussian-risk values above the inherited v14 feasibility threshold
# and an explicitly infeasible residual candidate.  Equal safety logits must
# still retain the complete v14 selector semantics in these edge cases.
candidate_position[:, :, -1, :, 0] += 20.0
candidate_risk = torch.rand(batch, modes, candidates, timesteps) * 0.9
quality_logits = torch.randn(batch, modes, candidates) * 0.1

planner.eval()
with torch.no_grad():
    zero_logits = planner._candidate_safety_logits(
        candidate_position, baseline_position, candidate_risk)
    assert torch.count_nonzero(zero_logits) == 0

    # Exercise the real v15 sparse-world branch as well.  Its final head is
    # deliberately zero-initialized, so adding trajectory-aligned Gaussian
    # context must remain finite and preserve v14 selection before training.
    num_gaussians = 32
    synthetic_scene = {
        'content': torch.randn(
            batch, timesteps, num_gaussians, planner.embed_dims),
        'future_xy': torch.randn(
            batch, timesteps, num_gaussians, 2),
        'scale_xy': torch.full(
            (batch, timesteps, num_gaussians, 2), 0.5),
        'opacity': torch.full(
            (batch, timesteps, num_gaussians, 1), 0.8),
        'dynamic_probability': torch.full(
            (batch, timesteps, num_gaussians, 1), 0.7),
        'importance': torch.full(
            (batch, timesteps, num_gaussians, 1), 0.6),
        'rotation_xy': torch.eye(2).reshape(1, 1, 2, 2).expand(
            batch, num_gaussians, -1, -1).clone(),
    }
    scene_logits = planner._candidate_safety_logits(
        candidate_position, baseline_position, candidate_risk,
        synthetic_scene)
    assert scene_logits.shape == candidate_risk.shape
    assert torch.isfinite(scene_logits).all()
    assert torch.count_nonzero(scene_logits) == 0

    # Both new heads are zero-initialized on the v12 checkpoint.  The complete
    # 4-NFE sampler, candidate construction and v15 selector must therefore be
    # an exact deterministic baseline fallback before any training update.
    ddim_output = planner._ddim_sample(
        baseline_displacement, baseline_position, synthetic_scene)
    assert torch.allclose(
        ddim_output['ego_fut_preds'], baseline_displacement, atol=1e-6)
    expected_candidates = baseline_displacement[:, :, None].expand(
        -1, -1, planner.num_inference_samples + 1, -1, -1)
    assert torch.allclose(
        ddim_output['ego_fut_candidates'], expected_candidates, atol=1e-6)
    assert torch.count_nonzero(ddim_output['ego_fut_selected_index']) == 0
    assert torch.count_nonzero(
        ddim_output['ego_fut_candidate_collision_logits']) == 0

    old_selection = v14_planner._select_candidates(
        candidate_position, baseline_position,
        candidate_risk, quality_logits)
    new_selection = planner._safety_first_selection(
        candidate_position, baseline_position,
        candidate_risk, quality_logits, zero_logits)
assert torch.equal(old_selection['selected'], new_selection['selected'])
assert not new_selection['safety_informative'].any()
assert torch.isfinite(new_selection['costs']).all()


# -------------------------------------------------------------------------
# 4. Metric-aligned SAT label contracts and calibration gradient isolation.
# -------------------------------------------------------------------------
sat = MetricAlignedVehicleSAT()
candidate_displacement = torch.zeros(1, 2, timesteps, 2)
candidate_displacement[:, :, :, 0] = 0.5
candidate_displacement[:, 1, 0, 1] = 10.0
safe_target = candidate_displacement[:, 1].clone()
ego_mask = torch.ones(1, timesteps)
attr = torch.zeros(1, 1, 34)
attr[..., 12:18] = 1.0
attr[..., 27] = 14.0
boxes = torch.zeros(1, 1, 7)
boxes[..., 0] = 2.5
boxes[..., 3] = 1.8
boxes[..., 4] = 4.0
boxes[..., 6] = -math.pi / 2.0

sat_output = sat(
    candidate_displacement, safe_target, ego_mask,
    attr, boxes, torch.ones(1, dtype=torch.bool))
assert sat_output['hard_target'][0, 0].any()
assert not sat_output['hard_target'][0, 1].any()
assert sat_output['valid'].sum() == 2 * timesteps

pedestrian_attr = attr.clone()
pedestrian_attr[..., 27] = 2.0
pedestrian_output = sat(
    candidate_displacement, safe_target, ego_mask,
    pedestrian_attr, boxes, torch.ones(1, dtype=torch.bool))
assert not pedestrian_output['hard_target'].any()

gt_collision_output = sat(
    candidate_displacement, candidate_displacement[:, 0], ego_mask,
    attr, boxes, torch.ones(1, dtype=torch.bool))
assert (gt_collision_output['valid'][0, :, 1:5].sum()
        < sat_output['valid'][0, :, 1:5].sum())

loss_probe = SafetyCalibratedResidualDDIMPlanLoss()
collision_logits = torch.zeros(
    1, modes, 2, timesteps, requires_grad=True)
residual_prediction = torch.zeros(
    1, modes, timesteps, 2, requires_grad=True)
candidate_probe = torch.zeros(
    1, modes, 2, timesteps, 2, requires_grad=True)
candidate_probe.data[:, :, :, :, 0] = 0.5
candidate_probe.data[:, :, 1, 0, 1] = 10.0
loss_inputs = {
    'ego_fut_residual_preds': residual_prediction,
    'ego_fut_ddim_preds': torch.zeros(1, modes, timesteps, 2),
    'ego_fut_ddim_targets': torch.zeros(1, modes, timesteps, 2),
    'ego_fut_generated_risk': torch.zeros(1, modes, timesteps),
    'ego_fut_candidates': candidate_probe,
    'ego_fut_candidate_risk': torch.zeros(1, modes, 2, timesteps),
    'ego_fut_candidate_quality_logits': torch.zeros(1, modes, 2),
    'ego_fut_candidate_collision_logits': collision_logits,
    'ego_fut_selected_index': torch.zeros(1, modes, dtype=torch.long),
    'ego_fut_gt': safe_target,
    'ego_fut_masks': ego_mask,
    'ego_fut_cmd': torch.tensor([[1.0, 0.0, 0.0]]),
    'attr_labels_planner': attr,
    'gt_boxes': boxes,
    'fut_valid_flag': torch.ones(1, dtype=torch.bool),
}
probe_total, probe_logs = loss_probe(loss_inputs)
probe_total.backward()
assert collision_logits.grad is not None
assert torch.isfinite(collision_logits.grad).all()
# SAT labels are detached.  Candidate tensors are used only for detached label
# construction and the v14 ranking target, so calibration cannot move them.
assert candidate_probe.grad is None
assert 'sat_oracle_all_candidates_collision_rate' in probe_logs

missing_annotation_inputs = dict(loss_inputs)
missing_annotation_inputs.pop('gt_boxes')
try:
    loss_probe(missing_annotation_inputs)
except KeyError as error:
    assert 'gt_boxes' in str(error)
else:
    raise AssertionError('missing metric SAT annotation did not fail fast')


print('v15 strict config/checkpoint/safety audit: OK')
print('baseline config:', baseline_path)
print('v14 config:', v14_path)
print('verified v12-fixempty continuation checkpoint:', checkpoint_path)
print('checkpoint epoch field:', checkpoint.get('epoch'))
print('matched v12-fixempty planner tensors:', len(baseline_state))
print('fresh v14/DDIM state keys:', len(v14_new_keys))
print('fresh v15-only safety state keys:', len(v15_new_keys - v14_new_keys))
print('legacy losses preserved:', len(baseline_losses))
print('new loss appended: 1 (v14 objective + detached SAT calibration)')
for key in (
        'max_epochs', 'lr', 'optimizer', 'eval_every_epochs',
        'find_unused_parameters', 'frozen_modules'):
    if key in baseline:
        print(f'baseline parity {key}:', config[key])
print('zero-logit v15 selection == v14 selection: OK')
print('trajectory-aligned Gaussian safety branch: OK')
print('vehicle-only SAT + GT-collision masking: OK')
print('metric SAT annotation mapping/fail-fast: OK')
print('safety calibration gradient -> safety head only: OK')
print('same-seed v14/v15 DDIM initialization parity: OK')
