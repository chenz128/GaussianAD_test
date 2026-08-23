"""Audit v14 baseline parity, checkpoint lineage and identity initialization.

Run in the complete GaussianAD environment before every training launch::

    VERIFIED_V12_CHECKPOINT=/absolute/path/to/audited_v12.pth python \\
      config/nuscenes_gs25600_gtbox_oracle_v14_ft_plan_residual_ddim/validate_v14.py
"""

import os
from pathlib import Path
import sys

import torch
from mmengine.config import Config
from mmengine.registry import MODELS


config_path = Path(__file__).with_name(
    'nuscenes_gs25600_gtbox_oracle_v14_ft_plan_residual_ddim.py')
repo_root = config_path.parents[2]
baseline_path = repo_root / (
    'config/nuscenes_gs25600_gtbox_oracle_v12_ft_plan_'
    'futattn_global_residual/'
    'nuscenes_gs25600_gtbox_oracle_v12_ft_plan_'
    'futattn_global_residual.py')
sys.path.insert(0, str(repo_root))

checkpoint_text = os.environ.get('VERIFIED_V12_CHECKPOINT', '')
if not checkpoint_text:
    raise RuntimeError('VERIFIED_V12_CHECKPOINT is required')
checkpoint_path = Path(checkpoint_text)
if not checkpoint_path.is_absolute():
    raise ValueError('VERIFIED_V12_CHECKPOINT must be an absolute path')
if 'gaussian_residual_dit' in checkpoint_text.replace('\\', '/').lower():
    raise ValueError('gaussian_residual_dit checkpoints are explicitly invalid')
if not checkpoint_path.is_file():
    raise FileNotFoundError(checkpoint_path)
if not baseline_path.is_file():
    raise FileNotFoundError(
        'Exact v12 global-residual baseline config is missing: '
        + str(baseline_path))

config = Config.fromfile(config_path)
baseline = Config.fromfile(baseline_path)

# Imports below register the new loss/planner and the inherited v12 planner.
from loss import OPENOCC_LOSS  # noqa: E402
from loss.residual_ddim_plan_loss import ResidualDDIMPlanLoss  # noqa: E402,F401
from model.planner.planner_v14_residual_ddim import (  # noqa: E402
    _positions_to_displacements,
)


def _assert_equal(actual, expected, label):
    if actual != expected:
        raise AssertionError(
            f'{label} changed from baseline:\n'
            f'  baseline={expected!r}\n  v14={actual!r}')


# -------------------------------------------------------------------------
# 1. Resolved-config parity: only the explicit v14 extension may differ.
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
    _assert_equal(config[key], baseline[key], f'top-level config {key!r}')

for key in sorted(set(config.model.keys()) | set(baseline.model.keys())):
    if key == 'planner_head':
        continue
    if key not in config.model or key not in baseline.model:
        raise AssertionError(f'unexpected non-planner model key: {key}')
    _assert_equal(
        config.model[key], baseline.model[key], f'model.{key}')

baseline_head = baseline.model.planner_head
v14_head = config.model.planner_head
assert baseline_head.type == 'VADHeadFutAttnGlobalResidual'
assert v14_head.type == 'VADHeadFutAttnResidualDDIM'
for key in sorted(baseline_head.keys()):
    if key != 'type':
        _assert_equal(
            v14_head[key], baseline_head[key], f'model.planner_head.{key}')

allowed_new_head_keys = {
    'residual_hidden_dims',
    'residual_num_layers',
    'residual_num_heads',
    'residual_dropout',
    'residual_scale',
    'residual_clip',
    'diffusion_sigma_max',
    'diffusion_train_t_min',
    'diffusion_sample_steps',
    'num_inference_samples',
    'fixed_noise_seed',
    'gaussian_topk',
    'gaussian_corridor_radius',
    'gaussian_importance_floor',
    'dynamic_semantic_dims',
    'risk_margin',
    'risk_uncertainty_growth',
    'detach_baseline',
    'detach_residual_reference',
    'detach_gaussian_context',
    'keep_baseline_eval',
    'selector_risk_weight',
    'selector_residual_weight',
    'selector_dynamics_weight',
    'selector_learned_weight',
    'selector_baseline_margin',
    'selector_risk_threshold',
    'selector_max_normalized_residual',
    'selector_max_acceleration',
    'selector_max_jerk',
}
unexpected_head_keys = (
    set(v14_head.keys()) - set(baseline_head.keys())
    - allowed_new_head_keys)
if unexpected_head_keys:
    raise AssertionError(
        'unapproved planner config additions: '
        + repr(sorted(unexpected_head_keys)))

baseline_loss_cfgs = list(baseline.loss.loss_cfgs)
v14_loss_cfgs = list(config.loss.loss_cfgs)
for key in sorted(set(config.loss.keys()) | set(baseline.loss.keys())):
    if key == 'loss_cfgs':
        continue
    if key not in config.loss or key not in baseline.loss:
        raise AssertionError(f'unexpected MultiLoss config key: {key}')
    _assert_equal(config.loss[key], baseline.loss[key], f'loss.{key}')
_assert_equal(
    v14_loss_cfgs[:-1], baseline_loss_cfgs,
    'legacy loss list/order')
assert len(v14_loss_cfgs) == len(baseline_loss_cfgs) + 1
assert v14_loss_cfgs[-1].type == 'ResidualDDIMPlanLoss'

for key, value in baseline.loss_input_convertion.items():
    _assert_equal(
        config.loss_input_convertion[key], value,
        f'loss_input_convertion.{key}')
allowed_new_loss_inputs = {
    'ego_fut_base_preds',
    'ego_fut_residual_preds',
    'ego_fut_ddim_preds',
    'ego_fut_ddim_targets',
    'ego_fut_candidates',
    'ego_fut_candidate_risk',
    'ego_fut_candidate_quality_logits',
    'ego_fut_selected_index',
    'ego_fut_generated_risk',
}
unexpected_loss_inputs = (
    set(config.loss_input_convertion.keys())
    - set(baseline.loss_input_convertion.keys())
    - allowed_new_loss_inputs)
if unexpected_loss_inputs:
    raise AssertionError(
        'unapproved loss-input additions: '
        + repr(sorted(unexpected_loss_inputs)))

assert config.load_from == checkpoint_text
assert config.resume_from == ''
assert MODELS.get('VADHeadFutAttnResidualDDIM') is not None
assert OPENOCC_LOSS.get('ResidualDDIMPlanLoss') is not None

# -------------------------------------------------------------------------
# 2. Checkpoint lineage: it must strictly load the exact v12 planner.
# -------------------------------------------------------------------------
checkpoint = torch.load(checkpoint_path, map_location='cpu')
state_dict = checkpoint.get('state_dict', checkpoint)
planner_state = {}
for key, value in state_dict.items():
    normalized = key[len('module.'):] if key.startswith('module.') else key
    marker = 'planner_head.'
    if marker in normalized:
        planner_state[normalized.split(marker, 1)[1]] = value
if not planner_state:
    raise AssertionError(
        'checkpoint has no full-model planner_head.* state; wrong artifact')

baseline_planner = MODELS.build(baseline_head)
baseline_planner.init_weights()
baseline_load = baseline_planner.load_state_dict(planner_state, strict=False)
if baseline_load.missing_keys or baseline_load.unexpected_keys:
    raise AssertionError(
        'checkpoint is not the exact v12 global-residual planner:\n'
        f'  missing={baseline_load.missing_keys!r}\n'
        f'  unexpected={baseline_load.unexpected_keys!r}')

planner = MODELS.build(v14_head)
planner.init_weights()
load_result = planner.load_state_dict(planner_state, strict=False)
allowed_missing_prefixes = (
    'noisy_residual_encoder.',
    'reference_encoder.',
    'gaussian_context_proj.',
    'gaussian_relative_encoder.',
    'horizon_embedding.',
    'mode_embedding.',
    'diffusion_time_mlp.',
    'residual_dit_blocks.',
    'residual_final_norm.',
    'residual_output.',
    'candidate_quality_mlp.',
    'residual_scale',
    'fixed_residual_noise',
)
illegal_missing = [
    key for key in load_result.missing_keys
    if not key.startswith(allowed_missing_prefixes)]
if illegal_missing:
    raise AssertionError('non-v14 missing keys: ' + repr(illegal_missing))
if load_result.unexpected_keys:
    raise AssertionError(
        'unexpected inherited planner keys: '
        + repr(load_result.unexpected_keys))

baseline_state = baseline_planner.state_dict()
v14_state = planner.state_dict()
for key, value in baseline_state.items():
    if key not in v14_state or not torch.equal(v14_state[key], value):
        raise AssertionError(
            f'inherited checkpoint tensor differs after load: {key}')

# The original branch keeps the baseline's training/gradient behavior. The new
# objective receives detached reference and Gaussian tensors only.
assert planner.detach_baseline is False
assert planner.detach_residual_reference is True
assert planner.detach_gaussian_context is True
assert planner.keep_baseline_eval is False
assert planner.planner_gaussian_grad_scale == 1.0
assert planner.planner_offset_grad_scale == 1.0
planner.train()
assert planner.training
assert torch.count_nonzero(planner.residual_output.weight) == 0
assert torch.count_nonzero(planner.residual_output.bias) == 0
assert torch.count_nonzero(planner.candidate_quality_mlp[-1].weight) == 0
assert planner.fixed_residual_noise.shape == (4, 6, 2)

# -------------------------------------------------------------------------
# 3. Diffusion/geometry contracts and exact zero-residual fallback.
# -------------------------------------------------------------------------
timestep = torch.tensor([0.0, 0.5, 1.0])
alpha, sigma = planner._diffusion_schedule(timestep)
assert torch.allclose(sigma[:1], torch.zeros_like(sigma[:1]))
assert torch.allclose(alpha[:1], torch.ones_like(alpha[:1]))
assert 0.0 < float(sigma[-1]) < 1.0
assert torch.all(alpha > 0.0)

position = torch.randn(2, 3, 6, 2)
displacement = _positions_to_displacements(position)
assert torch.allclose(displacement.cumsum(dim=-2), position, atol=1e-6)

batch, modes, timesteps, gaussians = 2, 3, 6, 32
scene = {
    'content': torch.randn(batch, timesteps, gaussians, planner.embed_dims),
    'future_xy': torch.randn(batch, timesteps, gaussians, 2),
    'scale_xy': torch.full((batch, timesteps, gaussians, 2), 0.5),
    'opacity': torch.full((batch, timesteps, gaussians, 1), 0.8),
    'dynamic_probability': torch.full(
        (batch, timesteps, gaussians, 1), 0.7),
    'importance': torch.full((batch, timesteps, gaussians, 1), 0.6),
    'rotation_xy': torch.eye(2).reshape(1, 1, 2, 2).expand(
        batch, gaussians, -1, -1).clone(),
}
baseline_displacement = 0.1 * torch.randn(batch, modes, timesteps, 2)
reference_position = baseline_displacement.cumsum(dim=-2)
tokens, risk = planner._select_gaussian_context(
    scene, reference_position, return_tokens=True)
assert tokens.shape == (
    batch, modes, timesteps, min(planner.gaussian_topk, gaussians),
    planner.residual_hidden_dims)
assert risk.shape == (batch, modes, timesteps)
assert torch.isfinite(tokens).all() and torch.isfinite(risk).all()
assert (risk >= 0).all() and (risk <= 1).all()

planner.eval()
with torch.no_grad():
    ddim_output = planner._ddim_sample(
        baseline_displacement, reference_position, scene)
assert torch.allclose(
    ddim_output['ego_fut_preds'], baseline_displacement, atol=1e-6)
assert torch.count_nonzero(ddim_output['ego_fut_selected_index']) == 0

print('v14 strict baseline/config/checkpoint audit: OK')
print('baseline config:', baseline_path)
print('verified continuation checkpoint:', checkpoint_path)
print('baseline planner:', baseline_head.type)
print('new planner:', v14_head.type)
print('matched inherited planner tensors:', len(baseline_state))
print('new v14 state keys:', *load_result.missing_keys, sep='\n  ')
print('legacy losses preserved:', len(baseline_loss_cfgs))
print('new losses appended: 1 (ResidualDDIMPlanLoss)')
for key in (
        'max_epochs', 'lr', 'optimizer', 'eval_every_epochs',
        'find_unused_parameters', 'frozen_modules'):
    if key in baseline:
        print(f'baseline parity {key}:', config[key])
print('zero-init two-NFE DDIM == baseline (synthetic): max error < 1e-6')
print('real fixed-mini-batch identity check is still required before training')
