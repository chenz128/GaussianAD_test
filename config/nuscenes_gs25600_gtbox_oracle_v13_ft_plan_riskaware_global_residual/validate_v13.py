"""Read-only registration/config validation for the v13 experiment."""

from pathlib import Path
import sys

import torch
from mmengine.config import Config
from mmengine.registry import MODELS


config_path = Path(__file__).with_name(
    'nuscenes_gs25600_gtbox_oracle_v13_ft_plan_riskaware_global_residual.py')
repo_root = config_path.parents[2]
sys.path.insert(0, str(repo_root))
print('loading config...', flush=True)
config = Config.fromfile(config_path)
print('custom modules loaded', flush=True)

baseline_path = repo_root / (
    'config/nuscenes_gs25600_gtbox_oracle_v12_ft_plan_'
    'futattn_global_residual/'
    'nuscenes_gs25600_gtbox_oracle_v12_ft_plan_'
    'futattn_global_residual.py')
baseline = Config.fromfile(baseline_path)
intentional_differences = {
    'custom_imports', 'model', 'loss', 'loss_input_convertion'}
for key in sorted(set(config.keys()) & set(baseline.keys())):
    if not key.startswith('_') and key not in intentional_differences:
        assert config[key] == baseline[key], (
            'unexpected base-parameter change: ' + key)

from loss import OPENOCC_LOSS  # noqa: E402
from loss.risk_aware_plan_loss import (  # noqa: E402
    HardNegativePlanAgentSATCollisionLoss,
)


assert config.model.planner_head.type == (
    'VADHeadFutAttnRiskAwareGlobalResidual')
assert config.load_from == (
    'exp/nuscenes_gs25600_v12_fixempty/checkpoints/epoch_15.pth')
assert config.max_epochs == 15
assert config.lr == 2e-4
assert config.model.planner_head.planner_gaussian_grad_scale == 1.0
assert config.model.planner_head.planner_offset_grad_scale == 1.0
assert MODELS.get('VADHeadFutAttnRiskAwareGlobalResidual') is not None
assert OPENOCC_LOSS.get('RiskAwarePlanLoss') is not None
assert OPENOCC_LOSS.get('RiskAwareGateLoss') is not None
plan_loss_config = next(
    item for item in config.loss.loss_cfgs
    if item.type == 'RiskAwarePlanLoss')
assert plan_loss_config.col_loss_weight == 0.1
assert plan_loss_config.col_sat is True

planner = MODELS.build(config.model.planner_head)
planner.init_weights()
assert torch.count_nonzero(planner.refine_gate_mlp[-1].weight) == 0
assert torch.count_nonzero(planner.refine_gate_mlp[-1].bias) == 0

batch, modes, timesteps, gaussians = 2, 3, 6, 64
gaussian_output = torch.zeros(batch, gaussians, 28)
gaussian_output[..., :2] = 8.0 * torch.rand(batch, gaussians, 2) - 4.0
gaussian_output[..., 2] = 0.5
gaussian_output[..., 3:6] = 0.5
gaussian_output[..., 6] = 1.0
gaussian_output[..., 10] = 0.8
gaussian_output[..., 11 + 3] = 1.0  # car semantic channel
offset = 0.2 * torch.randn(batch, gaussians, timesteps * 2)
prediction = 0.2 * torch.randn(batch, modes, timesteps, 2)
risk = planner._trajectory_risk(
    {'gaussian_output': gaussian_output, 'offset': offset}, prediction)
assert risk.shape == (batch, modes, timesteps)
assert torch.isfinite(risk).all()
assert (risk >= 0).all() and (risk <= 1).all()

# A discriminability test must compare trajectories in the same scene.  The
# earlier random min/max smoke test only checked numerical range and must not be
# interpreted as a classifier-quality measurement.
contrast_gaussians = 64
contrast_output = torch.zeros(1, contrast_gaussians, 28)
contrast_output[..., 0] = torch.linspace(
    1.0, 12.0, contrast_gaussians)
contrast_output[..., 1] = 0.0
contrast_output[..., 2] = 0.5
contrast_output[..., 3:6] = 0.35
contrast_output[..., 6] = 1.0
contrast_output[..., 10] = 0.8
contrast_output[..., 11 + 3] = 1.0

# GaussianAD defines Cov = R^T S^2 R, so R maps world-space deltas into
# Gaussian-local axes.  Lock that convention here: a +90 degree yaw swaps the
# absolute x/y axis contributions used by the conservative box projection.
rotated_output = contrast_output[:, :1].clone()
half_sqrt_two = 2.0 ** -0.5
rotated_output[..., 6] = half_sqrt_two
rotated_output[..., 9] = half_sqrt_two
_, _, _, rotated_xy = planner._future_geometry({
    'gaussian_output': rotated_output,
    'offset': torch.zeros(1, 1, timesteps * 2),
})
expected_abs_rotation = torch.tensor(
    [[0.0, 1.0], [1.0, 0.0]], dtype=rotated_xy.dtype)
assert torch.allclose(
    rotated_xy[0, 0].abs(), expected_abs_rotation, atol=1e-5)

contrast_prediction = torch.zeros(1, 2, timesteps, 2)
contrast_prediction[:, :, :, 0] = 2.0
contrast_prediction[:, 1, 0, 1] = 8.0
contrast_risk = planner._trajectory_risk({
    'gaussian_output': contrast_output,
    'offset': torch.zeros(1, contrast_gaussians, timesteps * 2),
}, contrast_prediction)
collision_risk = contrast_risk[:, 0].mean()
safe_risk = contrast_risk[:, 1].mean()
assert collision_risk > safe_risk + 0.2

collision_loss_fn = HardNegativePlanAgentSATCollisionLoss(
    loss_weight=0.1, safe_margin=0.5, temperature=0.2)
agents = 5
attr_labels = torch.zeros(batch, agents, 34)
attr_labels[..., 12:18] = 1.0
agent_boxes = torch.zeros(batch, agents, 7)
agent_boxes[..., 0] = torch.linspace(0.5, 10.0, agents)
agent_boxes[..., 3] = 1.8
agent_boxes[..., 4] = 4.0
ego_prediction = torch.zeros(batch, timesteps, 2, requires_grad=True)
ego_mask = torch.ones(batch, timesteps)
collision_loss = collision_loss_fn(
    ego_prediction, attr_labels, None, ego_mask, agent_boxes)
assert torch.isfinite(collision_loss)
collision_loss.backward()
assert ego_prediction.grad is not None
assert torch.isfinite(ego_prediction.grad).all()

print('v13 config and registry validation: OK')
print('planner:', config.model.planner_head.type)
print('load_from:', config.load_from)
print('epochs/lr:', config.max_epochs, config.lr)
print('all non-planner/loss base config keys unchanged: OK')
print('planner Gaussian/offset gradient scale:',
      config.model.planner_head.planner_gaussian_grad_scale)
print('random numerical smoke shape/range:', tuple(risk.shape),
      float(risk.min()), float(risk.max()))
print('contrast collision/safe risk:',
      float(collision_risk), float(safe_risk))
print('rotation-aware Gaussian local geometry: OK')
print('zero-init gate: OK')
print('hard-negative SAT forward/backward: OK', float(collision_loss))
print('default COL_W=0.1 and SAT enabled: OK')
