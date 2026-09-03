#!/usr/bin/env python3
"""Fail-fast audit for the frozen-OCC v16 experiment."""

import argparse
import os
from pathlib import Path
import re
import sys

import torch
from mmengine.config import Config
from mmengine.utils import import_modules_from_strings


REPO = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).with_name(
    'nuscenes_gs25600_v16_ft_plan_frozen_occ_truncated_ddim.py')
BASE_CONFIG = (
    REPO / 'config'
    / 'nuscenes_gs25600_v3_se3_ft_plan_futattn_global_residual_full'
    / 'nuscenes_gs25600_v3_se3_ft_plan_futattn_global_residual_full.py')
DEFAULT_CHECKPOINT = (
    REPO / 'exp'
    / 'nuscenes_gs25600_v3_se3_ft_plan_futattn_global_residual_full'
    / 'checkpoints' / 'epoch_16.pth')

ALLOWED_TOP_LEVEL_CHANGES = {
    'custom_imports',
    'load_from',
    'resume_from',
    'frozen_modules',
    'model',
    'loss',
    'loss_input_convertion',
}
NEW_PLANNER_STATE_PREFIXES = (
    'planner_head.noisy_residual_encoder.',
    'planner_head.reference_encoder.',
    'planner_head.gaussian_context_proj.',
    'planner_head.gaussian_relative_encoder.',
    'planner_head.horizon_embedding.',
    'planner_head.mode_embedding.',
    'planner_head.diffusion_time_embedding.',
    'planner_head.diffusion_time_mlp.',
    'planner_head.residual_dit_blocks.',
    'planner_head.residual_final_norm.',
    'planner_head.residual_output.',
    'planner_head.candidate_quality_mlp.',
    'planner_head.residual_scale',
    'planner_head.fixed_residual_noise',
    'planner_head.ego_footprint_samples',
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        '--build-model', action='store_true',
        help='also instantiate the full CPU model and audit load/trainability')
    return parser.parse_args()


def repo_path(path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (REPO / path).resolve()


def assert_equal(label, left, right):
    if left != right:
        raise AssertionError(f'{label} differs from the strong baseline')


def audit_config(checkpoint):
    os.environ['STRONG_OCC_CHECKPOINT'] = str(checkpoint)
    base = Config.fromfile(str(BASE_CONFIG)).to_dict()
    cfg = Config.fromfile(str(CONFIG)).to_dict()

    all_keys = set(base) | set(cfg)
    for key in sorted(all_keys - ALLOWED_TOP_LEVEL_CHANGES):
        # MMEngine retains helper variables from Python configs.  Leading
        # underscore fields do not participate in model/data/runner behavior.
        if key.startswith('_'):
            continue
        assert_equal(f'top-level config key {key}', cfg.get(key), base.get(key))

    base_model = dict(base['model'])
    new_model = dict(cfg['model'])
    base_planner = base_model.pop('planner_head')
    new_planner = new_model.pop('planner_head')
    base_type = base_model.pop('type')
    new_type = new_model.pop('type')
    if base_type != 'BEVSegmentorV3FuturePlanIsolated':
        raise AssertionError(f'unexpected strong baseline type: {base_type}')
    if new_type != 'BEVSegmentorV3FuturePlanFrozenFrontend':
        raise AssertionError(f'unexpected v16 segmentor guard: {new_type}')
    assert_equal('model outside planner_head', new_model, base_model)
    if new_planner.get('type') != 'VADHeadFrozenOccTruncatedResidualDDIM':
        raise AssertionError('unexpected v16 planner type')
    for key, value in base_planner.items():
        if key == 'type':
            continue
        assert_equal(f'inherited planner field {key}', new_planner.get(key), value)

    base_losses = list(base['loss']['loss_cfgs'])
    new_losses = list(cfg['loss']['loss_cfgs'])
    assert_equal('legacy loss prefix', new_losses[:-1], base_losses)
    if new_losses[-1].get('type') != 'FrozenOccTruncatedDDIMPlanLoss':
        raise AssertionError('v16 loss was not appended exactly once')
    for key, value in base['loss_input_convertion'].items():
        assert_equal(
            f'legacy loss mapping {key}',
            cfg['loss_input_convertion'].get(key), value)

    expected_frozen = {
        'img_backbone', 'img_neck', 'lifter', 'encoder',
        'temporal_encoder', 'decoder', 'map_decoder', 'head'}
    if set(cfg['frozen_modules']) != expected_frozen:
        raise AssertionError('frontend frozen_modules is incomplete')
    if repo_path(cfg['load_from']) != checkpoint.resolve():
        raise AssertionError('resolved load_from is not the audited checkpoint')
    if cfg.get('resume_from'):
        raise AssertionError('v16 must use weights-only load_from, not resume')
    return Config(cfg)


def audit_metric_source():
    metric_path = REPO / 'dataset' / 'metric_stp3.py'
    text = metric_path.read_text(encoding='utf-8')
    required = (
        r'^\s*pred_ego_fut_trajs\s*=\s*pred_ego_fut_trajs\.cumsum\(dim=-2\)',
        r'^\s*gt_ego_fut_trajs\s*=\s*gt_ego_fut_trajs\.cumsum\(dim=-2\)',
    )
    for pattern in required:
        if re.search(pattern, text, flags=re.MULTILINE) is None:
            raise AssertionError(
                'planning metric is not converting displacement trajectories '
                'to cumulative positions')


def checkpoint_state(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(str(path), map_location='cpu')
    state = checkpoint.get('state_dict', checkpoint)
    if not isinstance(state, dict) or not state:
        raise TypeError('checkpoint state_dict is empty or invalid')
    required_prefixes = (
        'img_backbone.', 'head.', 'map_decoder.', 'planner_head.')
    for prefix in required_prefixes:
        if not any(key.startswith(prefix) for key in state):
            raise AssertionError(f'checkpoint lacks required prefix {prefix}')
    return state


def audit_built_model(cfg, state):
    import_modules_from_strings(**cfg.custom_imports)
    from mmseg.models import build_segmentor

    model = build_segmentor(cfg.model)
    model.init_weights()
    model_state = model.state_dict()
    mismatched = [
        key for key, value in state.items()
        if key in model_state and tuple(value.shape) != tuple(model_state[key].shape)]
    if mismatched:
        raise AssertionError(f'shape-mismatched checkpoint keys: {mismatched[:20]}')
    incompatible = model.load_state_dict(state, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    if unexpected:
        raise AssertionError(f'unexpected checkpoint keys: {unexpected[:20]}')
    illegal_missing = [
        key for key in incompatible.missing_keys
        if not key.startswith(NEW_PLANNER_STATE_PREFIXES)]
    if illegal_missing:
        raise AssertionError(f'illegal missing keys: {illegal_missing[:20]}')

    # Every tensor inherited from the strong checkpoint must be restored
    # exactly.  This catches accidental same-shaped reinitialization that a
    # missing/unexpected-key audit alone would not expose.
    loaded_state = model.state_dict()
    unequal = [
        key for key, value in state.items()
        if key in loaded_state and not torch.equal(loaded_state[key], value)]
    if unequal:
        raise AssertionError(
            f'inherited checkpoint tensors changed after load: {unequal[:20]}')

    for name in cfg.frozen_modules:
        module = getattr(model, name, None)
        if module is None:
            raise AssertionError(f'frozen module does not exist: {name}')
        module.requires_grad_(False)
    model.train()
    frontend_training = [
        name for name in cfg.frozen_modules
        if getattr(model, name).training]
    if frontend_training:
        raise AssertionError(
            f'frozen frontend returned to train mode: {frontend_training}')
    residual_training = [
        name for name in model.planner_head._residual_child_names()
        if getattr(model.planner_head, name).training]
    if not residual_training:
        raise AssertionError('new residual planner modules are not in train mode')
    trainable = [name for name, value in model.named_parameters()
                 if value.requires_grad]
    illegal_trainable = [
        name for name in trainable
        if not name.startswith(NEW_PLANNER_STATE_PREFIXES)]
    if illegal_trainable:
        raise AssertionError(
            f'non-v16 parameters remain trainable: {illegal_trainable[:20]}')
    if not trainable:
        raise AssertionError('no v16 planner parameter is trainable')
    audit_planner_math(model)
    audit_loss_math(cfg)
    restored_count = sum(key in loaded_state for key in state)
    return len(incompatible.missing_keys), len(trainable), restored_count


def audit_planner_math(model):
    """Exercise v16-only tensor paths without loading a dataset or GPU."""
    planner = model.planner_head
    mro = [item.__name__ for item in type(planner).__mro__]
    required_order = (
        'VADHeadFrozenOccTruncatedResidualDDIM',
        'VADHeadFutAttnResidualDDIM',
        'VADHeadFutAttnGlobalResidualFutureGaussianIsolated',
        'VADHeadFutAttnGlobalResidual',
    )
    positions = [mro.index(name) for name in required_order]
    if positions != sorted(positions):
        raise AssertionError(f'unsafe planner MRO: {mro}')

    batch = 1
    timesteps = planner.fut_ts
    modes = planner.ego_fut_mode
    gaussian_count = max(
        planner.gaussian_topk, planner.guidance_gaussian_topk) + 4
    dtype = next(planner.parameters()).dtype
    future = torch.zeros(
        batch, timesteps, gaussian_count, 28, dtype=dtype)
    future[..., 0] = torch.linspace(
        2.0, 30.0, gaussian_count, dtype=dtype)
    future[..., 1] = torch.linspace(
        -4.0, 4.0, gaussian_count, dtype=dtype)
    future[..., 3:6] = 0.7
    future[..., 6] = 1.0
    future[..., 10] = 0.8
    future[..., planner.obstacle_semantic_indices[0] + 11] = 1.0
    padding = torch.zeros(
        batch, timesteps, gaussian_count, dtype=torch.bool)
    # Include padded high-evidence items to ensure masks suppress their risk.
    padding[..., -2:] = True
    future[..., -2:, 10] = 1.0
    content = torch.randn(
        batch, timesteps, gaussian_count, planner.embed_dims, dtype=dtype)
    scene = planner._build_gaussian_scene({
        'planner_future_gaussians': future,
        'planner_future_gaussian_mask': padding,
    }, future_content=content)

    baseline_displacement = torch.zeros(
        batch, modes, timesteps, 2, dtype=dtype)
    baseline_displacement[..., 0] = 1.0
    baseline_position = baseline_displacement.cumsum(dim=-2)
    planner.eval()
    planner.zero_grad(set_to_none=True)
    generated = planner._sample_residual_proposals(
        baseline_displacement, baseline_position, scene)
    expected_generated = (
        batch, modes, planner.num_inference_samples, timesteps, 2)
    if tuple(generated.shape) != expected_generated:
        raise AssertionError(
            f'generated proposal shape {tuple(generated.shape)} != '
            f'{expected_generated}')
    anchors = baseline_position[:, :, None].expand(
        -1, -1, planner._ANCHOR_CANDIDATE_COUNT, -1, -1)
    candidates = torch.cat([anchors, generated], dim=2)
    flat = candidates.reshape(batch, -1, timesteps, 2)
    _, risk = planner._select_gaussian_context(
        scene, flat, return_tokens=False)
    risk = risk.reshape(batch, modes, candidates.shape[2], timesteps)
    if not torch.isfinite(risk).all() or risk.min() < 0 or risk.max() > 1:
        raise AssertionError('planner OCC risk is not finite in [0, 1]')
    quality = planner._candidate_quality(
        candidates, baseline_position[:, :, None].expand_as(candidates), risk)
    selection = planner._select_candidates(
        candidates, baseline_position, risk, quality)
    if tuple(selection['selected'].shape) != (batch, modes):
        raise AssertionError('selector output shape mismatch')
    if not torch.equal(candidates[:, :, 0], baseline_position):
        raise AssertionError('candidate zero is not the exact baseline')

    generated.square().mean().backward()
    residual_names = planner._residual_child_names()
    has_new_gradient = any(
        parameter.grad is not None
        for name, child in planner.named_children()
        if name in residual_names
        for parameter in child.parameters())
    frozen_gradient = [
        name for name, child in planner.named_children()
        if name not in residual_names
        for parameter_name, parameter in child.named_parameters()
        if parameter.grad is not None]
    if not has_new_gradient:
        raise AssertionError('synthetic DDIM path produced no planner gradient')
    if frozen_gradient:
        raise AssertionError(
            f'synthetic DDIM path reached frozen anchor: {frozen_gradient[:10]}')


def audit_loss_math(cfg):
    """Check the planner-only loss, GT-box SAT path and gradients."""
    from loss.frozen_occ_truncated_ddim_loss import (
        FrozenOccTruncatedDDIMPlanLoss,
    )

    loss_cfg = dict(cfg.loss.loss_cfgs[-1])
    if loss_cfg.pop('type') != 'FrozenOccTruncatedDDIMPlanLoss':
        raise AssertionError('unexpected final loss while running smoke test')
    loss_module = FrozenOccTruncatedDDIMPlanLoss(**loss_cfg)
    batch, modes, timesteps = 1, 3, 6
    candidate_count = 3 + 4
    target = torch.zeros(batch, timesteps, 2)
    target[..., 0] = 1.0
    target[..., 1] = 1.0
    candidate = torch.zeros(
        batch, modes, candidate_count, timesteps, 2)
    candidate[..., 0] = 1.0
    candidate[:, :, 3:, :, 1] = torch.linspace(
        -0.2, 1.2, candidate_count - 3).reshape(1, 1, -1, 1)
    candidate.requires_grad_()
    position = candidate.cumsum(dim=-2)
    obstacle = position.new_tensor([4.0, 0.0])
    risk = torch.exp(
        -0.5 * (position - obstacle).square().sum(dim=-1))
    quality = torch.zeros(
        batch, modes, candidate_count, requires_grad=True)

    # Layout mirrors MetricAlignedVehicleSAT: agent displacement (12), mask
    # (6), one reserved value, lcf (9), yaw delta (6).
    attr = torch.zeros(batch, 1, 34)
    attr[..., 12:18] = 1.0
    attr[..., 27] = 14.0
    boxes = torch.zeros(batch, 1, 7)
    boxes[..., 0] = 4.0
    boxes[..., 3] = 2.0
    boxes[..., 4] = 4.0
    command = torch.zeros(batch, modes)
    command[:, 1] = 1.0
    inputs = {
        'ego_fut_gt': target,
        'ego_fut_masks': torch.ones(batch, timesteps),
        'ego_fut_cmd': command,
        'ego_fut_base_preds': candidate[:, :, 0].detach(),
        'ego_fut_candidates': candidate,
        'ego_fut_candidate_quality_logits': quality,
        'ego_fut_candidate_risk': risk,
        'attr_labels_planner': attr,
        'gt_boxes': boxes,
        'fut_valid_flag': torch.ones(batch),
    }
    total, logs = loss_module(inputs)
    if not torch.isfinite(total):
        raise AssertionError('synthetic v16 loss is not finite')
    total.backward()
    if candidate.grad is None or not torch.isfinite(candidate.grad).all():
        raise AssertionError('v16 candidate loss gradient is invalid')
    if quality.grad is None or not torch.isfinite(quality.grad).all():
        raise AssertionError('v16 ranking loss gradient is invalid')
    if 'loss_v16_sat' not in logs or 'loss_v16_rank' not in logs:
        raise AssertionError('v16 loss diagnostics are incomplete')


def main():
    args = parse_args()
    checkpoint = repo_path(args.checkpoint)
    cfg = audit_config(checkpoint)
    audit_metric_source()
    state = checkpoint_state(checkpoint)
    missing_count = None
    trainable_count = None
    restored_count = None
    if args.build_model:
        missing_count, trainable_count, restored_count = audit_built_model(
            cfg, state)
    print('[OK] v16 config matches the strong baseline outside allowed deltas')
    print(f'[OK] weights-only source: {checkpoint}')
    print(f'[OK] checkpoint tensors: {len(state)}')
    print('[OK] displacement -> cumulative-position metric conversion is active')
    if args.build_model:
        print(f'[OK] inherited tensors restored exactly: {restored_count}')
        print(f'[OK] new-state missing keys only: {missing_count}')
        print(f'[OK] trainable v16 parameter tensors: {trainable_count}')
        print('[OK] synthetic planner/loss forward and backward paths')


if __name__ == '__main__':
    sys.path.insert(0, str(REPO))
    main()
