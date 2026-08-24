"""One-real-batch audit for the v15 metric-aligned safety annotations."""

import os
from pathlib import Path
import sys

import torch
from mmengine.config import Config


config_path = Path(__file__).with_name(
    'nuscenes_gs25600_gtbox_oracle_v15_ft_plan_'
    'safety_calibrated_residual_ddim.py')
repo_root = config_path.parents[2]
sys.path.insert(0, str(repo_root))
os.chdir(repo_root)

config = Config.fromfile(config_path)

from dataset import get_dataloader  # noqa: E402
from loss.safety_calibrated_residual_ddim_loss import (  # noqa: E402
    MetricAlignedVehicleSAT,
)


def _tensor(value, name):
    if hasattr(value, 'data') and not torch.is_tensor(value):
        value = value.data
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if not torch.is_tensor(value):
        raise TypeError(f'{name} did not collate to a tensor: {type(value)!r}')
    return value


val_loader_config = dict(config.val_loader)
val_loader_config['batch_size'] = 1
val_loader_config['num_workers'] = 0
_, val_loader = get_dataloader(
    config.train_dataset_config,
    config.val_dataset_config,
    config.train_loader,
    val_loader_config,
    dist=False,
    val_only=True)
batch = next(iter(val_loader))

required = (
    'ego_fut_gt',
    'ego_fut_masks',
    'attr_labels_planner',
    'gt_boxes',
    'fut_valid_flag',
)
mapped = {}
for input_key in required:
    source_key = config.loss_input_convertion[input_key]
    mapped[input_key] = batch.get(source_key)
    if mapped[input_key] is None:
        raise KeyError(
            f'{input_key} -> {source_key} is missing from a real val batch')

target = _tensor(mapped['ego_fut_gt'], 'ego_fut_gt').float()
while target.dim() > 3 and target.shape[1] == 1:
    target = target.squeeze(1)
if target.dim() == 2:
    target = target[None]
if target.dim() != 3 or target.shape[-2:] != (6, 2):
    raise ValueError(f'unexpected ego_fut_gt shape: {tuple(target.shape)}')

mask = _tensor(mapped['ego_fut_masks'], 'ego_fut_masks').float()
while mask.dim() > 2 and mask.shape[1] == 1:
    mask = mask.squeeze(1)
if mask.dim() == 1:
    mask = mask[None]
if mask.shape != target.shape[:2]:
    raise ValueError(
        f'ego mask/target mismatch: {tuple(mask.shape)} vs '
        f'{tuple(target.shape)}')

# Candidate zero is GT and candidate one is a deliberately shifted trajectory;
# this checks annotation decoding and SAT tensor contracts, not model quality.
candidates = torch.stack([target, target.clone()], dim=1)
candidates[:, 1, 0, 1] += 10.0
sat = MetricAlignedVehicleSAT()
sat_output = sat(
    candidates,
    target,
    mask,
    mapped['attr_labels_planner'],
    gt_boxes=mapped['gt_boxes'],
    fut_valid_flag=mapped['fut_valid_flag'])

expected_shape = (target.shape[0], 2, target.shape[1])
for key in ('soft_target', 'hard_target', 'valid'):
    if sat_output[key].shape != expected_shape:
        raise AssertionError(
            f'{key} shape {tuple(sat_output[key].shape)} != {expected_shape}')
if not torch.isfinite(sat_output['soft_target']).all():
    raise AssertionError('real-batch SAT soft targets contain NaN/Inf')

print('v15 real-data annotation/SAT smoke: OK')
print('batch keys:', ', '.join(sorted(batch.keys())))
print('ego target shape:', tuple(target.shape))
print('SAT valid elements:', int(sat_output['valid'].sum().item()))
print('SAT hard collisions:', int(sat_output['hard_target'].sum().item()))
