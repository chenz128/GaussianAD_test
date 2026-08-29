#!/usr/bin/env python3
"""Build the isolated joint warm-start checkpoint without changing code."""

import argparse
import os
from pathlib import Path

import torch


DEFAULT_OCC = Path(
    '/data/chenz/GaussianAD/out/nuscenes_gs25600_v3_se3/'
    'checkpoints/epoch_15.pth'
)
DEFAULT_PLANNER = Path(
    '/data/xinyao/navsim_workspace/GaussianAD/exp/'
    'nuscenes_gs25600_v12_fixempty_ft_plan_futattn_global_residual/'
    'checkpoints/epoch_15.pth'
)
DEFAULT_OUTPUT = Path(
    '/data/xinyao/navsim_workspace/GaussianAD/exp/'
    'nuscenes_gs25600_v3_se3_ft_plan_futattn_global_residual_full/init/'
    'v3_se3_occ_plus_futattn_global_residual_planner_epoch15.pth'
)
BACKEND_PREFIXES = ('map_decoder.', 'planner_head.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--occ-checkpoint', type=Path, default=DEFAULT_OCC)
    parser.add_argument(
        '--planner-checkpoint', type=Path, default=DEFAULT_PLANNER)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--force', action='store_true')
    return parser.parse_args()


def load_state(path):
    checkpoint = torch.load(str(path), map_location='cpu')
    state = checkpoint.get('state_dict', checkpoint)
    if not isinstance(state, dict):
        raise TypeError(f'{path}: state_dict is not a mapping')
    return checkpoint, state


def main():
    args = parse_args()
    for source in (args.occ_checkpoint, args.planner_checkpoint):
        if not source.is_file():
            raise FileNotFoundError(source)
    if args.output.exists() and not args.force:
        raise FileExistsError(
            f'{args.output} already exists; pass --force to replace it')

    print(f'loading OCC frontend: {args.occ_checkpoint}', flush=True)
    occ_checkpoint, occ_state = load_state(args.occ_checkpoint)
    merged_state = dict(occ_state)
    occ_key_count = len(merged_state)
    del occ_checkpoint, occ_state

    print(f'loading planner backend: {args.planner_checkpoint}', flush=True)
    planner_checkpoint, planner_state = load_state(args.planner_checkpoint)
    selected = {
        key: value
        for key, value in planner_state.items()
        if key.startswith(BACKEND_PREFIXES)
    }
    if not selected:
        raise RuntimeError('no map_decoder/planner_head keys were selected')

    shape_conflicts = []
    for key, value in selected.items():
        if key in merged_state and merged_state[key].shape != value.shape:
            shape_conflicts.append(
                (key, tuple(merged_state[key].shape), tuple(value.shape)))
    if shape_conflicts:
        raise RuntimeError(f'backend shape conflicts: {shape_conflicts}')

    merged_state.update(selected)
    prefix_counts = {
        prefix: sum(key.startswith(prefix) for key in selected)
        for prefix in BACKEND_PREFIXES
    }
    del planner_checkpoint, planner_state, selected

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    payload = {
        'state_dict': merged_state,
        'meta': {
            'merge_policy': (
                'V3-SE3 state_dict, overwritten by planner checkpoint '
                'for map_decoder.* and planner_head.*'
            ),
            'occ_checkpoint': str(args.occ_checkpoint),
            'planner_checkpoint': str(args.planner_checkpoint),
            'occ_initial_key_count': occ_key_count,
            'backend_prefix_counts': prefix_counts,
            'merged_key_count': len(merged_state),
        },
    }
    print(
        f'saving {len(merged_state)} keys to {args.output} '
        f'(backend={prefix_counts})',
        flush=True,
    )
    torch.save(payload, str(temporary))
    os.replace(str(temporary), str(args.output))
    print('checkpoint merge complete', flush=True)


if __name__ == '__main__':
    main()
