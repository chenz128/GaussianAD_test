#!/usr/bin/env python3
"""Re-render a dumped GaussianAD future occupancy sample with the empty fix.

The legacy ``*_occflow.npz`` dump contains future labels rendered before the
empty Gaussian was included in ``forward_flow``. This utility reuses the saved
GaussianPrediction and future offsets, but re-renders each future step with the
scene-covering empty Gaussian restored. It does not load or alter a checkpoint.

The output NPZ is compatible with ``plot_occflow_anim.py`` and preserves the
original GT future occupancy grid.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

import model  # noqa: F401 - required to unpickle GaussianPrediction
from model.head.localagg.local_aggregate import LocalAggregator
from model.utils.utils import get_rotation_matrix


EMPTY_LABEL = 17
PC_MIN = (-30.0, -30.0, -2.0)
GRID_SIZE = 0.5
GRID_SHAPE = (120, 120, 8)
EMPTY_MEAN = (0.0, 0.0, -1.0)
EMPTY_SCALE = (60.0, 60.0, 4.0)
EMPTY_ROT = (1.0, 0.0, 0.0, 0.0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attr", required=True, help="*_gaussian_attr.pth")
    parser.add_argument("--future", required=True, help="*_future.npz")
    parser.add_argument("--legacy-occflow", required=True, help="legacy *_occflow.npz (GT source)")
    parser.add_argument("--checkpoint", required=True, help="checkpoint supplying head.empty_scalar")
    parser.add_argument("--out", required=True, help="fixed *_occflow.npz")
    parser.add_argument("--device", default="cuda", help="CUDA device, e.g. cuda:4")
    return parser.parse_args()


def filter_real_gaussians(means):
    pc_min = means.new_tensor(PC_MIN)
    grid = means.new_tensor((GRID_SIZE, GRID_SIZE, GRID_SIZE))
    integer = ((means - pc_min) / grid).to(torch.int)
    mask = ((integer[:, 0] >= 0) & (integer[:, 0] < GRID_SHAPE[0])
            & (integer[:, 1] >= 0) & (integer[:, 1] < GRID_SHAPE[1])
            & (integer[:, 2] >= 0) & (integer[:, 2] < GRID_SHAPE[2]))
    return torch.nonzero(mask, as_tuple=False).squeeze(-1)


def covariance_inverse(scales, rotations):
    count = scales.shape[0]
    scale_matrix = torch.zeros((count, 3, 3), dtype=scales.dtype, device=scales.device)
    scale_matrix[..., 0, 0] = scales[..., 0]
    scale_matrix[..., 1, 1] = scales[..., 1]
    scale_matrix[..., 2, 2] = scales[..., 2]
    rotation_matrix = get_rotation_matrix(rotations)
    matrix = torch.matmul(scale_matrix, rotation_matrix)
    covariance = torch.matmul(matrix.transpose(-1, -2), matrix)
    return torch.linalg.inv(covariance)


def main():
    args = parse_args()
    device = torch.device(args.device)

    attr = torch.load(args.attr, map_location="cpu")
    means = attr.means[0].to(device=device, dtype=torch.float32)
    scales = attr.scales[0].to(device=device, dtype=torch.float32)
    rotations = attr.rotations[0].to(device=device, dtype=torch.float32)
    semantics = attr.semantics[0].to(device=device, dtype=torch.float32)
    real_opacity = attr.opacities[0].to(device=device, dtype=torch.float32)
    if real_opacity.numel() == 0:
        real_opacity = torch.ones_like(semantics[:, :1])

    future = np.load(args.future)
    offset = torch.from_numpy(future["offset"].astype(np.float32)).to(device)
    # The visualization dump saves cumulative GT ego displacement, exactly what
    # fixed forward_flow subtracts from the moved real Gaussians.
    ego = torch.from_numpy(future["gt_ego"].astype(np.float32)).to(device)

    legacy = np.load(args.legacy_occflow)
    xyz = legacy["xyz"].astype(np.float32)
    xyz_t = torch.from_numpy(xyz).to(device).unsqueeze(0)
    gt = legacy["occ_fut_gt"].astype(np.int16)
    valid = legacy["valid"].astype(np.int8)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    empty_scalar = float(state_dict["head.empty_scalar"].reshape(-1)[0])

    # Add the missing empty class channel to real Gaussians, then append the
    # special scene-covering empty Gaussian. Use the exact scalar learned by
    # the checkpoint rather than its initialization value.
    sem18 = torch.cat([semantics, torch.zeros_like(semantics[:, :1])], dim=-1)
    empty_mean = means.new_tensor(EMPTY_MEAN).unsqueeze(0)
    empty_scale = means.new_tensor(EMPTY_SCALE).unsqueeze(0)
    empty_rot = means.new_tensor(EMPTY_ROT).unsqueeze(0)
    empty_sem = torch.zeros((1, sem18.shape[-1]), dtype=sem18.dtype, device=device)
    empty_sem[0, EMPTY_LABEL] = empty_scalar
    empty_opa = torch.ones((1, 1), dtype=real_opacity.dtype, device=device)

    aggregator = LocalAggregator(
        scale_multiplier=5, H=GRID_SHAPE[0], W=GRID_SHAPE[1], D=GRID_SHAPE[2],
        pc_min=list(PC_MIN), grid_size=GRID_SIZE).to(device).eval()

    pred_steps = []
    with torch.no_grad():
        for step in range(6):
            moved = means + torch.cat([offset[:, step], torch.zeros_like(offset[:, step, :1])], dim=-1)
            moved = moved - torch.cat([ego[step], torch.zeros_like(ego[step, :1])], dim=-1)
            real_idx = filter_real_gaussians(moved)

            future_means = torch.cat([moved[real_idx], empty_mean], dim=0).unsqueeze(0)
            future_scales = torch.cat([scales[real_idx], empty_scale], dim=0).unsqueeze(0)
            future_rots = torch.cat([rotations[real_idx], empty_rot], dim=0).unsqueeze(0)
            future_sem = torch.cat([sem18[real_idx], empty_sem], dim=0).unsqueeze(0)
            future_opa = torch.cat([real_opacity[real_idx], empty_opa], dim=0).unsqueeze(0)
            cov_inv = covariance_inverse(future_scales[0], future_rots[0]).unsqueeze(0)

            logits = aggregator(
                xyz_t.clone().float(), future_means,
                future_opa.reshape(1, -1), future_sem,
                future_scales, cov_inv)[None].transpose(1, 2)
            pred_steps.append(logits[0].argmax(0).cpu().numpy().astype(np.int16))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        xyz=xyz,
        occ_now=legacy["occ_now"].astype(np.int16),
        occ_fut=np.stack(pred_steps, axis=0),
        occ_fut_gt=gt,
        valid=valid,
    )
    empty_ratio = float((np.stack(pred_steps) == EMPTY_LABEL).mean())
    print(f"[OK] wrote {out_path}; empty_scalar={empty_scalar:.4f}; "
          f"future predicted-empty ratio={empty_ratio:.2%}")


if __name__ == "__main__":
    main()
