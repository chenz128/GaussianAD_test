"""
Offline cross-rendering comparison of per-ray depth variance (std_fg)
between ft_depth and concentrate checkpoints.

This directly measures whether ConcLoss achieved its optimization target:
lower Var[z] = sharper depth distribution along each ray.

Usage (on h20-old):
    /data/chenz/conda_env/splatting/bin/python tools/compare_depth_variance.py

No training code is modified. This loads:
  - val pkl (camera params)
  - gaussian_attr.pth from each run
  - pseudo_depth from disk (for foreground mask)
And renders depth variance using gsplat for N val samples.
"""

import os
import sys
import pickle
import numpy as np
import torch
from pathlib import Path

# ─── Config ───────────────────────────────────────────────────────────
PKL_PATH = "data/nuscenes_cam/nuscenes_infos_val_gaussian_ad_v4.pkl"
PSEUDO_DEPTH_ROOT = "/data/chenz/Gaussianflowocc_test/data/metric_3d_nusc"
PSEUDO_SEG_ROOT = "/data/chenz/Gaussianflowocc_test/data/grounded_sam_nusc"

CHECKPOINTS = {
    "ft_depth": "out/nuscenes_gs25600_base_ft_depth/vis/val_0_gaussian_attr.pth",
    "concentrate": "out/nuscenes_gs25600_concentrate/vis/val_0_gaussian_attr.pth",
}

# render config (must match training)
PSEUDO_LABEL_SCALE = 0.44
CROP_TOP_ORIG = 140  # pixels in original resolution
RENDER_H = 256  # int((900 - CROP_TOP_ORIG) * PSEUDO_LABEL_SCALE) ≈ 334 -> actually config uses 256
RENDER_W = 704  # int(1600 * PSEUDO_LABEL_SCALE) ≈ 704

NUM_SAMPLES = 20  # number of val frames to average over
DEVICE = "cuda:0"

CAM_ORDER = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
             'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']


# ─── Helpers ──────────────────────────────────────────────────────────

def load_gaussian_attr(path):
    """Load gaussian attributes from vis checkpoint."""
    g = torch.load(path, map_location='cpu')
    means = g.means.squeeze(0).float()       # (G, 3)
    scales = g.scales.squeeze(0).float()     # (G, 3)
    quats = g.rotations.squeeze(0).float()   # (G, 4)
    opacities = g.opacities.squeeze(0).float().squeeze(-1)  # (G,)
    return means, scales, quats, opacities


def build_camera_params(frame):
    """Extract gs_extrins (lidar2cam) and gs_intrins (scaled K) from pkl frame."""
    cams = frame['cams']
    
    intrinsics = []
    extrinsics = []
    
    for cam_name in CAM_ORDER:
        cam = cams[cam_name]
        
        # intrinsic: 3x3, scale and crop
        K = np.array(cam['cam_intrinsic'], dtype=np.float64).copy()  # (3, 3)
        K[0, 0] *= PSEUDO_LABEL_SCALE  # fx
        K[1, 1] *= PSEUDO_LABEL_SCALE  # fy
        K[0, 2] *= PSEUDO_LABEL_SCALE  # cx
        K[1, 2] *= PSEUDO_LABEL_SCALE  # cy
        crop_top = int(CROP_TOP_ORIG * PSEUDO_LABEL_SCALE)
        K[1, 2] -= crop_top            # cy adjust
        intrinsics.append(K)
        
        # extrinsic: lidar2cam = inv(cam2lidar)
        # cam2lidar = [sensor2lidar_rotation | sensor2lidar_translation]
        R_c2l = np.array(cam['sensor2lidar_rotation'], dtype=np.float64)  # (3,3)
        t_c2l = np.array(cam['sensor2lidar_translation'], dtype=np.float64)  # (3,)
        cam2lidar = np.eye(4, dtype=np.float64)
        cam2lidar[:3, :3] = R_c2l
        cam2lidar[:3, 3] = t_c2l
        lidar2cam = np.linalg.inv(cam2lidar)
        extrinsics.append(lidar2cam)
    
    gs_intrins = torch.from_numpy(np.stack(intrinsics)).float()  # (6, 3, 3)
    gs_extrins = torch.from_numpy(np.stack(extrinsics)).float()  # (6, 4, 4)
    return gs_extrins, gs_intrins


def load_pseudo_depth(frame, scene_name=None):
    """Load and downsample pseudo depth for foreground masking."""
    # Need scene_name and token to find the file
    token = frame.get('lidar_token', None)
    if token is None:
        return None
    
    if scene_name is None:
        return None
    
    depth_path = os.path.join(PSEUDO_DEPTH_ROOT, scene_name, token + '.npy')
    if not os.path.exists(depth_path):
        return None
    
    # (6, 900, 1600)
    raw = np.load(depth_path).astype(np.float32)
    depth = torch.from_numpy(raw)
    
    # downsample
    import torch.nn.functional as F
    depth = F.interpolate(
        depth.unsqueeze(1), scale_factor=PSEUDO_LABEL_SCALE, mode='bilinear', align_corners=False
    ).squeeze(1)  # (6, H_scaled, W_scaled)
    
    # crop top
    crop_top = int(CROP_TOP_ORIG * PSEUDO_LABEL_SCALE)
    depth = depth[:, crop_top:]
    
    # match render size
    if depth.shape[1] != RENDER_H or depth.shape[2] != RENDER_W:
        depth = F.interpolate(
            depth.unsqueeze(1),
            size=(RENDER_H, RENDER_W),
            mode='bilinear', align_corners=False
        ).squeeze(1)
    
    return depth  # (6, H, W)


def render_depth_variance(means, quats, scales, opacities, gs_extrins, gs_intrins,
                          device='cuda:0', eps=1e-4):
    """
    Render per-ray depth variance Var[z] for all cameras.
    Replicates GaussianRasterizer2D._render_depth_variance().
    
    Returns:
        rendered_var: (6, H, W) per-ray depth variance
        rendered_depth: (6, H, W) accumulated depth (for reference)
    """
    from gsplat import rasterization
    
    means = means.to(device)
    quats = quats.to(device)
    scales = scales.to(device)
    opacities = opacities.to(device)
    gs_extrins = gs_extrins.to(device)
    gs_intrins = gs_intrins.to(device)
    
    nC = gs_extrins.shape[0]
    
    # First: render depth + accumulation
    dummy = means.new_zeros((means.shape[0], 1))
    rendered_full, alpha_full, _ = rasterization(
        means=means, quats=quats, scales=scales, opacities=opacities,
        colors=dummy, viewmats=gs_extrins, Ks=gs_intrins,
        width=RENDER_W, height=RENDER_H, render_mode='RGB+D',
    )
    rendered_depth = rendered_full[..., 1]       # (nC, H, W) accumulated depth
    rendered_acc = alpha_full[..., 0]            # (nC, H, W)
    
    # Second: render z^2 per camera for variance
    ones = means.new_ones((means.shape[0], 1))
    means_h = torch.cat([means, ones], dim=-1)  # (G, 4)
    
    var_list = []
    for c in range(nC):
        cam_pts = means_h @ gs_extrins[c].transpose(0, 1)  # (G, 4)
        z = cam_pts[:, 2]                                   # (G,)
        z2 = (z * z).unsqueeze(-1)                          # (G, 1)
        
        out_c, alpha_c, _ = rasterization(
            means=means, quats=quats, scales=scales, opacities=opacities,
            colors=z2, viewmats=gs_extrins[c:c+1], Ks=gs_intrins[c:c+1],
            width=RENDER_W, height=RENDER_H, render_mode='RGB+D',
        )
        z2_acc = out_c[0, ..., 0]    # sum T a z^2
        d_acc = out_c[0, ..., 1]     # sum T a z (D channel)
        A_c = alpha_c[0, ..., 0].clamp_min(eps)
        E_z = d_acc / A_c
        E_z2 = z2_acc / A_c
        var_c = (E_z2 - E_z * E_z).clamp_min(0.0)
        var_list.append(var_c)
    
    rendered_var = torch.stack(var_list, dim=0)  # (nC, H, W)
    return rendered_var, rendered_depth, rendered_acc


def compute_stats(rendered_var, rendered_depth, rendered_acc, pseudo_depth=None):
    """Compute summary statistics."""
    stats = {}
    
    # Foreground mask: use pseudo_depth > 0.5 if available, else acc > 0.01
    if pseudo_depth is not None:
        fg_mask = pseudo_depth.to(rendered_var.device) > 0.5
    else:
        fg_mask = rendered_acc > 0.01
    
    var_fg = rendered_var[fg_mask]
    std_fg = var_fg.clamp_min(0).sqrt()
    
    stats['var_fg_mean'] = var_fg.mean().item()
    stats['std_fg_mean'] = std_fg.mean().item()
    stats['std_fg_median'] = std_fg.median().item()
    stats['std_fg_p75'] = std_fg.quantile(0.75).item()
    stats['std_fg_p90'] = std_fg.quantile(0.90).item()
    stats['depth_fg_mean'] = rendered_depth[fg_mask].mean().item()
    stats['acc_fg_mean'] = rendered_acc[fg_mask].mean().item()
    stats['fg_pixels'] = fg_mask.sum().item()
    
    return stats


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    # cd to project root
    project_root = os.environ.get('PROJECT_ROOT', '/data/chenz/GaussianAD')
    os.chdir(project_root)
    print(f"Working dir: {os.getcwd()}")
    
    # Load val pkl
    print(f"Loading {PKL_PATH} ...")
    with open(PKL_PATH, 'rb') as f:
        data = pickle.load(f)
    infos = data['infos']
    
    # Flatten scenes into frame list with scene info
    all_frames = []
    scene_names_map = {}  # scene_token -> scene_name (for pseudo label paths)
    for scene_token, frames in infos.items():
        # Try to find scene_name from metadata or frame
        for fr in frames:
            scene_name = fr.get('scene_name', None)
            if scene_name:
                scene_names_map[scene_token] = scene_name
                break
        if scene_token not in scene_names_map:
            # Fall back: try to get from metadata
            scene_names_map[scene_token] = f"scene-unknown-{scene_token[:8]}"
        for fr in frames:
            all_frames.append((scene_token, fr))
    
    print(f"Total val frames: {len(all_frames)}")
    
    # Load gaussian checkpoints
    gaussians = {}
    for name, path in CHECKPOINTS.items():
        print(f"Loading gaussians: {name} from {path}")
        gaussians[name] = load_gaussian_attr(path)
    
    # Sample frames (evenly spaced)
    n = min(NUM_SAMPLES, len(all_frames))
    indices = np.linspace(0, len(all_frames) - 1, n, dtype=int)
    
    print(f"\nRendering depth variance on {n} val frames...")
    print("=" * 70)
    
    # Accumulate per-model stats
    all_stats = {name: [] for name in CHECKPOINTS}
    
    for i, idx in enumerate(indices):
        scene_token, frame = all_frames[idx]
        scene_name = scene_names_map.get(scene_token, None)
        
        # Build camera params
        gs_extrins, gs_intrins = build_camera_params(frame)
        
        # Load pseudo depth (for fg mask)
        pseudo_depth = load_pseudo_depth(frame, scene_name)
        
        if i == 0:
            if pseudo_depth is not None:
                print(f"  Pseudo depth loaded: shape={pseudo_depth.shape}")
            else:
                print(f"  Pseudo depth NOT found, using acc>0.01 as fg mask")
        
        # Render each model
        for name in CHECKPOINTS:
            means, scales, quats, opacities = gaussians[name]
            with torch.no_grad():
                var, depth, acc = render_depth_variance(
                    means, quats, scales, opacities, gs_extrins, gs_intrins,
                    device=DEVICE
                )
            stats = compute_stats(var, depth, acc, pseudo_depth)
            all_stats[name].append(stats)
        
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  [{i+1}/{n}] frame {idx}: "
                  + " | ".join(f"{name}: std_fg={all_stats[name][-1]['std_fg_mean']:.4f}m"
                               for name in CHECKPOINTS))
    
    # ─── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY: Per-ray depth std (sqrt(Var[z])) on foreground pixels")
    print("=" * 70)
    print(f"{'Metric':<20}", end="")
    for name in CHECKPOINTS:
        print(f"{name:>15}", end="")
    print(f"{'delta':>12}")
    print("-" * 70)
    
    names = list(CHECKPOINTS.keys())
    metrics = ['std_fg_mean', 'std_fg_median', 'std_fg_p75', 'std_fg_p90',
               'var_fg_mean', 'depth_fg_mean', 'acc_fg_mean']
    
    for metric in metrics:
        vals = {}
        for name in names:
            vals[name] = np.mean([s[metric] for s in all_stats[name]])
        
        print(f"{metric:<20}", end="")
        for name in names:
            print(f"{vals[name]:>15.4f}", end="")
        
        # delta (concentrate - ft_depth)
        if len(names) == 2:
            delta = vals[names[1]] - vals[names[0]]
            pct = delta / max(abs(vals[names[0]]), 1e-8) * 100
            sign = "+" if delta > 0 else ""
            print(f"  {sign}{delta:.4f} ({sign}{pct:.1f}%)", end="")
        print()
    
    print("-" * 70)
    print("\nInterpretation:")
    v0 = np.mean([s['std_fg_mean'] for s in all_stats[names[0]]])
    v1 = np.mean([s['std_fg_mean'] for s in all_stats[names[1]]])
    if v1 < v0:
        pct = (v0 - v1) / v0 * 100
        print(f"  ✓ ConcLoss REDUCED depth std by {pct:.1f}% ({v0:.4f}m → {v1:.4f}m)")
        print(f"    This proves ConcLoss achieves its optimization target.")
    else:
        pct = (v1 - v0) / v0 * 100
        print(f"  ✗ ConcLoss did NOT reduce depth std (increased by {pct:.1f}%: {v0:.4f}m → {v1:.4f}m)")
        print(f"    ConcLoss failed to achieve its optimization target.")
    
    # Per-frame comparison
    print(f"\nPer-frame std_fg (m):")
    print(f"{'frame':<8}", end="")
    for name in names:
        print(f"{name:>12}", end="")
    print(f"{'winner':>10}")
    
    wins = {n: 0 for n in names}
    for i in range(n):
        print(f"  {i:<6}", end="")
        frame_vals = {}
        for name in names:
            val = all_stats[name][i]['std_fg_mean']
            frame_vals[name] = val
            print(f"{val:>12.4f}", end="")
        winner = min(frame_vals, key=frame_vals.get)
        wins[winner] += 1
        print(f"{'  ← ' + winner:>10}")
    
    print(f"\nWin count: ", end="")
    for name in names:
        print(f"{name}={wins[name]}/{n}  ", end="")
    print()


if __name__ == '__main__':
    main()
