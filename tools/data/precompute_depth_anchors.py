"""
Precompute initial anchor xyz from Metric3D depth maps.

Aggregates backprojected 3D points from a subset of training samples,
then voxel-downsamples to get `num_anchor` representative positions.

Output: .npy file of shape (num_anchor, 3) in LIDAR frame.

Usage:
    python tools/data/precompute_depth_anchors.py \
        --dataroot data/nuscenes \
        --depth-root data/metric_3d_nusc \
        --pkl data/nuscenes_cam/nuscenes_infos_train_gaussian_ad_v6.pkl \
        --num-anchor 25600 \
        --num-samples 200 \
        --output data/depth_anchor_init_25600.npy
"""

import os, sys, argparse, pickle
import numpy as np
from pyquaternion import Quaternion


# Camera order in depth .npy files (from GaussianFlowOcc)
DEPTH_CAM_ORDER = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                   'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
# GaussianAD sensor_types order
SENSOR_TYPES = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']


def get_reorder_indices():
    """Get indices to reorder from depth .npy order to sensor_types order."""
    return [DEPTH_CAM_ORDER.index(s) for s in SENSOR_TYPES]


def backproject_sample(info, depth_root, pc_range, max_depth=40.0, stride=8):
    """Backproject one sample's depth maps to 3D points in LIDAR frame."""
    scene_name = info.get('scene_name', None)
    if scene_name is None:
        return np.zeros((0, 3), dtype=np.float32)

    sample_token = info['token']
    depth_path = os.path.join(depth_root, scene_name, f'{sample_token}.npy')
    if not os.path.exists(depth_path):
        return np.zeros((0, 3), dtype=np.float32)

    raw_depth = np.load(depth_path).astype(np.float32)  # (6, 900, 1600)
    reorder = get_reorder_indices()
    raw_depth = raw_depth[reorder]

    # Build ego2lidar
    lidar2ego = np.eye(4, dtype=np.float64)
    lidar2ego[:3, :3] = Quaternion(info['lidar2ego_rotation']).rotation_matrix
    lidar2ego[:3, 3] = np.asarray(info['lidar2ego_translation'])
    ego2lidar = np.linalg.inv(lidar2ego)

    # Get cam params
    cams_info = info['cams_info']
    all_pts = []

    for c, cam_name in enumerate(SENSOR_TYPES):
        cam_info = cams_info[cam_name]
        # cam intrinsic
        K = np.array(cam_info['cam_intrinsic'], dtype=np.float64)[:3, :3]
        # cam2ego
        cam2ego = np.eye(4, dtype=np.float64)
        cam2ego[:3, :3] = Quaternion(cam_info['sensor2ego_rotation']).rotation_matrix
        cam2ego[:3, 3] = np.asarray(cam_info['sensor2ego_translation'])
        # cam2lidar
        cam2lidar = ego2lidar @ cam2ego

        depth_c = raw_depth[c]
        H, W = depth_c.shape
        v_grid, u_grid = np.mgrid[0:H:stride, 0:W:stride]
        d_vals = depth_c[v_grid, u_grid]
        valid = (d_vals > 0.5) & (d_vals < max_depth)
        u_valid = u_grid[valid].astype(np.float64)
        v_valid = v_grid[valid].astype(np.float64)
        d_valid = d_vals[valid].astype(np.float64)

        if len(d_valid) == 0:
            continue

        K_inv = np.linalg.inv(K)
        pts_cam = K_inv @ np.stack([u_valid * d_valid,
                                    v_valid * d_valid,
                                    d_valid], axis=0)
        pts_cam_h = np.vstack([pts_cam, np.ones((1, pts_cam.shape[1]))])
        pts_lidar = (cam2lidar @ pts_cam_h)[:3].T  # (N, 3)
        all_pts.append(pts_lidar.astype(np.float32))

    if len(all_pts) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    pts = np.concatenate(all_pts, axis=0)

    # Filter to pc_range
    mask = ((pts[:, 0] >= pc_range[0]) & (pts[:, 0] < pc_range[3]) &
            (pts[:, 1] >= pc_range[1]) & (pts[:, 1] < pc_range[4]) &
            (pts[:, 2] >= pc_range[2]) & (pts[:, 2] < pc_range[5]))
    return pts[mask]


def voxel_downsample(pts, voxel_size, num_target):
    """Voxel downsample, then random subsample/repeat to exact num_target."""
    if pts.shape[0] == 0:
        return np.random.uniform(-30, 30, (num_target, 3)).astype(np.float32)

    # Voxel downsample: average points per voxel
    voxel_indices = np.floor(pts / voxel_size).astype(np.int32)
    # Use a dict to accumulate
    voxel_dict = {}
    for i in range(pts.shape[0]):
        key = tuple(voxel_indices[i])
        if key not in voxel_dict:
            voxel_dict[key] = []
        voxel_dict[key].append(pts[i])

    # Compute voxel centers (mean of points in each voxel)
    centers = np.array([np.mean(v, axis=0) for v in voxel_dict.values()], dtype=np.float32)

    N = centers.shape[0]
    if N >= num_target:
        indices = np.random.choice(N, num_target, replace=False)
        return centers[indices]
    else:
        # Repeat + jitter
        repeats = (num_target // N) + 1
        pts_rep = np.tile(centers, (repeats, 1))[:num_target]
        jitter = np.random.randn(*pts_rep.shape).astype(np.float32) * 0.1
        pts_rep = pts_rep + jitter
        return pts_rep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='data/nuscenes')
    parser.add_argument('--depth-root', required=True)
    parser.add_argument('--pkl', required=True)
    parser.add_argument('--num-anchor', type=int, default=25600)
    parser.add_argument('--num-samples', type=int, default=200,
                        help='Number of training frames to aggregate')
    parser.add_argument('--max-depth', type=float, default=40.0)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--voxel-size', type=float, default=0.4)
    parser.add_argument('--pc-range', type=float, nargs=6,
                        default=[-30, -30, -2, 30, 30, 2])
    parser.add_argument('--output', default='data/depth_anchor_init_25600.npy')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    print(f"Loading PKL: {args.pkl}")
    with open(args.pkl, 'rb') as f:
        data = pickle.load(f)
    infos = data['infos'] if isinstance(data, dict) else data

    # Randomly select a subset of frames
    num_samples = min(args.num_samples, len(infos))
    indices = np.random.choice(len(infos), num_samples, replace=False)
    print(f"Processing {num_samples} frames...")

    all_pts = []
    for i, idx in enumerate(indices):
        info = infos[idx]
        pts = backproject_sample(info, args.depth_root, args.pc_range,
                                 args.max_depth, args.stride)
        all_pts.append(pts)
        if (i + 1) % 50 == 0:
            total = sum(p.shape[0] for p in all_pts)
            print(f"  [{i+1}/{num_samples}] accumulated {total:,} points")

    all_pts = np.concatenate(all_pts, axis=0)
    print(f"Total points after filtering: {all_pts.shape[0]:,}")

    # Voxel downsample to num_anchor
    print(f"Voxel downsampling (voxel_size={args.voxel_size}) to {args.num_anchor} points...")
    anchors = voxel_downsample(all_pts, args.voxel_size, args.num_anchor)
    print(f"Output shape: {anchors.shape}")
    print(f"  x range: [{anchors[:, 0].min():.2f}, {anchors[:, 0].max():.2f}]")
    print(f"  y range: [{anchors[:, 1].min():.2f}, {anchors[:, 1].max():.2f}]")
    print(f"  z range: [{anchors[:, 2].min():.2f}, {anchors[:, 2].max():.2f}]")

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    np.save(args.output, anchors)
    print(f"Saved to: {args.output}")


if __name__ == '__main__':
    main()
