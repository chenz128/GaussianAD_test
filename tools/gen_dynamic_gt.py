"""离线生成动静分离 2D 真值（render 分辨率，喂给 DynamicLoss）。

方案 2：纯 LiDAR 时序残差 + ego 速度门控。
  - 复用 visualize_dynamic_static_unsup.py 的逐点动静标签生成
    （RANSAC 地面 + 邻帧 ICP 配准 + 最近邻残差 + DBSCAN 簇投票）
  - ego 速度门控：高速帧（径向视差导致假阳）整帧标为 ignore
  - 把逐点标签投影到 6 个相机的 render 平面（与 gsplat 渲染同分辨率/内参），
    画小圆盘得到稠密 2D mask

输出（每个 sample 一个 .npy，扁平按 token 命名）：
  dynamic_gt_root/{token}.npy   shape (6, H, W) uint8
    0 = ignore（无 LiDAR 命中 / 高速帧整帧 / 远处）
    1 = static
    2 = dynamic
  相机顺序 = dataset.sensor_types:
    ['CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']
  render 内参/外参与 dataset.py pseudo 分支完全一致（scale=0.44, crop_top=140,
    lidar2cam = inv(cam2ego) @ lidar2ego），保证与 rendered_dynamic 像素对齐。

用法（远端 tmux，可分片并行）：
  /data/chenz/conda_env/splatting/bin/python tools/gen_dynamic_gt.py \
      --pkl data/nuscenes_cam/nuscenes_infos_train_gaussian_ad_v4.pkl \
      --data-root data/nuscenes \
      --out-dir data/dynamic_gt_nusc \
      --num-shards 8 --shard-id 0
"""
import os
import argparse

import numpy as np
import mmengine
from pyquaternion import Quaternion

from tools.visualize_dynamic_static_unsup import (
    load_lidar, lidar2global_mat, generate_dynamic_labels,
)

# dataset.sensor_types order（dataset 内部相机顺序，GT 直接按此顺序保存，免 reorder）
SENSOR_TYPES = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']


def lidar2cam_mat(cam_calib, lidar_calib):
    """lidar2cam = inv(cam2ego) @ lidar2ego，与 dataset.py pseudo 分支一致。"""
    cam2ego = np.eye(4)
    cam2ego[:3, :3] = Quaternion(cam_calib['rotation']).rotation_matrix
    cam2ego[:3, 3] = np.asarray(cam_calib['translation'])
    lidar2ego = np.eye(4)
    lidar2ego[:3, :3] = Quaternion(lidar_calib['rotation']).rotation_matrix
    lidar2ego[:3, 3] = np.asarray(lidar_calib['translation'])
    return np.linalg.inv(cam2ego) @ lidar2ego


def render_intrinsic(cam_calib, scale, crop_top):
    """render 内参 = camera_intrinsic 缩放 + crop 平移，与 dataset.py 一致。"""
    K = np.asarray(cam_calib['camera_intrinsic'], dtype=np.float64).copy()
    K[0, 0] *= scale  # fx
    K[1, 1] *= scale  # fy
    K[0, 2] *= scale  # cx
    K[1, 2] *= scale  # cy
    K[1, 2] -= crop_top
    return K


def paint_disk(mask, u, v, val, radius):
    """在 (H,W) mask 上以 (u,v) 为中心画半径 radius 的方形圆盘，写入 val。"""
    H, W = mask.shape
    u = int(round(u)); v = int(round(v))
    u0, u1 = max(0, u - radius), min(W, u + radius + 1)
    v0, v1 = max(0, v - radius), min(H, v + radius + 1)
    if u0 < u1 and v0 < v1:
        mask[v0:v1, u0:u1] = val


def project_and_paint(pts_lidar, is_dyn, lidar2cam, K, H, W,
                      static_radius, dyn_radius):
    """投影 lidar 点到 render 平面，画动静圆盘。返回 (H,W) uint8。"""
    mask = np.zeros((H, W), dtype=np.uint8)
    pts_h = np.concatenate([pts_lidar, np.ones((pts_lidar.shape[0], 1))], axis=1)
    cam = (lidar2cam @ pts_h.T).T            # (N,4)
    depth = cam[:, 2]
    front = depth > 0.1
    uv = (K @ cam[:, :3].T).T                 # (N,3)
    uv = uv[:, :2] / np.clip(uv[:, 2:3], 1e-3, None)
    inimg = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    vis = front & inimg
    # 先画静态(1)，再画动态(2)覆盖，保证动态优先
    sta = vis & (~is_dyn)
    for u, v in uv[sta]:
        paint_disk(mask, u, v, 1, static_radius)
    dyn = vis & is_dyn
    for u, v in uv[dyn]:
        paint_disk(mask, u, v, 2, dyn_radius)
    return mask


def ego_speed(scene, idx):
    """从全局位姿平移估计 ego 速度 (m/s)，关键帧间隔 0.5s。"""
    def center(i):
        ii = scene[i]
        l2g = lidar2global_mat(ii['data']['LIDAR_TOP']['calib'],
                               ii['data']['LIDAR_TOP']['pose'])
        return l2g[:3, 3]
    c0 = center(idx)
    if idx + 1 < len(scene):
        c1 = center(idx + 1)
    elif idx - 1 >= 0:
        c1 = center(idx - 1)
    else:
        return 0.0
    return float(np.linalg.norm(c1 - c0) / 0.5)


def gen_one(scene, idx, data_root, args):
    info = scene[idx]
    H, W = args.render_h, args.render_w
    out = np.zeros((6, H, W), dtype=np.uint8)

    # 边界帧无邻居 → 整帧 ignore
    if idx <= 0 or idx >= len(scene) - 1:
        return out, 'edge'

    spd = ego_speed(scene, idx)
    if spd > args.speed_thresh:
        return out, f'fast({spd:.1f}m/s)'   # 高速帧整帧 ignore

    pts_t = load_lidar(data_root, info)
    l2g_t = lidar2global_mat(info['data']['LIDAR_TOP']['calib'],
                             info['data']['LIDAR_TOP']['pose'])
    neighbors = []
    for off in range(-args.n_neighbor, args.n_neighbor + 1):
        if off == 0:
            continue
        j = idx + off
        if 0 <= j < len(scene):
            ni = scene[j]
            pts_n = load_lidar(data_root, ni)
            l2g_n = lidar2global_mat(ni['data']['LIDAR_TOP']['calib'],
                                     ni['data']['LIDAR_TOP']['pose'])
            neighbors.append((off, (pts_n, l2g_n)))

    is_dyn, score, ground_mask, overlay, pts_t_g, fitness, rmse = generate_dynamic_labels(
        pts_t, l2g_t, neighbors, args.dist_thresh,
        use_icp=True, vote_ratio=args.vote_ratio)

    # 配准失败帧 → 整帧 ignore（rmse 与车速无关，是可靠门控）
    if rmse > args.rmse_thresh:
        return out, f'badrmse({rmse:.2f})'

    lidar_calib = info['data']['LIDAR_TOP']['calib']
    for ci, cam_type in enumerate(SENSOR_TYPES):
        cam = info['data'][cam_type]
        l2c = lidar2cam_mat(cam['calib'], lidar_calib)
        K = render_intrinsic(cam['calib'], args.scale, args.crop_top)
        out[ci] = project_and_paint(pts_t, is_dyn, l2c, K, H, W,
                                    args.static_radius, args.dyn_radius)
    n_dyn = int((out == 2).sum())
    return out, f'ok(dyn_px={n_dyn},rmse={rmse:.2f},spd={spd:.1f})'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl', default='data/nuscenes_cam/nuscenes_infos_train_gaussian_ad_v4.pkl')
    parser.add_argument('--data-root', default='data/nuscenes')
    parser.add_argument('--out-dir', default='data/dynamic_gt_nusc')
    parser.add_argument('--scale', type=float, default=0.44)
    parser.add_argument('--crop-top', type=int, default=140)
    parser.add_argument('--render-h', type=int, default=256)
    parser.add_argument('--render-w', type=int, default=704)
    parser.add_argument('--dist-thresh', type=float, default=0.3)
    parser.add_argument('--rmse-thresh', type=float, default=0.35,
                        help='ICP inlier_rmse 超此判配准失败 → 整帧 ignore')
    parser.add_argument('--speed-thresh', type=float, default=3.0,
                        help='ego 速度超此判高速帧（径向视差假阳）→ 整帧 ignore')
    parser.add_argument('--vote-ratio', type=float, default=0.3)
    parser.add_argument('--n-neighbor', type=int, default=2)
    parser.add_argument('--static-radius', type=int, default=2)
    parser.add_argument('--dyn-radius', type=int, default=3)
    parser.add_argument('--num-shards', type=int, default=1,
                        help='总分片数（多 tmux 并行时设置）')
    parser.add_argument('--shard-id', type=int, default=0,
                        help='本进程负责的分片 id（0..num_shards-1）')
    parser.add_argument('--overwrite', action='store_true',
                        help='已存在的 .npy 也重新生成')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f'loading {args.pkl} ...')
    data = mmengine.load(args.pkl)
    scene_infos = data['infos']
    keyframes = data['metadata']   # list of (scene_token, frame_idx)

    # 分片
    my = [(i, kf) for i, kf in enumerate(keyframes)
          if i % args.num_shards == args.shard_id]
    print(f'shard {args.shard_id}/{args.num_shards}: {len(my)} / {len(keyframes)} keyframes')

    n_ok = n_skip = 0
    for cnt, (gi, (scene_token, frame_idx)) in enumerate(my):
        scene = scene_infos[scene_token]
        frame_idx = int(np.clip(frame_idx, 0, len(scene) - 1))
        token = scene[frame_idx]['token']
        out_path = os.path.join(args.out_dir, f'{token}.npy')
        if os.path.exists(out_path) and not args.overwrite:
            n_skip += 1
            continue
        try:
            gt, status = gen_one(scene, frame_idx, args.data_root, args)
        except Exception as e:
            print(f'  [ERR] {token[:12]} idx{frame_idx}: {e}')
            gt = np.zeros((6, args.render_h, args.render_w), dtype=np.uint8)
            status = f'error:{e}'
        np.save(out_path, gt)
        n_ok += 1
        if cnt % 50 == 0:
            print(f'  [{cnt}/{len(my)}] {token[:12]} idx{frame_idx} -> {status}')
    print(f'done shard {args.shard_id}: generated={n_ok}, skipped(existing)={n_skip}')


if __name__ == '__main__':
    main()
