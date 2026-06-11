"""动静解耦真值可视化。

按 GT box 的速度大小将物体分为动态 / 静态两类，输出：
  1. BEV 俯视图：自车在中心，box 按朝向绘制，动态红 / 静态蓝，画速度箭头
  2. 六相机投影图：3D box 投影到每个相机，按动静着色

真值来源（全部来自 nuScenes 标注，零噪声）：
  - gt_boxes  (N, 7): [x, y, z, dx, dy, dz, heading]  LIDAR 系
  - gt_velocity (N, 2): [vx, vy]  全局/LIDAR 系
  - 动态判据: |v| > --vel-thresh (默认 0.5 m/s)

用法:
  python tools/visualize_dynamic_static_gt.py \
      --pkl data/nuscenes_cam/nuscenes_infos_val_gaussian_ad_v4.pkl \
      --data-root data/nuscenes \
      --num-samples 5 --seed 0
"""
import os
import argparse

import numpy as np
import mmengine
from pyquaternion import Quaternion

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

SENSOR_TYPES = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

# 子图排布：前排 3 个，后排 3 个
CAM_GRID = [['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT'],
            ['CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']]

STATIC_COLOR = (0.20, 0.45, 0.95)   # 蓝
DYNAMIC_COLOR = (0.95, 0.25, 0.20)  # 红


def get_lidar2global(calib_dict, pose_dict):
    lidar2ego = np.eye(4)
    lidar2ego[:3, :3] = Quaternion(calib_dict['rotation']).rotation_matrix
    lidar2ego[:3, 3] = np.asarray(calib_dict['translation']).T
    ego2global = np.eye(4)
    ego2global[:3, :3] = Quaternion(pose_dict['rotation']).rotation_matrix
    ego2global[:3, 3] = np.asarray(pose_dict['translation']).T
    return ego2global @ lidar2ego


def get_img2global(calib_dict, pose_dict):
    cam2img = np.eye(4)
    cam2img[:3, :3] = np.asarray(calib_dict['camera_intrinsic'])
    img2cam = np.linalg.inv(cam2img)
    cam2ego = np.eye(4)
    cam2ego[:3, :3] = Quaternion(calib_dict['rotation']).rotation_matrix
    cam2ego[:3, 3] = np.asarray(calib_dict['translation']).T
    ego2global = np.eye(4)
    ego2global[:3, :3] = Quaternion(pose_dict['rotation']).rotation_matrix
    ego2global[:3, 3] = np.asarray(pose_dict['translation']).T
    return ego2global @ cam2ego @ img2cam


def box_to_corners_3d(box):
    """[x,y,z,dx,dy,dz,heading] -> (8,3) 角点（LIDAR 系）。"""
    x, y, z, dx, dy, dz, heading = box[:7]
    xc = np.array([1, 1, 1, 1, -1, -1, -1, -1]) * dx / 2
    yc = np.array([1, 1, -1, -1, 1, 1, -1, -1]) * dy / 2
    zc = np.array([1, -1, 1, -1, 1, -1, 1, -1]) * dz / 2
    corners = np.stack([xc, yc, zc], axis=1)  # (8,3)
    c, s = np.cos(heading), np.sin(heading)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    corners = corners @ rot.T
    corners += np.array([x, y, z])
    return corners


def box_to_bev_corners(box):
    """BEV 4 角点 (4,2)，已含朝向。"""
    x, y, _, dx, dy, _, heading = box[:7]
    xc = np.array([1, 1, -1, -1]) * dx / 2
    yc = np.array([1, -1, -1, 1]) * dy / 2
    corners = np.stack([xc, yc], axis=1)
    c, s = np.cos(heading), np.sin(heading)
    rot = np.array([[c, -s], [s, c]])
    corners = corners @ rot.T
    corners += np.array([x, y])
    return corners


def draw_bev(ax, gt_boxes, is_dynamic, gt_velocity, pc_range):
    ax.set_facecolor('white')
    # 自车
    ego = box_to_bev_corners([0, 0, 0, 4.084, 1.85, 0, 0])
    ax.add_patch(MplPolygon(ego, closed=True, fill=True,
                            facecolor=(0.1, 0.8, 0.1), edgecolor='k', alpha=0.6, zorder=5))
    ax.arrow(0, 0, 3, 0, head_width=0.8, head_length=1.0, fc='g', ec='g', zorder=6)

    for i, box in enumerate(gt_boxes):
        color = DYNAMIC_COLOR if is_dynamic[i] else STATIC_COLOR
        corners = box_to_bev_corners(box)
        ax.add_patch(MplPolygon(corners, closed=True, fill=False,
                                edgecolor=color, linewidth=1.8, zorder=4))
        cx, cy = box[0], box[1]
        if is_dynamic[i]:
            vx, vy = gt_velocity[i]
            spd = np.hypot(vx, vy)
            ax.arrow(cx, cy, vx, vy, head_width=0.6, head_length=0.8,
                     fc=color, ec=color, zorder=7, length_includes_head=True)
            ax.text(cx, cy, f'{spd:.1f}', fontsize=6, color=color, zorder=8)

    # nuScenes LIDAR: x 前, y 左 → BEV 习惯把 x 朝上
    ax.set_xlim(pc_range[1], pc_range[4])   # y
    ax.set_ylim(pc_range[0], pc_range[3])   # x
    ax.set_aspect('equal')
    ax.invert_xaxis()  # y 左为正 → 屏幕左侧
    ax.set_xlabel('y (left +)')
    ax.set_ylabel('x (forward +)')
    ax.set_title('BEV  (red=dynamic, blue=static)')
    ax.grid(True, alpha=0.2)


def project_boxes_to_cam(gt_boxes, lidar2img, img_w, img_h):
    results = []
    for box in gt_boxes:
        corners = box_to_corners_3d(box)             # (8,3)
        pts = np.concatenate([corners, np.ones((8, 1))], axis=1)  # (8,4)
        cam = (lidar2img @ pts.T).T                  # (8,4)
        depth = cam[:, 2]
        if np.all(depth < 0.1):
            results.append((None, False))
            continue
        uv = cam[:, :2] / np.clip(cam[:, 2:3], 1e-3, None)
        in_front = depth > 0.1
        in_img = (uv[:, 0] > -img_w * 0.3) & (uv[:, 0] < img_w * 1.3) & \
                 (uv[:, 1] > -img_h * 0.3) & (uv[:, 1] < img_h * 1.3)
        visible = bool(np.any(in_front & in_img))
        results.append((uv, visible))
    return results


# 3D box 的 12 条边
BOX_EDGES = [(0, 1), (0, 2), (1, 3), (2, 3),       # 前面（dx+）
             (4, 5), (4, 6), (5, 7), (6, 7),       # 后面（dx-）
             (0, 4), (1, 5), (2, 6), (3, 7)]       # 连接


def draw_cam(ax, img, gt_boxes, is_dynamic, lidar2img):
    h, w = img.shape[:2]
    ax.imshow(img)
    proj = project_boxes_to_cam(gt_boxes, lidar2img, w, h)
    for i, (uv, visible) in enumerate(proj):
        if uv is None or not visible:
            continue
        color = DYNAMIC_COLOR if is_dynamic[i] else STATIC_COLOR
        for a, b in BOX_EDGES:
            ax.plot([uv[a, 0], uv[b, 0]], [uv[a, 1], uv[b, 1]],
                    color=color, linewidth=1.2)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis('off')


def visualize_sample(info, data_root, vel_thresh, out_path):
    gt_boxes = np.asarray(info['gt_boxes'], dtype=np.float32)
    gt_velocity = np.asarray(info['gt_velocity'], dtype=np.float32)
    gt_names = np.asarray(info['gt_names'])

    if gt_boxes.shape[0] == 0:
        print(f'  [skip] no boxes: {info["token"]}')
        return

    gt_velocity = np.nan_to_num(gt_velocity, nan=0.0)
    speed = np.hypot(gt_velocity[:, 0], gt_velocity[:, 1])
    is_dynamic = speed > vel_thresh

    n_dyn = int(is_dynamic.sum())
    n_sta = int((~is_dynamic).sum())

    lidar_calib = info['data']['LIDAR_TOP']['calib']
    lidar_pose = info['data']['LIDAR_TOP']['pose']
    lidar2global = get_lidar2global(lidar_calib, lidar_pose)

    fig = plt.figure(figsize=(20, 11))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.3])

    for r in range(2):
        for c in range(3):
            cam_type = CAM_GRID[r][c]
            ax = fig.add_subplot(gs[r, c])
            cam_calib = info['data'][cam_type]['calib']
            cam_pose = info['data'][cam_type]['pose']
            img2global = get_img2global(cam_calib, cam_pose)
            lidar2img = np.linalg.inv(img2global) @ lidar2global
            img_path = os.path.join(data_root, info['data'][cam_type]['filename'])
            try:
                img = plt.imread(img_path)
            except FileNotFoundError:
                ax.text(0.5, 0.5, f'missing\n{cam_type}', ha='center')
                ax.axis('off')
                continue
            draw_cam(ax, img, gt_boxes, is_dynamic, lidar2img)
            ax.set_title(cam_type, fontsize=9)

    ax_bev = fig.add_subplot(gs[:, 3])
    pc_range = [-50, -50, -5, 50, 50, 5]
    draw_bev(ax_bev, gt_boxes, is_dynamic, gt_velocity, pc_range)

    fig.suptitle(
        f'token={info["token"][:12]}...  '
        f'dynamic={n_dyn} (red)  static={n_sta} (blue)  thresh={vel_thresh} m/s',
        fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {out_path}  (dyn={n_dyn}, sta={n_sta})')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl', default='data/nuscenes_cam/nuscenes_infos_val_gaussian_ad_v4.pkl')
    parser.add_argument('--data-root', default='data/nuscenes')
    parser.add_argument('--out-dir', default='out/dynamic_static_gt_vis')
    parser.add_argument('--num-samples', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--vel-thresh', type=float, default=0.5,
                        help='速度阈值 (m/s)，超过则判定为动态')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f'loading {args.pkl} ...')
    data = mmengine.load(args.pkl)
    scene_infos = data['infos']
    keyframes = data['metadata']  # list of (scene_token, index)

    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(keyframes), min(args.num_samples, len(keyframes)), replace=False)

    for k, idx in enumerate(pick):
        scene_token, frame_idx = keyframes[idx]
        info = scene_infos[scene_token][frame_idx]
        out_path = os.path.join(args.out_dir, f'sample_{k:03d}.jpg')
        print(f'[{k+1}/{len(pick)}] scene={scene_token[:8]} idx={frame_idx}')
        visualize_sample(info, args.data_root, args.vel_thresh, out_path)

    print(f'\ndone. results in {args.out_dir}/')


if __name__ == '__main__':
    main()
