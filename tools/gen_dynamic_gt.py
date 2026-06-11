"""离线生成动静分离 2D 真值（render 分辨率，喂给 DynamicLoss）。

LiDAR 主分支（方案 2）：纯 LiDAR 时序残差 + ego 速度门控。
  - 复用 visualize_dynamic_static_unsup.py 的逐点动静标签生成
    （RANSAC 地面 + 邻帧 ICP 配准 + 最近邻残差 + DBSCAN 簇投票）
  - ego 速度门控：高速帧（径向视差导致假阳）整帧标为 ignore
  - 把逐点标签投影到 6 个相机的 render 平面（与 gsplat 渲染同分辨率/内参），
    画小圆盘得到稠密 2D mask

光流融合分支（--use-flow，默认关）：RAFT 实际光流 - Metric3D 深度刚性流残差。
  - 见 tools/flow_dynamic.py：对 GroundedSAM 可移动连通域用残差中位数判动
  - 与 LiDAR 动态做像素级 union（动优先级最高），几何互补：
    光流强于径向/迫近运动，LiDAR 强于切向/同向运动
  - LiDAR 配准失败（高速帧）整帧 ignore 时，光流仍可补上径向运动车辆

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
  # 纯 LiDAR（原行为）
  /data/chenz/conda_env/splatting/bin/python tools/gen_dynamic_gt.py \
      --pkl data/nuscenes_cam/nuscenes_infos_train_gaussian_ad_v4.pkl \
      --data-root data/nuscenes \
      --out-dir data/dynamic_gt_nusc \
      --num-shards 8 --shard-id 0
  # LiDAR + 光流融合（加 --use-flow，需占一块 GPU 跑 RAFT）
  CUDA_VISIBLE_DEVICES=3 /data/chenz/conda_env/splatting/bin/python tools/gen_dynamic_gt.py \
      --pkl data/nuscenes_cam/nuscenes_infos_train_gaussian_ad_v4.pkl \
      --data-root data/nuscenes --out-dir data/dynamic_gt_nusc --use-flow
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


def _mat_from_rt(rotation, translation):
    """四元数 + 平移 → 4x4 齐次变换。"""
    M = np.eye(4)
    M[:3, :3] = Quaternion(rotation).rotation_matrix
    M[:3, 3] = np.asarray(translation)
    return M


def load_sweep_lidar(sweep):
    """读 sweep（非关键帧）LiDAR 点云 (N,3)，lidar 系。"""
    p = sweep['data_path']
    if p.startswith('./'):
        p = p[2:]
    pc = np.fromfile(p, dtype=np.float32).reshape(-1, 5)
    return pc[:, :3]


def sweep_l2g(sweep):
    """sweep lidar → global = ego2global @ sensor2ego，与关键帧 lidar2global 一致。"""
    e2g = _mat_from_rt(sweep['ego2global_rotation'], sweep['ego2global_translation'])
    s2e = _mat_from_rt(sweep['sensor2ego_rotation'], sweep['sensor2ego_translation'])
    return e2g @ s2e


def collect_adaptive_neighbors(scene, idx, data_root, target_dt, max_dt=0.5):
    """按目标时间间隔 target_dt（秒）自适应选邻帧。

    候选池 = 关键帧 ±1/±2 + 当前关键帧的 prev sweeps（过去侧）
            + 下一关键帧的 prev sweeps（未来侧）。
    在 past/future 两侧各选最接近 target_dt 与 2*target_dt 的候选（共最多 4 个）。
    target_dt 由 ego 速度决定：高速 → 小 dt → sweep（压低 radial 视差假阳）；
    低速 → 大 dt（0.5s）→ 关键帧（保留真动态物位移、检出灵敏）。

    返回 list[(off, (pts_lidar, l2g))]，off=1 标记最近邻（overlay 用）。
    """
    t0 = scene[idx]['timestamp']
    cands = []  # (signed_dt, kind, payload)
    for off in (-2, -1, 1, 2):
        j = idx + off
        if 0 <= j < len(scene):
            nj = scene[j]
            dt = (nj['timestamp'] - t0) / 1e6
            cands.append((dt, 'kf', nj))
    for sw in scene[idx].get('sweeps', []):
        dt = (sw['timestamp'] - t0) / 1e6
        if 1e-3 < abs(dt) <= max_dt + 1e-6:
            cands.append((dt, 'sw', sw))
    if idx + 1 < len(scene):
        for sw in scene[idx + 1].get('sweeps', []):
            dt = (sw['timestamp'] - t0) / 1e6
            if 1e-3 < dt <= max_dt + 1e-6:
                cands.append((dt, 'sw', sw))

    # 两侧各选最接近 target_dt 与 2*target_dt 的候选（按候选索引去重，避免 dict 比较）
    chosen_idx = []
    for sign in (-1, 1):
        side = [(i, c) for i, c in enumerate(cands) if np.sign(c[0]) == sign]
        if not side:
            continue
        for mult in (1.0, 2.0):
            tgt = min(target_dt * mult, max_dt)
            i_best = min(side, key=lambda ic: abs(abs(ic[1][0]) - tgt))[0]
            if i_best not in chosen_idx:
                chosen_idx.append(i_best)

    chosen = sorted((cands[i] for i in chosen_idx), key=lambda c: abs(c[0]))
    neighbors = []
    for rank, (dt, kind, payload) in enumerate(chosen):
        if kind == 'kf':
            pts = load_lidar(data_root, payload)
            l2g = lidar2global_mat(payload['data']['LIDAR_TOP']['calib'],
                                   payload['data']['LIDAR_TOP']['pose'])
        else:
            pts = load_sweep_lidar(payload)
            l2g = sweep_l2g(payload)
        off = 1 if rank == 0 else rank + 1   # 最近邻 off=1，供 overlay 选取
        neighbors.append((off, (pts, l2g)))
    return neighbors


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
                      static_radius, dyn_radius, is_uncertain=None):
    """投影 lidar 点到 render 平面，画动静圆盘。返回 (H,W) uint8。

    绘制顺序（后画覆盖先画）：静态(1) → 不确定(0,挖 ignore 洞) → 动态(2,最高优先)。
    is_uncertain：方案1 中被置信度过滤退回的低置信曾动态点，投影成 ignore(0)，不监督。
    """
    mask = np.zeros((H, W), dtype=np.uint8)
    pts_h = np.concatenate([pts_lidar, np.ones((pts_lidar.shape[0], 1))], axis=1)
    cam = (lidar2cam @ pts_h.T).T            # (N,4)
    depth = cam[:, 2]
    front = depth > 0.1
    uv = (K @ cam[:, :3].T).T                 # (N,3)
    uv = uv[:, :2] / np.clip(uv[:, 2:3], 1e-3, None)
    inimg = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    vis = front & inimg
    if is_uncertain is None:
        is_uncertain = np.zeros(pts_lidar.shape[0], dtype=bool)
    # 1) 静态(1) 铺底（排除动态与不确定点）
    sta = vis & (~is_dyn) & (~is_uncertain)
    for u, v in uv[sta]:
        paint_disk(mask, u, v, 1, static_radius)
    # 2) 不确定点 → 画 ignore(0) 盘，挖空该区域（既不教动态也不教静态）
    unc = vis & is_uncertain
    for u, v in uv[unc]:
        paint_disk(mask, u, v, 0, dyn_radius)
    # 3) 动态(2) 最高优先，覆盖一切
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


def gen_one(scene, idx, data_root, args, detector=None):
    """生成单帧 (6,H,W) 动静 GT。

    LiDAR 分支：时序残差 + ego 速度门控（主分支）。
    光流分支（detector 不为 None 时）：RAFT 光流-刚性流残差 union。
    二者几何互补：光流强于径向/迫近运动，LiDAR 强于切向/同向运动。
    关键：LiDAR 配准失败（高速帧）或单帧/无邻帧时 LiDAR 整帧 ignore，
    但光流仍能补上径向运动车辆（不依赖 LiDAR 邻帧，只靠相机 sweep）。
    """
    info = scene[idx]
    H, W = args.render_h, args.render_w
    out = np.zeros((6, H, W), dtype=np.uint8)

    # ---- LiDAR 主分支 ----
    lidar_status = 'ok'
    if len(scene) < 2:
        lidar_status = 'edge'
    else:
        # ego 速度自适应时间基线：高速 → 小 dt（sweep，压 radial 视差）；低速 → 0.5s 关键帧
        spd = ego_speed(scene, idx)
        target_dt = float(np.clip(args.dt_budget / max(spd, 0.5), args.min_dt, args.max_dt))
        neighbors = collect_adaptive_neighbors(scene, idx, data_root, target_dt, args.max_dt)
        if len(neighbors) == 0:
            lidar_status = 'edge'
        else:
            pts_t = load_lidar(data_root, info)
            l2g_t = lidar2global_mat(info['data']['LIDAR_TOP']['calib'],
                                     info['data']['LIDAR_TOP']['pose'])
            is_dyn, score, ground_mask, overlay, pts_t_g, fitness, rmse, is_uncertain = generate_dynamic_labels(
                pts_t, l2g_t, neighbors, args.dist_thresh,
                use_icp=True, vote_ratio=args.vote_ratio,
                min_dyn_cluster=args.min_dyn_cluster, strong_abs=args.strong_abs,
                max_dyn_range=args.max_dyn_range)
            # 配准失败帧 → LiDAR 整帧 ignore（rmse 与车速无关，是可靠门控），交由光流补
            if rmse > args.rmse_thresh:
                lidar_status = f'badrmse({rmse:.2f},spd={spd:.1f})'
            else:
                lidar_calib = info['data']['LIDAR_TOP']['calib']
                for ci, cam_type in enumerate(SENSOR_TYPES):
                    cam = info['data'][cam_type]
                    l2c = lidar2cam_mat(cam['calib'], lidar_calib)
                    K = render_intrinsic(cam['calib'], args.scale, args.crop_top)
                    out[ci] = project_and_paint(pts_t, is_dyn, l2c, K, H, W,
                                                args.static_radius, args.dyn_radius, is_uncertain)
                n_unc = int(is_uncertain.sum())
                lidar_status = f'ok(unc_pts={n_unc},rmse={rmse:.2f},spd={spd:.1f},dt={target_dt:.2f})'

    # ---- 光流分支 union（动优先级最高，覆盖 LiDAR 的 static/ignore）----
    flow_px = 0
    if detector is not None:
        try:
            # LiDAR 判动像素(render空间) → 给光流近距超大守卫做互证仲裁
            lidar_dyn = np.stack([(out[ci] == 2) for ci in range(6)], axis=0)
            flow_dyn = detector.detect(info['token'], SENSOR_TYPES, lidar_dyn=lidar_dyn)
            for ci in range(6):
                out[ci][flow_dyn[ci]] = 2
            flow_px = int(flow_dyn.sum())
        except Exception as e:
            return out, f'lidar={lidar_status} flow_err:{e}'

    n_dyn = int((out == 2).sum())
    return out, f'lidar={lidar_status} dyn_px={n_dyn} flow_px={flow_px}'


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
    parser.add_argument('--dt-budget', type=float, default=1.5,
                        help='位移预算(m)：target_dt = clip(budget/v_ego, min_dt, max_dt)')
    parser.add_argument('--min-dt', type=float, default=0.15,
                        help='自适应时间基线下限(s)，约 3 个 sweep')
    parser.add_argument('--max-dt', type=float, default=0.5,
                        help='自适应时间基线上限(s)，关键帧间隔')
    parser.add_argument('--speed-thresh', type=float, default=3.0,
                        help='[已弃用] 旧整帧高速门控，自适应基线后不再使用')
    parser.add_argument('--vote-ratio', type=float, default=0.3)
    parser.add_argument('--min-dyn-cluster', type=int, default=15,
                        help='方案1：动态簇最小点数，小于此退回 ignore')
    parser.add_argument('--strong-abs', type=float, default=0.45,
                        help='方案1：动态簇中位残差(m)下限，弱残差退回 ignore')
    parser.add_argument('--max-dyn-range', type=float, default=40.0,
                        help='方案1：动态簇质心最大水平距离(m)，更远退回 ignore')
    parser.add_argument('--n-neighbor', type=int, default=2,
                        help='[已弃用] 旧固定关键帧邻帧数，改用自适应 sweep 选取')
    parser.add_argument('--static-radius', type=int, default=2)
    parser.add_argument('--dyn-radius', type=int, default=3)
    parser.add_argument('--num-shards', type=int, default=1,
                        help='总分片数（多 tmux 并行时设置）')
    parser.add_argument('--shard-id', type=int, default=0,
                        help='本进程负责的分片 id（0..num_shards-1）')
    parser.add_argument('--overwrite', action='store_true',
                        help='已存在的 .npy 也重新生成')
    # —— 光流融合分支（与 LiDAR 动态做 union，默认关，向后兼容）——
    parser.add_argument('--use-flow', action='store_true',
                        help='开启光流-刚性流残差分支，与 LiDAR 动态做像素级 union')
    parser.add_argument('--nusc-version', default='v1.0-trainval',
                        help='光流分支需加载 nuScenes devkit（取相机 sweep 与位姿）')
    parser.add_argument('--m3d-root', default='/data/chenz/Gaussianflowocc_test/data/metric_3d_nusc',
                        help='Metric3D 单目深度根目录')
    parser.add_argument('--sam-root', default='/data/chenz/Gaussianflowocc_test/data/grounded_sam_nusc',
                        help='GroundedSAM 语义 mask 根目录')
    parser.add_argument('--flow-res-thresh', type=float, default=3.0,
                        help='光流残差阈值（render 分辨率下像素），调高提 precision')
    parser.add_argument('--device', default='cuda', help='光流 RAFT 推理设备')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 光流检测器（可选）：加载 nuScenes devkit + RAFT，与 LiDAR 动态 union
    detector = None
    if args.use_flow:
        from nuscenes.nuscenes import NuScenes
        from tools.flow_dynamic import FlowDynamicDetector
        print(f'loading NuScenes {args.nusc_version} for optical-flow branch ...')
        nusc = NuScenes(version=args.nusc_version, dataroot=args.data_root, verbose=False)
        detector = FlowDynamicDetector(
            nusc, args.data_root, args.m3d_root, args.sam_root, device=args.device,
            scale=args.scale, crop_top=args.crop_top,
            render_h=args.render_h, render_w=args.render_w,
            res_thresh=args.flow_res_thresh)
        print('optical-flow detector ready')

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
            gt, status = gen_one(scene, frame_idx, args.data_root, args, detector=detector)
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
