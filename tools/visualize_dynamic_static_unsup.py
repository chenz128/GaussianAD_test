"""无标注动静分离真值可视化（LiDAR 时序残差法）。

完全不使用 nuScenes 的 3D 框标注，只用：
  - LiDAR 点云（samples/LIDAR_TOP/*.pcd.bin）
  - ego/lidar 位姿（来自定位，非人工标注）

原理（ego-motion compensated scene residual）：
  把中心帧 t 与相邻帧 t-1 / t+1 的点云全部用 lidar2global 对齐到全局系。
  - 静态结构（路面、建筑、停放车）在不同帧完全重合 → 最近邻距离极小
  - 动态物体（行驶车辆、行人）发生位移 → 最近邻距离大
  逐点 dynamic_score = min(NN_dist_to_{t-1}, NN_dist_to_{t+1})
  阈值化 → 动态 mask（无标注伪真值）

输出三联图：
  1. 全局系点云叠加：t(蓝) vs t+1(橙) —— 直观看"拖影"
  2. BEV 动态分数着色：静态灰 → 动态红
  3. 六相机：把动态点投影到图像上（红点 = 模型该判为动态的区域）

用法:
  python tools/visualize_dynamic_static_unsup.py \
      --pkl data/nuscenes_cam/nuscenes_infos_val_gaussian_ad_v4.pkl \
      --data-root data/nuscenes --num-samples 5 --seed 0 \
      --dist-thresh 0.3
"""
import os
import argparse

import numpy as np
import mmengine
from pyquaternion import Quaternion
from scipy.spatial import cKDTree
import open3d as o3d
from sklearn.cluster import DBSCAN

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CAM_GRID = [['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT'],
            ['CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']]


def lidar2global_mat(calib_dict, pose_dict):
    lidar2ego = np.eye(4)
    lidar2ego[:3, :3] = Quaternion(calib_dict['rotation']).rotation_matrix
    lidar2ego[:3, 3] = np.asarray(calib_dict['translation']).T
    ego2global = np.eye(4)
    ego2global[:3, :3] = Quaternion(pose_dict['rotation']).rotation_matrix
    ego2global[:3, 3] = np.asarray(pose_dict['translation']).T
    return ego2global @ lidar2ego


def img2global_mat(calib_dict, pose_dict):
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


def load_lidar(data_root, info):
    fn = info['data']['LIDAR_TOP']['filename']
    pc = np.fromfile(os.path.join(data_root, fn), dtype=np.float32).reshape(-1, 5)
    return pc[:, :3]  # (N,3) lidar 系


def _axis_angle_from_R(R):
    """旋转矩阵 → (单位轴 k, 角度 θ)。"""
    cos_t = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_t)
    if theta < 1e-8:
        return np.array([0.0, 0.0, 1.0]), 0.0
    k = np.array([R[2, 1] - R[1, 2],
                  R[0, 2] - R[2, 0],
                  R[1, 0] - R[0, 1]]) / (2.0 * np.sin(theta))
    n = np.linalg.norm(k)
    if n < 1e-8:
        return np.array([0.0, 0.0, 1.0]), 0.0
    return k / n, theta


def deskew_points(pts_xyz, M_rel, dt_kf, sweep_dur=0.05, az_ref=0.5, sign=1.0):
    """旋转 LiDAR 逐点去畸变（运动补偿）。

    旋转式 LiDAR 转一圈 ~sweep_dur 秒，期间自车在运动 → 单帧点云扭曲。
    用方位角 φ=atan2(y,x) 反推每个点的扫描相位，按恒速假设把每个点
    补偿到统一参考相位 az_ref，消除扫描内畸变（病因 B）。

    M_rel: lidar_t → lidar_{t+1} 的相对运动（在 lidar_t 系下），由相邻帧位姿求得
    dt_kf: 关键帧间隔（秒，~0.5），用于把 sweep 内时间折算成运动比例
    恒速近似：平移线性、旋转沿固定轴按比例缩放（标准 constant-velocity de-skew）
    """
    R_rel = M_rel[:3, :3]
    tr_rel = M_rel[:3, 3]
    k, theta = _axis_angle_from_R(R_rel)

    az = np.arctan2(pts_xyz[:, 1], pts_xyz[:, 0])      # [-π, π]
    frac = (az + np.pi) / (2.0 * np.pi)                # [0,1] 扫描相位
    # 相对参考相位的时间偏移（秒），再折算成关键帧运动比例
    f = sign * (frac - az_ref) * sweep_dur / dt_kf    # (N,) 通常 ±0.05 量级

    # 逐点旋转：Rodrigues 矢量化（每点角度 = f_i * theta）
    kxp = np.cross(np.broadcast_to(k, pts_xyz.shape), pts_xyz)   # k × p
    kkxp = np.cross(np.broadcast_to(k, pts_xyz.shape), kxp)      # k × (k × p)
    theta_i = f * theta
    s = np.sin(theta_i)[:, None]
    c = (1.0 - np.cos(theta_i))[:, None]
    p_rot = pts_xyz + s * kxp + c * kkxp
    # 逐点平移（线性）
    p_corr = p_rot + f[:, None] * tr_rel
    return p_corr


def load_lidar_deskew(data_root, scene, idx, sweep_dur=0.05, sign=1.0):
    """加载第 idx 帧点云并做逐点去畸变。

    用 idx→idx+1（末帧用 idx-1 反向）的位姿差估计 sweep 内自车运动。
    """
    info = scene[idx]
    pts = load_lidar(data_root, info)
    l2g_t = lidar2global_mat(info['data']['LIDAR_TOP']['calib'],
                             info['data']['LIDAR_TOP']['pose'])
    # 取一个邻居估计运动方向
    if idx + 1 < len(scene):
        j, s = idx + 1, 1.0
    elif idx - 1 >= 0:
        j, s = idx - 1, -1.0
    else:
        return pts  # 单帧场景，无法去畸变
    nb = scene[j]
    l2g_n = lidar2global_mat(nb['data']['LIDAR_TOP']['calib'],
                             nb['data']['LIDAR_TOP']['pose'])
    dt_kf = abs(nb['timestamp'] - info['timestamp']) / 1e6   # 秒
    if dt_kf < 1e-3:
        return pts
    # 统一构造 lidar_t 系下的"正向"相对运动 M_rel（正 dt_kf）
    M_tn = np.linalg.inv(l2g_t) @ l2g_n   # lidar_t → lidar_j
    if s > 0:
        M_rel = M_tn          # j=idx+1，本身即正向
    else:
        M_rel = np.linalg.inv(M_tn)   # j=idx-1，取逆得到正向
    return deskew_points(pts, M_rel, dt_kf, sweep_dur=sweep_dur,
                         az_ref=0.5, sign=sign)


def transform(pts, mat):
    h = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    return (mat @ h.T).T[:, :3]


def voxel_downsample(pts, voxel=0.2):
    """体素下采样：归一化采样密度，消除扫描线稀疏带来的最近邻噪声。"""
    keys = np.floor(pts / voxel).astype(np.int64)
    _, uniq = np.unique(keys, axis=0, return_index=True)
    return pts[uniq]


def compute_dynamic_score(pts_t_lidar, l2g_t, neighbors, voxel=0.2):
    """逐点动态分数 = 到各相邻帧点云的最近邻距离（取 min，抗遮挡）。

    参考帧先体素下采样，使静态表面在固定分辨率下匹配，
    避免远处扫描线稀疏导致的伪动态。
    """
    pts_t_g = transform(pts_t_lidar, l2g_t)  # 中心帧 → 全局
    dists = []
    for pts_n_lidar, l2g_n in neighbors:
        pts_n_g = transform(pts_n_lidar, l2g_n)
        pts_n_g = voxel_downsample(pts_n_g, voxel)
        tree = cKDTree(pts_n_g)
        d, _ = tree.query(pts_t_g, k=1)
        dists.append(d)
    if len(dists) == 0:
        return np.zeros(pts_t_lidar.shape[0]), pts_t_g
    score = np.min(np.stack(dists, axis=0), axis=0)
    return score, pts_t_g


def range_adaptive_thresh(pts_lidar, base, slope=0.025, rmse=0.0, k=3.0):
    """距离 + 配准噪声自适应阈值。

    静态点的残差基线 ≈ 该帧 ICP 的 inlier_rmse（高速行驶时升高）。
    若阈值固定在 base 紧贴噪声地板，高速帧的静态点会大量越界误判为动态。
    故把阈值地板抬到 max(base, k*rmse)，让动态判据始终高于配准噪声 k 倍。
    thresh(r) = max(base, k*rmse) + slope * r，r 为到自车水平距离 (m)。
    """
    r = np.hypot(pts_lidar[:, 0], pts_lidar[:, 1])
    floor = max(base, k * rmse)
    return floor + slope * r


def icp_refine_transform(src_g, tgt_g, max_iter=30, multi_init=True):
    """coarse-to-fine point-to-plane ICP，估计把 src_g 对齐到 tgt_g 的修正变换。

    strong版（方案5，治"帧间配准陷局部最优"）：
    - 收敛域加大：4.0→2.0→1.0→0.5m，粗尺度先吸收大的位姿残差
    - 多偏航初值重启：对 yaw 加 {0,±3°,±6°} 扰动各跑一遍，取 fitness 最高的
      避免单一 eye(4) 初值在大旋转残差下陷入局部最优
    动态物体两帧位移大、占少数，配准仍由静态背景主导，不会被带偏。
    返回 (T, fitness, inlier_rmse)：inlier_rmse 是已匹配点的配准误差，
    与重叠率无关，是比 fitness 更可靠的配准质量/噪声地板指标。
    """
    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(src_g)
    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(tgt_g)
    tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))

    # 围绕 src/tgt 公共质心做 yaw 扰动，避免偏航扰动引入额外平移
    c = src_g.mean(axis=0)

    def yaw_init(deg):
        a = np.deg2rad(deg)
        ca, sa = np.cos(a), np.sin(a)
        R = np.array([[ca, -sa, 0, 0],
                      [sa,  ca, 0, 0],
                      [0,    0, 1, 0],
                      [0,    0, 0, 1]], dtype=np.float64)
        # 绕质心旋转： T_c->R->T_back
        Tc = np.eye(4); Tc[:3, 3] = -c
        Tb = np.eye(4); Tb[:3, 3] = c
        return Tb @ R @ Tc

    yaw_cands = [0.0, 3.0, -3.0, 6.0, -6.0] if multi_init else [0.0]

    best_T, best_fit, best_rmse = np.eye(4), -1.0, 0.0
    for deg in yaw_cands:
        T = yaw_init(deg)
        fitness = 0.0
        rmse = 0.0
        for max_corr in (4.0, 2.0, 1.0, 0.5):
            reg = o3d.pipelines.registration.registration_icp(
                src, tgt, max_corr, T,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter))
            T = reg.transformation
            fitness = reg.fitness
            rmse = reg.inlier_rmse
        if fitness > best_fit:
            best_fit, best_T, best_rmse = fitness, T, rmse
    return best_T, best_fit, best_rmse


def segment_ground_ransac(pts_lidar, dist_thresh=0.2, z_hint=-1.0):
    """RANSAC 拟合主地面平面，返回地面 mask。

    只对低处候选点拟合（避免墙面/车顶被当平面），再按点到平面距离判定。
    比固定 z 阈值鲁棒，能适应坡道/不平路面。
    """
    cand = pts_lidar[:, 2] < z_hint + 1.0
    if cand.sum() < 100:
        return pts_lidar[:, 2] < -1.5
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts_lidar[cand])
    try:
        plane, _ = pc.segment_plane(distance_threshold=dist_thresh,
                                    ransac_n=3, num_iterations=200)
    except RuntimeError:
        return pts_lidar[:, 2] < -1.5
    a, b, c, d = plane
    n = np.linalg.norm([a, b, c]) + 1e-9
    dist = np.abs(pts_lidar @ np.array([a, b, c]) + d) / n
    # 地面 = 距主平面近 且 位于低区域（防止远处屋顶平面误判）
    return (dist < dist_thresh * 2.0) & (pts_lidar[:, 2] < z_hint + 1.5)


def cluster_vote(pts_lidar, raw_dyn, ground_mask, eps=0.7, min_samples=5, vote_ratio=0.3):
    """DBSCAN 簇级投票：同一物体要么整体动、要么整体静。

    - 孤立噪声点（DBSCAN label=-1）→ 静态，剔除散落误报
    - 簇内动态点比例 > vote_ratio → 整簇动态，补全被遮挡/稀疏部分
    """
    is_dyn = np.zeros(pts_lidar.shape[0], dtype=bool)
    idx = np.where(~ground_mask)[0]
    if idx.shape[0] < min_samples:
        return is_dyn
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts_lidar[idx])
    raw_sub = raw_dyn[idx]
    for cl in np.unique(labels):
        if cl == -1:
            continue  # 噪声簇 → 静态
        cm = labels == cl
        if raw_sub[cm].mean() > vote_ratio:
            is_dyn[idx[cm]] = True
    return is_dyn


def confidence_filter(pts_lidar, is_dyn, score,
                      min_cluster=15, strong_abs=0.45, max_range=40.0,
                      eps=0.7, min_samples=5):
    """方案1：对动态点按簇做置信度分级，低置信簇退回 ignore（不监督）。

    纯 LiDAR 两帧残差 precision 偏低（慢速帧约 24%），噪声多为孤立点/小簇/弱残差/远处稀疏点。
    动态簇需 **同时** 满足三条才判高置信动态：
      - 点数 ≥ min_cluster      （真车/人在 LiDAR 上是连续点片；噪声多为零散小簇）
      - 簇内中位残差 ≥ strong_abs（真动态 0.5s 位移残差强；噪声刚擦过阈值）
      - 簇质心水平距离 ≤ max_range（远处点稀疏、残差噪声大，不可靠）
    其余动态点判为"不确定"，下游投影成 ignore(0)，既不教动态也不教静态——
    宁可不监督，也不引入假阳/假阴。用 precision 换掉脏标签。

    返回: is_dyn_hi(bool 高置信动态), is_uncertain(bool 低置信曾动态)
    """
    is_dyn_hi = np.zeros_like(is_dyn)
    is_uncertain = np.zeros_like(is_dyn)
    idx = np.where(is_dyn)[0]
    if idx.shape[0] == 0:
        return is_dyn_hi, is_uncertain
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts_lidar[idx])
    rng = np.linalg.norm(pts_lidar[idx][:, :2], axis=1)
    sc = score[idx]
    for cl in np.unique(labels):
        cm = labels == cl
        members = idx[cm]
        if cl == -1:
            # DBSCAN 噪声（孤立动态点）→ 不确定
            is_uncertain[members] = True
            continue
        n = int(cm.sum())
        med_res = float(np.median(sc[cm]))
        med_rng = float(np.median(rng[cm]))
        if (n >= min_cluster) and (med_res >= strong_abs) and (med_rng <= max_range):
            is_dyn_hi[members] = True
        else:
            is_uncertain[members] = True
    return is_dyn_hi, is_uncertain


def generate_dynamic_labels(pts_t, l2g_t, neighbors, base_thresh,
                            use_icp=True, voxel=0.2, vote_ratio=0.3,
                            dbscan_eps=0.7, dbscan_min=5, cover_chord=0.012,
                            min_dyn_cluster=15, strong_abs=0.45, max_dyn_range=40.0):
    """高质量无标注动静伪真值生成。

    pipeline: RANSAC 地面分割 → 邻帧 ICP 精配准 → 最近邻残差(min, 抗遮挡)
             → rmse 自适应阈值 → DBSCAN 簇级投票去噪/补全。
    返回: is_dyn(bool), score(残差 m), ground_mask, overlay(邻帧对齐后全局点), pts_t_g,
          mean_fitness(重叠率), mean_rmse(配准质量，门控用这个)
    neighbors: list of (off, (pts_n_lidar, l2g_n))，off 为帧偏移（±1/±2）
    """
    pts_t_g = transform(pts_t, l2g_t)
    ground_mask = segment_ground_ransac(pts_t)

    dists = []
    overlay = None
    fitnesses = []
    rmses = []
    for off, (pts_n, l2g_n) in neighbors:
        pts_n_g = transform(pts_n, l2g_n)
        pts_n_g = voxel_downsample(pts_n_g, voxel)
        if use_icp:
            T, fitness, rmse = icp_refine_transform(pts_n_g, pts_t_g)
            pts_n_g = transform(pts_n_g, T)
            fitnesses.append(fitness)
            rmses.append(rmse)
        tree = cKDTree(pts_n_g)
        d, _ = tree.query(pts_t_g, k=1)
        # 可见性检验：从该邻帧 LiDAR 视点看，t 点方向是否被邻帧点覆盖。
        # 高速行驶时大片静态区域邻帧根本没扫到（视野更替），这些点查最近邻
        # 只能指向点云边界 → 米级残差假阳。无覆盖方向一律视为"未知"，残差置 inf。
        origin_n = l2g_n[:3, 3]
        dir_t = pts_t_g - origin_n
        dir_n = pts_n_g - origin_n
        dir_t /= (np.linalg.norm(dir_t, axis=1, keepdims=True) + 1e-9)
        dir_n /= (np.linalg.norm(dir_n, axis=1, keepdims=True) + 1e-9)
        ang_d, _ = cKDTree(dir_n).query(dir_t, k=1)   # 单位向量弦长 ≈ 角距离
        covered = ang_d < cover_chord
        d = np.where(covered, d, np.inf)
        dists.append(d)
        if off == 1 or overlay is None:
            overlay = pts_n_g
    if dists:
        score = np.min(np.stack(dists, axis=0), axis=0)
        # 所有邻帧都未覆盖（min=inf）→ 无信息 → 判静态（残差置 0）
        score = np.where(np.isfinite(score), score, 0.0)
    else:
        score = np.zeros(pts_t.shape[0])
    mean_fitness = float(np.mean(fitnesses)) if fitnesses else 1.0
    # 取最佳（最小）rmse：select-min 残差用的是最近邻，阈值应对应最准那帧的噪声地板
    mean_rmse = float(np.min(rmses)) if rmses else 0.0

    thr = range_adaptive_thresh(pts_t, base_thresh, rmse=mean_rmse)
    raw_dyn = (score > thr) & (~ground_mask)
    is_dyn = cluster_vote(pts_t, raw_dyn, ground_mask, dbscan_eps, dbscan_min, vote_ratio)
    # 方案1：置信度分级，低置信动态点退回 ignore（提 precision）
    is_dyn, is_uncertain = confidence_filter(
        pts_t, is_dyn, score,
        min_cluster=min_dyn_cluster, strong_abs=strong_abs, max_range=max_dyn_range,
        eps=dbscan_eps, min_samples=dbscan_min)
    return is_dyn, score, ground_mask, overlay, pts_t_g, mean_fitness, mean_rmse, is_uncertain


def draw_overlay_bev(ax, pts_t_g, pts_next_g, center, rng=50):
    ax.set_facecolor('white')
    cx, cy = center[0], center[1]

    def crop(p):
        m = (np.abs(p[:, 0] - cx) < rng) & (np.abs(p[:, 1] - cy) < rng)
        return p[m]
    a = crop(pts_t_g)
    b = crop(pts_next_g)
    ax.scatter(a[:, 0], a[:, 1], s=0.4, c='#2060f0', alpha=0.5, label='frame t')
    ax.scatter(b[:, 0], b[:, 1], s=0.4, c='#f08020', alpha=0.5, label='frame t+1')
    ax.scatter([cx], [cy], s=60, c='lime', marker='*', edgecolors='k', zorder=5)
    ax.set_xlim(cx - rng, cx + rng)
    ax.set_ylim(cy - rng, cy + rng)
    ax.set_aspect('equal')
    ax.legend(markerscale=8, fontsize=8, loc='upper right')
    ax.set_title('global-frame overlay  (blue=t, orange=t+1)\nstatic aligns, dynamic = ghosting')
    ax.grid(True, alpha=0.2)


def draw_score_bev(ax, pts_t_lidar, score, is_dyn, ground_mask, rng=50):
    ax.set_facecolor('white')
    m = (np.abs(pts_t_lidar[:, 0]) < rng) & (np.abs(pts_t_lidar[:, 1]) < rng)
    p = pts_t_lidar[m]
    s = score[m]
    dyn = is_dyn[m]
    grd = ground_mask[m]
    # 地面：淡青灰；非地面静态：深灰；动态：按残差红
    ax.scatter(p[grd & ~dyn, 1], p[grd & ~dyn, 0], s=0.4, c='#d8e4d8', alpha=0.4)
    sta = (~grd) & (~dyn)
    ax.scatter(p[sta, 1], p[sta, 0], s=0.6, c='#909090', alpha=0.6)
    sc = ax.scatter(p[dyn, 1], p[dyn, 0], s=3.0,
                    c=np.clip(s[dyn], 0.3, 3.0), cmap='autumn_r', alpha=0.95)
    ax.scatter([0], [0], s=80, c='lime', marker='*', edgecolors='k', zorder=5)
    ax.arrow(0, 0, 0, 4, head_width=1.2, head_length=1.5, fc='g', ec='g', zorder=6)
    ax.set_xlim(rng, -rng)   # y 左为正 → 屏幕左
    ax.set_ylim(-rng, rng)
    ax.set_aspect('equal')
    ax.set_xlabel('y (left +)')
    ax.set_ylabel('x (forward +)')
    n_dyn = int(dyn.sum())
    ax.set_title(f'refined dynamic label (ICP+RANSAC+DBSCAN)\ndynamic pts (red)={n_dyn} / {len(p)}')
    if n_dyn > 0:
        plt.colorbar(sc, ax=ax, fraction=0.04, label='NN residual (m)')
    ax.grid(True, alpha=0.2)


def draw_cam(ax, img, pts_lidar, score, is_dyn, lidar2img):
    h, w = img.shape[:2]
    ax.imshow(img)
    pts_h = np.concatenate([pts_lidar, np.ones((pts_lidar.shape[0], 1))], axis=1)
    cam = (lidar2img @ pts_h.T).T
    depth = cam[:, 2]
    front = depth > 0.5
    uv = cam[:, :2] / np.clip(cam[:, 2:3], 1e-3, None)
    inimg = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    vis = front & inimg
    # 静态点：淡灰小点
    m_sta = vis & (~is_dyn)
    ax.scatter(uv[m_sta, 0], uv[m_sta, 1], s=1.0, c='#c0c0c0', alpha=0.25)
    # 动态点：红，按残差大小
    m_dyn = vis & is_dyn
    ax.scatter(uv[m_dyn, 0], uv[m_dyn, 1], s=6.0,
               c=np.clip(score[m_dyn], 0.3, 3.0), cmap='autumn_r', alpha=0.9)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis('off')


def visualize_sample(scene, idx, data_root, dist_thresh, out_path,
                     use_icp=True, vote_ratio=0.3, n_neighbor=2, fitness_thresh=0.6,
                     deskew=False, sweep_dur=0.05, deskew_sign=1.0, rmse_thresh=0.35):
    info = scene[idx]
    if deskew:
        pts_t = load_lidar_deskew(data_root, scene, idx, sweep_dur, deskew_sign)
    else:
        pts_t = load_lidar(data_root, info)
    l2g_t = lidar2global_mat(info['data']['LIDAR_TOP']['calib'],
                             info['data']['LIDAR_TOP']['pose'])

    # 采集 ±n_neighbor 帧邻居（多帧提高静态覆盖、抗遮挡）
    neighbors = []
    offsets = []
    for off in range(-n_neighbor, n_neighbor + 1):
        if off == 0:
            continue
        j = idx + off
        if 0 <= j < len(scene):
            ni = scene[j]
            if deskew:
                pts_n = load_lidar_deskew(data_root, scene, j, sweep_dur, deskew_sign)
            else:
                pts_n = load_lidar(data_root, ni)
            l2g_n = lidar2global_mat(ni['data']['LIDAR_TOP']['calib'],
                                     ni['data']['LIDAR_TOP']['pose'])
            neighbors.append((off, (pts_n, l2g_n)))
            offsets.append(off)

    is_dyn, score, ground_mask, overlay, pts_t_g, fitness, rmse, is_uncertain = generate_dynamic_labels(
        pts_t, l2g_t, neighbors, dist_thresh,
        use_icp=use_icp, vote_ratio=vote_ratio)

    center = l2g_t[:3, 3]
    n_dyn = int(is_dyn.sum())
    # 门控改用 inlier_rmse（配准质量，与重叠率/车速无关），而非 fitness
    low_quality = rmse > rmse_thresh

    fig = plt.figure(figsize=(22, 11))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1.1, 1.1])

    # 6 相机（占左两列）
    for r in range(2):
        for c in range(2):
            cam_type = CAM_GRID[r][c]
            ax = fig.add_subplot(gs[r, c])
            cam = info['data'][cam_type]
            i2g = img2global_mat(cam['calib'], cam['pose'])
            lidar2img = np.linalg.inv(i2g) @ l2g_t
            try:
                img = plt.imread(os.path.join(data_root, cam['filename']))
            except FileNotFoundError:
                ax.text(0.5, 0.5, f'missing {cam_type}', ha='center'); ax.axis('off'); continue
            draw_cam(ax, img, pts_t, score, is_dyn, lidar2img)
            ax.set_title(cam_type, fontsize=9)

    ax_ov = fig.add_subplot(gs[:, 2])
    if overlay is not None:
        draw_overlay_bev(ax_ov, pts_t_g, overlay, center)
    ax_sc = fig.add_subplot(gs[:, 3])
    draw_score_bev(ax_sc, pts_t, score, is_dyn, ground_mask)

    icp_tag = 'ICP-on' if use_icp else 'ICP-off'
    ds_tag = 'deskew' if deskew else 'raw'
    qtag = 'LOW-QUALITY(discard)' if low_quality else 'OK'
    qcolor = 'red' if low_quality else 'green'
    fig.suptitle(
        f'UNSUPERVISED dynamic/static (LiDAR, NO box labels)  [{icp_tag}, {ds_tag}, ±{n_neighbor}f, vote={vote_ratio}]  '
        f'token={info["token"][:12]}...  dynamic_pts={n_dyn}  rmse={rmse:.3f}m fit={fitness:.2f} [{qtag}]',
        fontsize=13, color=qcolor)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {out_path}  (dynamic_pts={n_dyn}, rmse={rmse:.3f}, fitness={fitness:.3f}, {"DISCARD" if low_quality else "keep"})')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl', default='data/nuscenes_cam/nuscenes_infos_val_gaussian_ad_v4.pkl')
    parser.add_argument('--data-root', default='data/nuscenes')
    parser.add_argument('--out-dir', default='out/dynamic_static_unsup_vis')
    parser.add_argument('--num-samples', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dist-thresh', type=float, default=0.3,
                        help='最近邻残差基准阈值 (m)，距离自适应增长')
    parser.add_argument('--no-icp', action='store_true', help='关闭 ICP 精配准（对比用）')
    parser.add_argument('--vote-ratio', type=float, default=0.3,
                        help='DBSCAN 簇内动态点比例超此则整簇判动')
    parser.add_argument('--n-neighbor', type=int, default=2,
                        help='前后各取几帧作参考（多帧提高静态覆盖）')
    parser.add_argument('--fitness-thresh', type=float, default=0.72,
                        help='ICP fitness 低于此值的帧判为低质量（生成训练 GT 时应丢弃）。'
                             '经验：转弯/畸变帧 fitness<0.72，清晰帧普遍 >0.8')
    parser.add_argument('--rmse-thresh', type=float, default=0.35,
                        help='ICP inlier_rmse 高于此值的帧判为低质量（真·配准失败）。'
                             'fitness 会被 ego 速度污染（高速→重叠率低），rmse 与重叠率'
                             '无关，是更可靠的质量门控。正常帧 rmse<0.25m')
    parser.add_argument('--deskew', action='store_true',
                        help='开启逐点去畸变（方位角反推扫描相位，恒速运动补偿）')
    parser.add_argument('--sweep-dur', type=float, default=0.05,
                        help='LiDAR 转一圈时长（秒），nuScenes 20Hz ≈ 0.05')
    parser.add_argument('--deskew-sign', type=float, default=1.0,
                        help='去畸变符号（+1/-1），方向不确定时可翻转验证')
    parser.add_argument('--scene', default=None,
                        help='直接指定 scene_token（前缀即可），配合 --idx 定点验证')
    parser.add_argument('--idx', type=int, default=None,
                        help='直接指定帧索引，配合 --scene 定点验证')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f'loading {args.pkl} ...')
    data = mmengine.load(args.pkl)
    scene_infos = data['infos']
    keyframes = data['metadata']

    # 定点验证模式：--scene + --idx
    if args.scene is not None and args.idx is not None:
        matched = [t for t in scene_infos if t.startswith(args.scene)]
        if not matched:
            raise SystemExit(f'no scene matches prefix {args.scene}')
        scene_token = matched[0]
        scene = scene_infos[scene_token]
        frame_idx = int(np.clip(args.idx, 1, len(scene) - 2))
        tag = 'deskew' if args.deskew else 'raw'
        out_path = os.path.join(args.out_dir, f'target_{scene_token[:8]}_{frame_idx}_{tag}.jpg')
        print(f'[target] scene={scene_token[:8]} idx={frame_idx} ({tag})')
        visualize_sample(scene, frame_idx, args.data_root, args.dist_thresh, out_path,
                         use_icp=not args.no_icp, vote_ratio=args.vote_ratio,
                         n_neighbor=args.n_neighbor, fitness_thresh=args.fitness_thresh,
                         deskew=args.deskew, sweep_dur=args.sweep_dur,
                         deskew_sign=args.deskew_sign, rmse_thresh=args.rmse_thresh)
        print(f'\ndone. results in {args.out_dir}/')
        return

    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(keyframes), min(args.num_samples, len(keyframes)), replace=False)

    for k, mi in enumerate(pick):
        scene_token, frame_idx = keyframes[mi]
        scene = scene_infos[scene_token]
        # 避开首尾帧（需要前后邻居）
        frame_idx = int(np.clip(frame_idx, 1, len(scene) - 2))
        out_path = os.path.join(args.out_dir, f'sample_{k:03d}.jpg')
        print(f'[{k+1}/{len(pick)}] scene={scene_token[:8]} idx={frame_idx}')
        visualize_sample(scene, frame_idx, args.data_root, args.dist_thresh, out_path,
                         use_icp=not args.no_icp, vote_ratio=args.vote_ratio,
                         n_neighbor=args.n_neighbor, fitness_thresh=args.fitness_thresh,
                         deskew=args.deskew, sweep_dur=args.sweep_dur,
                         deskew_sign=args.deskew_sign, rmse_thresh=args.rmse_thresh)

    print(f'\ndone. results in {args.out_dir}/')


if __name__ == '__main__':
    main()
