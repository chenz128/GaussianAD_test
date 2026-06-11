"""
验证脚本（只读，不改项目，不动训练）：
光流-刚性流残差能否点亮 LiDAR 最近邻残差漏掉的"径向运动"目标。

数据来源（全部现成，零人工标注）：
  - 图像 / 位姿 / 内参 / 标注（仅用于选帧+评估，不进入 GT）：nuscenes-devkit
  - 稠密单目度量深度 Metric3D：/data/chenz/Gaussianflowocc_test/data/metric_3d_nusc
  - 语义（取可移动类 mask 做门控）GroundedSAM：.../grounded_sam_nusc

方法：
  对相机 cam，在关键帧 t 与 t+1：
    f_real   = RAFT(img_t, img_{t+1})                      实际稠密光流
    f_rigid  = 用 Metric3D 深度 + ego/cam 位姿假设"世界静止"算出的刚性光流
    residual = ||f_real - f_rigid||                        运动证据（径向也能看见）
    dynamic  = (residual > tau) & movable_mask             门控降噪
  可视化：每相机一行 [原图 | f_real | f_rigid | residual | dynamic叠加(+GT框)]
"""
import os, sys, argparse
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from scipy.spatial import cKDTree
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points, BoxVisibility

NUSC_ROOT = '/data/chenz/GaussianAD/data/nuscenes'   # 有 sweeps + 完整 trainval
M3D_ROOT  = '/data/chenz/Gaussianflowocc_test/data/metric_3d_nusc'
SAM_ROOT  = '/data/chenz/Gaussianflowocc_test/data/grounded_sam_nusc'
OUT_DIR   = '/tmp/flowdyn'

# Metric3D / GroundedSAM 的相机顺序（与 generate_m3d_nusc.py 一致）
CAMS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
# 关键：GroundedSAM 用的是另一套相机顺序（见 generate_grounded_sam.py:304）
# depth 用 CAMS 顺序，seg 必须用 SAM_CAMS 顺序，否则 seg 与图像/深度错位！
SAM_CAMS = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_BACK_LEFT',
            'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_FRONT_RIGHT']
# nuScenes occ 16+1 类中"可移动且会动"的类别 id（GroundedSAM 用同一套）
# 0 noise,1 barrier,2 bicycle,3 bus,4 car,5 cons_veh,6 motorcycle,7 ped,
# 8 cone,9 trailer,10 truck,11 drive,12 flat,13 sidewalk,14 terrain,15 manmade,16 veg
MOVABLE = {2, 3, 4, 5, 6, 7, 9, 10}

RENDER_H, RENDER_W = 448, 800      # RAFT 输入分辨率（8 的倍数）
DEPTH_MIN, DEPTH_MAX = 2.8, 50.0   # Metric3D 训练上限 50m，远处外推不可靠
RES_THRESH = 3.0                   # 光流残差阈值（像素，RENDER 分辨率下）
LIDAR_THRESH = 0.30                # LiDAR 最近邻残差阈值（米，~0.05s sweep 间隔）
DEPTH_REL_ERR = 0.10               # Metric3D 相对深度误差（10%）
DEPTH_ABS_ERR = 0.6                # 深度绝对误差下限（米），近处大目标关键
SENS_K = 1.5                       # 深度敏感度门控系数：res_robust = max(0, res - K*sens)
RIGID_ALPHA = 0.12                 # 相对刚性流量级门控：阈值抬高 ALPHA*|f_rigid|


def pose_to_mat(translation, rotation):
    m = np.eye(4)
    m[:3, :3] = Quaternion(rotation).rotation_matrix
    m[:3, 3] = np.asarray(translation)
    return m


def cam_to_global(nusc, sd_token):
    """返回该 sample_data（某相机某时刻）的 cam->global 4x4 与内参 3x3。"""
    sd = nusc.get('sample_data', sd_token)
    cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
    ep = nusc.get('ego_pose', sd['ego_pose_token'])
    sensor2ego = pose_to_mat(cs['translation'], cs['rotation'])
    ego2global = pose_to_mat(ep['translation'], ep['rotation'])
    cam2global = ego2global @ sensor2ego
    K = np.array(cs['camera_intrinsic'])
    return cam2global, K, sd['filename']


def load_lidar(nusc, sd_token):
    """加载某 LIDAR_TOP sample_data 点云 (N,3) 与 lidar->global 4x4。"""
    sd = nusc.get('sample_data', sd_token)
    pts = np.fromfile(os.path.join(NUSC_ROOT, sd['filename']),
                      dtype=np.float32).reshape(-1, 5)[:, :3]
    cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
    ep = nusc.get('ego_pose', sd['ego_pose_token'])
    l2e = pose_to_mat(cs['translation'], cs['rotation'])
    e2g = pose_to_mat(ep['translation'], ep['rotation'])
    return pts, e2g @ l2e


def flow_to_image(flow):
    """flow (2,H,W) -> RGB uint8，HSV 颜色轮。"""
    fx, fy = flow[0], flow[1]
    mag = np.sqrt(fx**2 + fy**2)
    ang = np.arctan2(fy, fx)
    hsv = np.zeros((flow.shape[1], flow.shape[2], 3), np.uint8)
    hsv[..., 0] = ((ang + np.pi) / (2*np.pi) * 179).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / (np.percentile(mag, 99) + 1e-6) * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def heat(x, vmax):
    x = np.clip(x / (vmax + 1e-6), 0, 1)
    return cv2.applyColorMap((x*255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]


@torch.no_grad()
def run_raft(model, transforms, img1, img2, device):
    """img: HxWx3 uint8 RGB（已 resize）。返回 (2,H,W) 像素位移。"""
    # 关键：保持 uint8 传入 transforms，由其负责 dtype 转换与归一化到 [-1,1]。
    # 若预先 .float()（[0,255]）会绕过归一化，导致 RAFT 输出垃圾。
    t1 = torch.from_numpy(img1).permute(2, 0, 1)[None]  # uint8
    t2 = torch.from_numpy(img2).permute(2, 0, 1)[None]
    t1, t2 = transforms(t1, t2)
    flow = model(t1.to(device), t2.to(device))[-1][0].cpu().numpy()
    return flow


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=6, help='可视化帧数')
    ap.add_argument('--min-speed', type=float, default=4.0)
    ap.add_argument('--scan-scenes', type=int, default=80)
    ap.add_argument('--ego-static', action='store_true',
                    help='选自车近静止帧，验证刚性流正确性（背景残差应≈0）')
    ap.add_argument('--ego-max-speed', type=float, default=0.3,
                    help='自车近静止判定阈值 m/s')
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    device = 'cuda'

    print('Loading NuScenes...', flush=True)
    nusc = NuScenes(version='v1.0-trainval', dataroot=NUSC_ROOT, verbose=False)

    print('Loading RAFT...', flush=True)
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    weights = Raft_Large_Weights.C_T_SKHT_V2
    model = raft_large(weights=weights, progress=False).eval().to(device)
    transforms = weights.transforms()

    # ── 选帧：动态模式找高速车径向可见；静止模式找自车近静止（验证刚性流）──
    if args.ego_static:
        print('Selecting EGO-STATIC frames (validate rigid flow)...', flush=True)
    else:
        print('Selecting radial-motion frames...', flush=True)
    cand = []
    for scene in nusc.scene[:args.scan_scenes]:
        tok = scene['first_sample_token']
        prev_xyz, prev_t = None, None
        while tok:
            s = nusc.get('sample', tok)
            if s['next'] == '':
                break
            # 文件齐全检查（关键帧 tok 的 M3D/SAM）
            sam_f = os.path.join(SAM_ROOT, scene['name'], tok + '.npy')
            m3d_f = os.path.join(M3D_ROOT, scene['name'], tok + '.npy')
            if not (os.path.exists(sam_f) and os.path.exists(m3d_f)):
                tok = s['next']; continue

            if args.ego_static:
                # 用 LIDAR_TOP ego_pose 相邻关键帧位移估自车速度
                sd = nusc.get('sample_data', s['data']['LIDAR_TOP'])
                ep = nusc.get('ego_pose', sd['ego_pose_token'])
                xyz = np.array(ep['translation']); t = sd['timestamp'] / 1e6
                if prev_xyz is not None:
                    spd = float(np.linalg.norm(xyz - prev_xyz) / (t - prev_t + 1e-6))
                    if spd < args.ego_max_speed:
                        cand.append((spd, scene['name'], tok, 'CAM_FRONT'))
                prev_xyz, prev_t = xyz, t
            else:
                best = 0.0; bestcam = None
                for ann_t in s['anns']:
                    ann = nusc.get('sample_annotation', ann_t)
                    if not ann['category_name'].startswith('vehicle'):
                        continue
                    v = nusc.box_velocity(ann_t)
                    if np.isnan(v).any():
                        continue
                    spd = float(np.linalg.norm(v))
                    if spd < args.min_speed:
                        continue
                    # 投影中心到前/后相机判断是否径向可见
                    for cam in ('CAM_FRONT', 'CAM_BACK'):
                        c2g, K, _ = cam_to_global(nusc, s['data'][cam])
                        g2c = np.linalg.inv(c2g)
                        pc = g2c @ np.array([*ann['translation'], 1.0])
                        if pc[2] <= 1.0:
                            continue
                        u = K[0, 0]*pc[0]/pc[2] + K[0, 2]
                        vv = K[1, 1]*pc[1]/pc[2] + K[1, 2]
                        if 0 < u < 1600 and 0 < vv < 900 and spd > best:
                            best = spd; bestcam = cam
                if bestcam is not None:
                    cand.append((best, scene['name'], tok, bestcam))
            tok = s['next']
    # 静止模式速度升序（最静止优先）；动态模式速度降序（最快优先）
    cand.sort(reverse=not args.ego_static)
    # 去重 scene，挑多样
    picked, seen = [], set()
    for c in cand:
        if c[1] in seen:
            continue
        seen.add(c[1]); picked.append(c)
        if len(picked) >= args.n:
            break
    print(f'Picked {len(picked)} frames:', [(round(p[0],1), p[1], p[3]) for p in picked], flush=True)

    # GT box 级评估累加器（spd>1=真动为正）。每个 box 一条记录。
    # 列: is_moving, flow_hit, lidar_hit  (hit=该模态残差是否过阈值)
    EVAL = []

    for fi, (spd, scene_name, tok, focuscam) in enumerate(picked):
        s = nusc.get('sample', tok)
        # 两个相机看效果：聚焦相机 + 另一个方向
        show_cams = [focuscam] + (['CAM_BACK'] if focuscam == 'CAM_FRONT' else ['CAM_FRONT'])
        rows = []

        # ── LiDAR 最近邻残差（去自车运动后，静止结构≈0，动点大）──
        ld_sd = s['data']['LIDAR_TOP']
        pts_l, lt2g = load_lidar(nusc, ld_sd)
        ld_nx = nusc.get('sample_data', ld_sd)['next']
        if ld_nx:
            pts_l2, l2g2 = load_lidar(nusc, ld_nx)
            g1 = (lt2g @ np.concatenate([pts_l, np.ones((len(pts_l), 1))], 1).T).T[:, :3]
            g2 = (l2g2 @ np.concatenate([pts_l2, np.ones((len(pts_l2), 1))], 1).T).T[:, :3]
            dnn, _ = cKDTree(g2).query(g1, k=1)
        else:
            dnn = np.zeros(len(pts_l))

        for cam in show_cams:
            ci = CAMS.index(cam)
            sd_t = nusc.get('sample_data', s['data'][cam])
            # 关键：用相邻 sweep（~0.083s）而非下一关键帧（0.5s），避免 RAFT 大位移失效
            nb_tok = sd_t['next'] if sd_t['next'] else s['data'][cam]
            c2g_t, K, fn_t = cam_to_global(nusc, s['data'][cam])
            c2g_n, _, fn_n = cam_to_global(nusc, nb_tok)
            T = np.linalg.inv(c2g_n) @ c2g_t   # cam_t -> cam_{t+sweep}

            # 调试：T 的旋转角(度)与平移(米)，自车静止时应都≈0
            _R = T[:3, :3]; _tr = T[:3, 3]
            _ang = np.degrees(np.arccos(np.clip((np.trace(_R) - 1) / 2, -1, 1)))
            print(f'[{scene_name} {cam}] T rot={_ang:.2f}deg trans={np.linalg.norm(_tr):.3f}m '
                  f'dt={(nusc.get("sample_data", nb_tok)["timestamp"] - sd_t["timestamp"])/1e6:.3f}s', flush=True)

            img_t = cv2.cvtColor(cv2.imread(os.path.join(NUSC_ROOT, fn_t)), cv2.COLOR_BGR2RGB)
            img_n = cv2.cvtColor(cv2.imread(os.path.join(NUSC_ROOT, fn_n)), cv2.COLOR_BGR2RGB)
            depth = np.load(os.path.join(M3D_ROOT, scene_name, tok + '.npy'))[ci].astype(np.float32)
            seg = np.load(os.path.join(SAM_ROOT, scene_name, tok + '.npy'))[SAM_CAMS.index(cam)].astype(np.int64)

            # resize 到 RENDER 分辨率
            sx, sy = RENDER_W / 1600.0, RENDER_H / 900.0
            img_t_r = cv2.resize(img_t, (RENDER_W, RENDER_H))
            img_n_r = cv2.resize(img_n, (RENDER_W, RENDER_H))
            depth_r = cv2.resize(depth, (RENDER_W, RENDER_H), interpolation=cv2.INTER_NEAREST)
            seg_r = cv2.resize(seg, (RENDER_W, RENDER_H), interpolation=cv2.INTER_NEAREST)
            fx, fy = K[0, 0]*sx, K[1, 1]*sy
            cx, cy = K[0, 2]*sx, K[1, 2]*sy

            # LiDAR 点投影到当前相机（RENDER 分辨率），每点携带最近邻残差 dnn
            l2c = np.linalg.inv(c2g_t) @ lt2g
            pc_l = (l2c @ np.concatenate([pts_l, np.ones((len(pts_l), 1))], 1).T).T
            zc_l = pc_l[:, 2]
            ul = fx * pc_l[:, 0] / (zc_l + 1e-6) + cx
            vl = fy * pc_l[:, 1] / (zc_l + 1e-6) + cy
            inim = (zc_l > 0.5) & (ul >= 0) & (ul < RENDER_W) & (vl >= 0) & (vl < RENDER_H)
            pu = ul[inim].astype(int); pv = vl[inim].astype(int); pd = dnn[inim]

            # 实际光流
            f_real = run_raft(model, transforms, img_t_r, img_n_r, device)
            # RAFT 自检：同图喂两次，真实流必≈0
            f_self = run_raft(model, transforms, img_t_r, img_t_r, device)
            print(f'    SELFTEST raft(img,img)|mag|p50={np.median(np.hypot(f_self[0],f_self[1])):.3f} '
                  f'shape={f_real.shape}', flush=True)

            # 刚性光流
            uu, vv = np.meshgrid(np.arange(RENDER_W), np.arange(RENDER_H))
            Z = depth_r
            X = (uu - cx) / fx * Z
            Y = (vv - cy) / fy * Z
            P = np.stack([X, Y, Z, np.ones_like(Z)], 0).reshape(4, -1)
            Pc = T @ P
            Zc = Pc[2].reshape(RENDER_H, RENDER_W)
            up = (fx * Pc[0] / (Pc[2] + 1e-6) + cx).reshape(RENDER_H, RENDER_W)
            vp = (fy * Pc[1] / (Pc[2] + 1e-6) + cy).reshape(RENDER_H, RENDER_W)
            f_rigid = np.stack([up - uu, vp - vv], 0)

            # 深度扰动敏感度：把深度抬（相对 OR 绝对误差取大）重算刚性流，
            # 量出“仅因深度不确定就会摆动”的幅度。近处大目标（如贴近卡车）
            # 刚性流对深度极敏感，是高速自车静车假阳的根源。
            dZ = np.maximum(Z * DEPTH_REL_ERR, DEPTH_ABS_ERR)
            Zp = Z + dZ
            Pp = np.stack([(uu-cx)/fx*Zp, (vv-cy)/fy*Zp, Zp, np.ones_like(Zp)], 0).reshape(4, -1)
            Pcp = T @ Pp
            upp = (fx * Pcp[0] / (Pcp[2] + 1e-6) + cx).reshape(RENDER_H, RENDER_W)
            vpp = (fy * Pcp[1] / (Pcp[2] + 1e-6) + cy).reshape(RENDER_H, RENDER_W)
            sens = np.sqrt((upp - up)**2 + (vpp - vp)**2)
            rigid_mag = np.hypot(f_rigid[0], f_rigid[1])

            # 残差
            res_raw = np.sqrt((f_real[0]-f_rigid[0])**2 + (f_real[1]-f_rigid[1])**2)
            # 双重门控：扣掉深度敏感度 + 相对刚性流量级（压近处大目标假阳）
            res = np.maximum(0.0, res_raw - SENS_K * sens - RIGID_ALPHA * rigid_mag)
            valid = (Z > DEPTH_MIN) & (Z < DEPTH_MAX) & (Zc > 0.5)
            movable = np.isin(seg_r, list(MOVABLE))
            res_m = np.where(valid, res, 0)

            # 自检：非可移动区残差中位数应该小（刚性流对齐良好）
            static_res = res[valid & ~movable]
            mv_res = res[valid & movable]
            _bg = valid & ~movable
            print(f'    DBG real|mag|p50={np.median(np.hypot(f_real[0],f_real[1])[_bg]) if _bg.any() else 0:.2f} '
                  f'rigid|mag|p50={np.median(np.hypot(f_rigid[0],f_rigid[1])[_bg]) if _bg.any() else 0:.2f} '
                  f'valid_frac={valid.mean():.2f} Z_p50={np.median(Z[valid]) if valid.any() else 0:.1f}', flush=True)
            print(f'[{scene_name} {cam}] static_res_med={np.median(static_res) if static_res.size else 0:.2f}px '
                  f'movable_res_med={np.median(mv_res) if mv_res.size else 0:.2f}px '
                  f'movable_px={int(movable.sum())}', flush=True)
            # 对齐自检：movable 区的 seg 类别 + 质心(x归一化,0左1右)
            if movable.any():
                ys, xs = np.where(movable)
                cls_u, cls_c = np.unique(seg_r[movable], return_counts=True)
                print(f'    ALIGN movable_cls={dict(zip(cls_u.tolist(), cls_c.tolist()))} '
                      f'centroid_x={xs.mean()/RENDER_W:.2f} centroid_y={ys.mean()/RENDER_H:.2f}', flush=True)

            dynamic = valid & movable & (res > RES_THRESH)

            # ── gt box 速度诊断（仅评估，不进入 GT）：每辆车真实速度 vs 框内残差 ──
            _, boxes_cam, camK = nusc.get_sample_data(s['data'][cam], box_vis_level=BoxVisibility.ANY)
            gt_boxes_vis = []
            for box in boxes_cam:
                if not box.name.startswith('vehicle'):
                    continue
                vel = nusc.box_velocity(box.token)
                spd_b = float(np.linalg.norm(vel)) if not np.isnan(vel).any() else -1.0
                cor = view_points(box.corners(), camK, normalize=True)[:2]
                xs = cor[0] * sx; ys = cor[1] * sy
                x0, x1 = int(max(0, xs.min())), int(min(RENDER_W, xs.max()))
                y0, y1 = int(max(0, ys.min())), int(min(RENDER_H, ys.max()))
                if x1 <= x0 or y1 <= y0:
                    continue
                bm = np.zeros((RENDER_H, RENDER_W), bool); bm[y0:y1, x0:x1] = True
                sel = bm & movable & valid
                if sel.sum() < 30:
                    continue
                rmed = float(np.median(res[sel]))
                rraw = float(np.median(res_raw[sel]))
                gt_boxes_vis.append((spd_b, rmed, (x0, y0, x1, y1)))
                tag = 'MOVING' if spd_b > 1.0 else 'static'
                # box 内 lidar 投影点的最近邻残差中位数（互补诊断）
                lmed_b = -1.0
                if pu.size:
                    lsel = (pu >= x0) & (pu < x1) & (pv >= y0) & (pv < y1)
                    if int(lsel.sum()) >= 3:
                        lmed_b = float(np.median(pd[lsel]))
                hit = ('F' if rmed > RES_THRESH else '') + ('L' if lmed_b > LIDAR_THRESH else '')
                print(f'    GTBOX {tag} v={spd_b:.1f}m/s flow_raw={rraw:.1f}px flow={rmed:.1f}px '
                      f'lidar={lmed_b:.2f}m hit=[{hit}] px={int(sel.sum())}', flush=True)
                if spd_b >= 0:   # 速度有效才计入评估
                    EVAL.append((spd_b > 1.0, rmed > RES_THRESH, lmed_b > LIDAR_THRESH))

            # ── 拼图 ──
            vmax = max(np.percentile(res[valid], 95) if valid.any() else 1.0, 5.0)
            panel_real = flow_to_image(f_real)
            panel_rigid = flow_to_image(f_rigid)
            panel_res = heat(res_m, vmax)
            overlay = img_t_r.copy()
            # 按连通域做整车判定：用整车残差中位数决定红(动)/绿(静)，
            # 避免逐像素噪声把整辆车标乱（少数噪声像素不会翻转整车结论）
            mv_u8 = (movable & valid).astype(np.uint8)
            ncomp, comp = cv2.connectedComponents(mv_u8)
            inst_decisions = []  # (cx, cy, rmed, is_dyn, npx)
            for ic in range(1, ncomp):
                cmask = comp == ic
                npx = int(cmask.sum())
                if npx < 60:                      # 太小的碎片忽略（多为噪声）
                    continue
                rmed = float(np.median(res[cmask]))
                flow_dyn = rmed > RES_THRESH
                # LiDAR 残差：落在该连通域的 lidar 投影点的最近邻残差中位数
                lmed = -1.0; lidar_dyn = False
                if pu.size:
                    inc = cmask[pv, pu]
                    if int(inc.sum()) >= 3:
                        lmed = float(np.median(pd[inc]))
                        lidar_dyn = lmed > LIDAR_THRESH
                is_dyn = flow_dyn or lidar_dyn    # 光流 ∪ LiDAR 并集
                col = np.array([255, 0, 0]) if is_dyn else np.array([0, 200, 0])
                overlay[cmask] = (overlay[cmask]*0.45 + col*0.55).astype(np.uint8)
                ys2, xs2 = np.where(cmask)
                cxi, cyi = int(xs2.mean()), int(ys2.mean())
                inst_decisions.append((cxi, cyi, rmed, is_dyn, npx))
                # 标签 f=光流残差 l=lidar残差；命中源用 [F]/[L] 标出
                src = ('F' if flow_dyn else '') + ('L' if lidar_dyn else '')
                lab = f'f{rmed:.0f}' + (f' l{lmed:.2f}' if lmed >= 0 else '')
                if src:
                    lab += f' [{src}]'
                cv2.putText(overlay, lab, (cxi-22, cyi),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                            (255, 80, 80) if is_dyn else (80, 255, 80), 1)
            # gt box 矩形 + 速度/残差标注（黄=真动 v>1, 灰=真静）
            for spd_b, rmed, (x0, y0, x1, y1) in gt_boxes_vis:
                col = (255, 255, 0) if spd_b > 1.0 else (160, 160, 160)
                cv2.rectangle(overlay, (x0, y0), (x1, y1), col, 2)
                cv2.putText(overlay, f'v{spd_b:.0f} r{rmed:.0f}', (x0, max(12, y0-4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

            def label(im, txt):
                im = im.copy()
                cv2.putText(im, txt, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                return im
            row = np.concatenate([
                label(img_t_r, f'{cam} img_t'),
                label(panel_real, 'flow_real'),
                label(panel_rigid, 'flow_rigid'),
                label(panel_res, f'residual(<{vmax:.0f}px)'),
                label(overlay, 'flow|LiDAR union: red=dyn green=static (GTbox: yellow=move gray=still)'),
            ], axis=1)
            rows.append(row)
        fig = np.concatenate(rows, axis=0)
        outp = os.path.join(OUT_DIR, f'frame_{fi:02d}_{scene_name}.jpg')
        cv2.imwrite(outp, cv2.cvtColor(fig, cv2.COLOR_RGB2BGR))
        print('saved', outp, flush=True)

    # ── 汇总：GT box 级混淆矩阵 + precision/recall（三套判据）──
    if EVAL:
        E = np.array(EVAL)  # (N,3): moving, flow_hit, lidar_hit
        mv = E[:, 0]
        def pr(pred, name):
            tp = int((pred & mv).sum()); fp = int((pred & ~mv).sum())
            fn = int((~pred & mv).sum()); tn = int((~pred & ~mv).sum())
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            print(f'  {name:12s} TP={tp:3d} FP={fp:3d} FN={fn:3d} TN={tn:3d} '
                  f'P={prec:.2f} R={rec:.2f}', flush=True)
        n_mv = int(mv.sum()); n_st = int((~mv).sum())
        print(f'\n===== EVAL over {len(EVAL)} GT vehicle boxes '
              f'({n_mv} moving / {n_st} static) =====', flush=True)
        print(f'  thresholds: flow>{RES_THRESH}px  lidar>{LIDAR_THRESH}m', flush=True)
        pr(E[:, 1], 'flow-only')
        pr(E[:, 2], 'lidar-only')
        pr(E[:, 1] | E[:, 2], 'union')


if __name__ == '__main__':
    main()
