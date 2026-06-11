"""光流-刚性流残差动态检测模块（供 gen_dynamic_gt.py 融合）。

对每个相机，用相邻 sweep 的 RAFT 实际光流 与 Metric3D 深度 + ego/cam 位姿
算出的刚性光流做差，残差大的 GroundedSAM 可移动连通域判为"正在运动"。
与 LiDAR 时序残差几何互补：光流强于径向/迫近运动，LiDAR 强于切向/同向运动。

零人工标注：图像/位姿来自 nuScenes，深度来自 Metric3D，语义来自 GroundedSAM。

双重门控压制假阳（与验证脚本 _verify_flow_dynamic.py 一致）：
  res = max(0, ||f_real - f_rigid|| - sens_k*depth_sens - rigid_alpha*|f_rigid|)
  其中 depth_sens 是"仅因 Metric3D 深度不确定就会摆动"的刚性流幅度，
  rigid_alpha*|f_rigid| 压制近处大目标在高自车速下的刚性流极敏感假阳。

输出 render 空间 (scale=0.44, crop_top=140) 与 gen_dynamic_gt / dataset.py
pseudo 分支严格对齐，可直接与 LiDAR mask 做像素级 union。
"""
import os

import numpy as np
import torch
import cv2
from pyquaternion import Quaternion

# Metric3D 深度的相机顺序（generate_m3d_nusc.py）
DEPTH_CAMS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
              'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
# GroundedSAM 语义的相机顺序（generate_grounded_sam.py:304，与 depth 不同！）
SAM_CAMS = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_BACK_LEFT',
            'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_FRONT_RIGHT']
# nuScenes occ 16+1 类中"可移动且会动"的类别 id（GroundedSAM 同一套）
MOVABLE = {2, 3, 4, 5, 6, 7, 9, 10}
# 背景静态类：物理上不会自己动，语义先验即可可靠判静（不依赖光流残差）。
# barrier(1), traffic_cone(8), driveable_surface(11), other_flat(12),
# sidewalk(13), terrain(14), manmade(15), vegetation(16)。
# 故意排除所有 MOVABLE 类——同向/同速车的低残差是光流盲区，不可信，留 ignore。
STATIC_BG = {1, 8, 11, 12, 13, 14, 15, 16}


def _pose_to_mat(translation, rotation):
    m = np.eye(4)
    m[:3, :3] = Quaternion(rotation).rotation_matrix
    m[:3, 3] = np.asarray(translation)
    return m


class FlowDynamicDetector:
    """RAFT 光流 + Metric3D 深度的逐相机动态像素检测器。

    detect(token, sensor_types) -> (6, H, W) bool，True = 该像素属于"正在运动"
    的可移动目标。相机顺序与传入的 sensor_types 一致（GT 直接 union 即可）。
    """

    def __init__(self, nusc, data_root, m3d_root, sam_root, device='cuda',
                 scale=0.44, crop_top=140, render_h=256, render_w=704,
                 res_thresh=3.0, depth_min=2.8, depth_max=50.0,
                 depth_rel_err=0.10, depth_abs_err=0.6,
                 sens_k=1.5, rigid_alpha=0.12, min_comp_px=60,
                 dyn_ratio=0.6, min_dyn_depth=5.0, near_area_frac=0.15,
                 strong_mult=2.0, strong_frac=0.15,
                 bottom_frac=0.80, bottom_min_depth=4.0,
                 huge_area_frac=0.30, huge_near_depth=8.0, arbiter_frac=0.01):
        self.nusc = nusc
        self.data_root = data_root
        self.m3d_root = m3d_root
        self.sam_root = sam_root
        self.device = device
        self.scale = scale
        self.crop_top = crop_top
        self.H = render_h
        self.W = render_w
        self.res_thresh = res_thresh
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.depth_rel_err = depth_rel_err
        self.depth_abs_err = depth_abs_err
        self.sens_k = sens_k
        self.rigid_alpha = rigid_alpha
        self.min_comp_px = min_comp_px
        # 整车判定: 连通域内 res>res_thresh 的像素占比 >= dyn_ratio 才判动
        # (代替"中位数>阈值", 避免近端少数高残差像素把整辆停车拉成动态)
        self.dyn_ratio = dyn_ratio
        # 近处大面积守卫: 质心深度 < min_dyn_depth 且面积占图 > near_area_frac
        # 的连通域深度敏感度主导噪声, 退回不标(宁可不教)
        self.min_dyn_depth = min_dyn_depth
        self.near_area_frac = near_area_frac
        # 强运动核守卫: 连通域须有一簇 res>strong_mult*res_thresh 的强运动像素
        # (占比 >= strong_frac), 压制近地面/车轮勉强越阈的细条带噪声
        self.strong_mult = strong_mult
        self.strong_frac = strong_frac
        # 底部极近带守卫: 质心落在画面底部 bottom_frac 以下且中位深度
        # < bottom_min_depth 的连通域 → 车头引擎盖/地面反光假阳，排除
        self.bottom_frac = bottom_frac
        self.bottom_min_depth = bottom_min_depth
        # 近距超大面积守卫: 自车贴身驶过的静止大型结构(挂车/集装箱/墙/巴士)，
        # 极近(<huge_near_depth)时视差+深度敏感令刚性流估计整片出错、残差全饱和，
        # 连通域填满画面 >huge_area_frac。真动车几乎不可能在<8m处均匀填满30%+画面。
        # VRU(类2/6/7)豁免(近人不会填满30%)。
        self.huge_area_frac = huge_area_frac
        self.huge_near_depth = huge_near_depth
        # LiDAR 仲裁: 近距超大连通域即使命中几何守卫，若 LiDAR 在该区域独立
        # 判动(切向运动残差)占比 > arbiter_frac → 真动大车，不 skip（双模态互证，
        # 静止挂车 LiDAR 配准后残差≈0→仍 skip）。LiDAR 投影稀疏故阈值偏低。
        self.arbiter_frac = arbiter_frac
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
        weights = Raft_Large_Weights.C_T_SKHT_V2
        self.model = raft_large(weights=weights, progress=False).eval().to(device)
        self.transforms = weights.transforms()

    def _cam_to_global(self, sd_token):
        sd = self.nusc.get('sample_data', sd_token)
        cs = self.nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        ep = self.nusc.get('ego_pose', sd['ego_pose_token'])
        c2g = (_pose_to_mat(ep['translation'], ep['rotation'])
               @ _pose_to_mat(cs['translation'], cs['rotation']))
        K = np.array(cs['camera_intrinsic'])
        return c2g, K, sd['filename']

    @torch.no_grad()
    def _raft(self, img1, img2):
        # 关键：保持 uint8 传入 transforms，由其负责归一化到 [-1,1]；
        # 预先 .float() 会绕过归一化导致 RAFT 输出垃圾。
        t1 = torch.from_numpy(img1).permute(2, 0, 1)[None]
        t2 = torch.from_numpy(img2).permute(2, 0, 1)[None]
        t1, t2 = self.transforms(t1, t2)
        return self.model(t1.to(self.device), t2.to(self.device))[-1][0].cpu().numpy()

    def _to_render(self, arr, interp):
        """900x1600 → render 空间：resize(scale) 后裁掉顶部 crop_top 行。"""
        rh = int(round(900 * self.scale))
        rw = int(round(1600 * self.scale))
        r = cv2.resize(arr, (rw, rh), interpolation=interp)
        return r[self.crop_top:self.crop_top + self.H]

    def _render_K(self, K):
        """camera_intrinsic → render 内参：缩放 + crop 平移（与 dataset.py 一致）。"""
        Kr = K.astype(np.float64).copy()
        Kr[0, 0] *= self.scale  # fx
        Kr[1, 1] *= self.scale  # fy
        Kr[0, 2] *= self.scale  # cx
        Kr[1, 2] *= self.scale  # cy
        Kr[1, 2] -= self.crop_top
        return Kr

    def detect(self, token, sensor_types, lidar_dyn=None):
        """返回 (6, H, W) bool，相机顺序 = sensor_types。缺 M3D/SAM 文件则全 False。

        lidar_dyn: 可选 (6,H,W) bool，LiDAR 分支判动像素(render 空间，顺序同 sensor_types)，
                   供近距超大守卫做互证仲裁。
        """
        out = np.zeros((6, self.H, self.W), dtype=bool)
        sample = self.nusc.get('sample', token)
        scene_name = self.nusc.get('scene', sample['scene_token'])['name']
        m3d_f = os.path.join(self.m3d_root, scene_name, token + '.npy')
        sam_f = os.path.join(self.sam_root, scene_name, token + '.npy')
        if not (os.path.exists(m3d_f) and os.path.exists(sam_f)):
            return out
        depth_all = np.load(m3d_f).astype(np.float32)   # (6,900,1600) DEPTH_CAMS 顺序
        seg_all = np.load(sam_f).astype(np.int32)        # (6,900,1600) SAM_CAMS 顺序
        for oi, cam in enumerate(sensor_types):
            ld = lidar_dyn[oi] if lidar_dyn is not None else None
            out[oi] = self._detect_cam(sample, cam, depth_all, seg_all, ld)
        return out

    def static_bg(self, token, sensor_types):
        """返回 (6, H, W) bool，True = 背景静态类像素（render 空间，顺序同 sensor_types）。

        仅用 GroundedSAM 语义先验判定（STATIC_BG），不依赖光流残差——背景类物理上
        不会自己动，低残差可信；可移动类（同向/同速车的低残差是光流盲区）被排除在外，
        留作 ignore 不监督。用于 LiDAR 配准失败整帧 ignore 时补回 static(1) 背景层。
        缺 SAM 文件则全 False。
        """
        out = np.zeros((6, self.H, self.W), dtype=bool)
        sample = self.nusc.get('sample', token)
        scene_name = self.nusc.get('scene', sample['scene_token'])['name']
        sam_f = os.path.join(self.sam_root, scene_name, token + '.npy')
        if not os.path.exists(sam_f):
            return out
        seg_all = np.load(sam_f).astype(np.int32)        # (6,900,1600) SAM_CAMS 顺序
        bg_list = list(STATIC_BG)
        for oi, cam in enumerate(sensor_types):
            seg = seg_all[SAM_CAMS.index(cam)]
            seg_r = self._to_render(seg, cv2.INTER_NEAREST)
            out[oi] = np.isin(seg_r, bg_list)
        return out

    def _detect_cam(self, sample, cam, depth_all, seg_all, lidar_dyn_cam=None):
        H, W = self.H, self.W
        dyn = np.zeros((H, W), dtype=bool)

        sd_t_tok = sample['data'][cam]
        sd_t = self.nusc.get('sample_data', sd_t_tok)
        # 用相邻 sweep（~0.083s）而非下一关键帧（0.5s），避免 RAFT 大位移失效
        nb_tok = sd_t['next'] if sd_t['next'] else sd_t_tok
        c2g_t, K, fn_t = self._cam_to_global(sd_t_tok)
        c2g_n, _, fn_n = self._cam_to_global(nb_tok)
        T = np.linalg.inv(c2g_n) @ c2g_t   # cam_t -> cam_{t+sweep}

        img_t = cv2.cvtColor(cv2.imread(os.path.join(self.data_root, fn_t)), cv2.COLOR_BGR2RGB)
        img_n = cv2.cvtColor(cv2.imread(os.path.join(self.data_root, fn_n)), cv2.COLOR_BGR2RGB)
        depth = depth_all[DEPTH_CAMS.index(cam)]
        seg = seg_all[SAM_CAMS.index(cam)]

        img_t_r = np.ascontiguousarray(self._to_render(img_t, cv2.INTER_LINEAR))
        img_n_r = np.ascontiguousarray(self._to_render(img_n, cv2.INTER_LINEAR))
        depth_r = self._to_render(depth, cv2.INTER_NEAREST)
        seg_r = self._to_render(seg, cv2.INTER_NEAREST)

        Kr = self._render_K(K)
        fx, fy, cx, cy = Kr[0, 0], Kr[1, 1], Kr[0, 2], Kr[1, 2]

        f_real = self._raft(img_t_r, img_n_r)

        # 刚性光流：假设世界静止，仅由 ego/cam 运动产生的像素位移
        uu, vv = np.meshgrid(np.arange(W), np.arange(H))
        Z = depth_r
        X = (uu - cx) / fx * Z
        Y = (vv - cy) / fy * Z
        P = np.stack([X, Y, Z, np.ones_like(Z)], 0).reshape(4, -1)
        Pc = T @ P
        Zc = Pc[2].reshape(H, W)
        up = (fx * Pc[0] / (Pc[2] + 1e-6) + cx).reshape(H, W)
        vp = (fy * Pc[1] / (Pc[2] + 1e-6) + cy).reshape(H, W)
        f_rigid = np.stack([up - uu, vp - vv], 0)

        # 深度扰动敏感度：抬高深度（相对 OR 绝对误差取大）重算刚性流，
        # 量出"仅因深度不确定就会摆动"的幅度（近处大目标极敏感）。
        dZ = np.maximum(Z * self.depth_rel_err, self.depth_abs_err)
        Zp = Z + dZ
        Pp = np.stack([(uu - cx) / fx * Zp, (vv - cy) / fy * Zp, Zp, np.ones_like(Zp)],
                      0).reshape(4, -1)
        Pcp = T @ Pp
        upp = (fx * Pcp[0] / (Pcp[2] + 1e-6) + cx).reshape(H, W)
        vpp = (fy * Pcp[1] / (Pcp[2] + 1e-6) + cy).reshape(H, W)
        sens = np.sqrt((upp - up) ** 2 + (vpp - vp) ** 2)
        rigid_mag = np.hypot(f_rigid[0], f_rigid[1])

        res_raw = np.sqrt((f_real[0] - f_rigid[0]) ** 2 + (f_real[1] - f_rigid[1]) ** 2)
        # 双重门控：扣掉深度敏感度 + 相对刚性流量级
        res = np.maximum(0.0, res_raw - self.sens_k * sens - self.rigid_alpha * rigid_mag)
        valid = (Z > self.depth_min) & (Z < self.depth_max) & (Zc > 0.5)
        movable = np.isin(seg_r, list(MOVABLE))

        # 整车判定：用"超阈值像素占比"而非中位数，避免近端少数高残差像素
        # 把整辆停车翻成动态（真动车残差铺满全车，停车只有近端一条带高残差）
        mv = (movable & valid).astype(np.uint8)
        ncomp, comp = cv2.connectedComponents(mv)
        img_area = float(H * W)
        vv_grid = np.arange(H).reshape(H, 1)
        for ic in range(1, ncomp):
            cmask = comp == ic
            npx = int(cmask.sum())
            if npx < self.min_comp_px:
                continue
            comp_depth = Z[cmask]
            comp_depth = comp_depth[comp_depth > 0]
            med_depth = float(np.median(comp_depth)) if comp_depth.size else 0.0
            # 近处大面积守卫：极近(<min_dyn_depth)且占图过大的连通域深度敏感度
            # 主导噪声(贴近大车/车头反光/极近地面) → 退回不标
            if med_depth and med_depth < self.min_dyn_depth and npx > self.near_area_frac * img_area:
                continue
            # 底部极近带守卫：质心在画面底部且深度极近 → 车头引擎盖/地面反光
            # 仅对车辆类生效，对行人/骑行者(VRU,类2/6/7)豁免——近处VRU是最不能漏的
            if med_depth and med_depth < self.bottom_min_depth:
                seg_c = seg_r[cmask]
                is_vru = np.isin(seg_c, [2, 6, 7]).mean() > 0.5
                if not is_vru:
                    v_centroid = float((vv_grid * cmask).sum() / max(npx, 1))
                    if v_centroid > self.bottom_frac * H:
                        continue
            # 超阈值像素占比 >= dyn_ratio 才判动
            res_c = res[cmask]
            ratio = float((res_c > self.res_thresh).mean())
            if ratio < self.dyn_ratio:
                continue
            # 强运动核守卫：须有 strong_frac 比例像素超 strong_mult*res_thresh，
            # 真动目标有强运动核，近地面勉强越阈的噪声条带过不了这关
            strong_ratio = float((res_c > self.strong_mult * self.res_thresh).mean())
            if strong_ratio < self.strong_frac:
                continue
            # 近距超大面积守卫：极近+填满画面过大 → 自车贴身驶过的静止大型结构
            # (挂车/集装箱/墙)，视差使整片残差饱和。VRU 豁免。
            # LiDAR 仲裁：若 LiDAR 在该连通域独立判动(切向运动) → 真动大车，保留。
            if npx > self.huge_area_frac * img_area and \
               med_depth and med_depth < self.huge_near_depth:
                seg_c = seg_r[cmask]
                if not (np.isin(seg_c, [2, 6, 7]).mean() > 0.5):
                    lidar_backs = (lidar_dyn_cam is not None and
                                   float(lidar_dyn_cam[cmask].mean()) > self.arbiter_frac)
                    if not lidar_backs:
                        continue
            dyn[cmask] = True
        return dyn
