"""临时可视化：把融合动静 GT 叠回原图，肉眼判断 dynamic 贴合度。
每相机一行：[原图 | 原图叠GT(红动/绿静/原色=ignore) | box真值框(红动/绿静/黄灰区)]
"""
import os
import numpy as np
import cv2
import mmengine

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points, BoxVisibility
from tools.flow_dynamic import FlowDynamicDetector
from tools.gen_dynamic_gt import gen_one

NUSC_ROOT = 'data/nuscenes'
M3D = '/data/chenz/Gaussianflowocc_test/data/metric_3d_nusc'
SAM = '/data/chenz/Gaussianflowocc_test/data/grounded_sam_nusc'
PKL = 'data/nuscenes_cam/nuscenes_infos_train_gaussian_ad_v4.pkl'
OUT = '/tmp/dyn_vis'
SENSOR_TYPES = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
SCALE, CROP_TOP, RH, RW = 0.44, 140, 256, 704


class A:
    render_h = RH; render_w = RW; scale = SCALE; crop_top = CROP_TOP
    dist_thresh = 0.3; rmse_thresh = 0.35; dt_budget = 1.5; min_dt = 0.15; max_dt = 0.5
    vote_ratio = 0.3; min_dyn_cluster = 15; strong_abs = 0.45; max_dyn_range = 40.0
    static_radius = 2; dyn_radius = 3


# 可动类前缀 / 速度阈值（与 _eval_dyn_gt.py 一致）
MOVABLE_PREFIX = ('vehicle', 'human')
MOVING_SPD = 1.0   # > 此 = 真动态 → 红框
STATIC_SPD = 0.5   # <= 此 = 真静止 → 绿框；(0.5,1.0] 灰区 → 黄框


def render_img(nusc, sample, cam):
    """原图 -> render 空间 (RH,RW,3) RGB。"""
    sd = nusc.get('sample_data', sample['data'][cam])
    img = cv2.cvtColor(cv2.imread(os.path.join(NUSC_ROOT, sd['filename'])), cv2.COLOR_BGR2RGB)
    rh = int(round(900 * SCALE)); rw = int(round(1600 * SCALE))
    img = cv2.resize(img, (rw, rh))
    return img[CROP_TOP:CROP_TOP + RH]


def _draw_cube(img, c2d, color):
    """c2d: (2,8) render 平面像素坐标，画 3D box 线框。"""
    pts = c2d.T.astype(np.int32)  # (8,2)

    def rect(idx):
        prev = pts[idx[-1]]
        for i in idx:
            cur = pts[i]
            cv2.line(img, tuple(prev), tuple(cur), color, 1, cv2.LINE_AA)
            prev = cur

    rect([0, 1, 2, 3])      # 前面
    rect([4, 5, 6, 7])      # 后面
    for i in range(4):
        cv2.line(img, tuple(pts[i]), tuple(pts[i + 4]), color, 1, cv2.LINE_AA)


def draw_gt_boxes(img, nusc, sample, cam):
    """把该相机可见的可动类 box 按速度着色画到 render 空间图上。"""
    out = img.copy()
    sd_token = sample['data'][cam]
    _, boxes_cam, camK = nusc.get_sample_data(sd_token, box_vis_level=BoxVisibility.ANY)
    for box in boxes_cam:
        if not box.name.startswith(MOVABLE_PREFIX):
            continue
        vel = nusc.box_velocity(box.token)
        if np.isnan(vel).any():
            color = (160, 160, 160)   # 速度未知 → 灰
        else:
            spd = float(np.linalg.norm(vel))
            if spd > MOVING_SPD:
                color = (255, 0, 0)       # 真动 → 红
            elif spd <= STATIC_SPD:
                color = (0, 200, 0)       # 真静 → 绿
            else:
                color = (255, 210, 0)     # 灰区 → 黄
        cor = view_points(box.corners(), camK, normalize=True)[:2]  # (2,8) 原图像素
        c2d = np.empty_like(cor)
        c2d[0] = cor[0] * SCALE
        c2d[1] = cor[1] * SCALE - CROP_TOP
        # 跳过完全出界的
        if c2d[0].max() < 0 or c2d[0].min() > RW or c2d[1].max() < 0 or c2d[1].min() > RH:
            continue
        _draw_cube(out, c2d, color)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    nusc = NuScenes(version='v1.0-trainval', dataroot=NUSC_ROOT, verbose=False)
    det = FlowDynamicDetector(nusc, NUSC_ROOT, M3D, SAM, device='cuda',
                              scale=SCALE, crop_top=CROP_TOP, render_h=RH, render_w=RW)
    data = mmengine.load(PKL)
    si, kf = data['infos'], data['metadata']
    args = A()

    # 指定 token 模式（同帧对比）：环境变量 VIS_TOKENS 逗号分隔（可用前10位前缀）
    want = os.environ.get('VIS_TOKENS', '').strip()
    if want:
        want_set = set(want.split(','))
        cand = []
        for st, scene in si.items():
            for fr, info in enumerate(scene):
                if info['token'][:10] in want_set or info['token'] in want_set:
                    gt, status = gen_one(scene, fr, NUSC_ROOT, args, detector=det)
                    ndyn = int((gt == 2).sum())
                    cand.append((ndyn, 0, st, fr, gt))
                    print(f'token={info["token"][:10]} dyn_px={ndyn} {status}', flush=True)
        cand.sort(reverse=True)
        cand = cand[:3]
    else:
        # 扫一批帧，挑 dynamic 像素最多的 N 帧（最能看出贴合度）
        # VIS_SCAN=start,stop,step 可配置扫描区间（默认 0,600,7）；VIS_TOPK 取前几
        scan = os.environ.get('VIS_SCAN', '0,600,7').split(',')
        s0, s1, ss = int(scan[0]), int(scan[1]), int(scan[2])
        topk = int(os.environ.get('VIS_TOPK', '3'))
        cand = []
        for gi in range(s0, min(s1, len(kf)), ss):
            st, fr = kf[gi]; scene = si[st]; fr = int(np.clip(fr, 0, len(scene) - 1))
            try:
                gt, status = gen_one(scene, fr, NUSC_ROOT, args, detector=det)
            except Exception as e:
                continue
            ndyn = int((gt == 2).sum())
            cand.append((ndyn, gi, st, fr, gt))
            print(f'gi={gi} dyn_px={ndyn} {status}', flush=True)
            if len(cand) >= 60:
                break
        cand.sort(reverse=True)
        cand = cand[:topk]

    for rank, (ndyn, gi, st, fr, gt) in enumerate(cand):
        scene = si[st]; info = scene[fr]; token = info['token']
        sample = nusc.get('sample', token)
        rows = []
        for ci, cam in enumerate(SENSOR_TYPES):
            img = render_img(nusc, sample, cam)
            m = gt[ci]
            # 叠加：红=dynamic(2)，绿=static(1)，ignore(0)保持原色
            ov = img.copy()
            ov[m == 1] = (ov[m == 1] * 0.5 + np.array([0, 180, 0]) * 0.5).astype(np.uint8)
            ov[m == 2] = (ov[m == 2] * 0.35 + np.array([255, 0, 0]) * 0.65).astype(np.uint8)
            cv2.putText(ov, 'GT pix', (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
            # box 真值框视图：红=真动 绿=真静 黄=灰区 灰=速度未知
            boxv = draw_gt_boxes(img, nusc, sample, cam)
            cv2.putText(boxv, cam + ' box', (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
            sep = np.full((RH, 3, 3), 255, np.uint8)
            rows.append(np.concatenate([img, sep, ov, sep, boxv], axis=1))
        seph = np.full((3, rows[0].shape[1], 3), 255, np.uint8)
        grid = rows[0]
        for r in rows[1:]:
            grid = np.concatenate([grid, seph, r], axis=0)
        out_path = os.path.join(OUT, f'frame{rank}_{token[:10]}_dyn{ndyn}.jpg')
        cv2.imwrite(out_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f'SAVED {out_path}', flush=True)


if __name__ == '__main__':
    main()
