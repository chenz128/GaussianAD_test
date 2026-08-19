"""周报三联对比图：同一场景，纯 LiDAR vs 纯光流 vs 融合 的动静分离效果。

每个相机一行，横向 5 列：
  [原图 | LiDAR残差 | 光流 | 融合 | box真值]
  红 = dynamic，绿 = static，原色 = ignore/未判。

互补性评分自动挑帧（默认扫描模式）：优先选 LiDAR 和光流"各有独占动态像素"
的帧 —— 这种帧最能证明"融合 = 两者优势叠加，覆盖最全"。

三种 mask 来源：
  LiDAR-only : gen_one(detector=None)            -> 0/1/2
  Flow-only  : det.detect(token, lidar_dyn=None)  -> bool（纯光流，无仲裁）
  Fused      : gen_one(detector=det)             -> 0/1/2（完整融合，含 LiDAR 仲裁）

用法（远端，只用空闲 GPU 6 或 7）：
  # 自动挑互补性最强的帧
  CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. /data/chenz/conda_env/splatting/bin/python \
      tools/_vis_dyn_compare.py
  # 扫描区间 / 取前几张可调
  VIS_SCAN=0,1200,5 VIS_TOPK=4 CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. \
      /data/chenz/conda_env/splatting/bin/python tools/_vis_dyn_compare.py
  # 指定 token（逗号分隔，可用前10位前缀）
  VIS_TOKENS=abc1234567 CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. \
      /data/chenz/conda_env/splatting/bin/python tools/_vis_dyn_compare.py

输出：/tmp/dyn_cmp/cmp{rank}_{token[:10]}.jpg
"""
import os
import numpy as np
import cv2
import mmengine

from nuscenes.nuscenes import NuScenes
from tools.flow_dynamic import FlowDynamicDetector
from tools.gen_dynamic_gt import gen_one
from tools._vis_dyn_gt import (
    render_img, draw_gt_boxes, A, SENSOR_TYPES,
    NUSC_ROOT, M3D, SAM, PKL, RH, RW,
)

OUT = '/tmp/dyn_cmp'
# 每相机行至少要有这么多 union 动态像素才画（避免大量空行）
MIN_ROW_DYN = 25
# 字体
FONT = cv2.FONT_HERSHEY_SIMPLEX


def overlay(img, dyn_mask=None, static_mask=None):
    """红=dynamic，绿=static，其余原色。"""
    ov = img.copy()
    if static_mask is not None and static_mask.any():
        ov[static_mask] = (ov[static_mask] * 0.5 + np.array([0, 180, 0]) * 0.5).astype(np.uint8)
    if dyn_mask is not None and dyn_mask.any():
        ov[dyn_mask] = (ov[dyn_mask] * 0.35 + np.array([255, 0, 0]) * 0.65).astype(np.uint8)
    return ov


def label(img, text, color=(255, 255, 0)):
    out = img.copy()
    cv2.putText(out, text, (4, 14), FONT, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, (4, 14), FONT, 0.45, color, 1, cv2.LINE_AA)
    return out


def compute_three(scene, fr, det):
    """返回 (lgt, fdyn, fgt)：LiDAR(0/1/2), Flow(bool), Fused(0/1/2)。"""
    token = scene[fr]['token']
    lgt, _ = gen_one(scene, fr, NUSC_ROOT, A(), detector=None)     # LiDAR-only
    fdyn = det.detect(token, SENSOR_TYPES, lidar_dyn=None)          # Flow-only (无仲裁)
    fgt, _ = gen_one(scene, fr, NUSC_ROOT, A(), detector=det)       # Fused
    return lgt, fdyn, fgt


def complement_score(lgt, fdyn):
    """互补性评分：min(LiDAR独占动态, 光流独占动态)。两者都贡献独占越多越好。"""
    ldyn = (lgt == 2)
    l_uniq = int((ldyn & ~fdyn).sum())
    f_uniq = int((fdyn & ~ldyn).sum())
    return min(l_uniq, f_uniq), l_uniq, f_uniq


def render_frame(nusc, scene, fr, lgt, fdyn, fgt):
    """画一帧的 5 列对比图，只保留有动态的相机行。"""
    token = scene[fr]['token']
    sample = nusc.get('sample', token)
    rows = []
    for ci, cam in enumerate(SENSOR_TYPES):
        union_dyn = (lgt[ci] == 2) | fdyn[ci] | (fgt[ci] == 2)
        if int(union_dyn.sum()) < MIN_ROW_DYN:
            continue
        img = render_img(nusc, sample, cam)
        col_raw = label(img, cam)
        # 只显示 dynamic(红)，不画 static 绿底——聚焦"动态覆盖完整度"对比
        col_lid = label(overlay(img, dyn_mask=(lgt[ci] == 2)), 'LiDAR')
        col_flw = label(overlay(img, dyn_mask=fdyn[ci]), 'Flow')
        col_fus = label(overlay(img, dyn_mask=(fgt[ci] == 2)), 'Fused')
        col_box = label(draw_gt_boxes(img, nusc, sample, cam), 'GT box')
        sep = np.full((RH, 3, 3), 255, np.uint8)
        rows.append(np.concatenate(
            [col_raw, sep, col_lid, sep, col_flw, sep, col_fus, sep, col_box], axis=1))
    if not rows:
        return None
    seph = np.full((3, rows[0].shape[1], 3), 255, np.uint8)
    grid = rows[0]
    for r in rows[1:]:
        grid = np.concatenate([grid, seph, r], axis=0)
    return grid


def main():
    os.makedirs(OUT, exist_ok=True)
    nusc = NuScenes(version='v1.0-trainval', dataroot=NUSC_ROOT, verbose=False)
    det = FlowDynamicDetector(nusc, NUSC_ROOT, M3D, SAM, device='cuda',
                              scale=A.scale, crop_top=A.crop_top, render_h=RH, render_w=RW)
    data = mmengine.load(PKL)
    si, kf = data['infos'], data['metadata']

    want = os.environ.get('VIS_TOKENS', '').strip()
    cand = []   # (score, gi, st, fr, lgt, fdyn, fgt)
    if want:
        want_set = set(want.split(','))
        for st, scene in si.items():
            for fr, info in enumerate(scene):
                if info['token'][:10] in want_set or info['token'] in want_set:
                    lgt, fdyn, fgt = compute_three(scene, fr, det)
                    sc, lu, fu = complement_score(lgt, fdyn)
                    cand.append((sc, 0, st, fr, lgt, fdyn, fgt))
                    print(f'token={info["token"][:10]} complement={sc} '
                          f'lidar_uniq={lu} flow_uniq={fu}', flush=True)
    else:
        scan = os.environ.get('VIS_SCAN', '0,1500,5').split(',')
        s0, s1, ss = int(scan[0]), int(scan[1]), int(scan[2])
        for gi in range(s0, min(s1, len(kf)), ss):
            st, fr = kf[gi]
            scene = si[st]
            fr = int(np.clip(fr, 0, len(scene) - 1))
            try:
                lgt, fdyn, fgt = compute_three(scene, fr, det)
            except Exception as e:
                print(f'gi={gi} ERR {e}', flush=True)
                continue
            sc, lu, fu = complement_score(lgt, fdyn)
            cand.append((sc, gi, st, fr, lgt, fdyn, fgt))
            if sc > 0:
                print(f'gi={gi} complement={sc} lidar_uniq={lu} flow_uniq={fu}', flush=True)

    topk = int(os.environ.get('VIS_TOPK', '4'))
    cand.sort(key=lambda x: x[0], reverse=True)
    cand = cand[:topk]

    for rank, (sc, gi, st, fr, lgt, fdyn, fgt) in enumerate(cand):
        scene = si[st]
        token = scene[fr]['token']
        grid = render_frame(nusc, scene, fr, lgt, fdyn, fgt)
        if grid is None:
            print(f'rank{rank} {token[:10]} no dynamic row, skip', flush=True)
            continue
        out_path = os.path.join(OUT, f'cmp{rank}_{token[:10]}.jpg')
        cv2.imwrite(out_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f'SAVED {out_path} complement={sc}', flush=True)


if __name__ == '__main__':
    main()
