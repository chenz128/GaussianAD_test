"""用 nuScenes 3D box+速度真值量化评估融合动静 GT 的质量。

口径（box 级，与早先 flow 验证 P=0.83 一致）：
  - 真值：每个可动类(vehicle/human) box，box_velocity 速度 spd。
          spd > MOVING_SPD 判"真动态(正)"，spd <= STATIC_SPD 判"真静止(负)"，
          中间灰区(0.5~1.0)不计入，避免边界噪声污染。
  - 预测：box 投影到 render 平面(与 GT 同一套 render_intrinsic)，
          box 内 GT==2(dynamic) 像素数 >= MIN_HIT 判"预测动态"。
  - 累加 TP/FP/FN/TN → precision / recall / static 误报率。

只评估"有速度真值且投影可见"的 box，跳过被遮挡/出界/太小的 box。
背景(天空/地面/远处)无 box 真值，本就不进评估——评估只问"可动物体判对没"。

用法（单 GPU，远端）：
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=6 /data/chenz/conda_env/splatting/bin/python \
      tools/_eval_dyn_gt.py --num-frames 150 --step 180
"""
import argparse
import time

import numpy as np
import mmengine
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points, BoxVisibility

from tools.flow_dynamic import FlowDynamicDetector
from tools.gen_dynamic_gt import gen_one, SENSOR_TYPES

# 可动类前缀（nuScenes box.name）
MOVABLE_PREFIX = ('vehicle', 'human')
# 速度真值阈值（m/s）
MOVING_SPD = 1.0   # > 此值 = 真动态
STATIC_SPD = 0.5   # <= 此值 = 真静止；(0.5,1.0] 灰区不计


def build_args():
    class A:
        pass
    a = A()
    a.render_h = 256; a.render_w = 704; a.scale = 0.44; a.crop_top = 140
    a.dist_thresh = 0.3; a.rmse_thresh = 0.35
    a.dt_budget = 1.5; a.min_dt = 0.15; a.max_dt = 0.5
    a.vote_ratio = 0.3; a.min_dyn_cluster = 15
    a.strong_abs = 0.45; a.max_dyn_range = 40.0
    a.static_radius = 2; a.dyn_radius = 3
    return a


def eval_frame(nusc, gt, sample, scale, crop_top, H, W, min_hit, min_box_px):
    """对单帧 (6,H,W) GT 做 box 级评估，返回 list[(is_moving, pred_dyn, spd, npx_dyn)]."""
    recs = []
    for ci, cam_type in enumerate(SENSOR_TYPES):
        sd_token = sample['data'][cam_type]
        _, boxes_cam, camK = nusc.get_sample_data(sd_token, box_vis_level=BoxVisibility.ANY)
        gci = gt[ci]
        for box in boxes_cam:
            if not box.name.startswith(MOVABLE_PREFIX):
                continue
            vel = nusc.box_velocity(box.token)
            if np.isnan(vel).any():
                continue
            spd = float(np.linalg.norm(vel))
            # 灰区不计入
            if STATIC_SPD < spd <= MOVING_SPD:
                continue
            is_moving = spd > MOVING_SPD
            # corners → 原图像素 → render 平面
            cor = view_points(box.corners(), camK, normalize=True)[:2]
            xs = cor[0] * scale
            ys = cor[1] * scale - crop_top
            x0, x1 = int(max(0, np.floor(xs.min()))), int(min(W, np.ceil(xs.max())))
            y0, y1 = int(max(0, np.floor(ys.min()))), int(min(H, np.ceil(ys.max())))
            if x1 <= x0 or y1 <= y0:
                continue
            box_px = (x1 - x0) * (y1 - y0)
            if box_px < min_box_px:
                continue
            sub = gci[y0:y1, x0:x1]
            npx_dyn = int((sub == 2).sum())
            pred_dyn = npx_dyn >= min_hit
            recs.append((is_moving, pred_dyn, spd, npx_dyn))
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', default='data/nuscenes_cam/nuscenes_infos_train_gaussian_ad_v4.pkl')
    ap.add_argument('--data-root', default='data/nuscenes')
    ap.add_argument('--nusc-version', default='v1.0-trainval')
    ap.add_argument('--m3d-root', default='/data/chenz/Gaussianflowocc_test/data/metric_3d_nusc')
    ap.add_argument('--sam-root', default='/data/chenz/Gaussianflowocc_test/data/grounded_sam_nusc')
    ap.add_argument('--num-frames', type=int, default=150)
    ap.add_argument('--step', type=int, default=180)
    ap.add_argument('--min-hit', type=int, default=20, help='box 内 dynamic 像素数 >= 此值判预测动')
    ap.add_argument('--min-box-px', type=int, default=200, help='box render 面积下限')
    ap.add_argument('--use-flow', action='store_true', default=True)
    ap.add_argument('--no-flow', dest='use_flow', action='store_false')
    args_cli = ap.parse_args()

    args = build_args()
    t0 = time.time()
    nusc = NuScenes(version=args_cli.nusc_version, dataroot=args_cli.data_root, verbose=False)
    det = None
    if args_cli.use_flow:
        det = FlowDynamicDetector(
            nusc, args_cli.data_root, args_cli.m3d_root, args_cli.sam_root, device='cuda')
    data = mmengine.load(args_cli.pkl)
    si, kf = data['infos'], data['metadata']
    print(f'LOAD {time.time()-t0:.1f}s  use_flow={args_cli.use_flow}  '
          f'min_hit={args_cli.min_hit}', flush=True)

    all_recs = []
    n_done = 0
    for gi in range(0, len(kf), args_cli.step):
        if n_done >= args_cli.num_frames:
            break
        st, fr = kf[gi]
        scene = si[st]
        fr = int(np.clip(fr, 0, len(scene) - 1))
        info = scene[fr]
        try:
            gt, status = gen_one(scene, fr, args_cli.data_root, args, detector=det)
            sample = nusc.get('sample', info['token'])
            recs = eval_frame(nusc, gt, sample, args.scale, args.crop_top,
                              args.render_h, args.render_w,
                              args_cli.min_hit, args_cli.min_box_px)
        except Exception as e:
            print(f'  gi={gi} ERR {e}', flush=True)
            continue
        all_recs.extend(recs)
        n_done += 1
        if n_done % 20 == 0:
            print(f'  done {n_done} frames, boxes={len(all_recs)}', flush=True)

    recs = np.array([(int(m), int(p)) for m, p, _, _ in all_recs], dtype=int)
    if recs.shape[0] == 0:
        print('NO BOXES'); return
    mov = recs[:, 0].astype(bool)
    pred = recs[:, 1].astype(bool)

    TP = int((mov & pred).sum())
    FN = int((mov & ~pred).sum())
    FP = int((~mov & pred).sum())
    TN = int((~mov & ~pred).sum())
    n_mov = int(mov.sum()); n_sta = int((~mov).sum())

    prec = TP / max(TP + FP, 1)
    rec = TP / max(TP + FN, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    static_fp_rate = FP / max(n_sta, 1)

    print('\n================ DYN-GT 量化评估（box 级，nuScenes 速度真值）================')
    print(f'  评估帧数        : {n_done}')
    print(f'  可动 box 总数   : {recs.shape[0]}  (真动 {n_mov} / 真静 {n_sta})')
    print(f'  判定阈值        : box 内 dynamic 像素 >= {args_cli.min_hit}')
    print(f'  ------------------------------------------------------------')
    print(f'  TP={TP}  FP={FP}  FN={FN}  TN={TN}')
    print(f'  Precision(动态) : {prec:.3f}   ← 判为动的 box 里真在动的比例')
    print(f'  Recall(动态)    : {rec:.3f}   ← 真在动的 box 里被检出的比例')
    print(f'  F1              : {f1:.3f}')
    print(f'  静止误报率      : {static_fp_rate:.3f}   ← 停车被误标动态(假阳)的比例')
    print('============================================================================')


if __name__ == '__main__':
    main()
