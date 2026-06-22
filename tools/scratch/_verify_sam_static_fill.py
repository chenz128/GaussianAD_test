"""临时验证：SAM static fill（所有帧用 STATIC_BG 补 ignore→static）效果。

对比 fill 关 vs 开 时 ignore/static/dynamic 占比，并存可视化图（6相机×[图像|GT_off|GT_on]）。
不进 git。h20-new 仅用后 4 卡（CUDA_VISIBLE_DEVICES=4/5/6/7）。
"""
import argparse
import os
import numpy as np
import mmengine
import cv2


# nusc 17 类调色板（0=ignore 灰，1=static 绿，2=dynamic 红）
def colorize(out_cam):
    h, w = out_cam.shape
    rgb = np.full((h, w, 3), 80, np.uint8)        # ignore → 深灰
    rgb[out_cam == 1] = (0, 180, 0)               # static → 绿
    rgb[out_cam == 2] = (0, 0, 220)               # dynamic → 红
    return rgb


def stats(out):
    tot = out.size
    return (float((out == 0).sum()) / tot,
            float((out == 1).sum()) / tot,
            float((out == 2).sum()) / tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', default='data/nuscenes_cam/nuscenes_infos_train_gaussian_ad_v4.pkl')
    ap.add_argument('--data-root', default='data/nuscenes')
    ap.add_argument('--m3d-root', default='/data/chenz/Gaussianflowocc_test/data/metric_3d_nusc')
    ap.add_argument('--sam-root', default='/data/chenz/Gaussianflowocc_test/data/grounded_sam_nusc')
    ap.add_argument('--n-frames', type=int, default=12)
    ap.add_argument('--n-vis', type=int, default=4, help='存可视化的帧数')
    ap.add_argument('--vis-dir', default='/tmp/sam_fill_vis')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    os.makedirs(args.vis_dir, exist_ok=True)
    from tools.gen_dynamic_gt import gen_one, SENSOR_TYPES
    base_off = argparse.Namespace(
        scale=0.44, crop_top=140, render_h=256, render_w=704,
        dist_thresh=0.3, rmse_thresh=0.35, dt_budget=1.5, min_dt=0.15, max_dt=0.5,
        vote_ratio=0.3, min_dyn_cluster=15, strong_abs=0.45, max_dyn_range=40.0,
        static_radius=2, dyn_radius=3, flow_rescue_static=False, sam_static_fill=False,
    )
    base_on = argparse.Namespace(**{**vars(base_off), 'sam_static_fill': True})

    print(f'loading {args.pkl} ...', flush=True)
    data = mmengine.load(args.pkl)
    scene_infos = data['infos']
    keyframes = data['metadata']

    from nuscenes.nuscenes import NuScenes
    from tools.flow_dynamic import FlowDynamicDetector
    print('loading NuScenes v1.0-trainval ...', flush=True)
    nusc = NuScenes(version='v1.0-trainval', dataroot=args.data_root, verbose=False)
    detector = FlowDynamicDetector(
        nusc, args.data_root, args.m3d_root, args.sam_root, device=args.device,
        scale=base_off.scale, crop_top=base_off.crop_top,
        render_h=base_off.render_h, render_w=base_off.render_w, res_thresh=3.0)
    print('detector ready\n', flush=True)

    step = max(1, len(keyframes) // args.n_frames)
    picks = list(range(0, len(keyframes), step))[:args.n_frames]

    agg_off = np.zeros(3)
    agg_on = np.zeros(3)
    H, W = base_off.render_h, base_off.render_w
    for k, gi in enumerate(picks):
        scene_token, frame_idx = keyframes[gi]
        scene = scene_infos[scene_token]
        frame_idx = int(np.clip(frame_idx, 0, len(scene) - 1))
        token = scene[frame_idx]['token']

        out_off, st_off = gen_one(scene, frame_idx, args.data_root, base_off, detector=detector)
        out_on, st_on = gen_one(scene, frame_idx, args.data_root, base_on, detector=detector)
        s_off = stats(out_off)
        s_on = stats(out_on)
        agg_off += s_off
        agg_on += s_on
        print(f'{token[:10]} | OFF ign/sta/dyn={s_off[0]:.3f}/{s_off[1]:.3f}/{s_off[2]:.3f} | '
              f'ON ={s_on[0]:.3f}/{s_on[1]:.3f}/{s_on[2]:.3f} | {st_on}', flush=True)

        if k < args.n_vis:
            # 取每相机原图(render空间)拼图：行=相机，列=[图像|OFF|ON]
            rows = []
            for ci, cam in enumerate(SENSOR_TYPES):
                fn = scene[frame_idx]['data'][cam]['filename']
                img = cv2.imread(os.path.join(args.data_root, fn))
                rh, rw = int(round(900 * base_off.scale)), int(round(1600 * base_off.scale))
                img = cv2.resize(img, (rw, rh))[base_off.crop_top:base_off.crop_top + H]
                row = np.concatenate([img, colorize(out_off[ci]), colorize(out_on[ci])], axis=1)
                rows.append(row)
            canvas = np.concatenate(rows, axis=0)
            p = os.path.join(args.vis_dir, f'{k:02d}_{token[:10]}.jpg')
            cv2.imwrite(p, canvas)
            print(f'  saved vis -> {p}', flush=True)

    n = len(picks)
    mo, mn = agg_off / n, agg_on / n
    print(f'\n===== 平均 (n={n}) =====')
    print(f'OFF: ignore={mo[0]:.4f} static={mo[1]:.4f} dynamic={mo[2]:.4f}')
    print(f' ON: ignore={mn[0]:.4f} static={mn[1]:.4f} dynamic={mn[2]:.4f}')
    print(f'ignore 下降: {(mo[0]-mn[0]):.4f}  static 上升: {(mn[1]-mo[1]):.4f}')


if __name__ == '__main__':
    main()
