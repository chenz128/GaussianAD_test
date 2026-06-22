"""临时验证：对比 关光流 vs 开光流(+rescue) 时 static/dynamic/ignore 占比。

不进 git。在 h20-new 上跑（只用后 4 张卡之一，如 CUDA_VISIBLE_DEVICES=4）。
对每个采样帧分别跑 gen_one(detector=None) 和 gen_one(detector=on)，
统计三类像素占比，量化"光流补充 ignore"的实际收益。
"""
import argparse
import copy
import numpy as np
import mmengine


def stats(out):
    tot = out.size
    return (float((out == 0).sum()) / tot,   # ignore
            float((out == 1).sum()) / tot,   # static
            float((out == 2).sum()) / tot)   # dynamic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', default='data/nuscenes_cam/nuscenes_infos_train_gaussian_ad_v4.pkl')
    ap.add_argument('--data-root', default='data/nuscenes')
    ap.add_argument('--m3d-root', default='/data/chenz/Gaussianflowocc_test/data/metric_3d_nusc')
    ap.add_argument('--sam-root', default='/data/chenz/Gaussianflowocc_test/data/grounded_sam_nusc')
    ap.add_argument('--n-frames', type=int, default=20, help='等间隔采样帧数')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    # 复用 gen_dynamic_gt 的默认参数
    from tools.gen_dynamic_gt import gen_one
    base = argparse.Namespace(
        scale=0.44, crop_top=140, render_h=256, render_w=704,
        dist_thresh=0.3, rmse_thresh=0.35, dt_budget=1.5, min_dt=0.15, max_dt=0.5,
        vote_ratio=0.3, min_dyn_cluster=15, strong_abs=0.45, max_dyn_range=40.0,
        static_radius=2, dyn_radius=3, flow_rescue_static=True,
    )

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
        scale=base.scale, crop_top=base.crop_top,
        render_h=base.render_h, render_w=base.render_w, res_thresh=3.0)
    print('detector ready\n', flush=True)

    # 等间隔采样
    step = max(1, len(keyframes) // args.n_frames)
    picks = list(range(0, len(keyframes), step))[:args.n_frames]

    agg = {'off': np.zeros(3), 'on': np.zeros(3)}
    n_lidar_fail = 0
    rows = []
    for gi in picks:
        scene_token, frame_idx = keyframes[gi]
        scene = scene_infos[scene_token]
        frame_idx = int(np.clip(frame_idx, 0, len(scene) - 1))
        token = scene[frame_idx]['token']

        out_off, st_off = gen_one(scene, frame_idx, args.data_root, base, detector=None)
        out_on, st_on = gen_one(scene, frame_idx, args.data_root, base, detector=detector)

        s_off = stats(out_off)
        s_on = stats(out_on)
        agg['off'] += s_off
        agg['on'] += s_on

        lidar_failed = ('badrmse' in st_on) or ('lidar=edge' in st_on)
        n_lidar_fail += int(lidar_failed)
        rows.append((token[:10], lidar_failed, s_off, s_on, st_on))
        print(f'{token[:10]} fail={lidar_failed} | '
              f'OFF ign/sta/dyn={s_off[0]:.3f}/{s_off[1]:.3f}/{s_off[2]:.3f} | '
              f'ON  ={s_on[0]:.3f}/{s_on[1]:.3f}/{s_on[2]:.3f} | {st_on}', flush=True)

    n = len(picks)
    print('\n===== 平均 (n={}) =====' .format(n))
    for k in ('off', 'on'):
        m = agg[k] / n
        print(f'{k:>3}: ignore={m[0]:.4f}  static={m[1]:.4f}  dynamic={m[2]:.4f}')
    print(f'LiDAR 整帧失败帧数: {n_lidar_fail}/{n}')

    # 只看 LiDAR 失败帧的 rescue 收益
    fail_rows = [r for r in rows if r[1]]
    if fail_rows:
        off_m = np.mean([r[2] for r in fail_rows], axis=0)
        on_m = np.mean([r[3] for r in fail_rows], axis=0)
        print(f'\n--- 仅 LiDAR 失败帧 (n={len(fail_rows)}) ---')
        print(f'off: ignore={off_m[0]:.4f} static={off_m[1]:.4f} dynamic={off_m[2]:.4f}')
        print(f' on: ignore={on_m[0]:.4f} static={on_m[1]:.4f} dynamic={on_m[2]:.4f}')


if __name__ == '__main__':
    main()
