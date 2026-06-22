"""临时：统计当前 dynamic LiDAR 主分支的整帧 ignore 率（不开光流，快）。

整帧 ignore = lidar_status 为 'edge' 或 'badrmse(...)'（整帧 6 相机全 0）。
对比旧 GT ~70% 整帧 ignore，确认新代码降到多少。不进 git。
"""
import argparse
import numpy as np
import mmengine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', default='data/nuscenes_cam/nuscenes_infos_train_gaussian_ad_v4.pkl')
    ap.add_argument('--data-root', default='data/nuscenes')
    ap.add_argument('--n-frames', type=int, default=300)
    args = ap.parse_args()

    from tools.gen_dynamic_gt import gen_one
    base = argparse.Namespace(
        scale=0.44, crop_top=140, render_h=256, render_w=704,
        dist_thresh=0.3, rmse_thresh=0.35, dt_budget=1.5, min_dt=0.15, max_dt=0.5,
        vote_ratio=0.3, min_dyn_cluster=15, strong_abs=0.45, max_dyn_range=40.0,
        static_radius=2, dyn_radius=3, flow_rescue_static=False,
    )

    print(f'loading {args.pkl} ...', flush=True)
    data = mmengine.load(args.pkl)
    scene_infos = data['infos']
    keyframes = data['metadata']

    step = max(1, len(keyframes) // args.n_frames)
    picks = list(range(0, len(keyframes), step))[:args.n_frames]

    n_edge = n_badrmse = n_ok = 0
    agg = np.zeros(3)
    for k, gi in enumerate(picks):
        scene_token, frame_idx = keyframes[gi]
        scene = scene_infos[scene_token]
        frame_idx = int(np.clip(frame_idx, 0, len(scene) - 1))
        out, status = gen_one(scene, frame_idx, args.data_root, base, detector=None)
        if 'lidar=edge' in status:
            n_edge += 1
        elif 'badrmse' in status:
            n_badrmse += 1
        else:
            n_ok += 1
        tot = out.size
        agg += [(out == 0).sum() / tot, (out == 1).sum() / tot, (out == 2).sum() / tot]
        if k % 50 == 0:
            print(f'  [{k}/{len(picks)}] {status}', flush=True)

    n = len(picks)
    print(f'\n===== LiDAR-only 整帧 ignore 统计 (n={n}) =====')
    print(f'lidar=ok     : {n_ok:4d}  ({n_ok/n:.1%})')
    print(f'lidar=edge   : {n_edge:4d}  ({n_edge/n:.1%})')
    print(f'lidar=badrmse: {n_badrmse:4d}  ({n_badrmse/n:.1%})')
    print(f'整帧 ignore(edge+badrmse): {(n_edge+n_badrmse)/n:.1%}')
    m = agg / n
    print(f'平均像素占比: ignore={m[0]:.4f} static={m[1]:.4f} dynamic={m[2]:.4f}')


if __name__ == '__main__':
    main()
