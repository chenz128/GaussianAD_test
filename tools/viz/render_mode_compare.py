"""
render_mode_compare.py — 对比 RGB+D vs RGB+ED 渲染效果，不影响训练。
每张结果图列顺序: [GT_sem | GT_depth | sem(D) | sem(ED) | dep(D) | dep(ED) | |ED-D|]

运行（远端 GPU 6，不占用训练 GPU 0-5）:
    CUDA_VISIBLE_DEVICES=6 /data/chenz/conda_env/splatting/bin/python \\
        tools/viz/render_mode_compare.py \\
        --py-config config/nuscenes_gs25600_2D.py \\
        --work-dir out/nuscenes_gs25600_2D \\
        --out-dir /tmp/render_compare \\
        --num-samples 3
"""
import argparse, os, sys
import numpy as np
import torch
from PIL import Image

# file now lives in tools/viz/, go up 3 levels to reach repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ── colormaps ─────────────────────────────────────────────────────────────────

_NUSC_PALETTE = np.array([
    [  0,   0,   0],   # 0: invalid / free
    [112, 128, 144],   # 1: barrier
    [220,  20,  60],   # 2: bicycle
    [255, 127,  80],   # 3: bus
    [255, 158,   0],   # 4: car
    [233, 150,  70],   # 5: construction_vehicle
    [255,  61,  99],   # 6: motorcycle
    [  0,   0, 230],   # 7: pedestrian
    [ 47,  79,  79],   # 8: traffic_cone
    [255, 140,   0],   # 9: trailer
    [255,  99,  71],   # 10: truck
    [  0, 207, 191],   # 11: driveable_surface
    [175,   0,  75],   # 12: other_flat
    [ 75,   0,  75],   # 13: sidewalk
    [112, 180,  60],   # 14: terrain
    [222, 184, 135],   # 15: manmade
    [  0, 175,   0],   # 16: vegetation
], dtype=np.uint8)


def colorize_sem(m):
    """m: (H,W) int 0..16 -> (H,W,3) uint8"""
    return _NUSC_PALETTE[np.clip(m, 0, 16)]


def depth_to_rgb(d, vmin=0., vmax=40.):
    """d: (H,W) float -> (H,W,3) uint8; invalid (d<=0) -> gray"""
    norm = np.clip((d - vmin) / (vmax - vmin + 1e-6), 0., 1.)
    r = np.clip(norm * 4 - 2,               0, 1)
    g = np.clip(np.minimum(norm*4, 4-norm*4), 0, 1)
    b = np.clip(1 - norm * 4,               0, 1)
    rgb = (np.stack([r, g, b], -1) * 255).astype(np.uint8)
    rgb[d <= 0] = 128
    return rgb


def diff_to_rgb(diff):
    """diff: (H,W) absolute difference -> heat map (H,W,3) uint8"""
    n = np.clip(diff / (diff.max() + 1e-6), 0, 1)
    r = np.clip(n * 3 - 1,               0, 1)
    g = np.clip(np.minimum(n*3, 3-n*3),  0, 1)
    b = np.clip(1 - n * 3,               0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


# ── argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--py-config',   required=True)
    p.add_argument('--work-dir',    required=True)
    p.add_argument('--out-dir',     default='/tmp/render_compare')
    p.add_argument('--num-samples', type=int, default=3)
    p.add_argument('--resume-from', default='')
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from mmengine import Config
    from mmengine.runner import set_random_seed
    from mmengine.logging import MMLogger
    from mmseg.models import build_segmentor

    set_random_seed(0)
    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = args.work_dir

    log_file = os.path.join(args.out_dir, 'render_compare.log')
    logger = MMLogger('render_compare', log_file=log_file)
    MMLogger._instance_dict['render_compare'] = logger

    # ── build model ───────────────────────────────────────────────────────────
    import model as _m  # noqa: registers all model modules
    my_model = build_segmentor(cfg.model).cuda().eval()
    print(f'Model params: {sum(p.numel() for p in my_model.parameters()):,}')

    # ── load checkpoint ───────────────────────────────────────────────────────
    ckpt_path = args.resume_from or os.path.join(args.work_dir, 'latest.pth')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        my_model.load_state_dict(ckpt['state_dict'], strict=True)
        print(f'Loaded {ckpt_path}  epoch={ckpt.get("epoch","?")}')
    else:
        print(f'WARNING: no checkpoint at {ckpt_path}')

    # ── dataloader: train split (has gs_extrins/gs_intrins/pseudo labels) ────
    from dataset import get_dataloader

    lc = dict(batch_size=1, num_workers=2, shuffle=False)
    train_loader, _ = get_dataloader(
        cfg.train_dataset_config, cfg.val_dataset_config,
        lc, lc, dist=False, val_only=False,
    )

    # ── rasterizer ────────────────────────────────────────────────────────────
    rasterizer = my_model.head.rasterizer_2d
    if rasterizer is None:
        print('ERROR: rasterizer_2d is None — build model with render_config')
        return
    H, W = rasterizer.height, rasterizer.width
    print(f'Render resolution: {W}x{H}')

    from gsplat import rasterization

    # ── iterate samples ───────────────────────────────────────────────────────
    for idx, data in enumerate(train_loader):
        if idx >= args.num_samples:
            break
        print(f'\n=== Sample {idx} ===')

        for k in list(data):
            if isinstance(data[k], torch.Tensor):
                data[k] = data[k].cuda()
        imgs = data.pop('img')  # (B, F, N, C, H, W)

        with torch.no_grad():
            result = my_model(imgs=imgs, metas=data)

        g        = result['gaussian']
        extrins  = data['gs_extrins'][0]   # (nC, 4, 4)
        intrins  = data['gs_intrins'][0]   # (nC, 3, 3)
        ps_seg   = data.get('pseudo_seg')
        ps_depth = data.get('pseudo_depth')

        print(f'  Gaussians={g.means.shape[1]}  cameras={extrins.shape[0]}')
        print(f'  Scale  min={g.scales[0].min():.4f}  max={g.scales[0].max():.4f}  mean={g.scales[0].mean():.4f}')
        print(f'  Opacity min={g.opacities[0].min():.4f}  max={g.opacities[0].max():.4f}  mean={g.opacities[0].mean():.4f}')

        kw = dict(
            means    = g.means[0],
            quats    = g.rotations[0],
            scales   = g.scales[0],
            opacities= g.opacities[0, :, 0],
            colors   = g.semantics_logits[0],
            viewmats = extrins,
            Ks       = intrins,
            width=W, height=H,
        )
        with torch.no_grad():
            oD,  _, _ = rasterization(**kw, render_mode='RGB+D')
            oED, _, _ = rasterization(**kw, render_mode='RGB+ED')

        sD  = oD[..., :17].cpu().numpy()   # (nC, H, W, 17)
        dD  = oD[..., 17].cpu().numpy()    # (nC, H, W)
        sED = oED[..., :17].cpu().numpy()
        dED = oED[..., 17].cpu().numpy()

        pss = ps_seg[0].cpu().numpy()   if ps_seg   is not None else None
        psd = ps_depth[0].cpu().numpy() if ps_depth is not None else None

        rows = []
        for cam in range(extrins.shape[0]):
            diff = np.abs(dED[cam] - dD[cam])
            cols = [
                colorize_sem(np.argmax(sD[cam],  -1)),   # sem(D)
                colorize_sem(np.argmax(sED[cam], -1)),   # sem(ED)
                depth_to_rgb(dD[cam]),                    # dep(D)
                depth_to_rgb(dED[cam]),                   # dep(ED)
                diff_to_rgb(diff),                        # |ED-D|
            ]
            if pss is not None:
                cols = [colorize_sem(pss[cam]), depth_to_rgb(psd[cam])] + cols

            rows.append(np.concatenate(cols, axis=1))

            d1   = dD[cam][dD[cam]>0].mean()   if (dD[cam]>0).any()   else 0.
            d2   = dED[cam][dED[cam]>0].mean() if (dED[cam]>0).any() else 0.
            dmn  = diff[diff>0.01].mean()       if (diff>0.01).any()  else 0.
            cov  = (dD[cam] > 0).mean()
            print(f'  cam{cam}: dep(D)={d1:.2f}m dep(ED)={d2:.2f}m |diff|={dmn:.3f}m coverage={cov:.1%}')

        grid    = np.concatenate(rows, axis=0)
        out_pth = os.path.join(args.out_dir, f'sample_{idx:02d}.jpg')
        Image.fromarray(grid).save(out_pth, quality=92)
        print(f'  Saved: {out_pth}  size={grid.shape[1]}x{grid.shape[0]}')

    print(f'\nAll done. Results: {args.out_dir}')


if __name__ == '__main__':
    main()
