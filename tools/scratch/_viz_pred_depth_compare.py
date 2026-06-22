"""Standalone: load trained 2D-depth model, forward ONE training batch, capture
the REAL rendered_depth tensor, then:
  1. print depth distribution stats (percentiles / std / coverage)
  2. render OLD colormap vs NEW colormap side-by-side per camera for comparison.

Run on a FREE GPU (does not touch the running training):
  CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python tools/_viz_pred_depth_compare.py \
      --py-config config/nuscenes_gs25600_2D.py \
      --ckpt out/nuscenes_gs25600_2D_depth/checkpoints/epoch_2.pth \
      --out /tmp/pred_depth_compare.jpg
"""
import argparse
import os
import numpy as np
import torch


# ---- OLD colormap (the one currently in render_loss.py before fix) ----
def old_cmap(d, vmax=40.0):
    norm = np.clip(d / vmax, 0.0, 1.0)
    r = np.clip(norm * 4 - 2, 0, 1)
    g = np.clip(np.minimum(norm * 4, 4 - norm * 4), 0, 1)
    b = np.clip(1 - norm * 4, 0, 1)
    rgb = np.stack([r, g, b], -1)
    rgb[d <= 0] = 0.5
    return (rgb * 255).astype(np.uint8)


# ---- NEW colormap (jet, vmax=30 to match ±30m Gaussian range) ----
def new_cmap(d, vmax=30.0):
    norm = np.clip(d / vmax, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4 * norm - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * norm - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * norm - 1), 0, 1)
    rgb = np.stack([r, g, b], -1)
    rgb[d <= 0] = 0.5
    return (rgb * 255).astype(np.uint8)


def label(img, txt, cv2):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(img, txt, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return img


def stats(name, d):
    valid = d[d > 0]
    if valid.size == 0:
        print(f'[{name}] NO valid (>0) pixels!')
        return
    pct = np.percentile(valid, [1, 5, 25, 50, 75, 95, 99])
    print(f'[{name}] valid={valid.size}/{d.size} ({100*valid.size/d.size:.1f}%) '
          f'min={valid.min():.2f} max={valid.max():.2f} mean={valid.mean():.2f} '
          f'std={valid.std():.2f}')
    print(f'        pct[1,5,25,50,75,95,99]m = '
          + ' '.join(f'{p:.1f}' for p in pct))
    # fraction in the old-colormap green plateau (10-30m)
    plateau = ((valid >= 10) & (valid <= 30)).mean()
    print(f'        fraction in 10-30m (old green plateau) = {100*plateau:.1f}%')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--py-config', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', default='/tmp/pred_depth_compare.jpg')
    ap.add_argument('--n-batches', type=int, default=1)
    args = ap.parse_args()

    import cv2
    from mmengine import Config
    from mmseg.models import build_segmentor
    import model  # noqa: register modules
    from dataset import get_dataloader

    cfg = Config.fromfile(args.py_config)

    # build model
    net = build_segmentor(cfg.model)
    net.init_weights()
    ckpt = torch.load(args.ckpt, map_location='cpu')
    sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    print(net.load_state_dict(sd, strict=False))
    net = net.cuda().eval()
    print(f'loaded {args.ckpt} (epoch={ckpt.get("epoch", "?")})')

    # build train dataloader (has pseudo labels + gs_extrins/gs_intrins)
    train_loader, _ = get_dataloader(
        cfg.train_dataset_config, cfg.val_dataset_config,
        cfg.train_loader, cfg.val_loader, dist=False, iter_resume=False)

    # IMPORTANT: rasterizer only runs when self.training=True; keep train mode
    # but disable grad. BN/dropout effect on rendered depth is negligible here.
    net.train()

    blocks = []
    all_pred, all_gt = [], []
    seen = 0
    with torch.no_grad():
        for data in train_loader:
            for k in list(data.keys()):
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].cuda()
            imgs = data.pop('img')
            out = net(imgs=imgs, metas=data, global_iter=0)
            rd = out.get('rendered_depth')
            if rd is None:
                print('rendered_depth is None — rasterizer did not run.')
                return
            pd = data['pseudo_depth']
            rd = rd[0].detach().cpu().numpy()   # (nC, H, W)
            pd = pd[0].detach().cpu().numpy()    # (nC, H, W)
            all_pred.append(rd)
            all_gt.append(pd)

            nC = rd.shape[0]
            for cam in range(nC):
                p = rd[cam]
                g = pd[cam]
                row = np.concatenate([
                    label(old_cmap(p, 40), f'cam{cam} PRED | OLD cmap(vmax40)', cv2),
                    label(old_cmap(g, 40), f'cam{cam} GT | OLD cmap(vmax40)', cv2),
                    label(new_cmap(p, 30), f'cam{cam} PRED | NEW jet(vmax30)', cv2),
                    label(new_cmap(g, 30), f'cam{cam} GT | NEW jet(vmax30)', cv2),
                ], axis=1)
                sep = np.full((3, row.shape[1], 3), 40, np.uint8)
                blocks.append(sep)
                blocks.append(row)
            seen += 1
            if seen >= args.n_batches:
                break

    pred = np.concatenate([a.reshape(-1) for a in all_pred])
    gt = np.concatenate([a.reshape(-1) for a in all_gt])
    print('\n================ DEPTH DISTRIBUTION ================')
    stats('PRED rendered_depth', pred)
    stats('GT pseudo_depth', gt)
    print('===================================================\n')

    canvas = np.concatenate(blocks, axis=0)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.out, canvas)
    print('saved', args.out, canvas.shape)


if __name__ == '__main__':
    main()
