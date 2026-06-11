"""
4-column comparison visualizer to inspect 2D pseudo-label quality vs OCC GT.

For a small batch of samples, each camera view is shown as a row with 4 columns:
    col 1: OCC GT (3D occupancy semantics projected onto the camera view,
            drawn over a dimmed original image for context)
    col 2: 2D semantic pseudo-label  (grounded_sam, colorized)
    col 3: 2D depth pseudo-label     (metric_3d, colorized)
    col 4: original RGB image (aligned to the pseudo-label resolution)

Purpose: spot semantic conflicts between the OCC GT and the 2D semantic
pseudo-label on specific objects.

All 4 columns share the same camera intrinsics/extrinsics and spatial layout
(render resolution = original * pseudo_label_scale, top-cropped), so pixels
are directly comparable.

Run on the H20 server (data lives there):
    /data/chenz/conda_env/splatting/bin/python tools/vis_pseudo_vs_occ.py \
        --config config/nuscenes_gs25600_2D.py \
        --num-samples 8 --out-dir out/pseudo_vs_occ
"""
import os
import argparse

import numpy as np
import cv2
import torch
from mmengine import Config

import dataset  # noqa: F401  (registers NuScenesDataset into OPENOCC_DATASET)
from dataset import OPENOCC_DATASET


# ── nuScenes occupancy palette (RGB), 0-indexed for occ label 1..16 ──────────
# index = label - 1  (label 1=barrier ... 16=vegetation; 0=noise, 17=empty)
_NUSC_PALETTE = np.array([
    [112, 128, 144],  # 0: barrier
    [220,  20,  60],  # 1: bicycle
    [255, 127,  80],  # 2: bus
    [255, 158,   0],  # 3: car
    [233, 150,  70],  # 4: construction_vehicle
    [255,  61,  99],  # 5: motorcycle
    [  0,   0, 230],  # 6: pedestrian
    [ 47,  79,  79],  # 7: traffic_cone
    [255, 140,   0],  # 8: trailer
    [255,  99,  71],  # 9: truck
    [  0, 207, 191],  # 10: driveable_surface
    [175,   0,  75],  # 11: other_flat
    [ 75,   0,  75],  # 12: sidewalk
    [112, 180,  60],  # 13: terrain
    [222, 184, 135],  # 14: manmade
    [  0, 175,   0],  # 15: vegetation
    [  0,   0,   0],  # 16: free/empty
], dtype=np.uint8)

_CLASS_NAMES = [
    'barrier', 'bicycle', 'bus', 'car', 'constr_veh', 'motorcycle',
    'pedestrian', 'traffic_cone', 'trailer', 'truck', 'driveable',
    'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation',
]


def colorize_sem(cls_map):
    """cls_map: (H, W) int in [0,16] (pseudo_seg/occ label) -> RGB (H,W,3).

    label 0 (invalid/sky/background) -> gray; label 1..16 -> palette[label-1].
    """
    out = np.full((*cls_map.shape, 3), 80, dtype=np.uint8)  # gray bg
    valid = cls_map >= 1
    idx = np.clip(cls_map[valid] - 1, 0, 15)
    out[valid] = _NUSC_PALETTE[idx]
    return out


def depth_to_rgb(depth, vmin=0.0, vmax=40.0):
    """depth: (H,W) float -> RGB heatmap; invalid (<=0) -> gray."""
    norm = np.clip((depth - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)
    r = np.clip(norm * 4 - 2, 0, 1)
    g = np.clip(np.minimum(norm * 4, 4 - norm * 4), 0, 1)
    b = np.clip(1 - norm * 4, 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    rgb[depth <= 0] = 0.5
    return (rgb * 255).astype(np.uint8)


def to_numpy(x):
    """Unwrap mmcv DataContainer / torch.Tensor -> np.ndarray."""
    if hasattr(x, 'data'):
        x = x.data
    if isinstance(x, (list, tuple)):
        x = x[0]
        if hasattr(x, 'data'):
            x = x.data
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def project_occ(occ_xyz, occ_label, lidar2cam, K, H, W, base_img=None,
                radius=3, highlight_cls=None):
    """Project occupied voxels onto one camera view.

    occ_xyz: (N,3) LIDAR-frame voxel centers
    occ_label: (N,) int, 1..16 occupied (0 noise, 17 empty already removed)
    lidar2cam: (4,4)  K: (3,3)
    base_img: (H,W,3) uint8 RGB to draw on (dimmed); if None use black.
    radius: point radius in pixels.
    highlight_cls: if given (e.g. 15=manmade), draw those points larger so the
        class (e.g. poles) is easy to spot.
    """
    if base_img is not None:
        canvas = (base_img.astype(np.float32) * 0.35).astype(np.uint8)
    else:
        canvas = np.zeros((H, W, 3), dtype=np.uint8)

    if occ_xyz.shape[0] == 0:
        return canvas

    pts_h = np.concatenate([occ_xyz, np.ones((occ_xyz.shape[0], 1))], axis=1)  # (N,4)
    cam = (lidar2cam @ pts_h.T).T[:, :3]  # (N,3)
    z = cam[:, 2]
    front = z > 0.1
    cam, lbl, z = cam[front], occ_label[front], z[front]
    if cam.shape[0] == 0:
        return canvas

    uv = (K @ cam.T).T  # (N,3)
    u = uv[:, 0] / uv[:, 2]
    v = uv[:, 1] / uv[:, 2]
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, lbl, z = u[inb].astype(np.int32), v[inb].astype(np.int32), lbl[inb], z[inb]
    if u.shape[0] == 0:
        return canvas

    # z-buffer: draw far points first so near points overwrite
    order = np.argsort(-z)
    u, v, lbl = u[order], v[order], lbl[order]
    colors = _NUSC_PALETTE[np.clip(lbl - 1, 0, 15)]
    for ui, vi, col, cl in zip(u, v, colors, lbl):
        r = radius + 3 if (highlight_cls is not None and int(cl) == highlight_cls) else radius
        cv2.circle(canvas, (int(ui), int(vi)), r,
                   (int(col[0]), int(col[1]), int(col[2])), -1)
    return canvas


def put_label(img, text, color=(255, 255, 255)):
    cv2.putText(img, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                color, 1, cv2.LINE_AA)
    return img


def make_legend(width):
    """Horizontal class color legend (incl. invalid(0))."""
    sw = 18
    bar_h = 26
    names = ['invalid(0)'] + _CLASS_NAMES
    cols = [np.array([80, 80, 80], dtype=np.uint8)] + [c for c in _NUSC_PALETTE[:16]]
    n = len(names)
    cell_w = max(width // n, 90)
    legend = np.full((bar_h, cell_w * n, 3), 30, dtype=np.uint8)
    for i, (name, col) in enumerate(zip(names, cols)):
        x0 = i * cell_w
        legend[4:4 + sw, x0 + 4:x0 + 4 + sw] = col
        cv2.putText(legend, name, (x0 + 4 + sw + 3, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1, cv2.LINE_AA)
    if legend.shape[1] != width:
        legend = cv2.resize(legend, (width, bar_h), interpolation=cv2.INTER_AREA)
    return legend


def build_sample_image(dataset_obj, idx, disp_w=480, highlight_cls=None):
    data = dataset_obj[idx]

    pseudo_seg = to_numpy(data['pseudo_seg'])      # (6, H, W)
    pseudo_depth = to_numpy(data['pseudo_depth'])  # (6, H, W)
    gs_intrins = to_numpy(data['gs_intrins'])      # (6, 3, 3)
    gs_extrins = to_numpy(data['gs_extrins'])      # (6, 4, 4) lidar2cam
    occ_xyz = to_numpy(data['occ_xyz']).reshape(-1, 3)
    occ_label = to_numpy(data['occ_label']).reshape(-1).astype(np.int64)

    # keep only occupied semantic voxels (1..16); drop noise(0) and empty(17)
    occ_keep = (occ_label >= 1) & (occ_label <= 16)
    occ_xyz, occ_label = occ_xyz[occ_keep], occ_label[occ_keep]

    n_cam = pseudo_seg.shape[0]
    H, W = pseudo_seg.shape[1], pseudo_seg.shape[2]

    # resolve original image paths + crop info
    scene_token, sub_index = dataset_obj.keyframes[idx]
    info = dataset_obj.scene_infos[scene_token][sub_index]
    scale = dataset_obj.pseudo_label_scale
    crop_top = dataset_obj.pseudo_label_crop_top
    full_h = H + crop_top  # render height before top-crop
    max_d = dataset_obj.max_pseudo_depth

    rows = []
    for c in range(n_cam):
        cam_type = dataset_obj.sensor_types[c]
        # original image -> resize to render res -> top crop -> (H, W, 3) RGB
        img_path = os.path.join(dataset_obj.data_path, info['data'][cam_type]['filename'])
        bgr = cv2.imread(img_path)
        if bgr is None:
            img_rgb = np.zeros((H, W, 3), dtype=np.uint8)
        else:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (W, full_h), interpolation=cv2.INTER_AREA)
            img_rgb = rgb[crop_top:crop_top + H]

        # col 1: OCC GT projected over dimmed image (for localization)
        col_occ = project_occ(occ_xyz, occ_label, gs_extrins[c], gs_intrins[c],
                              H, W, base_img=img_rgb, radius=3,
                              highlight_cls=highlight_cls)
        # col 2: OCC GT projected on black (pure semantics, colors readable)
        col_occ_pure = project_occ(occ_xyz, occ_label, gs_extrins[c], gs_intrins[c],
                                  H, W, base_img=None, radius=3,
                                  highlight_cls=highlight_cls)
        col_sem = colorize_sem(pseudo_seg[c].astype(np.int64))
        col_dep = depth_to_rgb(pseudo_depth[c], vmin=0.0, vmax=max_d)
        col_img = img_rgb

        cells = [col_occ, col_occ_pure, col_sem, col_dep, col_img]
        disp_h = int(round(H * disp_w / W))
        cells = [cv2.resize(x, (disp_w, disp_h), interpolation=cv2.INTER_AREA) for x in cells]

        titles = ['OCC GT (on img)', 'OCC GT (pure sem)', '2D sem pseudo',
                  '2D depth pseudo', 'image']
        for j, (cell, t) in enumerate(zip(cells, titles)):
            tag = f'{cam_type} | {t}' if j == 0 else t
            put_label(cell, tag)
        rows.append(np.concatenate(cells, axis=1))  # horizontal

    grid = np.concatenate(rows, axis=0)  # vertical stack of cameras

    legend = make_legend(grid.shape[1])
    out = np.concatenate([grid, legend], axis=0)
    return out, info['token']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config/nuscenes_gs25600_2D.py')
    ap.add_argument('--out-dir', default='out/pseudo_vs_occ')
    ap.add_argument('--num-samples', type=int, default=20)
    ap.add_argument('--indices', type=int, nargs='*', default=None,
                    help='explicit dataset indices; overrides --num-samples')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--disp-w', type=int, default=480)
    ap.add_argument('--highlight-class', type=int, default=None,
                    help='OCC class id (1-16) to draw larger, e.g. 15=manmade '
                         '(poles/buildings/signs) to spot thin structures')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = Config.fromfile(args.config)
    ds_cfg = cfg.train_dataset_config  # only train cfg carries pseudo-label paths
    print('[build] constructing dataset (this loads nuScenes infos)...')
    ds = OPENOCC_DATASET.build(ds_cfg)
    assert ds.use_pseudo_label, 'dataset built without pseudo labels; check config'
    n = len(ds)
    print(f'[build] dataset ready, {n} samples, use_pseudo_label={ds.use_pseudo_label}')

    if args.indices:
        indices = args.indices
    else:
        rng = np.random.default_rng(args.seed)
        indices = sorted(rng.choice(n, size=min(args.num_samples, n), replace=False).tolist())
    print(f'[run] visualizing indices: {indices}')

    for i in indices:
        try:
            img, token = build_sample_image(ds, i, disp_w=args.disp_w,
                                            highlight_cls=args.highlight_class)
        except Exception as e:  # noqa: BLE001
            print(f'[skip] idx {i} failed: {e}')
            continue
        out_path = os.path.join(args.out_dir, f'sample_{i:06d}_{token}.jpg')
        cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f'[saved] {out_path}  ({img.shape[1]}x{img.shape[0]})')

    print('[done]')


if __name__ == '__main__':
    main()
