"""
Visualize nuScenes frames in the SAME layout as the reference figure:

  Inputs (left, 2x3)      : 6 surround cameras with corner labels
      [Front Left, Front, Front Right]
      [Back Left,  Back,  Back Right]
  Predictions (middle)    : BEV semantic occupancy (pred) + predicted 3D
                            detection boxes + predicted map vectors + planned
                            ego trajectory (pred vs GT).
  Ground Truth (right)    : BEV semantic occupancy (GT occ_label) + GT boxes +
                            GT map vectors + GT ego trajectory.

Everything comes from the project's own dataloader + model forward pass.

Usage:
  python viz/visualize_bev.py \
      --py-config config/nuscenes_gs25600_base_plan/nuscenes_gs25600_base_plan.py \
      --ckpt exp/nuscenes_gs25600_base_plan/checkpoints/epoch_15.pth \
      --out-dir viz/out --vis-index 0
"""
import os, sys, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ---------------------------------------------------------------------------
# nuScenes semantic occupancy palette (classic SurroundOcc colours).
# The model's raw argmax class indexes THIS table directly:
#   class 0..16 -> semantic colour ; class >=17 -> free / empty (white).
# ---------------------------------------------------------------------------
NUSCENES_CMAP = np.array([
    [  0,   0,   0],   # 0  others
    [255, 120,  50],   # 1  barrier
    [255, 192, 203],   # 2  bicycle
    [255, 255,   0],   # 3  bus
    [  0, 150, 245],   # 4  car
    [  0, 255, 255],   # 5  construction_vehicle
    [255, 127,   0],   # 6  motorcycle
    [255,   0,   0],   # 7  pedestrian
    [255, 240, 150],   # 8  traffic_cone
    [135,  60,   0],   # 9  trailer
    [160,  32, 240],   # 10 truck
    [255,   0, 255],   # 11 driveable_surface
    [139, 137, 137],   # 12 other_flat
    [ 75,   0,  75],   # 13 sidewalk
    [150, 240,  80],   # 14 terrain
    [230, 230, 250],   # 15 manmade
    [  0, 175,   0],   # 16 vegetation
], dtype=np.uint8)
FREE_CLASS = 17                      # occ free / empty
WHITE = np.array([255, 255, 255], dtype=np.uint8)

# map vector palette (MapTR: 0 divider, 1 ped_crossing, 2 boundary)
MAP_COLORS = {0: '#ff8c00', 1: '#0066ff', 2: '#00a000'}
MAP_NAMES = {0: 'divider', 1: 'ped_crossing', 2: 'boundary'}

# detection box palette (indexed by predicted/GT label, wraps around)
DET_PALETTE = ['#00e5ff', '#ffd400', '#ff4081', '#7c4dff', '#64dd17',
               '#ff6d00', '#00b8d4', '#c51162', '#aeea00', '#6200ea']

# camera display order matching the reference figure
CAM_ORDER = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT',  'CAM_BACK',  'CAM_BACK_RIGHT']
CAM_LABEL = {'CAM_FRONT_LEFT': 'Front Left', 'CAM_FRONT': 'Front',
             'CAM_FRONT_RIGHT': 'Front Right', 'CAM_BACK_LEFT': 'Back Left',
             'CAM_BACK': 'Back', 'CAM_BACK_RIGHT': 'Back Right'}
# dataset sensor order (order of img_metas['filename'])
DS_ORDER = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

# BEV grid (model/head/gaussian_head.py): pc_range +-30 m, 0.5 m cells.
PC = [-30.0, -30.0, -2.0, 30.0, 30.0, 2.0]
GRID = 120
CELL = 0.5
X_MAX, Y_MAX = PC[3], PC[4]


# ---------------------------------------------------------------------------
# coordinate helpers : lidar/ego (x forward, y left) -> BEV pixels
#   image row 0 = top = far ahead (+x) ; col 0 = left (+y)
# ---------------------------------------------------------------------------
def to_px(x, y):
    x = np.asarray(x); y = np.asarray(y)
    col = (Y_MAX - y) / CELL
    row = (X_MAX - x) / CELL
    return col, row


# ---------------------------------------------------------------------------
# cameras
# ---------------------------------------------------------------------------
def _cam_filenames(data):
    metas = data['img_metas'][0]
    inner = metas.data if hasattr(metas, 'data') else metas
    if isinstance(inner, list):
        inner = inner[0]
    if isinstance(inner, dict) and 'filename' in inner:
        return list(inner['filename'])
    return None


def load_cam_images(data):
    """Return dict cam_name -> RGB uint8, in CAM_ORDER. Paths in img_metas
    are already workspace-relative (data/nuscenes/...)."""
    imgs = {}
    fns = _cam_filenames(data)
    if fns:
        for name, fn in zip(DS_ORDER, fns):
            p = fn if os.path.isabs(fn) else os.path.join(ROOT, fn)
            if os.path.exists(p):
                imgs[name] = np.asarray(Image.open(p).convert('RGB'))
    return {c: imgs[c] for c in CAM_ORDER if c in imgs}


# ---------------------------------------------------------------------------
# occupancy -> BEV class map.
# BOTH pred and GT are dense voxel grids (115200 = 120x120x8), x-major:
#   index = ((x*120) + y)*8 + z   (axis0=x, axis1=y, axis2=z)
# So we take the topmost (highest z) non-free class per (x,y) column directly
# from the dense grid. No scatter => no truncation noise / no mosaic fragments.
def _dense_topmost(cls8):
    """cls8 (GRID,GRID,8) int class ids -> (GRID,GRID) topmost non-free map."""
    bev = np.full((GRID, GRID), FREE_CLASS, dtype=np.int32)
    for z in range(8):
        layer = cls8[:, :, z]
        mask = (layer > 0) & (bev == FREE_CLASS)
        bev[mask] = layer[mask]
    return bev


def occ_pred_to_bev(pred_occ_t, sampled_xyz):
    """pred_occ_t (C, N) logits, N = GRID*GRID*8 (x-major). sampled_xyz only
    used for shape sanity; the grid itself is dense."""
    cls = pred_occ_t.argmax(dim=0).detach().cpu().numpy().astype(np.int32)
    if cls.size != GRID * GRID * 8:
        raise ValueError(f"pred occ size {cls.size} != {GRID*GRID*8} -> not a dense grid")
    return _dense_topmost(cls.reshape(GRID, GRID, 8))


def occ_gt_to_bev(occ_label, occ_xyz):
    """occ_label (H,W,D) int, occ_xyz (H,W,D,3), x-major dense grid."""
    ol = occ_label.detach().cpu().numpy() if torch.is_tensor(occ_label) else np.asarray(occ_label)
    if ol.ndim == 4:
        ol = ol[0]
    if ol.shape != (GRID, GRID, 8):
        raise ValueError(f"GT occ shape {ol.shape} != {(GRID,GRID,8)}")
    return _dense_topmost(ol.astype(np.int32))


def bev_to_rgb(bev):
    """(GRID,GRID) class map -> RGB uint8. class>=FREE -> white."""
    rgb = np.empty((GRID, GRID, 3), dtype=np.uint8)
    free = bev >= FREE_CLASS
    idx = np.clip(bev, 0, len(NUSCENES_CMAP) - 1)
    rgb[:] = NUSCENES_CMAP[idx]
    rgb[free] = WHITE
    return rgb


# overlays : detection boxes, map vectors, trajectories
# ---------------------------------------------------------------------------
def draw_boxes(ax, boxes, labels=None, lw=1.4, default='#111111'):
    """boxes (N,>=7): x,y,z,dx,dy,dz,yaw(,...) in lidar/ego frame."""
    if boxes is None or len(boxes) == 0:
        return
    b = boxes.detach().cpu().numpy() if torch.is_tensor(boxes) else np.asarray(boxes)
    if b.ndim == 3:
        b = b[0]
    for i in range(b.shape[0]):
        x, y, dx, dy, yaw = b[i, 0], b[i, 1], b[i, 3], b[i, 4], b[i, 6]
        corners = np.array([[ dx/2,  dy/2], [ dx/2, -dy/2],
                            [-dx/2, -dy/2], [-dx/2,  dy/2], [dx/2, dy/2]])
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s], [s, c]])
        pts = corners @ R.T + np.array([x, y])
        col, row = to_px(pts[:, 0], pts[:, 1])
        color = default
        if labels is not None and i < len(labels):
            color = DET_PALETTE[int(labels[i]) % len(DET_PALETTE)]
        ax.plot(col, row, color=color, lw=lw, zorder=5)
        # heading tick
        hx, hy = to_px(x + c * dx / 2, y + s * dx / 2)
        cx, cy = to_px(x, y)
        ax.plot([cx, hx], [cy, hy], color=color, lw=lw, zorder=5)


def draw_map(ax, polylines, labels, lw=2.2):
    """polylines: list of (P,2) arrays in lidar/ego frame ; labels: cls idx."""
    for poly, lab in zip(polylines, labels):
        poly = np.asarray(poly)
        col, row = to_px(poly[:, 0], poly[:, 1])
        ax.plot(col, row, color=MAP_COLORS.get(int(lab), '#888888'),
                lw=lw, zorder=4, solid_capstyle='round')


def draw_traj(ax, pts, color, lw=2.6, ls='-', label=None):
    if pts is None or len(pts) < 2:
        return
    pts = np.asarray(pts)
    col, row = to_px(pts[:, 0], pts[:, 1])
    ax.plot(col, row, color=color, lw=lw, ls=ls, zorder=8, label=label,
            solid_capstyle='round')
    ax.scatter(col[-1], row[-1], color=color, s=28, zorder=9)


def draw_ego(ax, L=4.084, W=1.85):
    corners = np.array([[ L/2,  W/2], [ L/2, -W/2], [-L/2, -W/2],
                        [-L/2,  W/2], [L/2, W/2]])
    col, row = to_px(corners[:, 0], corners[:, 1])
    ax.plot(col, row, color='lime', lw=2.0, zorder=10)


def style_bev(ax, title, edge):
    ax.set_xlim(0, GRID); ax.set_ylim(GRID, 0)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')
    for sp in ax.spines.values():
        sp.set_edgecolor(edge); sp.set_linewidth(2.5)
    ax.set_title(title, fontsize=15, fontweight='bold')


# ---------------------------------------------------------------------------
# ego trajectory extraction (per-step deltas -> cumulative)
# ---------------------------------------------------------------------------
def cum_traj(deltas):
    d = deltas.detach().cpu().numpy() if torch.is_tensor(deltas) else np.asarray(deltas)
    d = np.nan_to_num(d, nan=0.0).reshape(-1, 2)
    traj = np.cumsum(d, axis=0)
    return np.concatenate([np.zeros((1, 2)), traj], axis=0)


# ---------------------------------------------------------------------------
# map GT / pred extraction
# ---------------------------------------------------------------------------
def get_gt_map(data):
    if 'gt_bboxes_3d' not in data:
        return [], []
    gb = data['gt_bboxes_3d'][0]
    el = gb.data if hasattr(gb, 'data') else gb
    polys, labs = [], []
    if hasattr(el, 'fixed_num_sampled_points'):
        pts = el.fixed_num_sampled_points
        pts = pts.detach().cpu().numpy() if torch.is_tensor(pts) else np.asarray(pts)
        lb = data.get('gt_labels_3d')
        lb = (lb[0].detach().cpu().numpy() if torch.is_tensor(lb) else np.asarray(lb[0])) if lb is not None else np.zeros(len(pts))
        for i in range(len(pts)):
            polys.append(pts[i])
            labs.append(int(lb[i]) if i < len(lb) else 0)
    return polys, labs


def get_pred_map(res, score_thr=0.3):
    if 'all_pts_preds' not in res or res['all_pts_preds'] is None:
        return [], []
    pts = res['all_pts_preds'][-1, 0].detach().cpu().numpy()      # (M,P,2) in [0,1]
    scr = res['all_cls_scores'][-1, 0].sigmoid().detach().cpu().numpy()  # (M,C)
    conf = scr.max(axis=-1); cls = scr.argmax(axis=-1)
    polys, labs = [], []
    for i in range(pts.shape[0]):
        if conf[i] < score_thr:
            continue
        p = pts[i].copy()
        p[:, 0] = p[:, 0] * (PC[3] - PC[0]) + PC[0]
        p[:, 1] = p[:, 1] * (PC[4] - PC[1]) + PC[1]
        polys.append(p); labs.append(int(cls[i]))
    return polys, labs


# ---------------------------------------------------------------------------
# figure composition
# ---------------------------------------------------------------------------
def compose(cam_imgs, pred, gt, save_path, title=None):
    fig = plt.figure(figsize=(26, 9), dpi=120)
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1.45, 1.45],
                          wspace=0.05, hspace=0.07)

    positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    for (r, c), name in zip(positions, CAM_ORDER):
        ax = fig.add_subplot(gs[r, c])
        if name in cam_imgs:
            ax.imshow(cam_imgs[name])
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor('#444'); sp.set_linewidth(1.2)
        ax.text(0.02, 0.95, CAM_LABEL[name], transform=ax.transAxes,
                fontsize=12, fontweight='bold', color='#ffd54f',
                va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.45, ec='none'))

    # Predictions panel
    axp = fig.add_subplot(gs[:, 3])
    axp.imshow(bev_to_rgb(pred['occ']), extent=[0, GRID, GRID, 0],
               origin='upper', interpolation='nearest', zorder=1)
    draw_map(axp, pred['map'][0], pred['map'][1])
    draw_boxes(axp, pred['boxes'], pred['labels'])
    draw_ego(axp)
    draw_traj(axp, pred['ego_gt'],  '#1e88e5', ls='-',  label='GT ego')
    draw_traj(axp, pred['ego_pred'], '#e53935', ls='--', label='Pred ego')
    style_bev(axp, 'Predictions', '#b25000')
    axp.legend(loc='lower right', fontsize=9, framealpha=0.75)

    # Ground Truth panel
    axg = fig.add_subplot(gs[:, 4])
    axg.imshow(bev_to_rgb(gt['occ']), extent=[0, GRID, GRID, 0],
               origin='upper', interpolation='nearest', zorder=1)
    draw_map(axg, gt['map'][0], gt['map'][1])
    draw_boxes(axg, gt['boxes'], gt['labels'])
    draw_ego(axg)
    draw_traj(axg, gt['ego_gt'], '#1e88e5', ls='-', label='GT ego')
    style_bev(axg, 'Ground Truth', '#b00020')
    axg.legend(loc='lower right', fontsize=9, framealpha=0.75)

    if title:
        fig.suptitle(title, fontsize=15)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.12, facecolor='white')
    plt.close(fig)
    print(f'saved -> {save_path}', flush=True)


# ---------------------------------------------------------------------------
def build_model(cfg, ckpt):
    import model  # noqa: register
    from mmseg.models import build_segmentor
    m = build_segmentor(cfg.model)
    if ckpt and os.path.exists(ckpt):
        ck = torch.load(ckpt, map_location='cpu')
        sd = ck['state_dict'] if 'state_dict' in ck else ck
        msg = m.load_state_dict(sd, strict=False)
        print(f'loaded {ckpt} | missing={len(msg.missing_keys)} '
              f'unexpected={len(msg.unexpected_keys)}', flush=True)
    return m.cuda().eval()



def _to_cuda_dev(x):
    import mmcv
    if isinstance(x, torch.Tensor):
        return x.cuda()
    if isinstance(x, dict):
        return {k: _to_cuda_dev(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_to_cuda_dev(v) for v in x)
    return x


def _move_to_cuda(data):
    for k in list(data.keys()):
        data[k] = _to_cuda_dev(data[k])
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--py-config', required=True)
    ap.add_argument('--ckpt', default='')
    ap.add_argument('--out-dir', default='viz/out')
    ap.add_argument('--vis-index', type=int, nargs='*', default=[0])
    ap.add_argument('--map-score', type=float, default=0.5)
    ap.add_argument('--det-score', type=float, default=0.5)
    args = ap.parse_args()

    from mmengine import Config
    cfg = Config.fromfile(args.py_config)
    os.makedirs(args.out_dir, exist_ok=True)

    from dataset import get_dataloader
    _, val_loader = get_dataloader(
        cfg.train_dataset_config, cfg.val_dataset_config,
        cfg.train_loader, cfg.val_loader, dist=False, val_only=True)
    my_model = build_model(cfg, args.ckpt)

    targets = sorted(set(args.vis_index))
    max_idx = max(targets)
    it = iter(val_loader)
    for cur in range(max_idx + 1):
        try:
            data = next(it)
        except StopIteration:
            print('reached end of loader', flush=True); break
        if cur not in targets:
            continue

        cam_imgs = load_cam_images(data)
        _move_to_cuda(data)

        with torch.no_grad():
            res = my_model(imgs=data['img'], metas=data)

        # ---- predictions ----
        pred_occ = occ_pred_to_bev(res['pred_occ'][-1][0], res['sampled_xyz'])
        fb = res['final_box_dicts'][0]
        pb = fb['pred_boxes']; ps = fb['pred_scores']; pl = fb['pred_labels']
        keep = (ps > args.det_score)
        pred_boxes = pb[keep].detach().cpu().numpy()
        pred_labels = pl[keep].detach().cpu().numpy()
        pred_map = get_pred_map(res, args.map_score)

        cmd = int(data['ego_fut_cmd'].argmax(dim=-1)[0].item())
        ego_pred = cum_traj(res['ego_fut_preds'][0, cmd])
        ego_gt = cum_traj(data['ego_fut_trajs'][0])

        pred = dict(occ=pred_occ, boxes=pred_boxes, labels=pred_labels,
                    map=pred_map, ego_pred=ego_pred, ego_gt=ego_gt)

        # ---- ground truth ----
        gt_occ = occ_gt_to_bev(data['occ_label'], data['occ_xyz'])
        gt_boxes = data['gt_boxes'][0].detach().cpu().numpy()
        gt_labels = gt_boxes[:, 9].astype(np.int32) if gt_boxes.shape[1] > 9 else None
        gt_map = get_gt_map(data)
        gt = dict(occ=gt_occ, boxes=gt_boxes, labels=gt_labels,
                  map=gt_map, ego_gt=ego_gt)

        fid = ''
        try:
            fid = data['frame_id'][0] if 'frame_id' in data else ''
        except Exception:
            pass
        compose(cam_imgs, pred, gt,
                save_path=os.path.join(args.out_dir, f'bev_{cur:05d}.png'),
                title=f'val index {cur}   {fid}')


if __name__ == '__main__':
    main()
