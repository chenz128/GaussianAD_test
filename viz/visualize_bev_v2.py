"""
Visualize nuScenes frames in the reference-figure layout (single row):

  [ cameras 3x2 (compact) ] | [Pred OCC][Pred route] | [GT OCC][GT route]

  * cameras   : 6 surround views, tightly packed (Front Left/Front/Front Right,
                Back Left/Back/Back Right).
  * OCC panel : ONLY the 4D semantic occupancy (BEV) + ego -> perception output.
  * route     : drawn SEPARATELY from the occupancy on a white canvas: lane /
                reference center-lines + ego + candidate & GT trajectories.

Coordinate frame (fixed for BOTH the occupancy image and every vector overlay):
  +x forward = UP, +y left = LEFT, ego at the (0,0) origin = panel centre.
  The raw occ grid is x-major (x,y increase along axis0,axis1) which imshow
  would render 180-deg rotated, so occ_display() flips it to match to_px.

Everything comes from the project's own dataloader + model forward pass.

Usage:
  python viz/visualize_bev_v2.py \
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

# ego candidate-trajectory palette (one distinct hue per planning mode)
TRAJ_PALETTE = ['#ff3d00', '#ffab00', '#00c853', '#2979ff', '#d500f9',
                '#00b8d4', '#c51162', '#76ff03', '#6200ea', '#ff6d00']

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
# STANDARD driving-BEV convention used for EVERY overlay AND the occ image:
#   image row 0 = top  = far ahead (+x forward)
#   image col 0 = left = ego-left  (+y)
# The occ grid is x-major with x,y increasing along axis0,axis1, so its raw
# imshow would place +x at the bottom and +y on the right (a 180 deg flip vs
# this convention).  occ_display() below flips the grid so the occupancy image
# and all vector overlays share exactly this frame (fixes the ego / transform).
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


def occ_display(bev):
    """Class map -> RGB image already flipped into the to_px frame so that
    +x(forward) is UP and +y(left) is LEFT.  imshow it with
    extent=[0, GRID, GRID, 0], origin='upper' and every to_px overlay lines up.
    """
    return bev_to_rgb(bev)[::-1, ::-1]


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
        # nuScenes/VAD box layout: col3=width, col4=length, col6=raw yaw.
        # Length runs along the heading; convert yaw to the lidar heading via
        # yaw' = -(yaw + pi/2) so the drawn box orientation matches the frame.
        x, y = b[i, 0], b[i, 1]
        length, width = b[i, 4], b[i, 3]
        yaw = -(b[i, 6] + np.pi / 2)
        corners = np.array([[ length/2,  width/2], [ length/2, -width/2],
                            [-length/2, -width/2], [-length/2,  width/2],
                            [ length/2,  width/2]])
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s], [s, c]])
        pts = corners @ R.T + np.array([x, y])
        col, row = to_px(pts[:, 0], pts[:, 1])
        color = default
        if labels is not None and i < len(labels):
            color = DET_PALETTE[int(labels[i]) % len(DET_PALETTE)]
        ax.plot(col, row, color=color, lw=lw, zorder=5)
        # heading tick (points along +length = vehicle forward)
        hx, hy = to_px(x + c * length / 2, y + s * length / 2)
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
    """Ego is at the LiDAR/ego origin (0,0). Draw the vehicle rectangle plus a
    forward-pointing heading triangle so its position AND orientation are
    unambiguous in the fixed to_px frame (nose points to +x = up)."""
    box = np.array([[ L/2,  W/2], [ L/2, -W/2], [-L/2, -W/2],
                    [-L/2,  W/2], [L/2, W/2]])
    col, row = to_px(box[:, 0], box[:, 1])
    ax.fill(col, row, color='#00e676', alpha=0.55, zorder=10)
    ax.plot(col, row, color='#004d40', lw=1.4, zorder=11)
    # heading triangle (nose forward)
    nose = np.array([[L/2 + 1.1, 0.0], [L/2 - 0.2, W/2], [L/2 - 0.2, -W/2]])
    c2, r2 = to_px(nose[:, 0], nose[:, 1])
    ax.fill(c2, r2, color='#00c853', zorder=12)
    # ego centre dot
    cc, rr = to_px(0.0, 0.0)
    ax.scatter([cc], [rr], s=18, color='#004d40', zorder=13)


def style_bev(ax, title, edge, y_half=None, fs=14):
    """y_half: if given, crop the lateral (+/-y) view to +/- y_half metres so
    the panel becomes a tall narrow strip (used for the route/plan panels)."""
    if y_half is None:
        ax.set_xlim(0, GRID)
    else:
        cL, _ = to_px(0.0,  y_half)      # +y (left)  -> smaller col
        cR, _ = to_px(0.0, -y_half)      # -y (right) -> larger col
        ax.set_xlim(float(cL), float(cR))
    ax.set_ylim(GRID, 0)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')
    for sp in ax.spines.values():
        sp.set_edgecolor(edge); sp.set_linewidth(2.5)
    ax.set_title(title, fontsize=fs, fontweight='bold')


# ---------------------------------------------------------------------------
# ego trajectory extraction (per-step deltas -> cumulative)
# ---------------------------------------------------------------------------
def cum_traj(deltas):
    """Per-step (dx,dy) displacements -> cumulative absolute trajectory with a
    leading (0,0) at the ego. Handles (T,2) single and (M,T,2) multi-mode."""
    d = deltas.detach().cpu().numpy() if torch.is_tensor(deltas) else np.asarray(deltas)
    d = np.nan_to_num(d, nan=0.0)
    if d.ndim == 2:
        traj = np.cumsum(d, axis=0)
        return np.concatenate([np.zeros((1, 2)), traj], axis=0)
    traj = np.cumsum(d, axis=1)
    z = np.zeros((d.shape[0], 1, 2))
    return np.concatenate([z, traj], axis=1)


def traj_to_lidar(traj):
    """Convert an ego trajectory from the planning frame (col0 = +right lateral,
    col1 = +forward) into the lidar/BEV frame used by to_px (x = forward,
    y = left):  x_forward = col1,  y_left = -col0.
    Handles (T,2) single-mode and (M,T,2) multi-mode arrays."""
    if traj is None:
        return None
    t = np.asarray(traj, dtype=np.float64)
    out = np.empty_like(t)
    out[..., 0] = t[..., 1]
    out[..., 1] = -t[..., 0]
    return out


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
# lateral half-width (metres) shown in the narrow route/plan panels
ROUTE_Y_HALF = 14.0


def _draw_occ_panel(ax, occ, boxes, labels, title, edge):
    """Pure 4D occupancy (BEV) + ego. Perception output only."""
    ax.imshow(occ_display(occ), extent=[0, GRID, GRID, 0],
              origin='upper', interpolation='nearest', zorder=1)
    draw_boxes(ax, boxes, labels, lw=1.2)
    draw_ego(ax)
    style_bev(ax, title, edge)


def _draw_route_panel(ax, ref_map, trajs, sel_traj, gt_traj, title, edge):
    """Route / planning panel drawn SEPARATELY from occupancy: white canvas +
    lane / reference center-lines + ego + candidate & GT trajectories."""
    ax.set_facecolor('white')
    if ref_map is not None and len(ref_map[0]):
        draw_map(ax, ref_map[0], ref_map[1], lw=1.8)
    draw_ego(ax)
    # all candidate trajectories (planning modes), thin
    if trajs is not None:
        for m, t in enumerate(trajs):
            col, row = to_px(t[:, 0], t[:, 1])
            c = TRAJ_PALETTE[m % len(TRAJ_PALETTE)]
            ax.plot(col, row, color=c, lw=1.6, alpha=0.85, zorder=6,
                    solid_capstyle='round')
            ax.scatter(col, row, s=10, color=c, zorder=7)
    # selected / predicted trajectory, thick red
    if sel_traj is not None:
        col, row = to_px(sel_traj[:, 0], sel_traj[:, 1])
        ax.plot(col, row, color='#e53935', lw=3.0, zorder=9,
                solid_capstyle='round', label='Pred ego')
        ax.scatter(col, row, s=22, color='#e53935', edgecolor='white',
                   lw=0.5, zorder=10)
    # GT reference trajectory, blue
    if gt_traj is not None:
        col, row = to_px(gt_traj[:, 0], gt_traj[:, 1])
        ax.plot(col, row, color='#1e88e5', lw=2.6, ls='--', zorder=8,
                solid_capstyle='round', label='GT ego')
        ax.scatter(col, row, s=22, marker='o', facecolor='white',
                   edgecolor='#1e88e5', lw=1.4, zorder=9)
    style_bev(ax, title, edge, y_half=ROUTE_Y_HALF, fs=12)


def compose(cam_imgs, pred, gt, save_path, title=None):
    """Layout (single row) matching the reference figure:

        [ cameras 3x2 (compact) ] | [Pred OCC][Pred route] | [GT OCC][GT route]

    OCC panels show ONLY the 4D occupancy (perception); the route panels are
    drawn separately with the lane center-lines + planned / GT trajectories.
    """
    fig = plt.figure(figsize=(30, 8.4), dpi=120)
    # outer columns: cameras block | pred-occ | pred-route | gt-occ | gt-route
    outer = fig.add_gridspec(1, 5, width_ratios=[3.05, 1.75, 0.62, 1.75, 0.62],
                             wspace=0.06)

    # ---- cameras: compact 2x3 sub-grid (tiny spacing) ----
    cam_gs = outer[0, 0].subgridspec(2, 3, wspace=0.015, hspace=0.02)
    positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    for (r, c), name in zip(positions, CAM_ORDER):
        ax = fig.add_subplot(cam_gs[r, c])
        if name in cam_imgs:
            ax.imshow(cam_imgs[name])
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor('#444'); sp.set_linewidth(1.0)
        ax.text(0.02, 0.94, CAM_LABEL[name], transform=ax.transAxes,
                fontsize=11, fontweight='bold', color='#ffd54f',
                va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.18', fc='black', alpha=0.45,
                          ec='none'))

    # ---- Predictions: OCC panel + route panel ----
    ax_p_occ = fig.add_subplot(outer[0, 1])
    _draw_occ_panel(ax_p_occ, pred['occ'], pred['boxes'], pred['labels'],
                    'Pred - 4D Occupancy', '#b25000')
    ax_p_rt = fig.add_subplot(outer[0, 2])
    ref_map = gt['map'] if (gt['map'] and len(gt['map'][0])) else pred['map']
    _draw_route_panel(ax_p_rt, ref_map, pred.get('trajs'), pred['ego_pred'],
                      pred['ego_gt'], 'Pred - Route', '#b25000')
    ax_p_rt.legend(loc='lower center', fontsize=8, framealpha=0.75,
                   ncol=2, bbox_to_anchor=(0.5, -0.02))

    # ---- Ground Truth: OCC panel + route panel ----
    ax_g_occ = fig.add_subplot(outer[0, 3])
    _draw_occ_panel(ax_g_occ, gt['occ'], gt['boxes'], gt['labels'],
                    'GT - 4D Occupancy', '#b00020')
    ax_g_rt = fig.add_subplot(outer[0, 4])
    _draw_route_panel(ax_g_rt, gt['map'], None, None, gt['ego_gt'],
                      'GT - Route', '#b00020')

    if title:
        fig.suptitle(title, fontsize=14)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, facecolor='white')
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
        # all candidate planning modes (M,T,2)->(M,T+1,2), and the selected one
        try:
            ego_trajs = traj_to_lidar(cum_traj(res['ego_fut_preds'][0]))
        except Exception:
            ego_trajs = None
        ego_pred = traj_to_lidar(cum_traj(res['ego_fut_preds'][0, cmd]))
        ego_gt = traj_to_lidar(cum_traj(data['ego_fut_trajs'][0]))

        pred = dict(occ=pred_occ, boxes=pred_boxes, labels=pred_labels,
                    map=pred_map, trajs=ego_trajs,
                    ego_pred=ego_pred, ego_gt=ego_gt)

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
