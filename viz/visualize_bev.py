"""
Visualize nuScenes frames in the SAME layout as the reference figure:
  LEFT  : 6 surround cameras (2 rows x 3 cols) with corner labels
          [Front Left, Front, Front Right]
          [Back Left,  Back, Back Right]
  RIGHT : BEV semantic occupancy (colored by class) + ego/agent trajectories

Usage (single GPU):
  python viz/visualize_bev.py \
      --py-config config/nuscenes_gs25600_gtbox_oracle_v12_ft_plan_futgau_detach_false_col/nuscenes_gs25600_gtbox_oracle_v12_ft_plan_futgau_detach_false_col.py \
      --ckpt exp/nuscenes_gs25600_v12_fixempty_ft_plan_futgau_detach_false/latest.pth \
      --out-dir viz/out \
      --num-samples 4 \
      --vis-index 0 50 120 300

The script reuses the project's own dataloader + model so the BEV occupancy and
trajectories come from the real forward pass. Camera images are read from disk
and de-normalized with the training img_norm_cfg so they look like raw photos.
"""
import os, sys, argparse, pickle
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ---- nuScenes semantic palette (class index -> RGB), matches vis.py ----
# index 0 = others/empty ; 1..16 = nuscenes occ classes
NUSCENES_CMAP = np.array([
    [  0,   0,   0],   # 0  others / empty
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

# camera display order matching the reference figure
CAM_ORDER = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT',  'CAM_BACK',  'CAM_BACK_RIGHT']
CAM_LABEL = {'CAM_FRONT_LEFT': 'Front Left', 'CAM_FRONT': 'Front',
             'CAM_FRONT_RIGHT': 'Front Right', 'CAM_BACK_LEFT': 'Back Left',
             'CAM_BACK': 'Back', 'CAM_BACK_RIGHT': 'Back Right'}

# image normalization used during training (config/_base_/surroundocc.py)
IMG_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMG_STD  = np.array([58.395,  57.12,  57.375], dtype=np.float32)

# BEV grid from model/head/gaussian_head.py
# pc_range = [x_min, y_min, z_min, x_max, y_max, z_max] (lidar, metres)
# grid_size = [0.5, 0.5, 0.5] -> grid 120 x 120 x 8 = 115200 voxels
PC_RANGE = [-30.0, -30.0, -2.0, 30.0, 30.0, 2.0]
GRID_X = 120  # x axis cells (forward)
GRID_Y = 120  # y axis cells (lateral)
DX = (PC_RANGE[3] - PC_RANGE[0]) / GRID_X   # 0.5
DY = (PC_RANGE[4] - PC_RANGE[1]) / GRID_Y   # 0.5
NUSC_RANGE = [-30.0, -30.0, 30.0, 30.0]      # xmin,ymin,xmax,ymax


def denorm_img(t):
    """(C,H,W) normalized tensor -> (H,W,3) uint8 RGB."""
    x = t.detach().cpu().float().numpy().transpose(1, 2, 0)  # H,W,C
    x = x * IMG_STD + IMG_MEAN
    # training used to_rgb=True, so channels are already RGB
    return np.clip(x, 0, 255).astype(np.uint8)


def _get_cam_filenames(data):
    """Return list of 6 absolute camera jpg paths in dataset sensor order."""
    try:
        metas = data['img_metas'][0]
        inner = metas.data if hasattr(metas, 'data') else metas
        if isinstance(inner, dict) and 'filename' in inner:
            return list(inner['filename'])
    except Exception:
        pass
    return None


# dataset sensor_types order (matching the pkl / img_metas filename list)
DS_ORDER = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']


def load_cam_images(data, data_root):
    """Return dict cam_name -> RGB uint8 array, in CAM_ORDER.

    Camera images are read from the raw jpg paths stored in img_metas.filename
    (the same ones the dataloader loaded & normalized), so they look like the
    original photos.
    """
    imgs = {}
    fns = _get_cam_filenames(data)
    if fns:
        for name, fn in zip(DS_ORDER, fns):
            p = fn if os.path.isabs(fn) else os.path.join(data_root, fn)
            if os.path.exists(p):
                imgs[name] = np.asarray(Image.open(p).convert('RGB'))
    return {c: imgs[c] for c in CAM_ORDER if c in imgs}


def occ_to_bev(pred_occ_t, sampled_xyz):
    """pred_occ[-1][idx]: (N, C) logits -> (GRID_X, GRID_Y) class map (BEV).

    sampled_xyz: (N, 3) lidar coords of each voxel (same order as pred_occ_t).
    We scatter each voxel's argmax class into the BEV grid using its (x,y).
    Args:
        pred_occ_t : (N, C) logits
        sampled_xyz: (N, 3) or (B, N, 3) -> take first sample
    """
    # pred_occ_t may be (C, N) [channels first] -> argmax over dim 0
    if pred_occ_t.dim() == 2 and pred_occ_t.shape[0] <= 20:
        cls = pred_occ_t.argmax(dim=0).detach().cpu().numpy().astype(np.int32)  # (N,)
    else:
        cls = pred_occ_t.argmax(dim=-1).detach().cpu().numpy().astype(np.int32)  # (N,)
    xyz = sampled_xyz if torch.is_tensor(sampled_xyz) else torch.from_numpy(np.asarray(sampled_xyz))
    xyz = xyz.detach().cpu().numpy()
    if xyz.ndim == 3:
        xyz = xyz[0]
    bev = np.zeros((GRID_X, GRID_Y), dtype=np.int32)
    xi = ((xyz[:, 0] - NUSC_RANGE[0]) / DX).astype(np.int32)
    yi = ((xyz[:, 1] - NUSC_RANGE[1]) / DY).astype(np.int32)
    ok = (xi >= 0) & (xi < GRID_X) & (yi >= 0) & (yi < GRID_Y)
    # keep the highest argmax-class voxel per (x,y) column (draw order not critical)
    sc = np.stack([xi[ok], yi[ok], cls[ok]], axis=-1)
    # for duplicate (x,y) prefer the non-empty class (largest index wins)
    key = sc[:, 0] * GRID_Y + sc[:, 1]
    order = np.argsort(key, kind='stable')
    sc = sc[order]
    # mimic z priority: we did not sort by z, so just take max class per cell
    for idx in range(len(sc)):
        x, y, c = int(sc[idx, 0]), int(sc[idx, 1]), int(sc[idx, 2])
        if c > 0:
            bev[x, y] = c
    return bev  # (x_idx, y_idx)


def bev_to_rgb(bev):
    """class map (x_idx, y_idx) -> (H, W, 3) uint8 image.

    +x (forward) points UP in the image, +y (lateral-left) points RIGHT.
    """
    rgb = NUSCENES_CMAP[np.clip(bev, 0, len(NUSCENES_CMAP) - 1)]  # (x, y, 3)
    # image rows: x_idx, with x_max (far ahead) at top -> flip x axis
    img = rgb[::-1, :, :]
    return img  # (row=x' , col=y)


def lidar_to_bev_px(pts):
    """(K,2) lidar xy (metres) -> (K,2) image pixel (col, row).

    col = y cell (lateral, 0..GRID_Y), row = x cell flipped (forward, top=far).
    """
    col = (pts[:, 1] - PC_RANGE[1]) / DY
    row = (PC_RANGE[3] - pts[:, 0]) / DX
    return np.stack([col, row], axis=-1)


def draw_traj(ax, pts_xy, color, lw=2.5, ls='-', zorder=5, label=None):
    if pts_xy is None or len(pts_xy) < 2:
        return
    px = lidar_to_bev_px(pts_xy)
    ax.plot(px[:, 0], px[:, 1], color=color, lw=lw, ls=ls, zorder=zorder, label=label)
    # arrow head at the end
    dx = px[-1, 0] - px[-2, 0]; dy = px[-1, 1] - px[-2, 1]
    if abs(dx) + abs(dy) > 1e-3:
        ax.annotate('', xy=(px[-1, 0], px[-1, 1]),
                    xytext=(px[-2, 0], px[-2, 1]),
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw),
                    zorder=zorder + 1)


def draw_ego_box(ax, center_xy=(0.0, 0.0), L=4.084, W=1.85, yaw=0.0):
    """draw ego footprint rectangle in BEV pixels."""
    half = np.array([[ L/2,  W/2], [ L/2, -W/2], [-L/2, -W/2], [-L/2,  W/2], [ L/2,  W/2]])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    pts = half @ R.T + np.array(center_xy)
    px = lidar_to_bev_px(pts)
    ax.plot(px[:, 0], px[:, 1], color='lime', lw=2.0, zorder=6)


def draw_agent_boxes(ax, gt_boxes, gt_names, fut_trajs=None, fut_masks=None):
    """gt_boxes (A[,B],7+3): x,y,z,w,l,h,yaw,vx,vy,vz ; draw current BEV footprint + future path.

    Handles optional leading batch dim (squeeze it off) and missing gt_names.
    fut_trajs[i] is typically (T*2,) or (T,2) per-step displacement; fut_masks[i] (T,).
    """
    if gt_boxes is None:
        return
    gb = gt_boxes.detach().cpu().numpy() if torch.is_tensor(gt_boxes) else np.asarray(gt_boxes)
    if gb.ndim == 3:
        gb = gb[0]
    if gb.ndim == 1:
        gb = gb[None]
    names = [None] * gb.shape[0]
    if gt_names is not None:
        nm = gt_names if isinstance(gt_names, (list, tuple)) else list(gt_names)
        for i in range(min(len(nm), gb.shape[0])):
            names[i] = nm[i]
    DYN = ('car','truck','bus','trailer','construction_vehicle','motorcycle','bicycle')
    for i in range(gb.shape[0]):
        x, y, w, l, h, yaw = gb[i, 0], gb[i, 1], gb[i, 3], gb[i, 4], gb[i, 5], gb[i, 6]
        half = np.array([[ l/2,  w/2], [ l/2, -w/2], [-l/2, -w/2], [-l/2,  w/2], [ l/2,  w/2]])
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s], [s, c]])
        pts = half @ R.T + np.array([x, y])
        px = lidar_to_bev_px(pts)
        is_dyn = names[i] in DYN
        ec = '#1565c0' if is_dyn else '#6a1b9a'
        ax.plot(px[:, 0], px[:, 1], color=ec, lw=1.4, zorder=4)
        # future trajectory (per-step displacement -> cumulative)
        if fut_trajs is not None:
            if torch.is_tensor(fut_trajs):
                ft = fut_trajs.detach().cpu().numpy()
                if ft.ndim == 3:
                    ft = ft[0]
                ft = ft[i]
            else:
                ft = np.asarray(fut_trajs)
                if ft.ndim == 3:
                    ft = ft[0]
                ft = ft[i]
            ft = ft.reshape(-1, 2)
            cum = np.cumsum(ft, axis=0) + np.array([x, y])
            if fut_masks is not None:
                fm = fut_masks.detach().cpu().numpy() if torch.is_tensor(fut_masks) else np.asarray(fut_masks)
                if fm.ndim == 3:
                    fm = fm[0]
                fm = fm[i].astype(bool)
                if len(fm) < len(cum):
                    fm = np.asarray([bool(fm[j]) if j < len(fm) else True for j in range(len(cum))])
                cum = cum[fm]
            if len(cum) >= 2:
                pxc = lidar_to_bev_px(cum)
                ax.plot(pxc[:, 0], pxc[:, 1], color='#e65100', lw=1.2, ls='--', zorder=4)


def compose(cam_imgs, bev_rgb, save_path,
            ego_pred=None, ego_gt=None,
            gt_boxes=None, gt_names=None, fut_trajs=None, fut_masks=None,
            title=None):
    fig = plt.figure(figsize=(24, 9), dpi=120)
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.15],
                          wspace=0.04, hspace=0.06)

    # ---- left: 6 cameras (2 x 3) ----
    positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    for (r, c), name in zip(positions, CAM_ORDER):
        ax = fig.add_subplot(gs[r, c])
        if name in cam_imgs:
            ax.imshow(cam_imgs[name])
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor('#444'); sp.set_linewidth(1.5)
        ax.text(0.02, 0.96, CAM_LABEL[name], transform=ax.transAxes,
                fontsize=13, fontweight='bold', color='#ffd54f',
                va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.45, ec='none'))

    # ---- right: BEV occupancy + trajectories ----
    axb = fig.add_subplot(gs[:, 3])
    axb.imshow(bev_rgb, extent=[0, GRID_Y, 0, GRID_X], origin='upper',
               interpolation='nearest', zorder=1)
    draw_ego_box(axb)
    draw_agent_boxes(axb, gt_boxes, gt_names, fut_trajs, fut_masks)
    draw_traj(axb, ego_gt,   color='#1e88e5', lw=2.6, ls='-',  zorder=7, label='GT ego')
    draw_traj(axb, ego_pred, color='#e53935', lw=2.6, ls='--', zorder=7, label='Pred ego')
    axb.set_xlim(0, GRID_Y); axb.set_ylim(GRID_X, 0)
    axb.set_xticks([]); axb.set_yticks([])
    axb.set_aspect('equal')
    for sp in axb.spines.values():
        sp.set_edgecolor('#c62828'); sp.set_linewidth(2.5)
    axb.legend(loc='lower right', fontsize=9, framealpha=0.7)

    if title:
        fig.suptitle(title, fontsize=14)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, facecolor='white')
    plt.close(fig)
    print(f'saved -> {save_path}')


def build_model(cfg, ckpt):
    import model  # noqa: register modules
    from mmseg.models import build_segmentor
    m = build_segmentor(cfg.model)
    if ckpt and os.path.exists(ckpt):
        ck = torch.load(ckpt, map_location='cpu')
        sd = ck['state_dict'] if 'state_dict' in ck else ck
        msg = m.load_state_dict(sd, strict=False)
        print(f'loaded ckpt {ckpt} | missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}')
    m.cuda().eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--py-config', required=True)
    ap.add_argument('--ckpt', default='')
    ap.add_argument('--out-dir', default='viz/out')
    ap.add_argument('--num-samples', type=int, default=4)
    ap.add_argument('--vis-index', type=int, nargs='*', default=None,
                    help='specific val indices to visualize (overrides num-samples)')
    ap.add_argument('--data-root', default='data/nuscenes/')
    ap.add_argument('--no-model', action='store_true',
                    help='skip model forward; only render cameras + GT BEV proxy')
    args = ap.parse_args()

    from mmengine import Config
    cfg = Config.fromfile(args.py_config)
    os.makedirs(args.out_dir, exist_ok=True)

    # dataloader
    from dataset import get_dataloader
    _, val_loader = get_dataloader(
        cfg.train_dataset_config, cfg.val_dataset_config,
        cfg.train_loader, cfg.val_loader, dist=False, val_only=True)

    my_model = None
    if not args.no_model:
        my_model = build_model(cfg, args.ckpt)

    targets = args.vis_index if args.vis_index else list(range(args.num_samples))
    loader_iter = iter(val_loader)

    for step, want_idx in enumerate(targets):
        # advance loader to the desired index
        data = None
        for cur in range(want_idx + 1):
            try:
                data = next(loader_iter)
            except StopIteration:
                print(f'reached end of val loader at idx {cur}'); return
        if data is None:
            continue

        # move tensors to cuda
        for k in list(data.keys()):
            if isinstance(data[k], torch.Tensor):
                data[k] = data[k].cuda()

        cam_imgs = load_cam_images(data, args.data_root)

        ego_pred = ego_gt = None
        pred_bev = None
        gt_boxes = gt_names = fut_trajs = fut_masks = None

        if my_model is not None:
            input_imgs = data.get('img')
            with torch.no_grad():
                res = my_model(imgs=input_imgs, metas=data)
            # BEV occupancy from the last occ head
            if 'pred_occ' in res and len(res['pred_occ']) > 0:
                sxyz = res.get('sampled_xyz')
                pred_bev = occ_to_bev(res['pred_occ'][-1][0], sxyz)
            # ego predicted trajectory (command mode), cumulative
            if 'ego_fut_preds' in res and 'ego_fut_cmd' in data:
                cmd = data['ego_fut_cmd'].argmax(dim=-1)
                pr = res['ego_fut_preds'].cumsum(dim=1)[0, cmd].detach().cpu().numpy().reshape(-1, 2)[:6]
                ego_pred = pr
            gt_boxes = data.get('gt_boxes')
            gt_names = data.get('gt_names')
            fut_trajs = data.get('gt_agent_fut_trajs')
            fut_masks = data.get('gt_agent_fut_masks')
        else:
            # no-model fallback: empty BEV
            pred_bev = np.zeros((GRID_X, GRID_Y), dtype=np.int32)

        # GT ego trajectory (cumulative per-step)
        if 'ego_fut_trajs' in data:
            egt = data['ego_fut_trajs']
            egt = egt[0].float().cpu().numpy() if torch.is_tensor(egt) else np.asarray(egt)[0]
            egt = np.nan_to_num(egt, nan=0.0).astype(np.float32)[:6, :2]
            ego_gt = np.cumsum(egt, axis=0)

        if pred_bev is None:
            pred_bev = np.zeros((GRID_X, GRID_Y), dtype=np.int32)
        bev_rgb = bev_to_rgb(pred_bev)

        compose(cam_imgs, bev_rgb,
                save_path=os.path.join(args.out_dir, f'bev_{want_idx:05d}.png'),
                ego_pred=ego_pred, ego_gt=ego_gt,
                gt_boxes=gt_boxes, gt_names=gt_names,
                fut_trajs=fut_trajs, fut_masks=fut_masks,
                title=f'val index {want_idx}')


if __name__ == '__main__':
    main()
