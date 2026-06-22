"""Plot BEV (top-down) views of saved gaussian attr.pth files.

Loads a GaussianPrediction object saved by vis.save_gaussian (means/scales/
rotations/opacities/semantics) and renders bird's-eye-view scatter/ellipse
plots to inspect spatial distribution and whether objects form clusters.
"""
import os, glob, argparse
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import torch

# import model so that pickle can resolve GaussianPrediction class
import model  # noqa: F401
from vis import get_nuscenes_colormap

# semantics channels 0-16 align with get_nuscenes_colormap (0=others ... 16=veg)
OCC_LABELS = ['others', 'barrier', 'bicycle', 'bus', 'car',
              'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone',
              'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk',
              'terrain', 'manmade', 'vegetation']
# movable foreground classes (things that can move) -> for grouping inspection
# 2 bicycle,3 bus,4 car,5 constr,6 motor,7 ped,9 trailer,10 truck
# (barrier=1 and traffic_cone=8 are static foreground -> excluded)
MOVABLE = {2, 3, 4, 5, 6, 7, 9, 10}


def quat_to_yaw(q):
    # q: (...,4) as (w,x,y,z)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def load_attr(path):
    g = torch.load(path, map_location='cpu')
    means = g.means[0].detach().cpu().numpy()
    scales = g.scales[0].detach().cpu().numpy()
    rots = g.rotations[0].detach().cpu().numpy()
    opas = g.opacities[0]
    if opas.numel() == 0:
        opas = torch.ones(means.shape[0], 1)
    opas = opas.squeeze(-1).detach().cpu().numpy()
    sems = g.semantics[0].detach().cpu().numpy()
    pred = sems.argmax(-1)
    dyn = None
    if getattr(g, 'dynamic_logits', None) is not None:
        dyn = torch.sigmoid(g.dynamic_logits[0].squeeze(-1)).detach().cpu().numpy()
    return means, scales, rots, opas, pred, dyn


def plot_bev(path, out_png, opa_thr=0.1):
    means, scales, rots, opas, pred, dyn = load_attr(path)
    cmap = get_nuscenes_colormap()
    yaw = quat_to_yaw(rots)

    keep = (opas > opa_thr) & (pred != 0)  # drop noise/others
    m, s, y, p, o = means[keep], scales[keep], yaw[keep], pred[keep], opas[keep]
    cols = np.array([cmap[int(c)] for c in p])

    mov = np.array([c in MOVABLE for c in p])

    fig, axes = plt.subplots(1, 3, figsize=(30, 10), dpi=120)

    # panel 1: all gaussians BEV scatter colored by semantics
    ax = axes[0]
    ax.scatter(m[:, 0], m[:, 1], s=3, c=cols, alpha=0.6, linewidths=0)
    ax.scatter([0], [0], s=120, marker='*', c='red', zorder=5, label='ego')
    ax.set_title('BEV all gaussians (semantic), N=%d' % len(m))
    ax.set_aspect('equal'); ax.set_xlabel('x'); ax.set_ylabel('y'); ax.legend()

    # panel 2: only movable-class gaussians (cluster inspection)
    ax = axes[1]
    if mov.sum() > 0:
        ax.scatter(m[mov, 0], m[mov, 1], s=8, c=cols[mov], alpha=0.8, linewidths=0)
    ax.scatter([0], [0], s=120, marker='*', c='red', zorder=5)
    ax.set_title('BEV movable-class only, N=%d' % int(mov.sum()))
    ax.set_aspect('equal'); ax.set_xlabel('x'); ax.set_ylabel('y')

    # panel 3: movable gaussians as oriented ellipses (shape/size)
    ax = axes[2]
    idxs = np.where(mov)[0]
    for i in idxs:
        e = Ellipse((m[i, 0], m[i, 1]),
                    width=2 * scales[keep][i, 0], height=2 * scales[keep][i, 1],
                    angle=np.degrees(y[i]), facecolor=cols[i], alpha=0.5,
                    edgecolor='k', linewidth=0.2)
        ax.add_patch(e)
    ax.scatter([0], [0], s=120, marker='*', c='red', zorder=5)
    if len(idxs) > 0:
        ax.set_xlim(m[mov, 0].min() - 2, m[mov, 0].max() + 2)
        ax.set_ylim(m[mov, 1].min() - 2, m[mov, 1].max() + 2)
    ax.set_title('movable gaussians as ellipses')
    ax.set_aspect('equal'); ax.set_xlabel('x'); ax.set_ylabel('y')

    plt.tight_layout()
    plt.savefig(out_png)
    plt.close(fig)
    print('saved', out_png, '| total kept=%d movable=%d' % (len(m), int(mov.sum())),
          '| dyn=%s' % ('yes' if dyn is not None else 'none'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--vis-dir', required=True)
    ap.add_argument('--opa-thr', type=float, default=0.1)
    args = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(args.vis_dir, '*_attr.pth')))
    print('found %d attr.pth' % len(paths))
    for pth in paths:
        name = os.path.basename(pth).replace('_attr.pth', '')
        out = os.path.join(args.vis_dir, name + '_bev.png')
        plot_bev(pth, out, args.opa_thr)
