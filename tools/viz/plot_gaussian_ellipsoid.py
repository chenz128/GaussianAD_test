"""Render saved gaussians as 3D ellipsoids from an elevated (tilted top-down) view.

Reads a GaussianPrediction saved by vis.save_gaussian (*_attr.pth) and draws each
gaussian as an oriented ellipsoid surface colored by semantics. Unlike the original
save_gaussian, the elevation/azimuth angle is configurable and the box aspect keeps
z from being stretched into a 'wall'. No model re-run needed.
"""
import os, glob, argparse
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from pyquaternion import Quaternion
import torch

import model  # noqa: F401  (so pickle resolves GaussianPrediction)
from vis import get_nuscenes_colormap

# semantics channels 0-16 align with get_nuscenes_colormap
MOVABLE = {2, 3, 4, 5, 6, 7, 9, 10}  # bicycle,bus,car,constr,motor,ped,trailer,truck


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
    return means, scales, rots, opas, pred


def render(path, out_png, elev, azim, opa_thr, scalar, movable_only,
           xlim, ylim, res, proj='persp', draw_ego=True, dist=None, exclude_cls=None):
    means, scales, rots, opas, pred = load_attr(path)
    cmap = get_nuscenes_colormap()

    keep = (opas > opa_thr) & (pred != 0)  # drop noise/others
    if movable_only:
        keep &= np.isin(pred, list(MOVABLE))
    if exclude_cls:
        keep &= ~np.isin(pred, list(exclude_cls))
    # crop to a region of interest so ellipsoids are visible (not a tiny blob)
    keep &= (means[:, 0] >= xlim[0]) & (means[:, 0] <= xlim[1])
    keep &= (means[:, 1] >= ylim[0]) & (means[:, 1] <= ylim[1])

    m, s, r, o, p = means[keep], scales[keep], rots[keep], opas[keep], pred[keep]
    n = m.shape[0]

    fig = plt.figure(figsize=(16, 12), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    try:
        ax.set_proj_type(proj)  # 'persp' gives third-person depth feel
    except Exception:
        pass
    ax.view_init(elev=elev, azim=azim)
    if dist is not None:
        try:
            ax.dist = dist  # smaller -> closer (zoom in)
        except Exception:
            pass

    # unit sphere mesh (low res for speed)
    u = np.linspace(0.0, 2.0 * np.pi, res)
    v = np.linspace(0.0, np.pi, res)
    su, sv = np.outer(np.cos(u), np.sin(v)), np.outer(np.sin(u), np.sin(v))
    cw = np.outer(np.ones_like(u), np.cos(v))

    for i in range(n):
        radii = s[i] * scalar
        Rm = Quaternion(r[i]).rotation_matrix.T
        x = radii[0] * su
        y = radii[1] * sv
        z = radii[2] * cw
        xyz = np.stack([x, y, z], axis=-1)
        xyz = (Rm[None, None] @ xyz[..., None]).squeeze(-1)
        xyz = xyz + m[i][None, None]
        ax.plot_surface(
            xyz[..., 0], xyz[..., 1], xyz[..., 2],
            rstride=1, cstride=1, color=cmap[int(p[i])],
            linewidth=0, alpha=float(min(1.0, o[i])), shade=True)

    if draw_ego:
        # ego vehicle marker at origin: red box + heading arrow (+x = forward)
        ax.scatter([0], [0], [0.5], c='red', s=450, marker='s', depthshade=False,
                   edgecolors='k', zorder=10)
        ax.quiver(0, 0, 0.5, 6, 0, 0, color='red', linewidth=3,
                  arrow_length_ratio=0.25, zorder=10)

    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(-3, 3)
    # keep z from being stretched: box aspect proportional to real ranges
    ax.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], 12))
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    ax.set_title('%s  N=%d  elev=%d azim=%d%s' % (
        os.path.basename(out_png), n, elev, azim,
        '  movable-only' if movable_only else ''))
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close(fig)
    print('saved', out_png, 'N=%d' % n)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--vis-dir', required=True)
    ap.add_argument('--elev', type=float, default=30.0, help='elevation angle (deg)')
    ap.add_argument('--azim', type=float, default=-60.0, help='azimuth angle (deg)')
    ap.add_argument('--opa-thr', type=float, default=0.1)
    ap.add_argument('--scalar', type=float, default=2.0, help='ellipsoid size scale')
    ap.add_argument('--movable-only', action='store_true')
    ap.add_argument('--xlim', type=float, nargs=2, default=[-30, 30])
    ap.add_argument('--ylim', type=float, nargs=2, default=[-30, 30])
    ap.add_argument('--res', type=int, default=8, help='ellipsoid mesh resolution')
    ap.add_argument('--suffix', type=str, default='_ellip')
    ap.add_argument('--proj', type=str, default='persp', choices=['persp', 'ortho'])
    ap.add_argument('--dist', type=float, default=None, help='camera distance (smaller=closer)')
    ap.add_argument('--frames', type=str, nargs='+', default=None,
                    help='only render attr files whose name contains one of these (e.g. val_0)')
    ap.add_argument('--exclude-cls', type=int, nargs='+', default=None,
                    help='semantic class ids to hide (e.g. 15 16 = manmade vegetation)')
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.vis_dir, '*_attr.pth')))
    if args.frames:
        paths = [p for p in paths if any(f in os.path.basename(p) for f in args.frames)]
    print('found %d attr.pth' % len(paths))
    for pth in paths:
        name = os.path.basename(pth).replace('_attr.pth', '')
        out = os.path.join(args.vis_dir, name + args.suffix + '.png')
        render(pth, out, args.elev, args.azim, args.opa_thr, args.scalar,
               args.movable_only, args.xlim, args.ylim, args.res,
               args.proj, True, args.dist, args.exclude_cls)
