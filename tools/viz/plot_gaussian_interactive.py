"""Interactive 3D gaussian visualization -> self-contained HTML (rotate/zoom/pan).

Reads a GaussianPrediction saved by vis.save_gaussian (*_attr.pth) and writes a
Plotly HTML you can open in any browser and freely orbit/zoom.

Two layers:
  - scatter3d of all kept gaussian centers, colored by semantics (light, smooth
    even for tens of thousands of points)
  - optional ellipsoid meshes for selected classes (e.g. movable) so you can see
    real gaussian shapes/orientation (kept small for performance)
"""
import os, glob, argparse
import numpy as np
from pyquaternion import Quaternion
import torch
import plotly.graph_objects as go

import model  # noqa: F401  (so pickle resolves GaussianPrediction)
from vis import get_nuscenes_colormap

OCC_LABELS = ['others', 'barrier', 'bicycle', 'bus', 'car',
              'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone',
              'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk',
              'terrain', 'manmade', 'vegetation']
MOVABLE = {2, 3, 4, 5, 6, 7, 9, 10}


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


def rgb_str(c):
    return 'rgb(%d,%d,%d)' % (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))


def unit_sphere(res):
    u = np.linspace(0.0, 2.0 * np.pi, res)
    v = np.linspace(0.0, np.pi, res)
    x = np.outer(np.cos(u), np.sin(v)).ravel()
    y = np.outer(np.sin(u), np.sin(v)).ravel()
    z = np.outer(np.ones_like(u), np.cos(v)).ravel()
    return np.stack([x, y, z], axis=-1)  # (res*res, 3)


def unit_sphere_mesh(res):
    """return (verts (V,3), faces (F,3)) for a triangulated unit sphere."""
    u = np.linspace(0.0, 2.0 * np.pi, res)
    v = np.linspace(0.0, np.pi, res)
    uu, vv = np.meshgrid(u, v, indexing='ij')
    x = (np.cos(uu) * np.sin(vv)).ravel()
    y = (np.sin(uu) * np.sin(vv)).ravel()
    z = (np.cos(vv) * np.ones_like(uu)).ravel()
    verts = np.stack([x, y, z], axis=-1)
    faces = []
    for a in range(res - 1):
        for b in range(res - 1):
            i0 = a * res + b
            i1 = (a + 1) * res + b
            i2 = (a + 1) * res + (b + 1)
            i3 = a * res + (b + 1)
            faces.append([i0, i1, i2])
            faces.append([i0, i2, i3])
    return verts, np.asarray(faces, dtype=np.int64)


def build(path, out_html, opa_thr, scalar, exclude_cls, ellipsoid_cls,
          max_ellip, ellip_res, point_size, no_points=False, mesh_opacity=1.0,
          real_opacity=False):
    means, scales, rots, opas, pred = load_attr(path)
    cmap = get_nuscenes_colormap()

    keep = (opas > opa_thr) & (pred != 0)
    if exclude_cls:
        keep &= ~np.isin(pred, list(exclude_cls))
    idx = np.where(keep)[0]
    m, s, r, p, o = means[idx], scales[idx], rots[idx], pred[idx], opas[idx]

    traces = []
    # ---- layer 1: scatter3d points per class (legend toggle by class) ----
    if not no_points:
        for cls in sorted(set(p.tolist())):
            sel = p == cls
            traces.append(go.Scatter3d(
                x=m[sel, 0], y=m[sel, 1], z=m[sel, 2],
                mode='markers',
                marker=dict(size=point_size, color=rgb_str(cmap[cls]), opacity=0.85),
                name='%d %s (%d)' % (cls, OCC_LABELS[cls], int(sel.sum()))))

    # ---- layer 2: ellipsoid meshes, merged per class -> one legend entry each ----
    # if ellipsoid_cls is None -> draw ALL kept classes as ellipsoids
    if ellipsoid_cls is None:
        emask = np.ones(len(p), dtype=bool)
    else:
        emask = np.isin(p, list(ellipsoid_cls))
    if emask.any():
        sv, sf = unit_sphere_mesh(ellip_res)  # verts (V,3), faces (F,3)
        V = sv.shape[0]
        eidx = np.where(emask)[0]
        if len(eidx) > max_ellip:
            eidx = np.random.RandomState(0).choice(eidx, max_ellip, replace=False)
        if real_opacity:
            # one Mesh3d per gaussian so each ellipsoid carries its OWN opacity.
            # legend grouped by class; only first ellipsoid of each class shows in legend.
            cls_count = {c: int((p[eidx] == c).sum()) for c in set(p[eidx].tolist())}
            cls_seen = set()
            for i in eidx:
                cls = int(p[i])
                radii = s[i] * scalar
                Rm = Quaternion(r[i]).rotation_matrix.T
                pts = (Rm @ (sv * radii).T).T + m[i]
                first = cls not in cls_seen
                cls_seen.add(cls)
                traces.append(go.Mesh3d(
                    x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                    i=sf[:, 0], j=sf[:, 1], k=sf[:, 2],
                    color=rgb_str(cmap[cls]),
                    opacity=float(np.clip(o[i], 0.0, 1.0)),
                    flatshading=True, hoverinfo='skip',
                    showlegend=first, legendgroup='cls%d' % cls,
                    name='%d %s (%d)' % (cls, OCC_LABELS[cls], cls_count[cls])))
        else:
            # group selected ellipsoids by class (single shared opacity)
            for cls in sorted(set(p[eidx].tolist())):
                cidx = eidx[p[eidx] == cls]
                vx, fx = [], []
                for k, i in enumerate(cidx):
                    radii = s[i] * scalar
                    Rm = Quaternion(r[i]).rotation_matrix.T
                    pts = (Rm @ (sv * radii).T).T + m[i]
                    vx.append(pts)
                    fx.append(sf + k * V)
                vx = np.concatenate(vx, axis=0)
                fx = np.concatenate(fx, axis=0)
                traces.append(go.Mesh3d(
                    x=vx[:, 0], y=vx[:, 1], z=vx[:, 2],
                    i=fx[:, 0], j=fx[:, 1], k=fx[:, 2],
                    color=rgb_str(cmap[cls]), opacity=mesh_opacity,
                    flatshading=True, hoverinfo='skip',
                    showlegend=True, legendgroup='cls%d' % cls,
                    name='%d %s (%d)' % (cls, OCC_LABELS[cls], len(cidx))))
    else:
        eidx = np.array([], dtype=int)

    # ego marker
    traces.append(go.Scatter3d(
        x=[0], y=[0], z=[0], mode='markers+text',
        marker=dict(size=6, color='red', symbol='diamond'),
        text=['ego'], textposition='top center', name='ego'))

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis_title='x (forward)', yaxis_title='y (left)', zaxis_title='z (up)',
            aspectmode='data',  # equal scale -> no z stretching
            bgcolor='white'),
        title=os.path.basename(out_html) + '  (drag=rotate, scroll=zoom)',
        margin=dict(l=0, r=0, t=30, b=0))
    # embed full plotly.js so the HTML works fully offline (no CDN needed)
    fig.write_html(out_html, include_plotlyjs=True)
    print('saved', out_html, 'points=%d' % len(m),
          'ellipsoids=%d' % len(eidx))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--vis-dir', required=True)
    ap.add_argument('--frames', type=str, nargs='+', default=None)
    ap.add_argument('--opa-thr', type=float, default=0.1)
    ap.add_argument('--scalar', type=float, default=2.0)
    ap.add_argument('--exclude-cls', type=int, nargs='+', default=None)
    ap.add_argument('--ellipsoid-cls', type=int, nargs='+', default=None,
                    help='classes to draw as ellipsoid meshes; omit = ALL classes')
    ap.add_argument('--max-ellip', type=int, default=2000)
    ap.add_argument('--ellip-res', type=int, default=7)
    ap.add_argument('--point-size', type=float, default=2.0)
    ap.add_argument('--no-points', action='store_true', help='hide scatter layer')
    ap.add_argument('--opacity', type=float, default=1.0,
                    help='ellipsoid opacity (1.0 = fully opaque)')
    ap.add_argument('--real-opacity', action='store_true',
                    help='render each ellipsoid with its own per-gaussian opacity')
    ap.add_argument('--suffix', type=str, default='_interactive')
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.vis_dir, '*_attr.pth')))
    if args.frames:
        paths = [p for p in paths if any(f in os.path.basename(p) for f in args.frames)]
    print('found %d attr.pth' % len(paths))
    for pth in paths:
        name = os.path.basename(pth).replace('_attr.pth', '')
        out = os.path.join(args.vis_dir, name + args.suffix + '.html')
        build(pth, out, args.opa_thr, args.scalar, args.exclude_cls,
              args.ellipsoid_cls, args.max_ellip, args.ellip_res, args.point_size,
              args.no_points, args.opacity, args.real_opacity)
