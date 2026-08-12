"""Occupancy-flow future animation + raw Gaussian overlay -> self-contained HTML.

Reads a *_occflow.npz dumped by visualize.py.  Optionally overlays the raw
Gaussians from a *_attr.pth.  When *_frontier_future.npz exists, each frame uses
the model's actual retained + newly generated Gaussian bank instead of offset
animation; new Gaussians are rendered as more opaque semantic ellipsoids.

Three toggle buttons: PRED / GT / Gaussians.
  PRED / GT   — animated future occupancy (Play/Pause + time slider).
  Gaussians   — oriented ellipsoid meshes; when a *_future.npz (per-Gaussian
                flow) sits next to the npz, the ellipsoids animate across the
                same future steps (centres shifted by the predicted flow).

Usage:
  python tools/viz/plot_occflow_anim.py \
      --npz  out/<run>/vis/val_0_occflow.npz \
      --attr-pth out/<run>/vis/val_0_gaussian_attr.pth \
      --out  val_0_occflow_gs.html
"""
import os
import sys
import argparse

import numpy as np
import plotly.graph_objects as go
from pyquaternion import Quaternion

try:
    import torch
except ImportError:
    torch = None


# ── nuScenes 17-class constants ──────────────────────────────────────────────

OCC_LABELS = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation',
]
MOVABLE = {2, 3, 4, 5, 6, 7, 9, 10}
EMPTY_LABEL = 17

_PALETTE = np.array([
    [  0,   0,   0], [255, 120,  50], [255, 192, 203], [255, 255,   0],
    [  0, 150, 245], [  0, 255, 255], [255, 127,   0], [255,   0,   0],
    [255, 240, 150], [135,  60,   0], [160,  32, 240], [255,   0, 255],
    [139, 137, 137], [ 75,   0,  75], [150, 240,  80], [230, 230, 250],
    [  0, 175,   0],
], dtype=np.float32)


def rgb_str(c):
    return 'rgb(%d,%d,%d)' % (int(c[0]), int(c[1]), int(c[2]))


# ── occupancy helpers ────────────────────────────────────────────────────────

def load_occflow(path):
    z = np.load(path)
    return (z['xyz'].astype(np.float32), z['occ_now'].astype(np.int16),
            z['occ_fut'].astype(np.int16), z['occ_fut_gt'].astype(np.int16),
            z['valid'].astype(np.int8))


def scatter_for(xyz, labels, only_movable, point_size):
    keep = labels != EMPTY_LABEL
    if only_movable:
        keep &= np.isin(labels, list(MOVABLE))
    x, y, z = xyz[keep, 0], xyz[keep, 1], xyz[keep, 2]
    lab = labels[keep]
    col = [rgb_str(_PALETTE[np.clip(l, 0, 16)]) for l in lab]
    return x, y, z, col


# ── Gaussian helpers ─────────────────────────────────────────────────────────

def _unit_sphere_mesh(res):
    """Triangulated unit sphere: returns (verts (V,3), faces (F,3))."""
    u = np.linspace(0.0, 2.0 * np.pi, res)
    v = np.linspace(0.0, np.pi, res)
    uu, vv = np.meshgrid(u, v, indexing='ij')
    verts = np.stack([
        np.cos(uu) * np.sin(vv),
        np.sin(uu) * np.sin(vv),
        np.cos(vv) * np.ones_like(uu),
    ], axis=-1).reshape(-1, 3)
    faces = []
    for a in range(res - 1):
        for b in range(res - 1):
            i0, i1 = a * res + b, (a + 1) * res + b
            i2, i3 = (a + 1) * res + (b + 1), a * res + (b + 1)
            faces += [[i0, i1, i2], [i0, i2, i3]]
    return verts, np.asarray(faces, dtype=np.int64)


def load_gaussian_traces(attr_pth, opa_thr,
                         ellip_cls, max_ellip, ellip_res, scalar,
                         offset=None, ego=None):
    """Return (traces, gs_meta).

    traces:  one Mesh3d trace per class (oriented ellipsoids), visible=False.
    gs_meta: per-class geometry (base vertices, faces, flow indices) so the
             caller can rebuild each future frame by shifting centres with the
             per-Gaussian flow ``offset``.
    """
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if root not in sys.path:
        sys.path.insert(0, root)
    import model  # noqa: F401  registers GaussianPrediction for torch.load

    g = torch.load(attr_pth, map_location='cpu')
    means  = g.means[0].detach().cpu().numpy()
    scales = g.scales[0].detach().cpu().numpy()
    rots   = g.rotations[0].detach().cpu().numpy()   # wxyz quaternion
    opas   = g.opacities[0].squeeze(-1).detach().cpu().numpy()
    sems   = g.semantics[0].detach().cpu().numpy()
    pred   = sems.argmax(-1)

    keep = (opas > opa_thr) & (pred != 0)   # drop noise class and near-transparent
    orig_idx = np.where(keep)[0]            # kept position -> original Gaussian index
    means, scales, rots, pred = means[keep], scales[keep], rots[keep], pred[keep]

    # ellipsoids: subsample across ALL kept classes (None = all), capped
    sv, sf = _unit_sphere_mesh(ellip_res)
    V = sv.shape[0]
    if ellip_cls is None:
        eidx = np.arange(len(pred))
    else:
        eidx = np.where(np.isin(pred, list(ellip_cls)))[0]
    if len(eidx) > max_ellip:
        eidx = np.random.RandomState(0).choice(eidx, max_ellip, replace=False)

    traces = []
    groups = []
    for cls in sorted(set(pred[eidx].tolist())):
        cidx = eidx[pred[eidx] == cls]          # positions into kept arrays
        n = len(cidx)
        # base ellipsoid vertices (oriented, scaled) BEFORE translation, (n,V,3)
        base = np.empty((n, V, 3), dtype=np.float32)
        for k, i in enumerate(cidx):
            radii = scales[i] * scalar
            Rm = Quaternion(rots[i]).rotation_matrix.T
            base[k] = (Rm @ (sv * radii).T).T
        fx = np.concatenate([sf + k * V for k in range(n)], 0)
        cmeans = means[cidx].astype(np.float32)         # (n,3)
        gidx = orig_idx[cidx]                            # (n,) -> flow rows
        verts = (base + cmeans[:, None, :]).reshape(-1, 3)
        traces.append(go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=fx[:, 0], j=fx[:, 1], k=fx[:, 2],
            color=rgb_str(_PALETTE[cls]), opacity=0.55,
            flatshading=True, hoverinfo='skip',
            visible=False, legendgroup='gs%d' % cls, showlegend=False,
            name='ellip %d %s' % (cls, OCC_LABELS[cls]),
        ))
        groups.append(dict(cls=cls, base=base, faces=fx, means=cmeans, gidx=gidx))

    gs_meta = dict(groups=groups, offset=offset, ego=ego)
    print('Gaussian layer: %d ellipsoids (%d classes, %d kept total)%s' %
          (len(eidx), len(groups), int(keep.sum()),
           '  [animated by flow+ego]' if offset is not None else ''))
    return traces, gs_meta


def build_gaussian_frame_data(gs_meta, step):
    """Rebuild the per-class Mesh3d vertex data for future frame ``step``.

    step 0 = current frame (no shift); step>=1 shifts each ellipsoid centre into
    the future ego frame: centre + flow ``offset[gidx, step-1]`` - ego
    displacement ``ego[step-1]`` (x, y), matching the occupancy animation.
    """
    offset = gs_meta['offset']
    ego = gs_meta['ego']
    data = []
    for grp in gs_meta['groups']:
        cmeans = grp['means']
        if offset is not None and step >= 1:
            shifted = cmeans.copy()
            shifted[:, :2] = shifted[:, :2] + offset[grp['gidx'], step - 1]
            if ego is not None:
                shifted[:, :2] = shifted[:, :2] - ego[step - 1]
        else:
            shifted = cmeans
        verts = (grp['base'] + shifted[:, None, :]).reshape(-1, 3)
        fx = grp['faces']
        data.append(dict(type='mesh3d',
                         x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                         i=fx[:, 0], j=fx[:, 1], k=fx[:, 2]))
    return data


def load_frontier_gaussian_frames(path, opa_thr, ellip_cls,
                                  max_ellip, ellip_res, scalar):
    """Build frame-wise meshes from the actual retained/generated banks."""
    z = np.load(path)
    means = z['means'].astype(np.float32)
    scales = z['scales'].astype(np.float32)
    rotations = z['rotations'].astype(np.float32)
    rotation_matrices = (
        z['rotation_matrices'].astype(np.float32)
        if 'rotation_matrices' in z.files else None)
    opacities = z['opacities'].astype(np.float32)
    semantics = z['semantics'].astype(np.float32)
    valid = (z['valid'].astype(bool) if 'valid' in z.files
             else np.ones(opacities.shape, dtype=bool))
    generated = (z['generated'].astype(bool) if 'generated' in z.files
                 else np.zeros(opacities.shape, dtype=bool))
    pred = semantics.argmax(-1)

    selected = []
    groups = set()
    for step in range(means.shape[0]):
        keep = valid[step] & (opacities[step] > opa_thr) & (pred[step] != 0)
        if ellip_cls is not None:
            keep &= np.isin(pred[step], list(ellip_cls))
        indices = np.where(keep)[0]
        if len(indices) > max_ellip:
            indices = np.random.RandomState(step).choice(
                indices, max_ellip, replace=False)
        selected.append(indices)
        groups.update(
            (bool(generated[step, index]), int(pred[step, index]))
            for index in indices)
    groups = sorted(groups, key=lambda item: (item[0], item[1]))

    sphere_vertices, sphere_faces = _unit_sphere_mesh(ellip_res)
    vertices_per_ellipsoid = sphere_vertices.shape[0]

    def mesh_data(step, is_generated, cls):
        indices = selected[step]
        indices = indices[
            (generated[step, indices] == is_generated)
            & (pred[step, indices] == cls)]
        vertices = []
        faces = []
        for mesh_index, index in enumerate(indices):
            radii = scales[step, index] * scalar
            rotation = (
                rotation_matrices[step, index]
                if rotation_matrices is not None
                else Quaternion(rotations[step, index]).rotation_matrix.T)
            base = (rotation @ (sphere_vertices * radii).T).T
            vertices.append(base + means[step, index])
            faces.append(sphere_faces + mesh_index * vertices_per_ellipsoid)
        if vertices:
            vertices = np.concatenate(vertices, 0)
            faces = np.concatenate(faces, 0)
        else:
            vertices = np.empty((0, 3), dtype=np.float32)
            faces = np.empty((0, 3), dtype=np.int64)
        return dict(
            type='mesh3d', x=vertices[:, 0], y=vertices[:, 1],
            z=vertices[:, 2], i=faces[:, 0], j=faces[:, 1], k=faces[:, 2])

    frame_data = []
    for step in range(means.shape[0]):
        frame_data.append([
            mesh_data(step, is_generated, cls)
            for is_generated, cls in groups])

    traces = []
    for group_index, (is_generated, cls) in enumerate(groups):
        mesh = frame_data[0][group_index]
        source = 'new' if is_generated else 'retained'
        traces.append(go.Mesh3d(
            x=mesh['x'], y=mesh['y'], z=mesh['z'],
            i=mesh['i'], j=mesh['j'], k=mesh['k'],
            color=rgb_str(_PALETTE[cls]),
            opacity=0.90 if is_generated else 0.45,
            flatshading=True, hoverinfo='skip', visible=False,
            showlegend=False, name='%s %s' % (source, OCC_LABELS[cls])))

    for step, indices in enumerate(selected):
        new_count = int(generated[step, indices].sum())
        print('Gaussian frame %d: %d ellipsoids (%d newly generated)' % (
            step, len(indices), new_count))
    return traces, frame_data


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True, help='*_occflow.npz path')
    ap.add_argument('--out', default=None, help='output HTML path')
    ap.add_argument('--title', default=None)
    ap.add_argument('--point-size', type=float, default=3.0)
    ap.add_argument('--only-movable', action='store_true')
    # Gaussian layer
    ap.add_argument('--attr-pth', default=None,
                    help='*_attr.pth from visualize.py --vis-gaussian')
    ap.add_argument('--opa-thr', type=float, default=0.3,
                    help='opacity threshold; higher removes floating weak Gaussians')
    ap.add_argument('--ellip-cls', type=int, nargs='+', default=None,
                    help='classes drawn as ellipsoids (default: ALL classes)')
    ap.add_argument('--max-ellip', type=int, default=4000,
                    help='max ellipsoids to draw (subsampled across all classes)')
    ap.add_argument('--ellip-res', type=int, default=6)
    ap.add_argument('--scalar', type=float, default=2.0,
                    help='ellipsoid scale multiplier')
    args = ap.parse_args()

    xyz, now, fut, gt, _ = load_occflow(args.npz)
    nfut = fut.shape[0]
    pred_steps = [now] + [fut[i] for i in range(nfut)]
    gt_steps   = [now] + [gt[i]  for i in range(nfut)]
    nsteps = len(pred_steps)

    occ_all = np.zeros(xyz.shape[0], dtype=bool)
    for s in pred_steps + gt_steps:
        m = s != EMPTY_LABEL
        if args.only_movable:
            m &= np.isin(s, list(MOVABLE))
        occ_all |= m
    pts = xyz[occ_all] if occ_all.any() else xyz
    pad = 2.0
    rng = [[float(pts[:, i].min()) - pad, float(pts[:, i].max()) + pad]
           for i in range(3)]
    dxyz  = [rng[i][1] - rng[i][0] for i in range(3)]
    dmax  = max(dxyz)
    aspect = dict(x=dxyz[0]/dmax, y=dxyz[1]/dmax, z=dxyz[2]/dmax)
    fixed_scene = dict(
        xaxis=dict(range=rng[0], autorange=False),
        yaxis=dict(range=rng[1], autorange=False),
        zaxis=dict(range=rng[2], autorange=False),
        aspectmode='manual', aspectratio=aspect)

    def make_occ_trace(step_labels, name, visible):
        x, y, z, col = scatter_for(xyz, step_labels, args.only_movable, args.point_size)
        return go.Scatter3d(
            x=x, y=y, z=z, mode='markers',
            marker=dict(size=args.point_size, color=col, opacity=1.0),
            name=name, visible=visible, showlegend=False)

    fig = go.Figure()
    fig.add_trace(make_occ_trace(pred_steps[0], 'pred', True))   # trace 0
    fig.add_trace(make_occ_trace(gt_steps[0],   'gt',   False))  # trace 1

    gs_traces = []
    gs_meta = None
    frontier_frame_data = None
    if args.attr_pth:
        prefix = os.path.splitext(args.npz)[0].replace('_occflow', '')
        frontier_npz = prefix + '_frontier_future.npz'
        if os.path.exists(frontier_npz):
            gs_traces, frontier_frame_data = load_frontier_gaussian_frames(
                frontier_npz, args.opa_thr, args.ellip_cls,
                args.max_ellip, args.ellip_res, args.scalar)
        else:
            offset = None
            ego = None
            fut_npz = prefix + '_future.npz'
            if os.path.exists(fut_npz):
                fz = np.load(fut_npz)
                offset = fz['offset'].astype(np.float32)
                if 'gt_ego' in fz.files:
                    ego = fz['gt_ego'].astype(np.float32)
            gs_traces, gs_meta = load_gaussian_traces(
                args.attr_pth, args.opa_thr,
                args.ellip_cls, args.max_ellip, args.ellip_res, args.scalar,
                offset, ego)
        for t in gs_traces:
            fig.add_trace(t)

    n_gs = len(gs_traces)
    vis_pred = [True,  False] + [False] * n_gs
    vis_gt   = [False, True]  + [False] * n_gs
    vis_gs   = [False, False] + [True]  * n_gs

    frames = []
    for t in range(nsteps):
        px, py, pz, pcol = scatter_for(xyz, pred_steps[t], args.only_movable, args.point_size)
        gx, gy, gz, gcol = scatter_for(xyz, gt_steps[t],   args.only_movable, args.point_size)
        fdata = [
            dict(type='scatter3d', x=px, y=py, z=pz,
                 marker=dict(size=args.point_size, color=pcol, opacity=1.0)),
            dict(type='scatter3d', x=gx, y=gy, z=gz,
                 marker=dict(size=args.point_size, color=gcol, opacity=1.0)),
        ]
        if frontier_frame_data is not None:
            fdata += frontier_frame_data[t]
        elif gs_meta is not None:
            fdata += build_gaussian_frame_data(gs_meta, t)
        frames.append(go.Frame(name=str(t), data=fdata,
                               layout=go.Layout(scene=fixed_scene)))
    fig.frames = frames

    slider_steps = []
    for t in range(nsteps):
        lbl = 'now' if t == 0 else '+%.1fs' % (0.5 * t)
        slider_steps.append(dict(
            method='animate', label=lbl,
            args=[[str(t)], dict(mode='immediate',
                                 frame=dict(duration=0, redraw=True),
                                 transition=dict(duration=0))]))

    toggle_buttons = [
        dict(label='PRED',  method='update', args=[dict(visible=vis_pred)]),
        dict(label='GT',    method='update', args=[dict(visible=vis_gt)]),
    ]
    if n_gs:
        toggle_buttons.append(
            dict(label='Gaussians', method='update', args=[dict(visible=vis_gs)]))

    fig.update_layout(
        title=args.title or os.path.basename(args.npz),
        updatemenus=[
            dict(type='buttons', showactive=False, x=0.05, y=0.05, xanchor='left',
                 buttons=[
                     dict(label='▶ Play', method='animate',
                          args=[None, dict(frame=dict(duration=600, redraw=True),
                                           fromcurrent=True, transition=dict(duration=0))]),
                     dict(label='⏸ Pause', method='animate',
                          args=[[None], dict(mode='immediate',
                                             frame=dict(duration=0, redraw=False),
                                             transition=dict(duration=0))]),
                 ]),
            dict(type='buttons', showactive=True, x=0.05, y=0.95, xanchor='left',
                 buttons=toggle_buttons),
        ],
        sliders=[dict(active=0, steps=slider_steps, x=0.15, len=0.7,
                      currentvalue=dict(prefix='t = '))],
        scene=dict(
            xaxis=dict(range=rng[0], title='x (m)', autorange=False),
            yaxis=dict(range=rng[1], title='y (m)', autorange=False),
            zaxis=dict(range=rng[2], title='z (m)', autorange=False),
            aspectmode='manual', aspectratio=aspect),
        legend=dict(itemsizing='constant', font=dict(size=10)),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    out = args.out or (os.path.splitext(args.npz)[0] + '_occflow_gs.html')
    fig.write_html(out, include_plotlyjs='cdn', auto_play=False)
    print('wrote', out)


if __name__ == '__main__':
    main()
