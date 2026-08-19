"""Combined future animation: PRED as gaussians, GT as occupancy points.

For each scene this overlays, in ONE self-contained Plotly HTML:
  * PRED = the model's gaussians drawn as OPAQUE ellipsoids, moving over the
    future horizon via  pos_i = means + offset[i] - gt_ego[i]  (ego-compensated
    with the GT ego trajectory, exactly like forward_flow does internally).
  * GT   = the ground-truth future OCCUPANCY (occ_fut_gt) drawn as coloured
    points on the fixed sampled grid `xyz`, label per frame.

Both live in the SAME ego-compensated current-frame coordinates (offset is pure
object motion in LIDAR frame and forward_flow removes ego with the GT ego
trajectory), so the gaussian prediction and the occ GT line up frame-by-frame.

Three view buttons: "Pred+GT", "Pred (gaussian)", "GT (occ)".
Play/Pause + time slider over t=0 (now) .. t=6 (+3.0s).

Inputs per scene (all dumped by visualize.py --vis-gaussian):
  <base>_gaussian_attr.pth   gaussian means/scales/rots/opacities/pred/dyn
  <base>_future.npz          offset (A,6,2), gt_ego (6,2 cumulative)
  <base>_occflow.npz         xyz (N,3), occ_now (N,), occ_fut_gt (6,N)

Usage:
  python tools/viz/plot_pred_gaussian_gt_occ_anim.py \
      --vis-dir out/<run>/vis --frames val_0 val_1 --suffix _pred_gs_gt_occ
"""
import os
import glob
import argparse
import numpy as np
from pyquaternion import Quaternion
import plotly.graph_objects as go

import model  # noqa: F401  (so pickle resolves GaussianPrediction)
from vis import get_nuscenes_colormap

try:
    from plot_gaussian_interactive import (
        OCC_LABELS, load_attr, rgb_str, unit_sphere_mesh)
    from plot_gaussian_future_anim import load_future, future_positions
except ImportError:
    from tools.viz.plot_gaussian_interactive import (
        OCC_LABELS, load_attr, rgb_str, unit_sphere_mesh)
    from tools.viz.plot_gaussian_future_anim import load_future, future_positions

EMPTY_LABEL = 17
# 17-class nuScenes occ palette (same ordering as OCC_LABELS / occ GT labels)
_PALETTE = np.array([
    [0, 0, 0], [255, 120, 50], [255, 192, 203], [255, 255, 0], [0, 150, 245],
    [0, 255, 255], [255, 127, 0], [255, 0, 0], [255, 240, 150], [135, 60, 0],
    [160, 32, 240], [255, 0, 255], [139, 137, 137], [75, 0, 75], [150, 240, 80],
    [230, 230, 250], [0, 175, 0],
], dtype=np.float32)


def load_occflow_gt(path):
    z = np.load(path)
    return (z['xyz'].astype(np.float32),
            z['occ_now'].astype(np.int16),
            z['occ_fut_gt'].astype(np.int16),
            z['valid'].astype(np.int8))


def build(attr_path, future_path, occflow_path, out_html,
          opa_thr=0.1, scalar=2.0, max_ellip=1500, ellip_res=6,
          gt_point_size=2.2, exclude_cls=None):
    means, scales, rots, opas, pred, dyn = load_attr(attr_path)
    fut = load_future(future_path)
    if fut is None:
        raise FileNotFoundError(f'missing future npz: {future_path}')
    offset, planner, _, gt_ego = fut
    assert offset.shape[0] == means.shape[0]
    xyz, occ_now, occ_fut_gt = load_occflow_gt(occflow_path)[:3]
    cmap = get_nuscenes_colormap()

    # ---- PRED gaussians: keep opaque, non-empty, above opacity thr ----
    keep = (opas > opa_thr) & (pred != 0)
    if exclude_cls:
        keep &= ~np.isin(pred, list(exclude_cls))
    idx = np.where(keep)[0]
    m, s, r, p = means[idx], scales[idx], rots[idx], pred[idx]
    off = offset[idx]

    steps = future_positions(m, off, planner, ego_comp=True, amplify=1.0,
                             extrap=0, gt_ego=gt_ego, use_gt_ego=True)
    T = len(steps)  # 7

    # ---- GT occ steps: fixed grid, label per frame ----
    gt_steps = [occ_now] + [occ_fut_gt[i] for i in range(occ_fut_gt.shape[0])]
    gt_steps = gt_steps[:T]

    # ---- ellipsoid mesh setup (subsample for size) ----
    sv, sf = unit_sphere_mesh(ellip_res)
    V = sv.shape[0]
    eidx = np.arange(len(p))
    if len(eidx) > max_ellip:
        eidx = np.random.RandomState(0).choice(eidx, max_ellip, replace=False)
    local_verts = {}
    for i in eidx:
        radii = s[i] * scalar
        Rm = Quaternion(r[i]).rotation_matrix.T
        local_verts[i] = (Rm @ (sv * radii).T).T

    traces = []
    cats = []          # 'pred' | 'gt'
    builders = []      # t -> frame data dict
    ttypes = []

    # ---- PRED ellipsoids grouped by semantic class ----
    sem_ell_classes = sorted(set(p[eidx].tolist()))
    for cls in sem_ell_classes:
        gidx = eidx[p[eidx] == cls]
        if len(gidx) == 0:
            continue
        lv = np.stack([local_verts[i] for i in gidx], axis=0)  # (G,V,3)
        faces = np.concatenate([sf + k * V for k in range(len(gidx))], axis=0)
        col = rgb_str(cmap[cls])

        def verts_at(t, gidx=gidx, lv=lv):
            centers = steps[t][gidx]
            vv = (lv + centers[:, None, :]).reshape(-1, 3)
            return dict(type='mesh3d', x=vv[:, 0], y=vv[:, 1], z=vv[:, 2])

        d0 = verts_at(0)
        traces.append(go.Mesh3d(
            x=d0['x'], y=d0['y'], z=d0['z'],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color=col, opacity=1.0, flatshading=True, hoverinfo='skip',
            name='PRED %d %s (%d)' % (cls, OCC_LABELS[cls], len(gidx)),
            showlegend=True))
        cats.append('pred')
        ttypes.append('mesh3d')
        builders.append(verts_at)

    # ---- GT occ as a single scatter, colour+points updated per frame ----
    def gt_at(t):
        lab = gt_steps[t]
        n = min(lab.shape[0], xyz.shape[0])
        lab = lab[:n]
        keepm = lab != EMPTY_LABEL
        if exclude_cls:
            keepm &= ~np.isin(lab, list(exclude_cls))
        xx = xyz[:n][keepm]
        cc = _PALETTE[np.clip(lab[keepm], 0, 16)]
        col = ['rgb(%d,%d,%d)' % (int(c[0]), int(c[1]), int(c[2])) for c in cc]
        return dict(type='scatter3d', x=xx[:, 0], y=xx[:, 1], z=xx[:, 2],
                    mode='markers',
                    marker=dict(size=gt_point_size, color=col, opacity=0.85))

    g0 = gt_at(0)
    traces.append(go.Scatter3d(
        x=g0['x'], y=g0['y'], z=g0['z'], mode='markers',
        marker=g0['marker'], name='GT occ (points)', showlegend=True))
    cats.append('gt')
    ttypes.append('scatter3d')
    builders.append(gt_at)

    # ---- ego marker ----
    traces.append(go.Scatter3d(
        x=[0], y=[0], z=[0], mode='markers+text',
        marker=dict(size=6, color='black', symbol='diamond'),
        text=['ego'], textposition='top center', name='ego'))
    cats.append('both')
    ttypes.append('scatter3d')
    builders.append(lambda t: dict(type='scatter3d', x=[0.0], y=[0.0], z=[0.0]))

    # ---- frames ----
    frame_labels = ['t=0 (now)'] + [
        't=%d (+%.1fs)' % (i + 1, 0.5 * (i + 1)) for i in range(T - 1)]

    # fixed axis range from GT occ grid (bounded) + t=0 gaussians
    allpts = np.concatenate([xyz, steps[0]], axis=0)
    pad = 3.0
    xr = [float(np.percentile(allpts[:, 0], 1) - pad),
          float(np.percentile(allpts[:, 0], 99) + pad)]
    yr = [float(np.percentile(allpts[:, 1], 1) - pad),
          float(np.percentile(allpts[:, 1], 99) + pad)]
    zr = [float(allpts[:, 2].min() - 2.0), float(allpts[:, 2].max() + 2.0)]
    dx, dy, dz = xr[1] - xr[0], yr[1] - yr[0], zr[1] - zr[0]
    dmax = max(dx, dy, dz)
    aspect = dict(x=dx / dmax, y=dy / dmax, z=dz / dmax)
    fixed_scene = dict(
        xaxis=dict(range=xr, autorange=False),
        yaxis=dict(range=yr, autorange=False),
        zaxis=dict(range=zr, autorange=False),
        aspectmode='manual', aspectratio=aspect, bgcolor='white')

    frames = []
    for t in range(T):
        fdata = [b(t) for b in builders]
        frames.append(go.Frame(
            name=str(t), data=fdata,
            layout=go.Layout(scene=fixed_scene)))

    fig = go.Figure(data=traces, frames=frames)

    cats_arr = np.array(cats)
    vis_both = [True for _ in cats_arr]
    vis_pred = [(c in ('pred', 'both')) for c in cats_arr]
    vis_gt = [(c in ('gt', 'both')) for c in cats_arr]

    play_menu = dict(
        type='buttons', direction='left', x=0.0, y=0.0,
        xanchor='left', yanchor='top', pad=dict(l=4, r=4, t=6, b=4),
        showactive=False,
        buttons=[
            dict(label='Play', method='animate',
                 args=[None, dict(frame=dict(duration=600, redraw=True),
                                  fromcurrent=True,
                                  transition=dict(duration=200))]),
            dict(label='Pause', method='animate',
                 args=[[None], dict(frame=dict(duration=0, redraw=False),
                                    mode='immediate',
                                    transition=dict(duration=0))]),
        ])
    view_menu = dict(
        type='buttons', direction='right', x=0.0, y=1.08,
        xanchor='left', yanchor='top', showactive=True,
        pad=dict(l=4, r=4, t=2, b=2), bgcolor='rgba(240,240,240,0.9)',
        buttons=[
            dict(label='Pred+GT', method='restyle', args=[{'visible': vis_both}]),
            dict(label='Pred (gaussian)', method='restyle',
                 args=[{'visible': vis_pred}]),
            dict(label='GT (occ)', method='restyle', args=[{'visible': vis_gt}]),
        ])
    slider = dict(
        active=0, x=0.1, y=0.0, len=0.85, xanchor='left', yanchor='top',
        pad=dict(t=4, b=4),
        currentvalue=dict(prefix='frame: ', visible=True),
        steps=[dict(method='animate', label=frame_labels[t],
                    args=[[str(t)], dict(frame=dict(duration=0, redraw=True),
                                         mode='immediate',
                                         transition=dict(duration=0))])
               for t in range(T)])

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='x (forward)', range=xr, autorange=False),
            yaxis=dict(title='y (left)', range=yr, autorange=False),
            zaxis=dict(title='z (up)', range=zr, autorange=False),
            aspectmode='manual', aspectratio=aspect, bgcolor='white'),
        title=os.path.basename(out_html)
              + '  (PRED=gaussian ellipsoids, GT=occ points; Play=future)',
        margin=dict(l=0, r=0, t=30, b=40),
        updatemenus=[play_menu, view_menu], sliders=[slider])

    fig.write_html(out_html, include_plotlyjs=True, auto_play=False)
    print('saved', out_html, 'pred_ellip=%d' % len(eidx), 'frames=%d' % T)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--vis-dir', required=True)
    ap.add_argument('--frames', type=str, nargs='+', default=None,
                    help='e.g. val_0 val_1 ; matches val_<idx>_gaussian_attr.pth')
    ap.add_argument('--opa-thr', type=float, default=0.1)
    ap.add_argument('--scalar', type=float, default=2.0)
    ap.add_argument('--max-ellip', type=int, default=1500)
    ap.add_argument('--ellip-res', type=int, default=6)
    ap.add_argument('--gt-point-size', type=float, default=2.2)
    ap.add_argument('--exclude-cls', type=int, nargs='+', default=None)
    ap.add_argument('--suffix', type=str, default='_pred_gs_gt_occ')
    args = ap.parse_args()

    attr_paths = sorted(glob.glob(os.path.join(args.vis_dir, '*_gaussian_attr.pth')))
    if args.frames:
        attr_paths = [p for p in attr_paths
                      if any(f in os.path.basename(p) for f in args.frames)]
    print('found %d attr.pth' % len(attr_paths))
    for ap_ in attr_paths:
        base = os.path.basename(ap_).replace('_gaussian_attr.pth', '')
        fut = os.path.join(args.vis_dir, base + '_future.npz')
        occ = os.path.join(args.vis_dir, base + '_occflow.npz')
        out = os.path.join(args.vis_dir, base + args.suffix + '.html')
        if not os.path.exists(occ):
            print('skip %s (no occflow npz)' % base)
            continue
        build(ap_, fut, occ, out, opa_thr=args.opa_thr, scalar=args.scalar,
              max_ellip=args.max_ellip, ellip_res=args.ellip_res,
              gt_point_size=args.gt_point_size, exclude_cls=args.exclude_cls)
