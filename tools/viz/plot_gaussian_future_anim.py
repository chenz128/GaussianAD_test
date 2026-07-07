"""Animated future-frame gaussian visualization -> self-contained Plotly HTML.

Reads a GaussianPrediction saved by vis.save_gaussian (*_attr.pth) PLUS the
per-gaussian future offsets saved by visualize.py (*_future.npz), then builds a
Plotly figure with a Play button + time slider so you can watch the predicted
future gaussians move (t=0 current -> t=1..6 future frames).

Coloring: two modes toggled by top buttons -- "Semantic" and "Dynamic/Static"
(red=dynamic, blue=static, from the model's per-gaussian dynamic_logits).
Ellipsoids are OPAQUE (no per-gaussian transparency) so shapes read clearly.

Future gaussian position at frame i (next-ego frame, static background stays put):
    pos_i = means + offset[i] - ego_planner[i]

Usage:
    python tools/viz/plot_gaussian_future_anim.py --vis-dir out/<run>/vis --frames val_3
"""
import os, glob, argparse
import numpy as np
from pyquaternion import Quaternion
import torch
import plotly.graph_objects as go

import model  # noqa: F401  (so pickle resolves GaussianPrediction)
from vis import get_nuscenes_colormap

# reuse helpers from the static plotter (same directory -> on sys.path[0])
try:
    from plot_gaussian_interactive import (
        OCC_LABELS, MOVABLE, load_attr, rgb_str, unit_sphere_mesh)
except ImportError:
    from tools.viz.plot_gaussian_interactive import (
        OCC_LABELS, MOVABLE, load_attr, rgb_str, unit_sphere_mesh)


def load_future(path):
    """load *_future.npz -> (offset (A,6,2), planner (6,2), pred_cls (A,)) or None."""
    if not os.path.exists(path):
        return None
    d = np.load(path)
    return (d['offset'].astype(np.float32),
            d['planner'].astype(np.float32),
            d['pred_cls'].astype(np.int64))


def motion_score(future_path):
    """movable-class raw offset (object world-motion) 90th-pct magnitude.

    Uses RAW offset (not ego-compensated): for movable objects the raw offset is
    the object's own predicted motion in the current ego frame, which is exactly
    what we want to rank scenes by. Subtracting the ego planner (~const across
    scenes) would swamp the signal.
    """
    f = load_future(future_path)
    if f is None:
        return -1.0
    offset, planner, pred_cls = f
    mag = np.linalg.norm(offset, axis=-1).max(axis=-1)   # (A,) max over 6 frames
    mov = np.isin(pred_cls, list(MOVABLE))
    if mov.sum() == 0:
        return 0.0
    return float(np.percentile(mag[mov], 90))


def future_positions(means, offset, planner, ego_comp=True, amplify=1.0):
    """return list of 7 (A,3) arrays: [current, f0..f5].

    amplify scales the per-gaussian offset (object motion) so tiny predicted
    motion is visible. Ego planner term is NOT amplified.
    """
    A = means.shape[0]
    steps = [means.copy()]
    off3 = np.concatenate([offset, np.zeros((A, 6, 1), np.float32)], axis=-1)  # (A,6,3)
    pl3 = np.concatenate([planner, np.zeros((6, 1), np.float32)], axis=-1)     # (6,3)
    for i in range(6):
        pos = means + amplify * off3[:, i, :]
        if ego_comp:
            pos = pos - pl3[i][None]
        steps.append(pos.astype(np.float32))
    return steps


def build_anim(attr_path, future_path, out_html, opa_thr=0.1, scalar=2.0,
               exclude_cls=None, max_ellip=1200, ellip_res=6, point_size=2.0,
               dyn_thr=0.0, ego_comp=True, amplify=1.0):
    means, scales, rots, opas, pred, dyn = load_attr(attr_path)
    fut = load_future(future_path)
    if fut is None:
        raise FileNotFoundError(f'missing future npz: {future_path}')
    offset, planner, _ = fut
    assert offset.shape[0] == means.shape[0], \
        f'offset A={offset.shape[0]} != means A={means.shape[0]}'
    cmap = get_nuscenes_colormap()

    keep = (opas > opa_thr) & (pred != 0)
    if exclude_cls:
        keep &= ~np.isin(pred, list(exclude_cls))
    idx = np.where(keep)[0]
    m, s, r, p = means[idx], scales[idx], rots[idx], pred[idx]
    off, pl = offset[idx], planner
    d = dyn[idx] if dyn is not None else None
    has_dyn = d is not None

    steps = future_positions(m, off, pl, ego_comp=ego_comp, amplify=amplify)  # 7 x (Nkeep,3)
    T = len(steps)

    # ellipsoid selection (subsample for size/perf)
    sv, sf = unit_sphere_mesh(ellip_res)
    V = sv.shape[0]
    eidx = np.arange(len(p))
    if len(eidx) > max_ellip:
        eidx = np.random.RandomState(0).choice(eidx, max_ellip, replace=False)
    # precompute per-gaussian local ellipsoid verts (shape/orientation fixed)
    local_verts = {}
    for i in eidx:
        radii = s[i] * scalar
        Rm = Quaternion(r[i]).rotation_matrix.T
        local_verts[i] = (Rm @ (sv * radii).T).T   # (V,3) centered at origin

    # ---- class groupings ----
    sem_classes = sorted(set(p.tolist()))
    sem_ell_classes = sorted(set(p[eidx].tolist()))
    de = d[eidx] if has_dyn else None

    traces = []
    cats = []   # 'sem' | 'dyn' | 'both'
    # movable index within each trace is via 'kind' so frames can update x/y/z.
    # record how to rebuild each trace's positions at a given timestep.
    builders = []   # list of callables: t -> (x,y,z)
    ttypes = []     # per-trace plotly type ('scatter3d'|'mesh3d') for frame data

    def scatter_pts(sel, col, name, cat):
        pos0 = steps[0][sel]
        traces.append(go.Scatter3d(
            x=pos0[:, 0], y=pos0[:, 1], z=pos0[:, 2], mode='markers',
            marker=dict(size=point_size, color=col, opacity=0.9), name=name))
        cats.append(cat)
        ttypes.append('scatter3d')
        builders.append(lambda t, sel=sel: (steps[t][sel][:, 0],
                                             steps[t][sel][:, 1],
                                             steps[t][sel][:, 2]))

    def ellip_group(gidx, col, name, cat):
        # gidx: indices (into kept arrays) of gaussians in this group
        lv = np.stack([local_verts[i] for i in gidx], axis=0)   # (G,V,3)
        faces = np.concatenate([sf + k * V for k in range(len(gidx))], axis=0)

        def verts_at(t):
            centers = steps[t][gidx]                # (G,3)
            vv = (lv + centers[:, None, :]).reshape(-1, 3)
            return vv[:, 0], vv[:, 1], vv[:, 2]
        x0, y0, z0 = verts_at(0)
        traces.append(go.Mesh3d(
            x=x0, y=y0, z=z0, i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color=col, opacity=1.0, flatshading=True, hoverinfo='skip',
            name=name, showlegend=True))
        cats.append(cat)
        ttypes.append('mesh3d')
        builders.append(verts_at)

    # semantic scatter (per class)
    for cls in sem_classes:
        sel = p == cls
        scatter_pts(sel, rgb_str(cmap[cls]),
                    '%d %s (%d)' % (cls, OCC_LABELS[cls], int(sel.sum())), 'sem')
    # dynamic/static scatter
    if has_dyn:
        is_dyn = d > dyn_thr
        scatter_pts(~is_dyn, rgb_str((0.20, 0.40, 0.95)),
                    'static (%d)' % int((~is_dyn).sum()), 'dyn')
        scatter_pts(is_dyn, rgb_str((0.95, 0.15, 0.15)),
                    'dynamic (%d)' % int(is_dyn.sum()), 'dyn')
    # semantic ellipsoids (opaque, per class)
    for cls in sem_ell_classes:
        gidx = eidx[p[eidx] == cls]
        if len(gidx) == 0:
            continue
        ellip_group(gidx, rgb_str(cmap[cls]),
                    '%d %s ellip (%d)' % (cls, OCC_LABELS[cls], len(gidx)), 'sem')
    # dynamic/static ellipsoids (opaque)
    if has_dyn:
        for label, sub, col in [
            ('static', de <= dyn_thr, (0.20, 0.40, 0.95)),
            ('dynamic', de > dyn_thr, (0.95, 0.15, 0.15)),
        ]:
            gidx = eidx[sub]
            if len(gidx) == 0:
                continue
            ellip_group(gidx, rgb_str(col), '%s ellip (%d)' % (label, len(gidx)), 'dyn')

    # ego marker (static, always visible)
    traces.append(go.Scatter3d(
        x=[0], y=[0], z=[0], mode='markers+text',
        marker=dict(size=6, color='black', symbol='diamond'),
        text=['ego'], textposition='top center', name='ego'))
    cats.append('both')
    ttypes.append('scatter3d')
    builders.append(lambda t: ([0.0], [0.0], [0.0]))

    # ---- build animation frames ----
    frame_labels = ['t=0 (now)'] + ['t=%d (+%.1fs)' % (i + 1, 0.5 * (i + 1))
                                    for i in range(6)]
    frames = []
    for t in range(T):
        fdata = []
        for b, tp in zip(builders, ttypes):
            x, y, z = b(t)
            fdata.append(dict(type=tp, x=x, y=y, z=z))
        frames.append(go.Frame(name=str(t), data=fdata))

    fig = go.Figure(data=traces, frames=frames)

    # default = semantic view
    cats_arr = np.array(cats)
    vis_sem = [(c in ('sem', 'both')) for c in cats_arr]
    vis_dyn = [(c in ('dyn', 'both')) for c in cats_arr]
    for tr, vs in zip(fig.data, vis_sem):
        tr.visible = True if vs else False

    # fixed axis ranges so scene doesn't rescale during playback
    allpos = np.concatenate(steps, axis=0)
    pad = 2.0
    xr = [float(allpos[:, 0].min() - pad), float(allpos[:, 0].max() + pad)]
    yr = [float(allpos[:, 1].min() - pad), float(allpos[:, 1].max() + pad)]
    zr = [float(allpos[:, 2].min() - pad), float(allpos[:, 2].max() + pad)]

    play_menu = dict(
        type='buttons', direction='left', x=0.0, y=0.0,
        xanchor='left', yanchor='top', pad=dict(l=4, r=4, t=6, b=4),
        showactive=False,
        buttons=[
            dict(label='Play', method='animate',
                 args=[None, dict(frame=dict(duration=500, redraw=True),
                                  fromcurrent=True,
                                  transition=dict(duration=200))]),
            dict(label='Pause', method='animate',
                 args=[[None], dict(frame=dict(duration=0, redraw=False),
                                    mode='immediate',
                                    transition=dict(duration=0))]),
        ])
    color_buttons = [dict(label='Semantic', method='restyle',
                          args=[{'visible': vis_sem}])]
    if has_dyn:
        color_buttons.append(dict(label='Dynamic/Static', method='restyle',
                                  args=[{'visible': vis_dyn}]))
    color_menu = dict(
        type='buttons', direction='right', x=0.0, y=1.08,
        xanchor='left', yanchor='top', showactive=True,
        buttons=color_buttons, pad=dict(l=4, r=4, t=2, b=2),
        bgcolor='rgba(240,240,240,0.9)')

    slider = dict(
        active=0, x=0.1, y=0.0, len=0.85, xanchor='left', yanchor='top',
        pad=dict(t=4, b=4),
        currentvalue=dict(prefix='frame: ', visible=True),
        steps=[dict(method='animate', label=frame_labels[t],
                    args=[[str(t)], dict(frame=dict(duration=0, redraw=True),
                                         mode='immediate',
                                         transition=dict(duration=0))])
               for t in range(T)])

    menus = [play_menu] + ([color_menu] if has_dyn else [])
    amp_txt = ('  [motion x%.0f amplified]' % amplify) if amplify != 1.0 else ''
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='x (forward)', range=xr),
            yaxis=dict(title='y (left)', range=yr),
            zaxis=dict(title='z (up)', range=zr),
            aspectmode='data', bgcolor='white'),
        title=os.path.basename(out_html) + '  (Play=future frames, drag=rotate)' + amp_txt,
        margin=dict(l=0, r=0, t=30, b=40),
        updatemenus=menus, sliders=[slider])

    fig.write_html(out_html, include_plotlyjs=True, auto_play=False)
    print('saved', out_html, 'kept=%d' % len(m), 'ellip=%d' % len(eidx),
          'frames=%d' % T, 'dyn=%s' % ('yes' if has_dyn else 'no'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--vis-dir', required=True)
    ap.add_argument('--frames', type=str, nargs='+', default=None,
                    help='e.g. val_3 ; matches val_<idx>_gaussian_attr.pth')
    ap.add_argument('--opa-thr', type=float, default=0.1)
    ap.add_argument('--scalar', type=float, default=2.0)
    ap.add_argument('--exclude-cls', type=int, nargs='+', default=None)
    ap.add_argument('--max-ellip', type=int, default=1200)
    ap.add_argument('--ellip-res', type=int, default=6)
    ap.add_argument('--point-size', type=float, default=2.0)
    ap.add_argument('--dyn-thr', type=float, default=0.0)
    ap.add_argument('--amplify', type=float, default=1.0,
                    help='scale object offset for visibility (labeled in title)')
    ap.add_argument('--no-ego-comp', action='store_true',
                    help='do NOT subtract ego motion (show raw means+offset)')
    ap.add_argument('--suffix', type=str, default='_future_anim')
    ap.add_argument('--scan', action='store_true',
                    help='only print motion score per frame, pick best sample')
    args = ap.parse_args()

    attr_paths = sorted(glob.glob(os.path.join(args.vis_dir, '*_gaussian_attr.pth')))
    if args.frames:
        attr_paths = [p for p in attr_paths
                      if any(f in os.path.basename(p) for f in args.frames)]

    if args.scan:
        scored = []
        for ap_ in attr_paths:
            base = os.path.basename(ap_).replace('_gaussian_attr.pth', '')
            fut = os.path.join(args.vis_dir, base + '_future.npz')
            sc = motion_score(fut)
            scored.append((sc, base))
            print('%-16s motion_score=%.3f' % (base, sc))
        scored.sort(reverse=True)
        if scored:
            print('BEST:', scored[0][1], 'score=%.3f' % scored[0][0])
        raise SystemExit

    print('found %d attr.pth' % len(attr_paths))
    for ap_ in attr_paths:
        base = os.path.basename(ap_).replace('_gaussian_attr.pth', '')
        fut = os.path.join(args.vis_dir, base + '_future.npz')
        out = os.path.join(args.vis_dir, base + args.suffix + '.html')
        build_anim(ap_, fut, out, opa_thr=args.opa_thr, scalar=args.scalar,
                   exclude_cls=args.exclude_cls, max_ellip=args.max_ellip,
                   ellip_res=args.ellip_res, point_size=args.point_size,
                   dyn_thr=args.dyn_thr, ego_comp=not args.no_ego_comp,
                   amplify=args.amplify)
