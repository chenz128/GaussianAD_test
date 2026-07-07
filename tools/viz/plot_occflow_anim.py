"""Occupancy-flow future animation -> self-contained Plotly HTML.

Reads a *_occflow.npz dumped by visualize.py, containing the model's REAL future
occupancy (occ_flow): 6 future frames (0.5s..3.0s) plus the current frame, each a
per-voxel semantic label over a fixed sampled grid `xyz`.

Unlike the old gaussian-offset animation (which incorrectly added the raw 30m
offset to gaussian centers), this uses the model's genuine forward_flow output:
each future frame is the occupancy re-rendered after moving gaussians AND
compensating ego motion, so the scene evolves naturally over time.

Output: a Plotly HTML with Play/Pause + a time slider over
    [t=0 now, t=1..6 future +0.5s..+3.0s].
Toggle buttons switch between PRED and GT future occupancy.
Voxels of the empty class (17) are dropped.
"""
import os
import argparse
import numpy as np
import plotly.graph_objects as go

OCC_LABELS = ['others', 'barrier', 'bicycle', 'bus', 'car',
              'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone',
              'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk',
              'terrain', 'manmade', 'vegetation']
MOVABLE = {2, 3, 4, 5, 6, 7, 9, 10}
EMPTY_LABEL = 17

_PALETTE = np.array([
    [0, 0, 0], [255, 120, 50], [255, 192, 203], [255, 255, 0], [0, 150, 245],
    [0, 255, 255], [255, 127, 0], [255, 0, 0], [255, 240, 150], [135, 60, 0],
    [160, 32, 240], [255, 0, 255], [139, 137, 137], [75, 0, 75], [150, 240, 80],
    [230, 230, 250], [0, 175, 0],
], dtype=np.float32)


def rgb_str(c):
    return 'rgb(%d,%d,%d)' % (int(c[0]), int(c[1]), int(c[2]))


def load_occflow(path):
    z = np.load(path)
    return (z['xyz'].astype(np.float32), z['occ_now'].astype(np.int16),
            z['occ_fut'].astype(np.int16), z['occ_fut_gt'].astype(np.int16),
            z['valid'].astype(np.int8))


def build_frames(xyz, now, fut, gt, only_movable=False):
    """Return list of (label_array,) per time step for PRED and GT.

    t=0 uses `now` for both pred and gt (current frame is the anchor).
    t=1..6 use fut[i-1] / gt[i-1].
    """
    nfut = fut.shape[0]
    pred_steps = [now] + [fut[i] for i in range(nfut)]
    # GT current frame == pred current frame (no GT-now dumped); reuse now.
    gt_steps = [now] + [gt[i] for i in range(nfut)]
    return pred_steps, gt_steps


def scatter_for(xyz, labels, only_movable, point_size):
    keep = labels != EMPTY_LABEL
    if only_movable:
        keep &= np.isin(labels, list(MOVABLE))
    x, y, z = xyz[keep, 0], xyz[keep, 1], xyz[keep, 2]
    lab = labels[keep]
    colors = _PALETTE[np.clip(lab, 0, 16)]
    col = ['rgb(%d,%d,%d)' % (int(c[0]), int(c[1]), int(c[2])) for c in colors]
    return x, y, z, col


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True, help='*_occflow.npz path')
    ap.add_argument('--out', default=None, help='output html path')
    ap.add_argument('--point-size', type=float, default=3.0)
    ap.add_argument('--only-movable', action='store_true',
                    help='show only movable classes (cars/peds/...)')
    ap.add_argument('--title', default=None)
    args = ap.parse_args()

    xyz, now, fut, gt, valid = load_occflow(args.npz)
    pred_steps, gt_steps = build_frames(xyz, now, fut, gt, args.only_movable)
    nsteps = len(pred_steps)

    # fixed axis ranges from all occupied voxels across all steps
    occ_all = np.zeros(xyz.shape[0], dtype=bool)
    for s in pred_steps + gt_steps:
        m = s != EMPTY_LABEL
        if args.only_movable:
            m &= np.isin(s, list(MOVABLE))
        occ_all |= m
    pts = xyz[occ_all]
    if pts.shape[0] == 0:
        pts = xyz
    pad = 2.0
    rng = [[float(pts[:, i].min()) - pad, float(pts[:, i].max()) + pad] for i in range(3)]

    def make_trace(step_labels, name, visible):
        x, y, z, col = scatter_for(xyz, step_labels, args.only_movable, args.point_size)
        return go.Scatter3d(
            x=x, y=y, z=z, mode='markers',
            marker=dict(size=args.point_size, color=col, opacity=1.0),
            name=name, visible=visible, showlegend=False)

    # base traces: PRED@t0 visible, GT@t0 hidden
    fig = go.Figure()
    fig.add_trace(make_trace(pred_steps[0], 'pred', True))
    fig.add_trace(make_trace(gt_steps[0], 'gt', False))

    # frames: each frame updates BOTH traces (pred + gt) for that time step
    frames = []
    for t in range(nsteps):
        px, py, pz, pcol = scatter_for(xyz, pred_steps[t], args.only_movable, args.point_size)
        gx, gy, gz, gcol = scatter_for(xyz, gt_steps[t], args.only_movable, args.point_size)
        frames.append(go.Frame(name=str(t), data=[
            dict(type='scatter3d', x=px, y=py, z=pz, marker=dict(size=args.point_size, color=pcol, opacity=1.0)),
            dict(type='scatter3d', x=gx, y=gy, z=gz, marker=dict(size=args.point_size, color=gcol, opacity=1.0)),
        ]))
    fig.frames = frames

    steps = []
    for t in range(nsteps):
        lbl = 'now' if t == 0 else '+%.1fs' % (0.5 * t)
        steps.append(dict(method='animate', label=lbl,
                          args=[[str(t)], dict(mode='immediate',
                                               frame=dict(duration=0, redraw=True),
                                               transition=dict(duration=0))]))

    # PRED/GT toggle: control which base trace is visible (frames update both, so
    # we rely on visibility to pick which one is shown)
    updatemenus = [
        dict(type='buttons', showactive=False, x=0.05, y=0.05, xanchor='left',
             buttons=[
                 dict(label='Play', method='animate',
                      args=[None, dict(frame=dict(duration=600, redraw=True),
                                       fromcurrent=True,
                                       transition=dict(duration=0))]),
                 dict(label='Pause', method='animate',
                      args=[[None], dict(mode='immediate',
                                         frame=dict(duration=0, redraw=False),
                                         transition=dict(duration=0))]),
             ]),
        dict(type='buttons', showactive=True, x=0.05, y=0.95, xanchor='left',
             buttons=[
                 dict(label='PRED', method='update',
                      args=[dict(visible=[True, False])]),
                 dict(label='GT', method='update',
                      args=[dict(visible=[False, True])]),
             ]),
    ]

    title = args.title or os.path.basename(args.npz)
    fig.update_layout(
        title=title,
        updatemenus=updatemenus,
        sliders=[dict(active=0, steps=steps, x=0.15, len=0.7,
                      currentvalue=dict(prefix='t = '))],
        scene=dict(
            xaxis=dict(range=rng[0], title='x (m)'),
            yaxis=dict(range=rng[1], title='y (m)'),
            zaxis=dict(range=rng[2], title='z (m)'),
            aspectmode='data'),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    out = args.out or (os.path.splitext(args.npz)[0] + '_occanim.html')
    fig.write_html(out, include_plotlyjs='cdn', auto_play=False)
    print('wrote', out)


if __name__ == '__main__':
    main()
