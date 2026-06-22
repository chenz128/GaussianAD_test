"""
Render occupancy voxels to a self-contained interactive Plotly HTML, with a
per-class statistics table beside the 3D scene.

Reads one or more *_occ.npz files (produced by export_occ_npz.py). Each npz has
xyz/pred/gt. For each requested source (pred or gt) a separate HTML is written:
  - 3D scene: one voxel = one colored cube marker, colored by nuScenes palette
  - table: per-class voxel count; for pred sources also IoU/precision/recall
           vs the GT stored in the same npz.

Usage:
  python tools/viz/plot_occ_interactive.py \
    --npz out/run_a/vis/val_0_occ.npz:predA \
          out/run_b/vis/val_0_occ.npz:predB \
          out/run_a/vis/val_0_occ.npz:gt \
    --out-dir out/occ_compare --tag val_0
"""
import argparse
import os
import os.path as osp

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# nuScenes 17-class palette (label 0..16); label 17 = empty (skip)
_PALETTE = np.array([
    [  0,   0,   0],   # 0  others/noise
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

_NAMES = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation',
]

EMPTY = 17
# classes shown in the table (skip 0=others/noise which is rarely meaningful)
TABLE_CLASSES = list(range(1, 17))


def rgb_str(c):
    return 'rgb(%d,%d,%d)' % (int(c[0]), int(c[1]), int(c[2]))


def per_class_stats(pred, gt):
    """Return dict cls -> (count_pred, iou, prec, reca) for pred vs gt."""
    out = {}
    for c in TABLE_CLASSES:
        tp = int(np.sum((pred == c) & (gt == c)))
        n_pred = int(np.sum(pred == c))
        n_gt = int(np.sum(gt == c))
        denom = n_pred + n_gt - tp
        iou = tp / denom if denom > 0 else float('nan')
        prec = tp / n_pred if n_pred > 0 else float('nan')
        reca = tp / n_gt if n_gt > 0 else float('nan')
        out[c] = (n_pred, n_gt, iou, prec, reca)
    return out


def build_scene_traces(xyz, label, point_size):
    traces = []
    for c in TABLE_CLASSES + [0]:
        idx = np.where(label == c)[0]
        if idx.size == 0:
            continue
        p = xyz[idx]
        traces.append(go.Scatter3d(
            x=p[:, 0], y=p[:, 1], z=p[:, 2],
            mode='markers',
            marker=dict(size=point_size, color=rgb_str(_PALETTE[c]),
                        symbol='square', opacity=1.0),
            name='%d %s (%d)' % (c, _NAMES[c], idx.size),
            legendgroup='cls%d' % c,
            showlegend=True,
        ))
    return traces


def build_html(xyz, label, gt, out_html, title, point_size, is_pred):
    def font_for(fill):
        r, g, b = [int(x) for x in fill[4:-1].split(',')]
        return 'black' if (r + g + b) > 380 else 'white'

    if is_pred:
        stats = per_class_stats(label, gt)
        header = ['class', 'count', 'IoU%', 'prec', 'recall']
        rows = [[], [], [], [], []]
        fills, fonts = [], []
        for c in TABLE_CLASSES:
            n_pred, n_gt, iou, prec, reca = stats[c]
            if n_pred == 0 and n_gt == 0:
                continue
            rows[0].append('%d %s' % (c, _NAMES[c]))
            rows[1].append(str(n_pred))
            rows[2].append('%.2f' % (iou * 100) if iou == iou else '-')
            rows[3].append('%.2f' % prec if prec == prec else '-')
            rows[4].append('%.2f' % reca if reca == reca else '-')
            fc = rgb_str(_PALETTE[c])
            fills.append(fc)
            fonts.append(font_for(fc))
        # overall mIoU over classes present in GT
        ious = [stats[c][2] for c in TABLE_CLASSES if stats[c][1] > 0]
        miou = np.nanmean(ious) * 100 if ious else float('nan')
        rows[0].append('mIoU (GT classes)')
        rows[1].append('')
        rows[2].append('%.2f' % miou if miou == miou else '-')
        rows[3].append('')
        rows[4].append('')
        fills.append('rgb(40,40,40)')
        fonts.append('white')
        ncol = len(header)
        table = go.Table(
            columnwidth=[3, 1.4, 1.2, 1.2, 1.2],
            header=dict(values=header, fill_color='rgb(30,30,30)',
                        font=dict(color='white', size=12), align='left'),
            cells=dict(values=rows,
                       fill_color=[fills] * ncol,
                       align='left', height=22,
                       font=dict(color=[fonts] * ncol, size=11)),
        )
    else:
        header = ['class', 'count']
        names_col, cnt_col, fills, fonts = [], [], [], []
        for c in TABLE_CLASSES:
            n = int(np.sum(label == c))
            if n == 0:
                continue
            names_col.append('%d %s' % (c, _NAMES[c]))
            cnt_col.append(str(n))
            fc = rgb_str(_PALETTE[c])
            fills.append(fc)
            fonts.append(font_for(fc))
        table = go.Table(
            columnwidth=[3, 1.4],
            header=dict(values=header, fill_color='rgb(30,30,30)',
                        font=dict(color='white', size=12), align='left'),
            cells=dict(values=[names_col, cnt_col],
                       fill_color=[fills, fills],
                       align='left', height=22,
                       font=dict(color=[fonts, fonts], size=11)),
        )

    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.7, 0.3],
        specs=[[{'type': 'scene'}, {'type': 'table'}]],
        horizontal_spacing=0.02,
    )
    for tr in build_scene_traces(xyz, label, point_size):
        fig.add_trace(tr, row=1, col=1)
    fig.add_trace(table, row=1, col=2)

    fig.update_layout(
        title=title,
        scene=dict(
            aspectmode='data',
            xaxis=dict(title='x', backgroundcolor='rgb(20,20,20)'),
            yaxis=dict(title='y', backgroundcolor='rgb(20,20,20)'),
            zaxis=dict(title='z', backgroundcolor='rgb(20,20,20)'),
            camera=dict(eye=dict(x=1.4, y=1.4, z=1.0)),
        ),
        paper_bgcolor='rgb(245,245,245)',
        legend=dict(itemsizing='constant', font=dict(size=10)),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.write_html(out_html, include_plotlyjs=True)
    nonempty = int(np.sum(label != EMPTY))
    print(f'saved {out_html}  nonempty_voxels={nonempty}')


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    for spec in args.npz:
        path, _, kind = spec.partition(':')
        kind = kind or 'pred'
        d = np.load(path)
        xyz, pred, gt = d['xyz'], d['pred'], d['gt']
        if kind == 'gt':
            label = gt
            is_pred = False
            name = 'gt'
        else:
            label = pred
            is_pred = True
            name = kind
        out_html = osp.join(args.out_dir, f'{args.tag}_{name}_occ.html')
        build_html(xyz, label, gt, out_html,
                   title=f'{args.tag}  [{name}]  occupancy',
                   point_size=args.point_size, is_pred=is_pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', nargs='+', required=True,
                        help='path:kind  (kind = gt or a label like predV4)')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--tag', default='val_0')
    parser.add_argument('--point-size', type=float, default=2.5)
    args = parser.parse_args()
    main(args)
