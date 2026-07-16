"""Readable 3D occupancy animation of GT-box future trajectories.

Visual encoding:
  * gray translucent voxels: current-frame SurroundOcc occupied volume;
  * colored translucent solid cuboids: animated future GT-box occupancy;
  * colored glowing line on each cuboid roof: real future GT-box center track.

Unlike the prior wireframe view, boxes are solid occupancy volumes and the
camera remains freely orbitable in self-contained Plotly HTML.
"""
import argparse
import os

import numpy as np
import plotly.graph_objects as go

try:
    from plot_gtbox_future_occ_anim import (
        COLORS, DT, STEPS, choose_index, load_infos, trajectory_data,
    )
except ImportError:
    from tools.viz.plot_gtbox_future_occ_anim import (
        COLORS, DT, STEPS, choose_index, load_infos, trajectory_data,
    )


def load_occ_voxels(info, data_root, max_voxels=18000):
    """Load true current SurroundOcc occupied voxel centers in the model BEV crop."""
    path = str(info.get('occ_path', ''))
    path = path if os.path.isabs(path) else os.path.join(data_root, path)
    if not path or not os.path.exists(path):
        print(f'[warn] missing Occ file: {path}')
        return np.empty((0, 3), np.float32)
    occ = np.load(path)
    i, j, k, label = occ[:, 0], occ[:, 1], occ[:, 2], occ[:, 3]
    keep = ((i >= 40) & (i < 160) & (j >= 40) & (j < 160) &
            (k >= 6) & (k < 14) & (label != 17))
    xyz = np.stack([
        (i[keep] - 40) * .5 + .25 - 30.,
        (j[keep] - 40) * .5 + .25 - 30.,
        (k[keep] - 6) * .5 + .25 - 2.,
    ], axis=1).astype(np.float32)
    if len(xyz) > max_voxels:
        xyz = xyz[np.random.RandomState(0).choice(len(xyz), max_voxels, replace=False)]
    return xyz


def cuboid_mesh(box, yaw):
    """Vertices and triangle faces for one yaw-rotated solid GT-box occupancy volume."""
    x, y, z, dx, dy, dz = box[:6]
    xy = np.array([[-dx / 2, -dy / 2], [dx / 2, -dy / 2],
                   [dx / 2, dy / 2], [-dx / 2, dy / 2]], np.float32)
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, -s], [s, c]], np.float32)
    xy = xy @ rotation.T + np.array([x, y], np.float32)
    low, high = z - dz / 2, z + dz / 2
    vertices = np.array([
        [xy[0, 0], xy[0, 1], low], [xy[1, 0], xy[1, 1], low],
        [xy[2, 0], xy[2, 1], low], [xy[3, 0], xy[3, 1], low],
        [xy[0, 0], xy[0, 1], high], [xy[1, 0], xy[1, 1], high],
        [xy[2, 0], xy[2, 1], high], [xy[3, 0], xy[3, 1], high],
    ], np.float32)
    faces = np.array([
        [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
        [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0],
    ], np.int32)
    return vertices, faces


def box_at_step(boxes, cumulative, cumulative_yaw, box_id, step):
    box = boxes[box_id].copy()
    yaw = float(box[6])
    if step:
        box[:2] += cumulative[box_id, step - 1]
        yaw += cumulative_yaw[box_id, step - 1]
    return box, yaw


def build(info, index, out, data_root, max_boxes, min_motion):
    boxes, names, cumulative, cumulative_yaw, valid, extent, movable = trajectory_data(info)
    use = movable & valid.any(axis=1) & (extent >= min_motion)
    selected = np.flatnonzero(use)
    selected = selected[np.argsort(extent[selected])[::-1][:max_boxes]]
    if not len(selected):
        raise ValueError('No visible moving trajectories; choose another sample or lower --min-motion.')
    context = load_occ_voxels(info, data_root)

    traces = [go.Scatter3d(
        x=context[:, 0], y=context[:, 1], z=context[:, 2], mode='markers',
        marker=dict(size=2.2, color='rgba(105,105,105,0.20)', symbol='square'),
        name='current SurroundOcc occupied voxels (context)', hoverinfo='skip')]
    mesh_indices = []
    all_path_points = []
    for local_id, box_id in enumerate(selected):
        color = COLORS[local_id % len(COLORS)]
        base = boxes[box_id]
        path_xy = np.concatenate([base[None, :2], base[None, :2] + cumulative[box_id]], axis=0)
        path_z = np.full(STEPS + 1, base[2] + base[5] / 2 + .2, np.float32)
        all_path_points.append(np.column_stack([path_xy, path_z]))
        label = f'#{box_id} {names[box_id]} | GT path {extent[box_id]:.1f}m'
        # Wide transparent line first produces a simple 3D "light ray" effect.
        traces += [
            go.Scatter3d(x=path_xy[:, 0], y=path_xy[:, 1], z=path_z, mode='lines',
                         line=dict(color=color, width=18), opacity=.16,
                         showlegend=False, hoverinfo='skip'),
            go.Scatter3d(x=path_xy[:, 0], y=path_xy[:, 1], z=path_z,
                         mode='lines+markers', line=dict(color=color, width=5),
                         marker=dict(size=4.5, color=color), name=label,
                         hovertemplate=label + '<extra></extra>'),
        ]
        initial, faces = cuboid_mesh(*box_at_step(boxes, cumulative, cumulative_yaw, box_id, 0))
        mesh_indices.append(len(traces))
        traces.append(go.Mesh3d(
            x=initial[:, 0], y=initial[:, 1], z=initial[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], color=color, opacity=.78,
            flatshading=True, name=f'animated GT occupancy | #{box_id} {names[box_id]}',
            hovertemplate=f'GT box occupancy<br>#{box_id} {names[box_id]}<extra></extra>'))
    traces.append(go.Scatter3d(
        x=[0], y=[0], z=[0], mode='markers+text', text=['ego / LiDAR'],
        textposition='top center', marker=dict(size=6, color='black', symbol='diamond'),
        name='ego / LiDAR'))

    frames = []
    for step in range(STEPS + 1):
        data = []
        for box_id in selected:
            vertices, _ = cuboid_mesh(*box_at_step(boxes, cumulative, cumulative_yaw, box_id, step))
            data.append(dict(type='mesh3d', x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2]))
        frames.append(go.Frame(name=str(step), traces=mesh_indices, data=data))

    path_points = np.concatenate(all_path_points, axis=0)
    if len(context):
        # Keep the camera focused around the tracked motion, not all 60m context.
        path_points = np.concatenate([path_points, np.array([[0, 0, 0]], np.float32)])
    pad = 7.
    xr = [max(-30., float(path_points[:, 0].min() - pad)), min(30., float(path_points[:, 0].max() + pad))]
    yr = [max(-30., float(path_points[:, 1].min() - pad)), min(30., float(path_points[:, 1].max() + pad))]
    zmin = max(-2., float(np.min(boxes[selected, 2] - boxes[selected, 5] / 2) - 1.))
    zmax = min(4., float(np.max(boxes[selected, 2] + boxes[selected, 5] / 2) + 1.5))
    dx, dy, dz = xr[1] - xr[0], yr[1] - yr[0], zmax - zmin
    largest = max(dx, dy, dz)
    labels = ['t=0 (now)'] + [f't={t} (+{t * DT:.1f}s)' for t in range(1, STEPS + 1)]
    slider = dict(active=0, x=.12, y=0, len=.83, currentvalue=dict(prefix='frame: '),
                  steps=[dict(label=labels[t], method='animate', args=[
                      [str(t)], dict(frame=dict(duration=0, redraw=True), mode='immediate',
                                     transition=dict(duration=0))]) for t in range(STEPS + 1)])
    buttons = dict(type='buttons', direction='left', x=0, y=0, showactive=False, buttons=[
        dict(label='Play', method='animate', args=[None, dict(
            frame=dict(duration=700, redraw=True), fromcurrent=True, transition=dict(duration=150))]),
        dict(label='Pause', method='animate', args=[[None], dict(
            frame=dict(duration=0, redraw=False), mode='immediate', transition=dict(duration=0))]),
    ])
    fig = go.Figure(data=traces, frames=frames)
    fig.update_layout(
        title=(f'3D GT-box trajectory supervision | sample {index} | {len(selected)} moving instances<br>'
               'gray voxels = current SurroundOcc context; colored solids = animated GT-box occupancy; '
               'colored rays = real future GT-box center paths'),
        margin=dict(l=0, r=0, t=60, b=42), updatemenus=[buttons], sliders=[slider],
        scene=dict(
            xaxis=dict(title='x: forward (m)', range=xr, backgroundcolor='white', gridcolor='#ddd'),
            yaxis=dict(title='y: left (m)', range=yr, backgroundcolor='white', gridcolor='#ddd'),
            zaxis=dict(title='z: up (m)', range=[zmin, zmax], backgroundcolor='white', gridcolor='#ddd'),
            aspectmode='manual', aspectratio=dict(x=dx / largest, y=dy / largest, z=dz / largest),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.05)), bgcolor='white'),
        legend=dict(x=0.01, y=.99, bgcolor='rgba(255,255,255,.75)'), width=1280, height=900)
    fig.write_html(out, include_plotlyjs=True, auto_play=False)
    print(f'saved {out}')
    print(f'index={index} token={info.get("token", "")} selected={selected.tolist()}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pkl', required=True)
    parser.add_argument('--data-root', default='.')
    parser.add_argument('--out', required=True)
    parser.add_argument('--index', type=int, default=-1)
    parser.add_argument('--max-boxes', type=int, default=6)
    parser.add_argument('--min-motion', type=float, default=1.0)
    args = parser.parse_args()
    infos = load_infos(args.pkl)
    index = args.index if args.index >= 0 else choose_index(infos, args.max_boxes, args.min_motion)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    build(infos[index], index, args.out, args.data_root, args.max_boxes, args.min_motion)


if __name__ == '__main__':
    main()