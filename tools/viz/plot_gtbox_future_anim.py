"""Animate GT-box future tracks stored in GaussianAD infos PKL.

The displayed trajectory is exactly the supervision consumed by v7
``PhysicsLoss.loss_traj``: ``gt_agent_fut_trajs`` is the sequence of
per-step center displacements obtained by following each current GT box's
nuScenes annotation ``next`` chain.  This script accumulates these deltas,
then animates the corresponding GT boxes over t=0..6 (0.5 s per step).

The self-contained HTML contains Play/Pause controls, a draggable time
slider, 3D orbit/zoom, translucent trajectory "light rays", and current
instantaneous-velocity reference rays for comparison.

Example (run on the H20 server):
    /data/chenz/conda_env/splatting/bin/python \
        tools/viz/plot_gtbox_future_anim.py \
        --pkl data/nuscenes_cam/nuscenes_infos_val_gaussian_ad_v4.pkl \
        --index -1 --out viz_gtbox/gt_tracks_auto.html
"""
import argparse
import os
import pickle

import numpy as np
import plotly.graph_objects as go


DT = 0.5
FUTURE_STEPS = 6
MOVABLE_NAMES = {
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'motorcycle', 'bicycle', 'pedestrian',
}


def load_infos(pkl_path):
    """Load either a list or the scene-token -> frame-list PKL layout."""
    with open(pkl_path, 'rb') as handle:
        data = pickle.load(handle)
    infos = data['infos'] if isinstance(data, dict) and 'infos' in data else data
    if isinstance(infos, dict):
        return [frame for frames in infos.values() for frame in frames]
    return infos


def valid_track_mask(fut_delta, fut_mask, min_displacement):
    """Return boxes with a valid, non-trivial future center trajectory."""
    cumulative = np.cumsum(fut_delta, axis=1)
    valid = fut_mask > 0.5
    distance = np.linalg.norm(cumulative, axis=-1)
    return valid.any(axis=1) & (np.where(valid, distance, 0.0).max(axis=1) >= min_displacement)


def select_sample(infos, min_displacement, max_boxes):
    """Choose a frame rich in valid, visibly moving GT tracks."""
    best_index, best_score = 0, -1.0
    for index, info in enumerate(infos):
        boxes = np.asarray(info.get('gt_boxes', []), dtype=np.float32)
        traj = np.asarray(info.get('gt_agent_fut_trajs', []), dtype=np.float32)
        mask = np.asarray(info.get('gt_agent_fut_masks', []), dtype=np.float32)
        if boxes.ndim != 2 or boxes.shape[0] == 0 or traj.shape[0] != boxes.shape[0]:
            continue
        traj = traj.reshape(-1, FUTURE_STEPS, 2)
        mask = mask.reshape(-1, FUTURE_STEPS)
        moving = valid_track_mask(traj, mask, min_displacement)
        cumulative = np.cumsum(traj, axis=1)
        endpoint = np.linalg.norm(cumulative[:, -1], axis=-1)
        score = min(int(moving.sum()), max_boxes) * 100.0 + float(endpoint[moving].sum())
        if score > best_score:
            best_index, best_score = index, score
    return best_index


def box_corners(box, yaw):
    """Return one closed 3D polyline for a [x,y,z,dx,dy,dz] box."""
    x, y, z, dx, dy, dz = box[:6]
    local_xy = np.array([
        [-dx / 2, -dy / 2], [dx / 2, -dy / 2],
        [dx / 2, dy / 2], [-dx / 2, dy / 2],
    ], dtype=np.float32)
    rotation = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw), np.cos(yaw)],
    ], dtype=np.float32)
    xy = local_xy @ rotation.T + np.array([x, y], dtype=np.float32)
    z_bottom, z_top = z - dz / 2, z + dz / 2
    corners = np.array([
        [xy[0, 0], xy[0, 1], z_bottom], [xy[1, 0], xy[1, 1], z_bottom],
        [xy[2, 0], xy[2, 1], z_bottom], [xy[3, 0], xy[3, 1], z_bottom],
        [xy[0, 0], xy[0, 1], z_bottom], [xy[0, 0], xy[0, 1], z_top],
        [xy[1, 0], xy[1, 1], z_top], [xy[1, 0], xy[1, 1], z_bottom],
        [xy[2, 0], xy[2, 1], z_bottom], [xy[2, 0], xy[2, 1], z_top],
        [xy[3, 0], xy[3, 1], z_top], [xy[3, 0], xy[3, 1], z_bottom],
        [xy[3, 0], xy[3, 1], z_top], [xy[0, 0], xy[0, 1], z_top],
        [xy[1, 0], xy[1, 1], z_top], [xy[2, 0], xy[2, 1], z_top],
        [xy[3, 0], xy[3, 1], z_top],
    ], dtype=np.float32)
    return corners


def trace_box(box, yaw, color, name, visible=True):
    corners = box_corners(box, yaw)
    return go.Scatter3d(
        x=corners[:, 0], y=corners[:, 1], z=corners[:, 2],
        mode='lines', line=dict(color=color, width=6), name=name,
        hovertemplate=name + '<extra></extra>', visible=visible,
    )


def build_animation(info, index, out_html, min_displacement, max_boxes,
                    moving_only, show_velocity):
    boxes = np.asarray(info['gt_boxes'], dtype=np.float32)
    names = np.asarray(info.get('gt_names', ['unknown'] * len(boxes))).astype(str)
    velocity = np.nan_to_num(np.asarray(
        info.get('gt_velocity', np.zeros((len(boxes), 2))), dtype=np.float32))
    delta = np.nan_to_num(np.asarray(info['gt_agent_fut_trajs'], dtype=np.float32))
    valid = np.nan_to_num(np.asarray(info['gt_agent_fut_masks'], dtype=np.float32)) > 0.5
    yaw_delta = np.nan_to_num(np.asarray(
        info.get('gt_agent_fut_yaw', np.zeros((len(boxes), FUTURE_STEPS))),
        dtype=np.float32))

    delta = delta.reshape(-1, FUTURE_STEPS, 2)
    valid = valid.reshape(-1, FUTURE_STEPS)
    yaw_delta = yaw_delta.reshape(-1, FUTURE_STEPS)
    cumulative = np.cumsum(delta, axis=1)
    cumulative_yaw = np.cumsum(yaw_delta, axis=1)
    has_track = valid_track_mask(delta, valid, min_displacement)
    movable = np.array([name in MOVABLE_NAMES for name in names])
    selected = has_track if moving_only else valid.any(axis=1)
    selected &= movable

    # Prefer long, visible motion; cap keeps a self-contained HTML responsive.
    extent = np.where(valid, np.linalg.norm(cumulative, axis=-1), 0.0).max(axis=1)
    selected_index = np.flatnonzero(selected)
    selected_index = selected_index[np.argsort(extent[selected_index])[::-1][:max_boxes]]
    if len(selected_index) == 0:
        raise ValueError('no valid moving GT-box tracks; lower --min-displacement or choose another --index')

    colors = ['#ff3b30', '#007aff', '#34c759', '#ff9500', '#af52de', '#00c7be', '#ff2d55', '#5856d6']
    trajectories = []
    traces = []
    box_trace_indices = []

    for local_id, box_id in enumerate(selected_index):
        color = colors[local_id % len(colors)]
        base = boxes[box_id].copy()
        path = np.concatenate([base[None, :2], base[None, :2] + cumulative[box_id]], axis=0)
        trajectories.append(path)
        label = f'#{box_id} {names[box_id]} | {extent[box_id]:.1f}m / {valid[box_id].sum() * DT:.1f}s'

        # "Light ray": broad translucent halo under a narrow saturated core.
        traces.append(go.Scatter3d(
            x=path[:, 0], y=path[:, 1], z=np.full(FUTURE_STEPS + 1, base[2] + base[5] / 2 + 0.15),
            mode='lines', line=dict(color=color, width=16), opacity=0.18,
            name=label + ' trajectory glow', hoverinfo='skip', showlegend=False))
        traces.append(go.Scatter3d(
            x=path[:, 0], y=path[:, 1], z=np.full(FUTURE_STEPS + 1, base[2] + base[5] / 2 + 0.15),
            mode='lines+markers', line=dict(color=color, width=5),
            marker=dict(size=4, color=color), name=label,
            hovertemplate=label + '<extra></extra>'))

        if show_velocity:
            velocity_path = np.stack([
                base[:2], base[:2] + velocity[box_id] * FUTURE_STEPS * DT], axis=0)
            traces.append(go.Scatter3d(
                x=velocity_path[:, 0], y=velocity_path[:, 1],
                z=np.full(2, base[2] + base[5] / 2 + 0.05), mode='lines',
                line=dict(color='rgba(80,80,80,0.8)', width=3, dash='dash'),
                name=label + ' constant-velocity reference', showlegend=False,
                hovertemplate='dashed: current velocity straight-line reference<extra></extra>'))

        box_trace_indices.append(len(traces))
        traces.append(trace_box(base, float(base[6]), color, label))

    # Ego origin makes the LIDAR coordinate convention explicit.
    traces.append(go.Scatter3d(
        x=[0], y=[0], z=[0], mode='markers+text',
        marker=dict(size=7, color='black', symbol='diamond'), text=['ego / LiDAR'],
        textposition='top center', name='ego / LiDAR'))

    frames = []
    for step in range(FUTURE_STEPS + 1):
        frame_data = []
        for local_id, box_id in enumerate(selected_index):
            base = boxes[box_id].copy()
            if step > 0:
                base[:2] += cumulative[box_id, step - 1]
                yaw = float(base[6] + cumulative_yaw[box_id, step - 1])
            else:
                yaw = float(base[6])
            corners = box_corners(base, yaw)
            frame_data.append(dict(
                type='scatter3d', x=corners[:, 0], y=corners[:, 1], z=corners[:, 2],
            ))
        frames.append(go.Frame(name=str(step), data=frame_data, traces=box_trace_indices))

    all_xy = np.concatenate(trajectories, axis=0)
    span = max(float(np.ptp(all_xy[:, 0])), float(np.ptp(all_xy[:, 1])), 20.0)
    pad = max(8.0, span * 0.25)
    xr = [float(all_xy[:, 0].min() - pad), float(all_xy[:, 0].max() + pad)]
    yr = [float(all_xy[:, 1].min() - pad), float(all_xy[:, 1].max() + pad)]
    zmax = float(np.max(boxes[selected_index, 2] + boxes[selected_index, 5] / 2) + 2.0)
    zmin = float(np.min(boxes[selected_index, 2] - boxes[selected_index, 5] / 2) - 1.0)
    dx, dy, dz = xr[1] - xr[0], yr[1] - yr[0], zmax - zmin
    longest = max(dx, dy, dz)

    slider_labels = ['t=0 (now)'] + [f't={step} (+{step * DT:.1f}s)' for step in range(1, FUTURE_STEPS + 1)]
    play_menu = dict(
        type='buttons', direction='left', x=0.0, y=0.0, xanchor='left', yanchor='top',
        showactive=False, buttons=[
            dict(label='Play', method='animate', args=[None, dict(
                frame=dict(duration=650, redraw=True), fromcurrent=True,
                transition=dict(duration=150))]),
            dict(label='Pause', method='animate', args=[[None], dict(
                frame=dict(duration=0, redraw=False), mode='immediate',
                transition=dict(duration=0))]),
        ])
    slider = dict(
        active=0, x=0.11, y=0.0, len=0.84, currentvalue=dict(prefix='frame: '),
        steps=[dict(label=slider_labels[step], method='animate', args=[
            [str(step)], dict(frame=dict(duration=0, redraw=True), mode='immediate',
                              transition=dict(duration=0))])
            for step in range(FUTURE_STEPS + 1)])

    title = (
        f'GT-box future tracks | sample={index} | token={info.get("token", "")[:8]} | '
        f'{len(selected_index)} moving instances<br>'
        'solid glow = GT future-box center track; dashed gray = current-velocity straight-line reference'
    )
    fig = go.Figure(data=traces, frames=frames)
    fig.update_layout(
        title=title, margin=dict(l=0, r=0, t=55, b=45),
        updatemenus=[play_menu], sliders=[slider],
        scene=dict(
            xaxis=dict(title='x: forward (m)', range=xr),
            yaxis=dict(title='y: left (m)', range=yr),
            zaxis=dict(title='z: up (m)', range=[zmin, zmax]),
            aspectmode='manual',
            aspectratio=dict(x=dx / longest, y=dy / longest, z=dz / longest),
            bgcolor='white', camera=dict(eye=dict(x=1.45, y=-1.45, z=1.1)),
        ),
    )
    fig.write_html(out_html, include_plotlyjs=True, auto_play=False)
    print(f'saved {out_html}')
    print(f'index={index} token={info.get("token", "")} selected={len(selected_index)} '
          f'box_ids={selected_index.tolist()}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pkl', required=True, help='GaussianAD train/val infos PKL')
    parser.add_argument('--out', required=True, help='output self-contained HTML path')
    parser.add_argument('--index', type=int, default=-1,
                        help='flattened sample index; -1 automatically selects a rich moving frame')
    parser.add_argument('--min-displacement', type=float, default=0.5,
                        help='minimum valid 3 s center displacement (m) to display')
    parser.add_argument('--max-boxes', type=int, default=12,
                        help='maximum moving GT tracks to show')
    parser.add_argument('--all-valid', action='store_true',
                        help='also show valid tracks below --min-displacement')
    parser.add_argument('--no-velocity-reference', action='store_true',
                        help='hide dashed current-velocity straight-line references')
    args = parser.parse_args()

    infos = load_infos(args.pkl)
    index = args.index if args.index >= 0 else select_sample(
        infos, args.min_displacement, args.max_boxes)
    if index >= len(infos):
        raise IndexError(f'index={index} out of range [0, {len(infos) - 1}]')
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    build_animation(
        infos[index], index, args.out, args.min_displacement, args.max_boxes,
        moving_only=not args.all_valid, show_velocity=not args.no_velocity_reference)


if __name__ == '__main__':
    main()