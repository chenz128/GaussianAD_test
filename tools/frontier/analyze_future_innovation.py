"""Measure explained versus innovative future occupancy.

Static future voxels are "explained" when a same-class current voxel, warped
with the full current-LiDAR to future-LiDAR pose, lies within ``match_radius``.
All dynamic voxels and unmatched static voxels are treated as innovation.

The GT-only pass estimates how much of future occupancy is genuinely new.  If
prediction NPZ files are provided, the script also reports full-scene and
innovation-only occupancy/semantic metrics for those samples.
"""
import argparse
import os
import pickle
from collections import defaultdict

import numpy as np
from pyquaternion import Quaternion
from scipy.spatial import cKDTree


EMPTY = 17
STATIC_CLASSES = np.array([1, 8, 11, 12, 13, 14, 15, 16], dtype=np.int16)
DYNAMIC_CLASSES = np.array([2, 3, 4, 5, 6, 7, 9, 10], dtype=np.int16)
CLASS_NAMES = [
    'others', 'barrier', 'bicycle', 'bus', 'car',
    'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone',
    'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation',
]


def lidar2global(info):
    lidar = info['data']['LIDAR_TOP']
    lidar2ego = np.eye(4, dtype=np.float64)
    lidar2ego[:3, :3] = Quaternion(
        lidar['calib']['rotation']).rotation_matrix
    lidar2ego[:3, 3] = np.asarray(
        lidar['calib']['translation'], dtype=np.float64)
    ego2global = np.eye(4, dtype=np.float64)
    ego2global[:3, :3] = Quaternion(
        lidar['pose']['rotation']).rotation_matrix
    ego2global[:3, 3] = np.asarray(
        lidar['pose']['translation'], dtype=np.float64)
    return ego2global @ lidar2ego


def grid_xyz():
    indices = np.indices((120, 120, 8), dtype=np.float32)
    return np.stack([
        indices[0] * 0.5 + 0.25 - 30.0,
        indices[1] * 0.5 + 0.25 - 30.0,
        indices[2] * 0.5 + 0.25 - 2.0,
    ], axis=-1).reshape(-1, 3)


XYZ = grid_xyz()


def load_occ(info):
    labels = np.full((200, 200, 16), EMPTY, dtype=np.int16)
    raw = np.load(info['occ_path'])
    labels[raw[:, 0], raw[:, 1], raw[:, 2]] = raw[:, 3]
    return labels[40:160, 40:160, 6:14].reshape(-1)


def transform_points(points, transform):
    return points @ transform[:3, :3].T + transform[:3, 3]


def explained_lookup(current_labels, current_to_future, radius):
    """Return (N,17) bool lookup for same-class static explainability."""
    lookup = np.zeros((XYZ.shape[0], 17), dtype=bool)
    for cls in STATIC_CLASSES:
        current_points = XYZ[current_labels == cls]
        if current_points.size == 0:
            continue
        warped = transform_points(current_points, current_to_future)
        distance = cKDTree(warped).query(XYZ, k=1, workers=-1)[0]
        lookup[:, cls] = distance <= radius
    return lookup


def innovation_labels(labels, explained):
    result = np.full(labels.shape, EMPTY, dtype=np.int16)
    dynamic = np.isin(labels, DYNAMIC_CLASSES)
    static = np.isin(labels, STATIC_CLASSES)
    indices = np.arange(labels.size)
    unexplained_static = static & ~explained[indices, labels.clip(0, 16)]
    innovation = dynamic | unexplained_static
    result[innovation] = labels[innovation]
    return result, dynamic, unexplained_static


def confusion_update(confusion, pred, target):
    valid = (target >= 0) & (target <= EMPTY)
    encoded = target[valid].astype(np.int64) * 18 + pred[valid].astype(np.int64)
    confusion += np.bincount(encoded, minlength=18 * 18).reshape(18, 18)


def summarize_confusion(confusion, classes):
    intersection = np.diag(confusion).astype(np.float64)
    union = confusion.sum(0) + confusion.sum(1) - intersection
    valid = union[classes] > 0
    miou = np.mean(intersection[classes][valid] / union[classes][valid])
    target_nonempty = confusion[classes].sum()
    pred_nonempty = confusion[:, classes].sum()
    geo_intersection = confusion[np.ix_(classes, classes)].sum()
    geo_union = target_nonempty + pred_nonempty - geo_intersection
    return 100.0 * miou, 100.0 * geo_intersection / max(geo_union, 1)


def build_keyframes(data):
    keyframes = [tuple(item) for item in data['metadata']]
    sorted_keys = sorted(
        keyframes, key=lambda item: item[0] + str(item[1]).zfill(3))
    per_scene = defaultdict(list)
    for item in keyframes:
        per_scene[item[0]].append(item)
    positions = {
        item: position
        for scene_frames in per_scene.values()
        for position, item in enumerate(scene_frames)
    }
    return sorted_keys, per_scene, positions


def future_info(data, per_scene, positions, key, step):
    frames = per_scene[key[0]]
    position = positions[key] + step
    if position >= len(frames):
        return None
    scene, frame_index = frames[position]
    return data['infos'][scene][frame_index]


def gt_statistics(data, keys, per_scene, positions, sample_indices, radius):
    by_step = [defaultdict(int) for _ in range(6)]
    by_class = defaultdict(lambda: defaultdict(int))
    valid_pairs = 0
    for dataset_index in sample_indices:
        key = keys[dataset_index]
        current = data['infos'][key[0]][key[1]]
        current_labels = load_occ(current)
        current_pose = lidar2global(current)
        for step in range(1, 7):
            future = future_info(data, per_scene, positions, key, step)
            if future is None:
                continue
            future_labels = load_occ(future)
            transform = np.linalg.inv(lidar2global(future)) @ current_pose
            explained = explained_lookup(current_labels, transform, radius)
            innovative, dynamic, new_static = innovation_labels(
                future_labels, explained)
            occupied = (future_labels >= 1) & (future_labels <= 16)
            innovation = innovative != EMPTY
            mapped_current_window = transform_points(XYZ, np.linalg.inv(transform))
            frontier = occupied & (
                (np.abs(mapped_current_window[:, 0]) >= 30.0)
                | (np.abs(mapped_current_window[:, 1]) >= 30.0)
                | (mapped_current_window[:, 2] < -2.0)
                | (mapped_current_window[:, 2] >= 2.0))
            values = by_step[step - 1]
            values['occupied'] += int(occupied.sum())
            values['innovation'] += int(innovation.sum())
            values['dynamic'] += int(dynamic.sum())
            values['new_static'] += int(new_static.sum())
            values['frontier'] += int(frontier.sum())
            for cls in range(1, 17):
                cls_mask = future_labels == cls
                by_class[cls]['total'] += int(cls_mask.sum())
                by_class[cls]['innovation'] += int(
                    (cls_mask & innovation).sum())
            valid_pairs += 1
    return by_step, by_class, valid_pairs


def prediction_statistics(data, keys, per_scene, positions, prediction_specs,
                          radius):
    full_confusion = np.zeros((18, 18), dtype=np.int64)
    innovation_confusion = np.zeros((18, 18), dtype=np.int64)
    results = []
    for dataset_index, path in prediction_specs:
        archive = np.load(path)
        key = keys[dataset_index]
        current = data['infos'][key[0]][key[1]]
        current_labels = load_occ(current)
        current_pose = lidar2global(current)
        sample_values = defaultdict(int)
        for step in range(1, 7):
            if not archive['valid'][step - 1]:
                continue
            future = future_info(data, per_scene, positions, key, step)
            if future is None:
                continue
            target = archive['occ_fut_gt'][step - 1].astype(np.int16)
            pred = archive['occ_fut'][step - 1].astype(np.int16)
            transform = np.linalg.inv(lidar2global(future)) @ current_pose
            explained = explained_lookup(current_labels, transform, radius)
            target_innovation, target_dynamic, target_static = innovation_labels(
                target, explained)
            pred_innovation, _, _ = innovation_labels(pred, explained)
            confusion_update(full_confusion, pred, target)
            confusion_update(
                innovation_confusion, pred_innovation, target_innovation)
            innovation = target_innovation != EMPTY
            occupied_pred = (pred >= 1) & (pred <= 16)
            sample_values['innovation'] += int(innovation.sum())
            sample_values['occupancy_hit'] += int(
                (innovation & occupied_pred).sum())
            sample_values['semantic_hit'] += int(
                (innovation & (pred == target)).sum())
            sample_values['dynamic'] += int(target_dynamic.sum())
            sample_values['dynamic_hit'] += int(
                (target_dynamic & (pred == target)).sum())
            sample_values['new_static'] += int(target_static.sum())
            sample_values['new_static_hit'] += int(
                (target_static & (pred == target)).sum())
        results.append((dataset_index, sample_values))
    return full_confusion, innovation_confusion, results


def ratio(numerator, denominator):
    return 100.0 * numerator / max(denominator, 1)


def parse_prediction(value):
    dataset_index, path = value.split(':', 1)
    return int(dataset_index), path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pkl', default='data/nuscenes_cam/nuscenes_infos_val_gaussian_ad_v4.pkl')
    parser.add_argument('--num-gt-samples', type=int, default=300)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--match-radius', type=float, default=0.76)
    parser.add_argument(
        '--prediction', action='append', default=[],
        help='dataset_index:path_to_occflow_npz; may be repeated')
    args = parser.parse_args()

    with open(args.pkl, 'rb') as handle:
        data = pickle.load(handle)
    keys, per_scene, positions = build_keyframes(data)
    valid_indices = np.array([
        index for index, key in enumerate(keys)
        if positions[key] + 6 < len(per_scene[key[0]])], dtype=np.int64)
    rng = np.random.RandomState(args.seed)
    count = min(args.num_gt_samples, len(valid_indices))
    sampled = rng.choice(valid_indices, count, replace=False)

    by_step, by_class, pairs = gt_statistics(
        data, keys, per_scene, positions, sampled, args.match_radius)
    print(f'GT samples={count} valid_pairs={pairs} radius={args.match_radius:.2f}m')
    for step, values in enumerate(by_step, 1):
        print(
            'step%d occupied=%d innovation=%d (%.2f%%) '
            'dynamic=%.2f%% new_static=%.2f%% frontier=%.2f%%' % (
                step, values['occupied'], values['innovation'],
                ratio(values['innovation'], values['occupied']),
                ratio(values['dynamic'], values['occupied']),
                ratio(values['new_static'], values['occupied']),
                ratio(values['frontier'], values['occupied'])))
    print('GT innovation by class:')
    for cls in range(1, 17):
        values = by_class[cls]
        print('  %2d %-22s %8d/%8d = %6.2f%%' % (
            cls, CLASS_NAMES[cls], values['innovation'], values['total'],
            ratio(values['innovation'], values['total'])))

    prediction_specs = [parse_prediction(value) for value in args.prediction]
    if not prediction_specs:
        return
    full, innovation, sample_results = prediction_statistics(
        data, keys, per_scene, positions, prediction_specs,
        args.match_radius)
    classes = np.arange(1, 17)
    full_miou, full_geo = summarize_confusion(full, classes)
    innovation_miou, innovation_geo = summarize_confusion(
        innovation, classes)
    print(
        'Prediction aggregate: full mIoU=%.2f geoIoU=%.2f | '
        'innovation mIoU=%.2f geoIoU=%.2f' % (
            full_miou, full_geo, innovation_miou, innovation_geo))
    for dataset_index, values in sample_results:
        print(
            '  idx=%d innovation=%d occ_recall=%.2f%% semantic_recall=%.2f%% '
            'dynamic_sem_recall=%.2f%% new_static_sem_recall=%.2f%%' % (
                dataset_index, values['innovation'],
                ratio(values['occupancy_hit'], values['innovation']),
                ratio(values['semantic_hit'], values['innovation']),
                ratio(values['dynamic_hit'], values['dynamic']),
                ratio(values['new_static_hit'], values['new_static'])))


if __name__ == '__main__':
    main()