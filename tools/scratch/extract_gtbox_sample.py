"""Extract one sample's GT 3D boxes from a gaussian_ad PKL into a compact JSON.

Only depends on numpy + pickle (no torch / plotly needed).
Box format in info: gt_boxes (N,7)=[x,y,z,dx,dy,dz,heading] in LIDAR frame,
gt_names (N,), gt_velocity (N,2)=[vx,vy].
"""
import argparse
import json
import os
import pickle

import numpy as np


def load_infos(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    infos = data['infos'] if isinstance(data, dict) and 'infos' in data else data
    # infos may be {scene_token: [frame, ...]} -> flatten to a list of frames
    if isinstance(infos, dict):
        flat = []
        for _tok, frames in infos.items():
            if isinstance(frames, list):
                flat.extend(frames)
            else:
                flat.append(frames)
        return flat
    return infos


def pick_sample(infos, v_thresh=0.5, min_moving=3):
    """Pick the sample with the most 'balanced' static/dynamic box mix."""
    best_i, best_score = 0, -1
    for i, info in enumerate(infos):
        gb = np.asarray(info['gt_boxes'])
        if gb.ndim != 2 or gb.shape[0] == 0:
            continue
        vel = np.asarray(info['gt_velocity'])
        speed = np.linalg.norm(np.nan_to_num(vel), axis=1)
        n_move = int((speed > v_thresh).sum())
        n_static = int((speed <= v_thresh).sum())
        if n_move < min_moving:
            continue
        # prefer samples with a healthy count of both static & moving
        score = min(n_move, 10) + min(n_static, 15)
        if score > best_score:
            best_score, best_i = score, i
    return best_i


def load_occ_points(occ_path, data_root='.'):
    """Load occ npy -> list of [x, y, z, label] voxel centers in LIDAR frame.

    occ npy: (M,4)=[i,j,k,label] on a 200x200x16 grid. Loader crops
    [40:160,40:160,6:14] -> (120,120,8); meshgrid range [-30,30]x[-30,30]x[-2,2]
    reso 0.5. Mapping (full index -> LIDAR xyz):
      x=(i-40)*0.5+0.25-30, y=(j-40)*0.5+0.25-30, z=(k-6)*0.5+0.25-2.
    Label 17 = empty (never stored); keep 0..16.
    """
    if not occ_path:
        return []
    path = occ_path if os.path.isabs(occ_path) else os.path.join(data_root, occ_path)
    if not os.path.exists(path):
        print(f'[occ] not found: {path}')
        return []
    vox = np.load(path)  # (M,4)
    i, j, k, lab = vox[:, 0], vox[:, 1], vox[:, 2], vox[:, 3]
    m = ((i >= 40) & (i < 160) & (j >= 40) & (j < 160) &
         (k >= 6) & (k < 14) & (lab != 17))
    i, j, k, lab = i[m], j[m], k[m], lab[m]
    x = (i - 40) * 0.5 + 0.25 - 30.0
    y = (j - 40) * 0.5 + 0.25 - 30.0
    z = (k - 6) * 0.5 + 0.25 - 2.0
    return [[round(float(a), 2), round(float(b), 2), round(float(c), 2), int(l)]
            for a, b, c, l in zip(x, y, z, lab)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--index', type=int, default=-1,
                    help='sample index; -1 = auto pick balanced sample')
    ap.add_argument('--v-thresh', type=float, default=0.5)
    ap.add_argument('--data-root', default='.')
    args = ap.parse_args()

    infos = load_infos(args.pkl)
    print(f'loaded {len(infos)} samples from {args.pkl}')

    idx = args.index if args.index >= 0 else pick_sample(infos, args.v_thresh)
    info = infos[idx]

    gb = np.asarray(info['gt_boxes'], dtype=np.float64)          # (N,7)
    names = list(map(str, info['gt_names']))
    vel = np.nan_to_num(np.asarray(info['gt_velocity'], dtype=np.float64))  # (N,2)
    speed = np.linalg.norm(vel, axis=1)

    boxes = []
    for j in range(gb.shape[0]):
        boxes.append({
            'x': gb[j, 0], 'y': gb[j, 1], 'z': gb[j, 2],
            'dx': gb[j, 3], 'dy': gb[j, 4], 'dz': gb[j, 5],
            'heading': gb[j, 6],
            'vx': vel[j, 0], 'vy': vel[j, 1],
            'speed': float(speed[j]),
            'dynamic': bool(speed[j] > args.v_thresh),
            'name': names[j],
        })

    occ = load_occ_points(info.get('occ_path'), args.data_root)

    out = {
        'index': int(idx),
        'token': str(info.get('token', '')),
        'scene': str(info.get('scene_name', info.get('scene_token', ''))),
        'v_thresh': args.v_thresh,
        'n_boxes': len(boxes),
        'n_dynamic': int((speed > args.v_thresh).sum()),
        'pc_range': [-30.0, -30.0, -2.0, 30.0, 30.0, 2.0],
        'boxes': boxes,
        'occ': occ,
    }
    with open(args.out, 'w') as f:
        json.dump(out, f)
    print(f'sample idx={idx} token={out["token"]} '
          f'n_boxes={out["n_boxes"]} n_dynamic={out["n_dynamic"]} '
          f'n_occ={len(occ)} -> {args.out}')


if __name__ == '__main__':
    main()
