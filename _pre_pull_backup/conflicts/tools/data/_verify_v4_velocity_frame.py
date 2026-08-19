#!/usr/bin/env python3
"""Spot-check whether v4 PKL's gt_velocity (vx,vy) is stored in the LIDAR frame.

velocity_norm alone cannot answer this (norm is rotation-invariant). This script
recomputes the correct LIDAR-frame velocity per box via nusc.box_velocity + the
same global->ego->lidar rotation used by the v6 converter, matches boxes by
nearest center, and compares DIRECTION.

For each sampled frame we report, over moving boxes (|v|>1 m/s):
  - cos(angle) between pkl velocity and LIDAR-frame velocity  -> ~1.0 = LIDAR ok
  - cos(angle) between pkl velocity and GLOBAL-frame velocity -> ~1.0 = still global
  - median magnitude ratio
So we can tell which frame the pkl velocity actually lives in.

Usage (remote GaussianAD env):
  /data/chenz/conda_env/GaussianAD/bin/python tools/data/_verify_v4_velocity_frame.py \
      --pkl data/nuscenes_cam/nuscenes_infos_val_gaussian_ad_v4.pkl \
      --dataroot data/nuscenes --version v1.0-trainval --num-frames 20
"""
import argparse
import numpy as np
import mmengine
from nuscenes import NuScenes
from pyquaternion import Quaternion


def gather_infos(data):
    infos = data.get("infos", data)
    if isinstance(infos, dict):
        flat = []
        for v in infos.values():
            if isinstance(v, list):
                flat.extend(v)
        return flat
    return infos


def global_to_lidar(v_xy, info):
    v3 = np.array([float(v_xy[0]), float(v_xy[1]), 0.0], dtype=np.float64)
    ego_rot = Quaternion(info["ego2global_rotation"]).rotation_matrix
    v_ego = np.linalg.inv(ego_rot) @ v3
    lidar_rot = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
    v_lidar = np.linalg.inv(lidar_rot) @ v_ego
    return v_lidar[:2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--dataroot", default="data/nuscenes")
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--num-frames", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--move-thresh", type=float, default=1.0)
    args = ap.parse_args()

    data = mmengine.load(args.pkl)
    infos = gather_infos(data)
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    rng = np.random.default_rng(args.seed)
    idxs = rng.choice(len(infos), size=min(args.num_frames, len(infos)), replace=False)

    cos_lidar_all, cos_global_all, ratio_all = [], [], []
    n_matched = 0

    for fi in idxs:
        info = infos[int(fi)]
        token = info["token"]
        try:
            sample = nusc.get("sample", token)
        except Exception:
            continue
        lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        # box centers stored in pkl (LIDAR frame): gt_boxes[:, :3]
        gt_boxes = np.asarray(info["gt_boxes"], dtype=np.float64)
        gt_vel_pkl = np.asarray(info["gt_velocity"], dtype=np.float64)  # (N,2)
        if gt_boxes.shape[0] == 0:
            continue

        # Build annotation centers in LIDAR frame for matching.
        ego_pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
        cs = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
        ego_t = np.array(ego_pose["translation"])
        ego_R = Quaternion(ego_pose["rotation"]).rotation_matrix
        lid_t = np.array(cs["translation"])
        lid_R = Quaternion(cs["rotation"]).rotation_matrix

        for ann_token in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            c_global = np.array(ann["translation"])
            c_ego = ego_R.T @ (c_global - ego_t)
            c_lidar = lid_R.T @ (c_ego - lid_t)  # center in LIDAR frame

            # nearest pkl box
            d = np.linalg.norm(gt_boxes[:, :3] - c_lidar[None, :], axis=1)
            j = int(np.argmin(d))
            if d[j] > 1.0:
                continue  # no reliable match

            v_global = np.asarray(nusc.box_velocity(ann_token))[:2]
            if not np.all(np.isfinite(v_global)):
                continue
            v_lidar = global_to_lidar(v_global, info)
            v_pkl = gt_vel_pkl[j]

            speed = np.linalg.norm(v_lidar)
            if speed < args.move_thresh:
                continue
            if np.linalg.norm(v_pkl) < 1e-3:
                continue

            n_matched += 1
            cl = float(np.dot(v_pkl, v_lidar) / (np.linalg.norm(v_pkl) * np.linalg.norm(v_lidar) + 1e-9))
            cg = float(np.dot(v_pkl, v_global) / (np.linalg.norm(v_pkl) * np.linalg.norm(v_global) + 1e-9))
            cos_lidar_all.append(cl)
            cos_global_all.append(cg)
            ratio_all.append(float(np.linalg.norm(v_pkl) / (speed + 1e-9)))

    print("=" * 70)
    print(f"matched moving boxes: {n_matched}")
    if n_matched == 0:
        print("no matches — cannot conclude")
        return
    cl = np.array(cos_lidar_all)
    cg = np.array(cos_global_all)
    rt = np.array(ratio_all)
    print(f"cos(pkl, LIDAR-frame vel)  : mean={cl.mean():.4f} median={np.median(cl):.4f} "
          f"frac>0.95={np.mean(cl > 0.95):.3f}")
    print(f"cos(pkl, GLOBAL-frame vel) : mean={cg.mean():.4f} median={np.median(cg):.4f} "
          f"frac>0.95={np.mean(cg > 0.95):.3f}")
    print(f"magnitude ratio |v_pkl|/|v_lidar| : median={np.median(rt):.4f}")
    print("-" * 70)
    if cl.mean() > 0.95 and cl.mean() > cg.mean():
        print("VERDICT: pkl gt_velocity is in the LIDAR frame  ✓ (vel_w target correct)")
    elif cg.mean() > 0.95 and cg.mean() > cl.mean():
        print("VERDICT: pkl gt_velocity is in the GLOBAL frame  ✗ (vel_w direction WRONG)")
    else:
        print("VERDICT: inconclusive / mixed — inspect per-frame")
    print("=" * 70)


if __name__ == "__main__":
    main()
