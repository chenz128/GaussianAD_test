"""Step 1 (frontier diffusion): pure-geometry visualization of the frontier region U_tau.

No model, no teacher, no GPU. Uses only the ego poses stored in the info pkl and
the SAME lidar2global convention as GaussianTemporalEncoder.warp_anchor, so the
geometry we verify here is exactly what the training pipeline will use.

Frontier definition (all in the CURRENT-frame LIDAR coords, where gaussians live):
    W        = occ window = [-30, 30] x [-30, 30]   (gaussian_head pc_min/grid: 120*0.5)
    cur2fut  = inv(l2g_fut) @ l2g_cur   maps a current-frame point -> future frame
    fut2cur  = inv(l2g_cur) @ l2g_fut   maps a future-frame  point -> current frame
    U_tau    = { x : cur2fut(x) in W  AND  x not in W }
             = region that becomes visible at t+tau but was NOT visible at t.

Run:
    python tools/frontier/viz_frontier_region.py \
        --pkl data/nuscenes_cam/nuscenes_infos_train_gaussian_ad_v4.pkl \
        --tau 6 --out tools/frontier/viz
"""
import argparse
import os
import pickle
from collections import defaultdict

import numpy as np
from pyquaternion import Quaternion

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import Polygon as MplPolygon

# occ / gaussian window in LIDAR frame (gaussian_head.py: pc_min=[-30,-30], grid=0.5, 120 cells)
WIN = 30.0
WIN_CORNERS = np.array([
    [-WIN, -WIN],
    [WIN, -WIN],
    [WIN, WIN],
    [-WIN, WIN],
], dtype=np.float64)


def get_lidar2global(calib_dict, pose_dict):
    """Identical to dataset/utils.py:get_lidar2global."""
    lidar2ego = np.eye(4)
    lidar2ego[:3, :3] = Quaternion(calib_dict["rotation"]).rotation_matrix
    lidar2ego[:3, 3] = np.asarray(calib_dict["translation"]).T
    ego2global = np.eye(4)
    ego2global[:3, :3] = Quaternion(pose_dict["rotation"]).rotation_matrix
    ego2global[:3, 3] = np.asarray(pose_dict["translation"]).T
    return ego2global @ lidar2ego


def lidar2global_of(info):
    lt = info["data"]["LIDAR_TOP"]
    return get_lidar2global(lt["calib"], lt["pose"])


def transform_xy(T, xy):
    """Apply a 4x4 SE3 to (N,2) points at z=0. Returns (N,2)."""
    n = xy.shape[0]
    pts = np.concatenate([xy, np.zeros((n, 1)), np.ones((n, 1))], axis=1)  # (N,4)
    out = (T @ pts.T).T  # (N,4)
    return out[:, :2]


def build_keyframes_per_scene(metadata):
    """Replicate dataset.get_scene_index: per-scene list in metadata order."""
    kf = defaultdict(list)
    for frame in metadata:
        kf[frame[0]].append(tuple(frame))
    return kf


def frontier_mask(fut2cur, grid_xy):
    """grid_xy: (M,2) current-frame cell centers. Returns bool mask of U_tau."""
    cur2fut = np.linalg.inv(fut2cur)
    in_cur = (np.abs(grid_xy[:, 0]) < WIN) & (np.abs(grid_xy[:, 1]) < WIN)
    mapped = transform_xy(cur2fut, grid_xy)
    in_fut = (np.abs(mapped[:, 0]) < WIN) & (np.abs(mapped[:, 1]) < WIN)
    return in_fut & (~in_cur), in_cur, in_fut


def draw_sample(info_cur, info_fut, tau, out_path, tag):
    l2g_cur = lidar2global_of(info_cur)
    l2g_fut = lidar2global_of(info_fut)
    fut2cur = np.linalg.inv(l2g_cur) @ l2g_fut  # future-frame pt -> current-frame

    # ego displacement + heading change (of future ego expressed in current frame)
    ego_fut_in_cur = fut2cur[:2, 3]
    yaw = np.degrees(np.arctan2(fut2cur[1, 0], fut2cur[0, 0]))
    disp = float(np.linalg.norm(ego_fut_in_cur))

    # rasterize a BEV grid large enough to contain BOTH windows fully, so the
    # frontier band is never clipped even for large (highway) displacements.
    lim = WIN + disp + 8.0
    step = 0.5
    gx = np.arange(-lim, lim, step) + step / 2
    gy = np.arange(-lim, lim, step) + step / 2
    GX, GY = np.meshgrid(gx, gy)
    grid_xy = np.stack([GX.ravel(), GY.ravel()], axis=1)
    mask_u, in_cur, in_fut = frontier_mask(fut2cur, grid_xy)

    fig, ax = plt.subplots(figsize=(8, 8))
    # frontier cells
    ax.scatter(grid_xy[mask_u, 0], grid_xy[mask_u, 1], s=4, c="tab:red",
               marker="s", label=f"U_tau (frontier) N={int(mask_u.sum())}")
    # current window outline
    ax.add_patch(MplPolygon(WIN_CORNERS, closed=True, fill=False,
                            edgecolor="black", lw=2, label="current window W(t)"))
    # future window mapped to current frame
    fut_quad = transform_xy(fut2cur, WIN_CORNERS)
    ax.add_patch(MplPolygon(fut_quad, closed=True, fill=False,
                            edgecolor="tab:blue", lw=2, ls="--",
                            label="future window W(t+tau)"))
    # ego markers
    ax.scatter([0], [0], c="black", s=60, marker="o", zorder=5, label="ego(t)")
    ax.scatter([ego_fut_in_cur[0]], [ego_fut_in_cur[1]], c="tab:blue", s=60,
               marker="^", zorder=5, label="ego(t+tau)")
    ax.annotate("", xy=ego_fut_in_cur, xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="tab:green", lw=2))

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m, forward)")
    ax.set_ylabel("y (m, left)")
    ax.set_title(f"{tag}\ntau={tau}  disp={disp:.2f}m  yaw={yaw:+.1f}deg  "
                 f"frontier cells={int(mask_u.sum())}")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    frac = float(mask_u.sum()) / max(1, int(in_fut.sum()))
    return dict(disp=disp, yaw=yaw, n_frontier=int(mask_u.sum()), frontier_frac=frac)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="data/nuscenes_cam/nuscenes_infos_train_gaussian_ad_v4.pkl")
    ap.add_argument("--tau", type=int, default=6)
    ap.add_argument("--out", default="tools/frontier/viz")
    ap.add_argument("--scan", type=int, default=4000, help="how many keyframes to scan for candidates")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with open(args.pkl, "rb") as f:
        data = pickle.load(f)
    infos = data["infos"]
    metadata = data["metadata"]
    kf_per_scene = build_keyframes_per_scene(metadata)

    # collect candidate (cur, fut) pairs with valid t+tau in same scene, measure motion
    cands = []
    scanned = 0
    for scene_token, frames in kf_per_scene.items():
        for pos in range(len(frames) - args.tau):
            st, idx_cur = frames[pos]
            _, idx_fut = frames[pos + args.tau]
            info_cur = infos[st][idx_cur]
            info_fut = infos[st][idx_fut]
            l2g_cur = lidar2global_of(info_cur)
            l2g_fut = lidar2global_of(info_fut)
            fut2cur = np.linalg.inv(l2g_cur) @ l2g_fut
            disp = float(np.linalg.norm(fut2cur[:2, 3]))
            yaw = abs(np.degrees(np.arctan2(fut2cur[1, 0], fut2cur[0, 0])))
            cands.append((disp, yaw, st, idx_cur, idx_fut))
            scanned += 1
            if scanned >= args.scan:
                break
        if scanned >= args.scan:
            break

    cands = np.array(cands, dtype=object)
    disps = np.array([c[0] for c in cands], dtype=float)
    yaws = np.array([c[1] for c in cands], dtype=float)
    print(f"[scan] {len(cands)} candidate (t, t+{args.tau}) pairs")
    print(f"[scan] disp  m : mean={disps.mean():.2f} p50={np.percentile(disps,50):.2f} "
          f"p90={np.percentile(disps,90):.2f} max={disps.max():.2f}")
    print(f"[scan] |yaw| deg: mean={yaws.mean():.2f} p50={np.percentile(yaws,50):.2f} "
          f"p90={np.percentile(yaws,90):.2f} max={yaws.max():.2f}")

    picks = []
    # 2 straight fast: large disp, small yaw
    order_straight = sorted(range(len(cands)),
                            key=lambda i: (-disps[i] + 5.0 * yaws[i]))
    picks += [("straight", i) for i in order_straight[:2]]
    # 2 turning: large yaw
    order_turn = sorted(range(len(cands)), key=lambda i: -yaws[i])
    picks += [("turn", i) for i in order_turn[:2]]
    # 1 near-static
    order_static = sorted(range(len(cands)), key=lambda i: disps[i])
    picks += [("static", order_static[0])]

    print("\n[render]")
    for tag, i in picks:
        disp, yaw, st, idx_cur, idx_fut = cands[i]
        info_cur = infos[st][idx_cur]
        info_fut = infos[st][idx_fut]
        fname = f"{tag}_{st}_{idx_cur}_to_{idx_fut}.png"
        out_path = os.path.join(args.out, fname)
        stats = draw_sample(info_cur, info_fut, args.tau, out_path, f"{tag}: {st} idx {idx_cur}->{idx_fut}")
        print(f"  [{tag:8s}] {fname}  disp={stats['disp']:.2f}m "
              f"yaw={stats['yaw']:+.1f}deg  frontier_cells={stats['n_frontier']}  "
              f"frontier_frac={stats['frontier_frac']:.2%} of future window")

    print(f"\nSaved {len(picks)} figures to {args.out}/")


if __name__ == "__main__":
    main()
