#!/usr/bin/env python3
"""Sanity-check / quality-stats for a converted GaussianAD PKL (v6).

Usage:
    python tools/stats_gaussianad_pkl.py \
        --pkl data/nuscenes_cam/nuscenes_infos_train_gaussian_ad_v6.pkl \
        [--pkl-ref path/to/author/original.pkl] \
        [--filter-min-points-in-gt 1]

If `--pkl-ref` is supplied, key statistics are printed side-by-side so the
deviation from the author's PKL is obvious.

The check covers:
1. Top-level structure   (infos: Dict[scene_token, List], metadata: List)
2. Required keys per frame info
3. Field shapes and dtypes
4. gt_ego_fut_cmd distribution (RIGHT / STRAIGHT / LEFT)
5. fut_valid_flag ratio  (frames with all 6 future steps)
6. Agent-future coverage (any mask > 0)
7. Per-frame counts: gt_boxes, gt_map elements per class
8. Velocity / num_lidar_pts distributions
9. Pseudo-label readiness (scene_token & scene_name present)
"""

import argparse
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import mmengine
import numpy as np


# Keys the GaussianAD dataset loader actually touches (see dataset/dataset.py:1700+).
REQUIRED_FRAME_KEYS = [
    "token",
    "lidar_path",
    "timestamp",
    "ego2global_rotation",
    "ego2global_translation",
    "lidar2ego_rotation",
    "lidar2ego_translation",
    "data",
    "gt_boxes",
    "gt_names",
    "gt_velocity",
    "num_lidar_pts",
    "num_radar_pts",
    "gt_map",
    "sweeps",
    "fut_valid_flag",
    "map_location",
    "gt_ego_his_trajs",
    "gt_ego_fut_trajs",
    "gt_ego_fut_masks",
    "gt_ego_fut_cmd",
    "gt_ego_lcf_feat",
    "gt_agent_fut_trajs",
    "gt_agent_fut_masks",
    "gt_agent_fut_goal",
    "gt_agent_lcf_feat",
    "gt_agent_fut_yaw",
    "occ_path",
    "has_surroundocc",
]
PSEUDO_LABEL_KEYS = ["scene_token", "scene_name"]
CMD_NAMES = ["RIGHT", "STRAIGHT", "LEFT"]


# ------------------------------- core stats ------------------------------- #
def _gather_infos(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    infos = data.get("infos", data)
    if isinstance(infos, dict):
        flat = []
        for v in infos.values():
            if isinstance(v, list):
                flat.extend(v)
        return flat
    if isinstance(infos, list):
        return infos
    raise TypeError(f"Unrecognised infos type: {type(infos)}")


def _safe_arr(x: Any) -> np.ndarray:
    try:
        return np.asarray(x)
    except Exception:
        return np.array([])


def _percentile_dict(arr: np.ndarray) -> Dict[str, float]:
    if arr.size == 0:
        return {"min": float("nan"), "p25": float("nan"), "median": float("nan"), "p75": float("nan"), "max": float("nan"), "mean": float("nan")}
    return {
        "min": float(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def compute_stats(pkl_path: str, filter_min_pts: int = 1) -> Dict[str, Any]:
    print(f"[load] {pkl_path}")
    data = mmengine.load(pkl_path)
    stats: Dict[str, Any] = {"path": pkl_path}

    # ── Top-level layout ──
    stats["has_infos_dict"] = isinstance(data.get("infos"), dict)
    stats["has_metadata"] = "metadata" in data and isinstance(data["metadata"], list)
    if stats["has_metadata"]:
        stats["metadata_len"] = len(data["metadata"])
    if stats["has_infos_dict"]:
        stats["num_scenes"] = len(data["infos"])

    all_infos = _gather_infos(data)
    n_frames = len(all_infos)
    stats["num_frames"] = n_frames
    if n_frames == 0:
        return stats

    # ── Key presence ──
    missing_key_counter: Counter = Counter()
    for info in all_infos:
        for k in REQUIRED_FRAME_KEYS + PSEUDO_LABEL_KEYS:
            if k not in info:
                missing_key_counter[k] += 1
    stats["missing_keys"] = dict(missing_key_counter)

    # ── Per-frame aggregations ──
    cmd_count = np.zeros(3, dtype=np.int64)
    fut_valid_count = 0
    fut_mask_steps_dist: Counter = Counter()
    agent_any_count = 0
    agent_full_count = 0
    box_counts: List[int] = []
    map_counts: Dict[str, List[int]] = {"divider": [], "ped_crossing": [], "boundary": []}
    velocity_norms: List[float] = []
    n_lidar_pts_all: List[int] = []
    n_radar_pts_all: List[int] = []
    n_lidar_pts_nonzero_rate = 0
    has_surroundocc_count = 0
    map_locations: Counter = Counter()
    lcf_feat_nonzero_dims: Counter = Counter()  # which dims of ego_lcf_feat are non-zero
    agent_lcf_feat_nonzero_dims: Counter = Counter()
    gt_name_count: Counter = Counter()
    gt_boxes_filtered: List[int] = []  # mimicking dataset's num_lidar_pts filter

    for info in all_infos:
        # ego_fut_cmd
        cmd = _safe_arr(info.get("gt_ego_fut_cmd"))
        if cmd.size == 3:
            cmd_count[int(np.argmax(cmd))] += 1

        # fut_valid_flag
        fut_flag = _safe_arr(info.get("fut_valid_flag")).reshape(-1)
        if fut_flag.size > 0 and bool(fut_flag[0]):
            fut_valid_count += 1

        # ego fut masks number of valid steps
        em = _safe_arr(info.get("gt_ego_fut_masks")).reshape(-1)
        fut_mask_steps_dist[int((em > 0).sum())] += 1

        # agent fut coverage
        am = _safe_arr(info.get("gt_agent_fut_masks"))
        n_agents = int(am.shape[0]) if am.ndim >= 1 else 0
        if n_agents > 0 and (am > 0).any():
            agent_any_count += 1
            row_full = (am > 0).sum(axis=1) == am.shape[1] if am.ndim == 2 else np.array([])
            if row_full.any():
                agent_full_count += 1

        # gt boxes
        gb = _safe_arr(info.get("gt_boxes"))
        n_box = int(gb.shape[0]) if gb.ndim >= 1 else 0
        box_counts.append(n_box)

        # gt_names freq
        for nm in _safe_arr(info.get("gt_names")).tolist():
            gt_name_count[str(nm)] += 1

        # filtered box count (mimic dataset filter)
        n_lp = _safe_arr(info.get("num_lidar_pts")).reshape(-1)
        if n_lp.size == n_box and n_box > 0:
            gt_boxes_filtered.append(int((n_lp >= filter_min_pts).sum()))
            n_lidar_pts_all.extend(n_lp.tolist())
            if (n_lp > 0).any():
                n_lidar_pts_nonzero_rate += 1
        else:
            gt_boxes_filtered.append(n_box)  # no filter applicable

        n_rp = _safe_arr(info.get("num_radar_pts")).reshape(-1)
        if n_rp.size > 0:
            n_radar_pts_all.extend(n_rp.tolist())

        # gt_map elements
        gmap = info.get("gt_map") or {}
        for k in map_counts:
            map_counts[k].append(len(gmap.get(k, [])))

        # velocity magnitudes (only of nonzero ones)
        vel = _safe_arr(info.get("gt_velocity"))
        if vel.size > 0 and vel.ndim == 2 and vel.shape[1] >= 2:
            mag = np.linalg.norm(vel[:, :2], axis=1)
            velocity_norms.extend(mag[np.isfinite(mag)].tolist())

        # has_surroundocc
        hocc = _safe_arr(info.get("has_surroundocc")).reshape(-1)
        if hocc.size > 0 and bool(hocc[0]):
            has_surroundocc_count += 1

        # map_location
        loc = info.get("map_location")
        if loc:
            map_locations[str(loc)] += 1

        # ego_lcf_feat
        elf = _safe_arr(info.get("gt_ego_lcf_feat")).reshape(-1)
        for d in range(min(9, elf.size)):
            if abs(float(elf[d])) > 1e-6:
                lcf_feat_nonzero_dims[d] += 1

        # agent_lcf_feat (per-row nonzero summary)
        alf = _safe_arr(info.get("gt_agent_lcf_feat"))
        if alf.ndim == 2 and alf.shape[1] >= 9:
            for d in range(9):
                if (np.abs(alf[:, d]) > 1e-6).any():
                    agent_lcf_feat_nonzero_dims[d] += 1

    # ── Pack ──
    stats["cmd_distribution"] = {
        name: int(cmd_count[i]) for i, name in enumerate(CMD_NAMES)
    }
    stats["cmd_ratio"] = {
        name: float(cmd_count[i] / max(n_frames, 1)) for i, name in enumerate(CMD_NAMES)
    }
    stats["fut_valid_rate"] = float(fut_valid_count / n_frames)
    stats["fut_mask_steps_distribution"] = dict(sorted(fut_mask_steps_dist.items()))
    stats["agent_any_future_rate"] = float(agent_any_count / n_frames)
    stats["agent_full_future_rate"] = float(agent_full_count / n_frames)
    stats["gt_boxes_per_frame"] = _percentile_dict(np.asarray(box_counts, dtype=np.float32))
    stats["gt_boxes_per_frame_filtered"] = _percentile_dict(
        np.asarray(gt_boxes_filtered, dtype=np.float32)
    )
    stats["gt_name_top10"] = dict(gt_name_count.most_common(10))
    for k, lst in map_counts.items():
        stats[f"map_{k}_per_frame"] = _percentile_dict(np.asarray(lst, dtype=np.float32))
    stats["velocity_norm"] = _percentile_dict(np.asarray(velocity_norms, dtype=np.float32))
    stats["num_lidar_pts"] = _percentile_dict(np.asarray(n_lidar_pts_all, dtype=np.float32))
    stats["num_radar_pts"] = _percentile_dict(np.asarray(n_radar_pts_all, dtype=np.float32))
    stats["frames_with_any_lidar_pts"] = int(n_lidar_pts_nonzero_rate)
    stats["frames_with_any_lidar_pts_rate"] = float(n_lidar_pts_nonzero_rate / n_frames)
    stats["has_surroundocc_rate"] = float(has_surroundocc_count / n_frames)
    stats["map_locations"] = dict(map_locations)
    stats["ego_lcf_feat_nonzero_dim_count"] = {
        f"dim{d}": int(lcf_feat_nonzero_dims.get(d, 0)) for d in range(9)
    }
    stats["agent_lcf_feat_nonzero_dim_count_frames"] = {
        f"dim{d}": int(agent_lcf_feat_nonzero_dims.get(d, 0)) for d in range(9)
    }

    return stats


# --------------------- "expected healthy" reference ---------------------- #
HEALTHY_REFERENCE = {
    "cmd_ratio.STRAIGHT": (0.55, 0.85, "STRAIGHT command ratio"),
    "cmd_ratio.LEFT":     (0.05, 0.25, "LEFT command ratio"),
    "cmd_ratio.RIGHT":    (0.05, 0.25, "RIGHT command ratio"),
    "fut_valid_rate":     (0.80, 1.00, "fraction of frames with 6 valid future steps"),
    "agent_any_future_rate": (0.65, 1.00, "fraction of frames where at least one agent has a future"),
    "frames_with_any_lidar_pts_rate": (0.85, 1.00, "fraction of frames whose annotations actually have lidar pts"),
    "map_divider_per_frame.mean":     (4.0, 20.0, "avg divider lines"),
    "map_ped_crossing_per_frame.mean":(0.5, 6.0,  "avg ped_crossings"),
    "map_boundary_per_frame.mean":    (2.0, 14.0, "avg road boundaries"),
    "gt_boxes_per_frame.mean":        (15.0, 80.0, "avg gt_boxes per frame"),
}


def _get_nested(d: Dict[str, Any], path: str) -> Any:
    cur: Any = d
    for k in path.split("."):
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return None
    return cur


def health_check(stats: Dict[str, Any]) -> List[str]:
    issues = []
    for key, (lo, hi, desc) in HEALTHY_REFERENCE.items():
        v = _get_nested(stats, key)
        if v is None:
            issues.append(f"[MISS] {key} ({desc}) absent")
            continue
        if not (lo <= v <= hi):
            issues.append(f"[WARN] {key}={v:.3f} outside healthy range [{lo}, {hi}] ({desc})")
    # missing key check
    mk = stats.get("missing_keys", {})
    n_frames = stats["num_frames"]
    for k, cnt in mk.items():
        if k in PSEUDO_LABEL_KEYS and cnt > 0:
            issues.append(f"[FAIL] pseudo-label key '{k}' missing in {cnt}/{n_frames} frames")
        elif k in REQUIRED_FRAME_KEYS and cnt > 0:
            issues.append(f"[FAIL] required key '{k}' missing in {cnt}/{n_frames} frames")
    return issues


# ------------------------------- printing ------------------------------- #
def _fmt_pct(v: Any) -> str:
    if isinstance(v, float):
        return f"{v*100:6.2f}%"
    return str(v)


def _fmt_float(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:8.3f}"
    return str(v)


def print_stats(stats: Dict[str, Any], stats_ref: Optional[Dict[str, Any]] = None) -> None:
    def both(key: str, fmt=_fmt_float) -> Tuple[str, str]:
        v = _get_nested(stats, key)
        vr = _get_nested(stats_ref, key) if stats_ref is not None else None
        return fmt(v), fmt(vr) if vr is not None else "—"

    print("=" * 80)
    print(f"PKL : {stats['path']}")
    if stats_ref:
        print(f"REF : {stats_ref['path']}")
    print("=" * 80)

    print(f"\n[1] Top-level structure")
    print(f"    has_infos_dict           : {stats.get('has_infos_dict')}")
    print(f"    has_metadata             : {stats.get('has_metadata')}")
    print(f"    num_scenes               : {stats.get('num_scenes')}"
          + (f"  (ref={stats_ref.get('num_scenes')})" if stats_ref else ""))
    print(f"    num_frames               : {stats.get('num_frames')}"
          + (f"  (ref={stats_ref.get('num_frames')})" if stats_ref else ""))
    print(f"    metadata_len             : {stats.get('metadata_len')}")

    print(f"\n[2] Missing keys")
    mk = stats.get("missing_keys", {})
    if not mk:
        print("    (none)")
    else:
        for k, v in sorted(mk.items(), key=lambda x: -x[1]):
            tag = "PSEUDO" if k in PSEUDO_LABEL_KEYS else "REQ"
            print(f"    [{tag}] {k:30s} missing in {v}/{stats['num_frames']} frames")

    print(f"\n[3] Ego future command distribution")
    for name in CMD_NAMES:
        cnt = stats["cmd_distribution"][name]
        ratio = stats["cmd_ratio"][name]
        if stats_ref:
            r_ratio = stats_ref["cmd_ratio"].get(name, 0.0)
            print(f"    {name:9s}: {cnt:7d}  ({ratio*100:5.2f}%)   ref={r_ratio*100:5.2f}%")
        else:
            print(f"    {name:9s}: {cnt:7d}  ({ratio*100:5.2f}%)")

    print(f"\n[4] Future planning coverage")
    a, b = both("fut_valid_rate", _fmt_pct)
    print(f"    fut_valid_rate (all 6 future steps): {a}   ref={b}")
    a, b = both("agent_any_future_rate", _fmt_pct)
    print(f"    agent_any_future_rate              : {a}   ref={b}")
    a, b = both("agent_full_future_rate", _fmt_pct)
    print(f"    agent_full_future_rate             : {a}   ref={b}")
    print(f"    fut_mask_steps_distribution        : {stats['fut_mask_steps_distribution']}")

    print(f"\n[5] gt_boxes per frame")
    for k in ("min", "p25", "median", "mean", "p75", "max"):
        a, b = both(f"gt_boxes_per_frame.{k}")
        print(f"    {k:6s}: {a}   ref={b}")
    print(f"    gt_boxes_per_frame_filtered (num_lidar_pts >= 1):")
    for k in ("median", "mean", "max"):
        a, b = both(f"gt_boxes_per_frame_filtered.{k}")
        print(f"      {k:6s}: {a}   ref={b}")

    print(f"\n[6] gt_map elements per frame")
    for cls in ("divider", "ped_crossing", "boundary"):
        a, b = both(f"map_{cls}_per_frame.mean")
        med_a, med_b = both(f"map_{cls}_per_frame.median")
        max_a, max_b = both(f"map_{cls}_per_frame.max")
        print(f"    {cls:13s}: mean={a} (ref={b})  median={med_a} (ref={med_b})  max={max_a} (ref={max_b})")

    print(f"\n[7] Velocity & num_lidar_pts")
    a, b = both("velocity_norm.mean")
    print(f"    velocity_norm.mean      : {a}   ref={b}  (m/s in LIDAR frame)")
    a, b = both("velocity_norm.p75")
    print(f"    velocity_norm.p75       : {a}   ref={b}")
    a, b = both("num_lidar_pts.mean")
    print(f"    num_lidar_pts.mean      : {a}   ref={b}")
    a, b = both("num_lidar_pts.median")
    print(f"    num_lidar_pts.median    : {a}   ref={b}")
    a, b = both("frames_with_any_lidar_pts_rate", _fmt_pct)
    print(f"    frames_with_any_lidar_pts_rate: {a}   ref={b}")

    print(f"\n[8] Pseudo-label readiness & misc")
    a, b = both("has_surroundocc_rate", _fmt_pct)
    print(f"    has_surroundocc_rate   : {a}   ref={b}")
    print(f"    map_locations          : {stats['map_locations']}")
    print(f"    gt_name_top10          : {stats['gt_name_top10']}")
    print(f"    ego_lcf_feat nonzero per dim: {stats['ego_lcf_feat_nonzero_dim_count']}")
    print(f"    agent_lcf_feat nonzero per dim (frame-level): {stats['agent_lcf_feat_nonzero_dim_count_frames']}")

    print(f"\n[9] Health check vs heuristic ranges")
    issues = health_check(stats)
    if not issues:
        print("    [OK] all heuristics passed")
    else:
        for s in issues:
            print(f"    {s}")
    print("=" * 80)


# ------------------------------- entrypoint ------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True, help="GaussianAD-format PKL to inspect")
    parser.add_argument("--pkl-ref", default=None, help="Optional reference PKL (author's) to compare against")
    parser.add_argument("--filter-min-points-in-gt", type=int, default=1)
    args = parser.parse_args()

    stats = compute_stats(args.pkl, filter_min_pts=args.filter_min_points_in_gt)
    stats_ref = None
    if args.pkl_ref and os.path.isfile(args.pkl_ref):
        stats_ref = compute_stats(args.pkl_ref, filter_min_pts=args.filter_min_points_in_gt)
    print_stats(stats, stats_ref)


if __name__ == "__main__":
    main()
