"""
=============================================================================
 三算法横向对比可视化启动命令（各 10 个场景，场景索引一致便于对比）
=============================================================================
说明：
- 三个实验均使用各自 checkpoints/epoch_15.pth。
- 10 个场景索引 = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135]，均匀覆盖
  验证集 150 个场景（0-149）。每个场景产出：七图 GIF/MP4 + 轨迹 GIF/MP4
  + 每帧轨迹 PNG（七图 PNG 默认清理，--keep-seven-png 可保留）。
- 每个实验以独立 --out-dir 保存，输出目录结构相同
  （scene_<token>/），便于横向对比。

[1] nuscenes_gs25600_base_plan_new（base planner 基线，物理 GPU0）
cd /data/xinyao/navsim_workspace/GaussianAD
/data/chenz/conda_env/splatting/bin/python tools/viz/visualize_occ_map_plan_results.py \
    --py-config config/nuscenes_gs25600_base_plan_new/nuscenes_gs25600_base_plan_new.py \
    --work-dir exp/nuscenes_gs25600_base_plan_new \
    --resume-from exp/nuscenes_gs25600_base_plan_new/checkpoints/epoch_15.pth \
    --scene-indices 0 15 30 45 60 75 90 105 120 135 \
    --device cuda:0 \
    --gif-ms 120 \
    --out-dir exp/nuscenes_gs25600_base_plan_new/vis_occ_map_plan_results_cmp

[2] nuscenes_gs25600_v12_fixempty_ft_plan_futattn_global_residual（v12 fut-attn global-residual，物理 GPU1）
注意：cumm/spconv 在非 cuda:0 设备号上会崩溃（CUDA illegal memory access），
必须用 CUDA_VISIBLE_DEVICES=<物理卡> 把目标卡映射为进程内 cuda:0。
cd /data/xinyao/navsim_workspace/GaussianAD
CUDA_VISIBLE_DEVICES=1 /data/chenz/conda_env/splatting/bin/python tools/viz/visualize_occ_map_plan_results.py \
    --py-config config/nuscenes_gs25600_gtbox_oracle_v12_ft_plan_futattn_global_residual/nuscenes_gs25600_gtbox_oracle_v12_ft_plan_futattn_global_residual.py \
    --work-dir exp/nuscenes_gs25600_v12_fixempty_ft_plan_futattn_global_residual \
    --resume-from exp/nuscenes_gs25600_v12_fixempty_ft_plan_futattn_global_residual/checkpoints/epoch_15.pth \
    --scene-indices 0 15 30 45 60 75 90 105 120 135 \
    --device cuda:0 \
    --gif-ms 120 \
    --out-dir exp/nuscenes_gs25600_v12_fixempty_ft_plan_futattn_global_residual/vis_occ_map_plan_results_cmp

[3] nuscenes_gs25600_v14_ft_plan_residual_ddim（v14 residual-DDIM，物理 GPU2，需 VERIFIED_V12_CHECKPOINT）
cd /data/xinyao/navsim_workspace/GaussianAD
export VERIFIED_V12_CHECKPOINT=/data/xinyao/navsim_workspace/GaussianAD/exp/nuscenes_gs25600_v12_fixempty_ft_plan_futattn_global_residual/checkpoints/epoch_15.pth
CUDA_VISIBLE_DEVICES=2 /data/chenz/conda_env/splatting/bin/python tools/viz/visualize_occ_map_plan_results.py \
    --py-config config/nuscenes_gs25600_gtbox_oracle_v14_ft_plan_residual_ddim/nuscenes_gs25600_gtbox_oracle_v14_ft_plan_residual_ddim.py \
    --work-dir exp/nuscenes_gs25600_v14_ft_plan_residual_ddim \
    --resume-from exp/nuscenes_gs25600_v14_ft_plan_residual_ddim/checkpoints/epoch_15.pth \
    --scene-indices 0 15 30 45 60 75 90 105 120 135 \
    --device cuda:0 \
    --gif-ms 120 \
    --out-dir exp/nuscenes_gs25600_v14_ft_plan_residual_ddim/vis_occ_map_plan_results_cmp

常用参数：
- --scene-indices A B C：自定义场景序号（--list-scenes 可查看全部 150 个场景序号）
- --scenes N [--scene-start S]：连续 N 个场景
- --gif-ms：GIF 每帧时长（默认 350ms）
- --no-video：只保存 PNG 不合成视频；--keep-seven-png：保留七图帧 PNG
=============================================================================
"""

#!/usr/bin/env python3

# 确保 import model / dataset 解析到当前工作区（而非 sys.path 里残留的
# /data/chenz/GaussianAD 等旧路径）
import os
import sys as _sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _REPO not in _sys.path:
    _sys.path.insert(0, _REPO)
if os.getcwd() not in _sys.path:
    _sys.path.insert(0, os.getcwd())

import argparse
import os.path as osp
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import cv2
import numpy as np
from PIL import Image
import torch

from mmengine import Config
from mmseg.models import build_segmentor


NUSC_OCC_COLORS = np.array(
    [
        [0, 0, 0],
        [255, 120, 50],
        [255, 192, 203],
        [255, 255, 0],
        [0, 150, 245],
        [0, 255, 255],
        [255, 127, 0],
        [255, 0, 0],
        [255, 240, 150],
        [135, 60, 0],
        [160, 32, 240],
        [255, 0, 255],
        [139, 137, 137],
        [75, 0, 75],
        [150, 240, 80],
        [230, 230, 250],
        [0, 175, 0],
        [245, 245, 245],
    ],
    dtype=np.float32,
) / 255.0

NUSC_OCC_CLASS_NAMES = [
    "others",
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
    "driveable_surface",
    "other_flat",
    "sidewalk",
    "terrain",
    "manmade",
    "vegetation",
    "empty",
]

MAP_CLASSES = ("divider", "ped_crossing", "boundary")
MAP_COLORS = {
    "divider": "darkorange",
    "ped_crossing": "royalblue",
    "boundary": "crimson",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize occupancy, map, and plan results in a single 7-panel figure."
    )
    parser.add_argument("--py-config", required=True, help="Path to config file.")
    parser.add_argument("--work-dir", required=True, help="Experiment directory used to find latest.pth.")
    parser.add_argument("--resume-from", default="", help="Checkpoint path. Default: <work_dir>/latest.pth.")
    parser.add_argument("--out-dir", default="", help="Directory to save figures.")
    parser.add_argument("--split", choices=("val", "train"), default="val", help="Dataset split to visualize.")
    parser.add_argument("--num-samples", type=int, default=4, help="How many samples to draw.")
    parser.add_argument("--vis-index", type=int, nargs="+", default=None,
                        help="Explicit keyframe indices (keeps scene continuity, e.g. a whole scene's 40 "
                             "frames: --vis-index 0 1 2 ... 39). Overrides --num-samples sampling.")
    parser.add_argument("--scenes", type=int, default=0,
                        help="Number of leading complete scenes to visualize (e.g. 10). Each scene "
                             "yields one GIF/MP4 with exactly --frames-per-scene frames.")
    parser.add_argument("--frames-per-scene", type=int, default=40,
                        help="How many frames per scene to take (default 40). Scenes with more "
                             "frames are truncated; with fewer they keep all their frames.")
    parser.add_argument("--scene-start", type=int, default=0,
                        help="Start scene index (0-based). Combined with --scenes N, visualizes "
                             "scenes [scene_start, scene_start+N).")
    parser.add_argument("--scene-indices", type=int, nargs="+", default=None,
                        help="Custom scene indices (0-based, see --list-scenes) to visualize in the "
                             "given order, e.g. --scene-indices 3 7 12. Each selected scene yields "
                             "its own GIF/MP4; overrides --scenes / --vis-index.")
    parser.add_argument("--list-scenes", action="store_true",
                        help="Print all scene indices (idx: scene_token: frames) and exit without "
                             "loading the model.")
    parser.add_argument("--start-index", type=int, default=0, help="Start index in the chosen dataloader.")
    parser.add_argument("--score-thresh", type=float, default=0.35, help="Map prediction score threshold.")
    parser.add_argument("--device", default="cuda:0", help="Device for inference.")
    parser.add_argument("--dpi", type=int, default=180, help="Saved figure DPI.")
    parser.add_argument("--gif-ms", type=int, default=350, help="Frame duration in milliseconds for scene GIF output.")
    parser.add_argument("--no-video", action="store_true",
                        help="Only save per-frame PNGs; skip scene GIF/MP4 synthesis.")
    parser.add_argument("--no-png", action="store_true",
                        help="Keep only GIF/MP4; delete per-frame PNGs after scene synthesis.")
    parser.add_argument("--keep-seven-png", action="store_true",
                        help="Keep per-frame 7-panel PNGs after scene animation synthesis "
                             "(default: remove them, keeping only the 7-panel videos).")
    parser.add_argument("--grid-shape", type=int, nargs=3, default=(200, 200, 16), help="Occupancy grid shape as X Y Z.")
    parser.add_argument("--empty-label", type=int, default=17, help="Class id used for empty voxels.")
    return parser.parse_args()


def _unwrap_data_container(value):
    if hasattr(value, "data"):
        value = value.data
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    return value


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _empty_map_dict():
    return {name: [] for name in MAP_CLASSES}


def _normalize_map_dict(map_dict):
    normalized = _empty_map_dict()
    if not isinstance(map_dict, dict):
        return normalized
    for name in MAP_CLASSES:
        for pts in map_dict.get(name, []):
            pts_np = _to_numpy(pts)
            if pts_np.size == 0:
                continue
            normalized[name].append(pts_np.reshape(-1, 2))
    return normalized


def _traj_to_polyline(traj):
    traj_np = _to_numpy(traj).reshape(-1, 2)
    cumulative = np.cumsum(traj_np, axis=0)
    origin = np.zeros((1, 2), dtype=np.float32)
    return np.concatenate([origin, cumulative], axis=0)


def _sample_token_from_batch(batch):
    img_meta = _unwrap_data_container(batch["img_metas"][0])
    return str(img_meta.get("sample_idx", "sample"))


def _resolve_dataset_keyframes(dataset):
    current = dataset
    while current is not None:
        keyframes = getattr(current, "keyframes", None)
        if keyframes is not None:
            return keyframes
        current = getattr(current, "dataset", None)
    return None


def _scene_info_from_loader(loader, batch_index):
    keyframes = _resolve_dataset_keyframes(getattr(loader, "dataset", None))
    if keyframes is None or batch_index >= len(keyframes):
        return None, None

    scene_token, scene_frame_index = keyframes[batch_index]
    return str(scene_token), int(scene_frame_index)


def _scene_output_dir(root_out_dir, scene_token):
    scene_dir = root_out_dir / f"scene_{scene_token}"
    scene_dir.mkdir(parents=True, exist_ok=True)
    return scene_dir


def _dataset_len(cfg, split):
    """返回未抽样情况下数据集的总帧数（用 num_samples=0 + 全索引 vis_indices）。"""
    from pathlib import Path
    imageset = cfg.val_dataset_config.get('imageset') if split == 'val' else \
        cfg.train_dataset_config.get('imageset')
    import mmengine
    data = mmengine.load(imageset)
    return len(data['metadata'])


def _checkpoint_path(args):
    if args.resume_from:
        return args.resume_from
    latest = osp.join(args.work_dir, "latest.pth")
    if osp.exists(latest):
        return latest
    raise FileNotFoundError(f"No checkpoint found. Expected: {latest}")


def _load_model(cfg, checkpoint_path, device):
    import model  # noqa: F401

    model_instance = build_segmentor(cfg.model)
    model_instance.init_weights()

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    try:
        load_msg = model_instance.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        model_state = model_instance.state_dict()
        filtered = {}
        for key, value in state_dict.items():
            if key in model_state and hasattr(value, "shape") and model_state[key].shape == value.shape:
                filtered[key] = value
        load_msg = model_instance.load_state_dict(filtered, strict=False)

    print(load_msg)
    model_instance = model_instance.to(device)
    model_instance.eval()
    return model_instance


def _build_loader(cfg, split):
    from dataset import get_dataloader

    cfg.train_loader["batch_size"] = 1
    cfg.train_loader["num_workers"] = 0
    cfg.train_loader["shuffle"] = False
    cfg.val_loader["batch_size"] = 1
    cfg.val_loader["num_workers"] = 0

    if split == "train":
        train_loader, _ = get_dataloader(
            cfg.train_dataset_config,
            cfg.val_dataset_config,
            cfg.train_loader,
            cfg.val_loader,
            dist=False,
            val_only=False,
        )
        return train_loader

    _, val_loader = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_loader,
        cfg.val_loader,
        dist=False,
        val_only=True,
    )
    return val_loader


def _prediction_map_dict(result_dict, score_thresh):
    from dataset.dataset import output_to_vecs

    map_dict = _empty_map_dict()
    for vec in output_to_vecs(result_dict):
        score = _to_float(vec["score"])
        if score < score_thresh:
            continue
        label = int(_to_float(vec["label"]))
        if not 0 <= label < len(MAP_CLASSES):
            continue
        pts = _to_numpy(vec["pts"]).reshape(-1, 2)
        map_dict[MAP_CLASSES[label]].append(pts)
    return map_dict


def _camera_image_path(batch, camera_index):
    img_meta = _unwrap_data_container(batch["img_metas"][0])
    filenames = img_meta.get("filename") or img_meta.get("img_filename")
    if not filenames:
        return None

    if isinstance(filenames, (list, tuple)) and filenames and isinstance(filenames[0], (list, tuple)):
        filenames = filenames[0]
    if isinstance(filenames, str):
        camera_path = filenames
    else:
        if len(filenames) <= camera_index:
            return None
        camera_path = filenames[camera_index]

    if not camera_path:
        return None
    return osp.abspath(camera_path)


def _load_camera_image(batch, camera_index, camera_name):
    camera_path = _camera_image_path(batch, camera_index)
    if camera_path is None or not osp.exists(camera_path):
        return None

    try:
        return mpimg.imread(camera_path)
    except Exception as exc:
        print(f"[WARN] failed to load {camera_name} image {camera_path}: {exc}")
        return None


def _reshape_occ(volume, grid_shape):
    if isinstance(volume, torch.Tensor):
        volume = volume.detach().cpu().numpy()
    volume = np.asarray(volume)
    if volume.shape == tuple(grid_shape):
        return volume
    return volume.reshape(*grid_shape)


def _infer_grid_shape(volume, requested_shape, fallback_shape=None):
    if isinstance(volume, torch.Tensor):
        volume = volume.detach().cpu().numpy()
    volume = np.asarray(volume)
    if volume.ndim == 3:
        return tuple(volume.shape)

    numel = int(volume.size)
    candidates = []
    if requested_shape is not None:
        candidates.append(tuple(int(x) for x in requested_shape))
    if fallback_shape is not None:
        candidates.append(tuple(int(x) for x in fallback_shape))

    for shape in candidates:
        if np.prod(shape) == numel:
            return shape

    raise ValueError(f"cannot infer occupancy grid shape for numel={numel}; tried {candidates}")


def _topdown_projection(volume, empty_label):
    x_dim, y_dim, z_dim = volume.shape
    projected = np.full((x_dim, y_dim), empty_label, dtype=np.int32)
    for z_index in range(z_dim - 1, -1, -1):
        layer = volume[:, :, z_index]
        mask = (projected == empty_label) & (layer != empty_label)
        projected[mask] = layer[mask]
    return projected


def _plot_camera_image(ax, image, title):
    if image is None:
        ax.text(0.5, 0.5, "image unavailable", ha="center", va="center", fontsize=12)
        ax.set_title(title)
        ax.axis("off")
        return
    ax.imshow(image)
    ax.set_title(title)
    ax.axis("off")


def _plot_topdown(ax, occ_map, title):
    cmap = ListedColormap(NUSC_OCC_COLORS)
    ax.imshow(occ_map.T, origin="lower", cmap=cmap, vmin=0, vmax=len(NUSC_OCC_COLORS) - 1)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def _occ_legend_handles():
    handles = []
    for class_index, class_name in enumerate(NUSC_OCC_CLASS_NAMES):
        handles.append(
            Patch(
                facecolor=NUSC_OCC_COLORS[class_index],
                edgecolor="black" if class_name == "empty" else "none",
                linewidth=0.3,
                label=f"{class_index}: {class_name}",
            )
        )
    return handles


def _draw_map(ax, map_dict, title_prefix, alpha, linestyle, linewidth, add_legend):
    legend_done = set()
    for name in MAP_CLASSES:
        color = MAP_COLORS[name]
        for pts in map_dict.get(name, []):
            label = None
            if add_legend and name not in legend_done:
                label = f"{title_prefix} {name}"
                legend_done.add(name)
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=alpha,
                label=label,
            )


def _draw_traj(ax, points, label, color, linestyle, linewidth):
    ax.plot(
        points[:, 0],
        points[:, 1],
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        marker="o",
        markersize=3,
        label=label,
    )


def _setup_map_axis(ax, x_range, y_range, title):
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    ax.axhline(0.0, color="gray", linewidth=0.6, alpha=0.4)
    ax.axvline(0.0, color="gray", linewidth=0.6, alpha=0.4)
    ax.scatter([0.0], [0.0], color="black", marker="*", s=60, label="ego")
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


def _save_combined_figure(
    pred_occ,
    gt_occ,
    gt_map,
    pred_map,
    gt_traj,
    pred_traj,
    sample_token,
    out_file,
    empty_label,
    x_range,
    y_range,
    command_idx,
    dpi,
    front_image,
    back_image,
):
    fig = plt.figure(figsize=(24, 12), dpi=dpi)
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.05], hspace=0.28)
    top = outer[0].subgridspec(1, 4, wspace=0.18)
    bottom = outer[1].subgridspec(1, 3, wspace=0.18)

    ax_front = fig.add_subplot(top[0, 0])
    ax_gt_occ = fig.add_subplot(top[0, 1])
    ax_pred_occ = fig.add_subplot(top[0, 2])
    ax_back = fig.add_subplot(top[0, 3])

    ax_gt = fig.add_subplot(bottom[0, 0])
    ax_pred = fig.add_subplot(bottom[0, 1])
    ax_overlay = fig.add_subplot(bottom[0, 2])

    _plot_camera_image(ax_front, front_image, "CAM_FRONT")
    _plot_camera_image(ax_back, back_image, "CAM_BACK")
    _plot_topdown(ax_gt_occ, _topdown_projection(gt_occ, empty_label), "GT top-down OCC")
    _plot_topdown(ax_pred_occ, _topdown_projection(pred_occ, empty_label), "Pred top-down OCC")

    _setup_map_axis(ax_gt, x_range, y_range, "GT Map + GT Plan")
    _draw_map(ax_gt, gt_map, "GT", alpha=0.95, linestyle="-", linewidth=2.0, add_legend=True)
    _draw_traj(ax_gt, gt_traj, "GT trajectory", "black", "--", 2.0)
    ax_gt.legend(fontsize=8, loc="upper right")

    _setup_map_axis(ax_pred, x_range, y_range, f"Pred Map + Pred Plan (mode {command_idx})")
    _draw_map(ax_pred, pred_map, "Pred", alpha=0.95, linestyle="-", linewidth=2.2, add_legend=True)
    _draw_traj(ax_pred, pred_traj, "Pred trajectory", "limegreen", "-", 2.3)
    ax_pred.legend(fontsize=8, loc="upper right")

    _setup_map_axis(ax_overlay, x_range, y_range, "Map/Plan Overlay")
    _draw_map(ax_overlay, gt_map, "GT", alpha=0.35, linestyle="--", linewidth=1.5, add_legend=True)
    _draw_map(ax_overlay, pred_map, "Pred", alpha=0.95, linestyle="-", linewidth=2.2, add_legend=True)
    _draw_traj(ax_overlay, gt_traj, "GT trajectory", "black", "--", 2.0)
    _draw_traj(ax_overlay, pred_traj, "Pred trajectory", "limegreen", "-", 2.3)
    ax_overlay.legend(fontsize=8, loc="upper right")

    fig.legend(
        handles=_occ_legend_handles(),
        loc="lower center",
        ncol=6,
        fontsize=8,
        frameon=False,
        title="Occupancy semantic legend",
        title_fontsize=10,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle(f"sample: {sample_token}", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.1, 1.0, 0.95))
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


def _save_traj_figure(
    gt_map,
    pred_map,
    gt_traj,
    pred_traj,
    sample_token,
    out_file,
    x_range,
    y_range,
    command_idx,
    dpi,
):
    """Standalone trajectory comparison figure: GT vs Pred plan only."""
    fig = plt.figure(figsize=(8, 8), dpi=dpi)
    ax = fig.add_subplot(1, 1, 1)

    _setup_map_axis(ax, x_range, y_range, f"Trajectory GT vs Pred (mode {command_idx})")
    _draw_map(ax, gt_map, "GT", alpha=0.25, linestyle="--", linewidth=1.2, add_legend=True)
    _draw_map(ax, pred_map, "Pred", alpha=0.40, linestyle=":", linewidth=1.2, add_legend=True)
    _draw_traj(ax, gt_traj, "GT trajectory", "black", "--", 2.2)
    _draw_traj(ax, pred_traj, "Pred trajectory", "limegreen", "-", 2.5)
    ax.legend(fontsize=9, loc="upper right")

    fig.suptitle(f"sample: {sample_token}", fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


def _cleanup_scene_pngs(seven_paths, traj_paths, scene_token, keep_seven=False, keep_traj=True):
    if not keep_seven and seven_paths:
        removed = 0
        for tmp_png in seven_paths:
            if os.path.exists(tmp_png):
                os.remove(tmp_png)
                removed += 1
        print(f"[cleanup] removed {removed} 7-panel PNGs for {scene_token}")
    if not keep_traj and traj_paths:
        removed = 0
        for tmp_png in traj_paths:
            if os.path.exists(tmp_png):
                os.remove(tmp_png)
                removed += 1
        print(f"[cleanup] removed {removed} trajectory PNGs for {scene_token}")


def _save_scene_gif(scene_token, image_paths, scene_dir, gif_ms, first_batch_index, last_batch_index, file_suffix=""):
    if not image_paths:
        return None

    frames = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            frames.append(image.convert("RGBA").copy())

    gif_path = scene_dir / f"scene_{scene_token}{file_suffix}_{first_batch_index:06d}_{last_batch_index:06d}.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=gif_ms,
        loop=0,
    )
    return gif_path


def _save_scene_mp4(scene_token, image_paths, scene_dir, gif_ms, first_batch_index, last_batch_index, file_suffix=""):
    if not image_paths:
        return None

    frames = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            frames.append(np.array(image.convert("RGB")))

    frame_height, frame_width = frames[0].shape[:2]
    fps = max(1.0, 1000.0 / max(1, gif_ms))
    mp4_path = scene_dir / f"scene_{scene_token}{file_suffix}_{first_batch_index:06d}_{last_batch_index:06d}.mp4"
    writer = cv2.VideoWriter(
        str(mp4_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer for {mp4_path}")

    try:
        for frame in frames:
            if frame.shape[:2] != (frame_height, frame_width):
                frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    return mp4_path


def _save_scene_animations(scene_token, image_paths, scene_dir, gif_ms, first_batch_index, last_batch_index, file_suffix=""):
    gif_path = _save_scene_gif(
        scene_token,
        image_paths,
        scene_dir,
        gif_ms,
        first_batch_index,
        last_batch_index,
        file_suffix,
    )
    mp4_path = _save_scene_mp4(
        scene_token,
        image_paths,
        scene_dir,
        gif_ms,
        first_batch_index,
        last_batch_index,
        file_suffix,
    )
    return gif_path, mp4_path


def main():
    args = parse_args()
    torch.backends.cudnn.benchmark = True

    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = args.work_dir
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.work_dir) / "occ_map_plan_vis"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_grid_shape = tuple(int(x) for x in getattr(cfg, "grid_size", args.grid_shape))

    # ---------- 场景边界统计（构建一次"未抽样"数据集）----------
    # 关键：统计场景时必须禁用 dataset 的随机抽样（num_samples>0 会按
    # subsample_seed 抽帧，导致场景统计与真实 keyframes 错位），因此显式
    # 用 num_samples=0 + 完整 vis_indices 构建一次"未抽样"数据集来数帧。
    from collections import OrderedDict
    frames_per_scene = getattr(args, 'frames_per_scene', 40)
    _full_vis = list(range(_dataset_len(cfg, args.split)))
    _save_cfg = dict(cfg.val_dataset_config)
    cfg.val_dataset_config.update({"vis_indices": _full_vis, "num_samples": 0})
    loader_tmp = _build_loader(cfg, args.split)
    keyframes = _resolve_dataset_keyframes(loader_tmp.dataset)
    cfg.val_dataset_config.clear()
    cfg.val_dataset_config.update(_save_cfg)
    scene_frame_counts = OrderedDict()
    for token, _idx in keyframes:
        scene_frame_counts.setdefault(token, 0)
        scene_frame_counts[token] += 1
    scene_items = list(scene_frame_counts.items())

    if args.list_scenes:
        # --list-scenes：仅打印场景序号列表（不加载模型、不推理）
        print(f"[SCENES] {len(scene_items)} scenes in split '{args.split}', in dataset order:")
        for idx, (token, count) in enumerate(scene_items):
            print(f"  [{idx:03d}] scene={token} frames={count}")
        print("[SCENES] use --scene-indices <idx...> to visualize any subset, e.g. --scene-indices 3 7 12")
        return

    checkpoint_path = _checkpoint_path(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_instance = _load_model(cfg, checkpoint_path, device)

    if args.scene_indices is not None and len(args.scene_indices) > 0:
        # --scene-indices A B C：用户按场景序号（--list-scenes 输出）任意挑选
        # 场景，按给定顺序依次可视化，每个场景独立产出 GIF/MP4 + 每帧轨迹图。
        # 优先级高于 --vis-index / --scenes。
        vis_indices = []
        picked = []
        for sidx in args.scene_indices:
            if not 0 <= sidx < len(scene_items):
                print(f"[WARN] scene index {sidx} out of range [0, {len(scene_items)}), skipped")
                continue
            token, count = scene_items[sidx]
            take = min(count, frames_per_scene)
            scene_offset = sum(c for _t, c in scene_items[:sidx])
            vis_indices.extend(range(scene_offset, scene_offset + take))
            picked.append((sidx, token, take))
        if not vis_indices:
            raise SystemExit("No valid scene indices given via --scene-indices.")
        cfg.val_dataset_config.update({"vis_indices": vis_indices, "num_samples": 0})
        args.num_samples = len(vis_indices)
        for sidx, token, take in picked:
            print(f"[SCENES] selected [{sidx:03d}] scene={token} frames={take}")
    elif args.vis_index is not None and len(args.vis_index) > 0:
        # 显式索引会覆盖随机抽样，保持同 scene 帧连续（视频需要）
        cfg.val_dataset_config.update({"vis_indices": args.vis_index, "num_samples": 0})
        # 关键修复：--vis-index 显式列出所有帧时，覆盖默认 --num-samples
        # （默认 4 会在 4 帧后 break，导致 40 帧场景只出 4 帧）
        args.num_samples = len(args.vis_index)
    elif args.scenes > 0:
        # --scenes N [--scene-start S]：自动取 [S, S+N) 连续场景，每场景固定取
        # 前 frames_per_scene 帧（默认 40），按场景边界对齐，超出部分自动截断。
        scenes_list = scene_items[args.scene_start:args.scene_start + args.scenes]
        scene_start = sum(cnt for _tok, cnt in scene_items[:args.scene_start])
        vis_indices = []
        for token, count in scenes_list:
            take = min(count, frames_per_scene)
            vis_indices.extend(range(scene_start, scene_start + take))
            scene_start += count
        cfg.val_dataset_config.update({"vis_indices": vis_indices, "num_samples": 0})
        args.num_samples = len(vis_indices)
    loader = _build_loader(cfg, args.split)

    pc_range = np.asarray(cfg.pc_range, dtype=np.float32)
    x_range = (float(pc_range[0]), float(pc_range[3]))
    y_range = (float(pc_range[1]), float(pc_range[4]))

    saved = 0
    saved_gifs = 0
    saved_mp4s = 0
    current_scene_token = None
    current_scene_dir = None
    current_scene_image_paths = []
    current_scene_traj_paths = []
    current_scene_first_batch_index = None
    current_scene_last_batch_index = None

    with torch.no_grad():
        for batch_index, data in enumerate(loader):
            if batch_index < args.start_index:
                continue
            if args.num_samples > 0 and saved >= args.num_samples:
                break

            sample_token = _sample_token_from_batch(data)
            scene_token, scene_frame_index = _scene_info_from_loader(loader, batch_index)
            if scene_token is None:
                scene_token = f"sample_{sample_token}"
            if scene_frame_index is None:
                scene_frame_index = batch_index

            if current_scene_token is not None and scene_token != current_scene_token:
                if not args.no_video:
                    gif_path, mp4_path = _save_scene_animations(
                        current_scene_token,
                        current_scene_image_paths,
                        current_scene_dir,
                        args.gif_ms,
                        current_scene_first_batch_index,
                        current_scene_last_batch_index,
                    )
                    if gif_path is not None:
                        print(f"[OK] saved {gif_path}")
                        saved_gifs += 1
                    if mp4_path is not None:
                        print(f"[OK] saved {mp4_path}")
                        saved_mp4s += 1
                    traj_gif_path, traj_mp4_path = _save_scene_animations(
                        current_scene_token,
                        current_scene_traj_paths,
                        current_scene_dir,
                        args.gif_ms,
                        current_scene_first_batch_index,
                        current_scene_last_batch_index,
                        file_suffix="_traj",
                    )
                    if traj_gif_path is not None:
                        print(f"[OK] saved {traj_gif_path}")
                        saved_gifs += 1
                    if traj_mp4_path is not None:
                        print(f"[OK] saved {traj_mp4_path}")
                        saved_mp4s += 1
                    _cleanup_scene_pngs(
                        current_scene_image_paths,
                        current_scene_traj_paths,
                        current_scene_token,
                        keep_seven=args.keep_seven_png,
                        keep_traj=not args.no_png,
                    )
                current_scene_image_paths = []
                current_scene_traj_paths = []
                current_scene_first_batch_index = None
                current_scene_last_batch_index = None

            if current_scene_token != scene_token:
                current_scene_token = scene_token
                current_scene_dir = _scene_output_dir(out_dir, scene_token)
                current_scene_first_batch_index = batch_index

            gt_map = _normalize_map_dict(data["gt_map"][0])
            front_image = _load_camera_image(data, 0, "CAM_FRONT")
            back_image = _load_camera_image(data, 3, "CAM_BACK")

            for key in list(data.keys()):
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].to(device)

            input_imgs = data.pop("img")
            result_dict = model_instance(imgs=input_imgs, metas=data)

            pred_occ = result_dict["pred_occ"][-1][0].argmax(0)
            gt_occ = result_dict["sampled_label"][0]
            grid_shape = _infer_grid_shape(pred_occ, args.grid_shape, cfg_grid_shape)
            pred_occ = _reshape_occ(pred_occ, grid_shape)
            gt_occ = _reshape_occ(gt_occ, grid_shape)

            metas = result_dict.get("metas", data)
            ego_cmd = metas["ego_fut_cmd"]
            if isinstance(ego_cmd, torch.Tensor):
                command_idx = int(torch.argmax(ego_cmd[0]).item())
            else:
                command_idx = int(np.argmax(np.asarray(ego_cmd)[0]))

            gt_traj = _traj_to_polyline(metas["ego_fut_trajs"][0])
            pred_traj = _traj_to_polyline(result_dict["ego_fut_preds"][0, command_idx])
            pred_map = _prediction_map_dict(result_dict, args.score_thresh)

            save_path = current_scene_dir / f"batch_{batch_index:06d}_frame_{scene_frame_index:03d}_{sample_token}.png"
            traj_save_path = current_scene_dir / f"batch_{batch_index:06d}_frame_{scene_frame_index:03d}_{sample_token}_traj.png"
            _save_combined_figure(
                pred_occ,
                gt_occ,
                gt_map,
                pred_map,
                gt_traj,
                pred_traj,
                sample_token,
                save_path,
                args.empty_label,
                x_range,
                y_range,
                command_idx,
                args.dpi,
                front_image,
                back_image,
            )
            _save_traj_figure(
                gt_map,
                pred_map,
                gt_traj,
                pred_traj,
                sample_token,
                traj_save_path,
                x_range,
                y_range,
                command_idx,
                args.dpi,
            )
            print(f"[OK] saved {save_path}")
            print(f"[OK] saved {traj_save_path}")
            current_scene_image_paths.append(save_path)
            current_scene_traj_paths.append(traj_save_path)
            current_scene_last_batch_index = batch_index
            saved += 1

    if current_scene_image_paths and not args.no_video:
        gif_path, mp4_path = _save_scene_animations(
            current_scene_token,
            current_scene_image_paths,
            current_scene_dir,
            args.gif_ms,
            current_scene_first_batch_index,
            current_scene_last_batch_index,
        )
        if gif_path is not None:
            print(f"[OK] saved {gif_path}")
            saved_gifs += 1
        if mp4_path is not None:
            print(f"[OK] saved {mp4_path}")
            saved_mp4s += 1
        traj_gif_path, traj_mp4_path = _save_scene_animations(
            current_scene_token,
            current_scene_traj_paths,
            current_scene_dir,
            args.gif_ms,
            current_scene_first_batch_index,
            current_scene_last_batch_index,
            file_suffix="_traj",
        )
        if traj_gif_path is not None:
            print(f"[OK] saved {traj_gif_path}")
            saved_gifs += 1
        if traj_mp4_path is not None:
            print(f"[OK] saved {traj_mp4_path}")
            saved_mp4s += 1
        _cleanup_scene_pngs(
            current_scene_image_paths,
            current_scene_traj_paths,
            current_scene_token,
            keep_seven=args.keep_seven_png,
            keep_traj=not args.no_png,
        )

    if saved == 0:
        raise SystemExit("No samples were visualized. Check --start-index and dataset length.")

    print(f"[DONE] saved {saved} figure(s), {saved_gifs} scene gif(s), and {saved_mp4s} scene mp4(s) under {out_dir}")


if __name__ == "__main__":
    main()

'''
#!/usr/bin/env python3

# 确保 import model / dataset 解析到当前工作区（而非 sys.path 里残留的
# /data/chenz/GaussianAD 等旧路径）
import os
import sys as _sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _REPO not in _sys.path:
    _sys.path.insert(0, _REPO)
if os.getcwd() not in _sys.path:
    _sys.path.insert(0, os.getcwd())

import argparse
import os.path as osp
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import torch

from mmengine import Config
from mmseg.models import build_segmentor


MAP_CLASSES = ("divider", "ped_crossing", "boundary")
MAP_COLORS = {
    "divider": "darkorange",
    "ped_crossing": "royalblue",
    "boundary": "crimson",
}

NUSC_OCC_COLORS = np.array(
    [
        [0, 0, 0],
        [255, 120, 50],
        [255, 192, 203],
        [255, 255, 0],
        [0, 150, 245],
        [0, 255, 255],
        [255, 127, 0],
        [255, 0, 0],
        [255, 240, 150],
        [135, 60, 0],
        [160, 32, 240],
        [255, 0, 255],
        [139, 137, 137],
        [75, 0, 75],
        [150, 240, 80],
        [230, 230, 250],
        [0, 175, 0],
        [245, 245, 245],
    ],
    dtype=np.float32,
) / 255.0

NUSC_OCC_CLASS_NAMES = [
    "others",
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
    "driveable_surface",
    "other_flat",
    "sidewalk",
    "terrain",
    "manmade",
    "vegetation",
    "empty",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize OCC, map, and plan predictions in a single 7-panel figure."
    )
    parser.add_argument("--py-config", required=True, help="Path to config file.")
    parser.add_argument("--work-dir", required=True, help="Experiment directory used to find latest.pth.")
    parser.add_argument("--resume-from", default="", help="Checkpoint path. Default: <work_dir>/latest.pth.")
    parser.add_argument("--out-dir", default="", help="Directory to save figures.")
    parser.add_argument("--split", choices=("val", "train"), default="val", help="Dataset split to visualize.")
    parser.add_argument("--num-samples", type=int, default=4, help="How many samples to draw.")
    parser.add_argument("--vis-index", type=int, nargs="+", default=None,
                        help="Explicit keyframe indices (keeps scene continuity, e.g. a whole scene's 40 "
                             "frames: --vis-index 0 1 2 ... 39). Overrides --num-samples sampling.")
    parser.add_argument("--start-index", type=int, default=0, help="Start index in the chosen dataloader.")
    parser.add_argument("--score-thresh", type=float, default=0.35, help="Map prediction score threshold.")
    parser.add_argument("--device", default="cuda:0", help="Device for inference.")
    parser.add_argument("--dpi", type=int, default=180, help="Saved figure DPI.")
    parser.add_argument("--grid-shape", type=int, nargs=3, default=(200, 200, 16), help="Occupancy grid shape as X Y Z.")
    parser.add_argument("--empty-label", type=int, default=17, help="Class id used for empty voxels.")
    parser.add_argument("--no-video", action="store_true",
                            help="Only save per-frame PNGs; skip scene GIF/MP4 synthesis.")
    return parser.parse_args()


def _unwrap_data_container(value):
    if hasattr(value, "data"):
        value = value.data
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    return value


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _empty_map_dict():
    return {name: [] for name in MAP_CLASSES}


def _normalize_map_dict(map_dict):
    normalized = _empty_map_dict()
    if not isinstance(map_dict, dict):
        return normalized
    for name in MAP_CLASSES:
        for pts in map_dict.get(name, []):
            pts_np = _to_numpy(pts)
            if pts_np.size == 0:
                continue
            normalized[name].append(pts_np.reshape(-1, 2))
    return normalized


def _traj_to_polyline(traj):
    traj_np = _to_numpy(traj).reshape(-1, 2)
    cumulative = np.cumsum(traj_np, axis=0)
    origin = np.zeros((1, 2), dtype=np.float32)
    return np.concatenate([origin, cumulative], axis=0)


def _sample_token_from_batch(batch):
    img_meta = _unwrap_data_container(batch["img_metas"][0])
    return str(img_meta.get("sample_idx", "sample"))


def _checkpoint_path(args):
    if args.resume_from:
        return args.resume_from
    latest = osp.join(args.work_dir, "latest.pth")
    if osp.exists(latest):
        return latest
    raise FileNotFoundError(f"No checkpoint found. Expected: {latest}")


def _load_model(cfg, checkpoint_path, device):
    import model  # noqa: F401

    model_instance = build_segmentor(cfg.model)
    model_instance.init_weights()

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    try:
        load_msg = model_instance.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        model_state = model_instance.state_dict()
        filtered = {}
        for key, value in state_dict.items():
            if key in model_state and hasattr(value, "shape") and model_state[key].shape == value.shape:
                filtered[key] = value
        load_msg = model_instance.load_state_dict(filtered, strict=False)

    print(load_msg)
    model_instance = model_instance.to(device)
    model_instance.eval()
    return model_instance


def _build_loader(cfg, split):
    from dataset import get_dataloader

    cfg.train_loader["batch_size"] = 1
    cfg.train_loader["num_workers"] = 0
    cfg.train_loader["shuffle"] = False
    cfg.val_loader["batch_size"] = 1
    cfg.val_loader["num_workers"] = 0

    if split == "train":
        train_loader, _ = get_dataloader(
            cfg.train_dataset_config,
            cfg.val_dataset_config,
            cfg.train_loader,
            cfg.val_loader,
            dist=False,
            val_only=False,
        )
        return train_loader

    _, val_loader = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_loader,
        cfg.val_loader,
        dist=False,
        val_only=True,
    )
    return val_loader


def _camera_image_path(batch, camera_index):
    img_meta = _unwrap_data_container(batch["img_metas"][0])
    filenames = img_meta.get("filename") or img_meta.get("img_filename")
    if not filenames:
        return None

    if isinstance(filenames, (list, tuple)) and filenames and isinstance(filenames[0], (list, tuple)):
        filenames = filenames[0]
    if isinstance(filenames, str):
        camera_path = filenames
    else:
        if len(filenames) <= camera_index:
            return None
        camera_path = filenames[camera_index]

    if not camera_path:
        return None
    return osp.abspath(camera_path)


def _load_camera_image(batch, camera_index, camera_name):
    camera_path = _camera_image_path(batch, camera_index)
    if camera_path is None or not osp.exists(camera_path):
        return None

    try:
        return mpimg.imread(camera_path)
    except Exception as exc:
        print(f"[WARN] failed to load {camera_name} image {camera_path}: {exc}")
        return None


def _reshape_occ(volume, grid_shape):
    if isinstance(volume, torch.Tensor):
        volume = volume.detach().cpu().numpy()
    volume = np.asarray(volume)
    if volume.shape == tuple(grid_shape):
        return volume
    return volume.reshape(*grid_shape)


def _infer_grid_shape(volume, requested_shape, fallback_shape=None):
    if isinstance(volume, torch.Tensor):
        volume = volume.detach().cpu().numpy()
    volume = np.asarray(volume)
    if volume.ndim == 3:
        return tuple(volume.shape)

    numel = int(volume.size)
    candidates = []
    if requested_shape is not None:
        candidates.append(tuple(int(x) for x in requested_shape))
    if fallback_shape is not None:
        candidates.append(tuple(int(x) for x in fallback_shape))

    for shape in candidates:
        if np.prod(shape) == numel:
            return shape

    raise ValueError(f"cannot infer occupancy grid shape for numel={numel}; tried {candidates}")


def _topdown_projection(volume, empty_label):
    x_dim, y_dim, z_dim = volume.shape
    projected = np.full((x_dim, y_dim), empty_label, dtype=np.int32)
    for z_index in range(z_dim - 1, -1, -1):
        layer = volume[:, :, z_index]
        mask = (projected == empty_label) & (layer != empty_label)
        projected[mask] = layer[mask]
    return projected


def _prediction_map_dict(result_dict, score_thresh):
    from dataset.dataset import output_to_vecs

    map_dict = _empty_map_dict()
    for vec in output_to_vecs(result_dict):
        score = _to_float(vec["score"])
        if score < score_thresh:
            continue
        label = int(_to_float(vec["label"]))
        if not 0 <= label < len(MAP_CLASSES):
            continue
        pts = _to_numpy(vec["pts"]).reshape(-1, 2)
        map_dict[MAP_CLASSES[label]].append(pts)
    return map_dict


def _plot_camera_image(ax, image, title):
    if image is None:
        ax.axis("off")
        ax.set_title(f"{title} missing")
        ax.text(0.5, 0.5, "image not found", ha="center", va="center", fontsize=12, transform=ax.transAxes)
        return
    ax.imshow(image)
    ax.set_title(title)
    ax.axis("off")


def _plot_occ_topdown(ax, occ_map, cmap, title):
    ax.imshow(occ_map.T, origin="lower", cmap=cmap, vmin=0, vmax=len(NUSC_OCC_COLORS) - 1)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")


def _draw_map(ax, map_dict, title_prefix, alpha, linestyle, linewidth, add_legend):
    legend_done = set()
    for name in MAP_CLASSES:
        color = MAP_COLORS[name]
        for pts in map_dict.get(name, []):
            label = None
            if add_legend and name not in legend_done:
                label = f"{title_prefix} {name}"
                legend_done.add(name)
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=alpha,
                label=label,
            )


def _draw_traj(ax, points, label, color, linestyle, linewidth):
    ax.plot(
        points[:, 0],
        points[:, 1],
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        marker="o",
        markersize=3,
        label=label,
    )


def _setup_plan_map_axis(ax, x_range, y_range, title):
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    ax.axhline(0.0, color="gray", linewidth=0.6, alpha=0.4)
    ax.axvline(0.0, color="gray", linewidth=0.6, alpha=0.4)
    ax.scatter([0.0], [0.0], color="black", marker="*", s=60, label="ego")
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


def _occ_legend_handles():
    handles = []
    for class_index, class_name in enumerate(NUSC_OCC_CLASS_NAMES):
        handles.append(
            Patch(
                facecolor=NUSC_OCC_COLORS[class_index],
                edgecolor="black" if class_name == "empty" else "none",
                linewidth=0.3,
                label=f"{class_index}: {class_name}",
            )
        )
    return handles


def _save_combined_figure(
    pred_occ,
    gt_occ,
    front_image,
    back_image,
    gt_map,
    pred_map,
    gt_traj,
    pred_traj,
    sample_token,
    out_file,
    x_range,
    y_range,
    command_idx,
    empty_label,
    dpi,
):
    cmap = ListedColormap(NUSC_OCC_COLORS)
    fig = plt.figure(figsize=(24, 12), dpi=dpi)
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.05], hspace=0.24)
    top = outer[0].subgridspec(1, 4, wspace=0.18)
    bottom = outer[1].subgridspec(1, 3, wspace=0.20)

    ax_front = fig.add_subplot(top[0, 0])
    ax_occ_gt = fig.add_subplot(top[0, 1])
    ax_occ_pred = fig.add_subplot(top[0, 2])
    ax_back = fig.add_subplot(top[0, 3])
    ax_gt = fig.add_subplot(bottom[0, 0])
    ax_pred = fig.add_subplot(bottom[0, 1])
    ax_overlay = fig.add_subplot(bottom[0, 2])

    _plot_camera_image(ax_front, front_image, "CAM_FRONT")
    _plot_camera_image(ax_back, back_image, "CAM_BACK")
    _plot_occ_topdown(ax_occ_gt, _topdown_projection(gt_occ, empty_label), cmap, "GT top-down OCC")
    _plot_occ_topdown(ax_occ_pred, _topdown_projection(pred_occ, empty_label), cmap, "Pred top-down OCC")

    _setup_plan_map_axis(ax_gt, x_range, y_range, "GT Map + GT Plan")
    _draw_map(ax_gt, gt_map, "GT", alpha=0.95, linestyle="-", linewidth=2.0, add_legend=True)
    _draw_traj(ax_gt, gt_traj, "GT trajectory", "black", "--", 2.0)
    ax_gt.legend(fontsize=7, loc="upper right")

    _setup_plan_map_axis(ax_pred, x_range, y_range, f"Pred Map + Pred Plan (mode {command_idx})")
    _draw_map(ax_pred, pred_map, "Pred", alpha=0.95, linestyle="-", linewidth=2.2, add_legend=True)
    _draw_traj(ax_pred, pred_traj, "Pred trajectory", "limegreen", "-", 2.3)
    ax_pred.legend(fontsize=7, loc="upper right")

    _setup_plan_map_axis(ax_overlay, x_range, y_range, "Overlay Map + Plan")
    _draw_map(ax_overlay, gt_map, "GT", alpha=0.35, linestyle="--", linewidth=1.5, add_legend=True)
    _draw_map(ax_overlay, pred_map, "Pred", alpha=0.95, linestyle="-", linewidth=2.2, add_legend=True)
    _draw_traj(ax_overlay, gt_traj, "GT trajectory", "black", "--", 2.0)
    _draw_traj(ax_overlay, pred_traj, "Pred trajectory", "limegreen", "-", 2.3)
    ax_overlay.legend(fontsize=7, loc="upper right")

    fig.suptitle(f"sample: {sample_token}", fontsize=16)
    fig.legend(
        handles=_occ_legend_handles(),
        loc="lower center",
        ncol=6,
        fontsize=8,
        frameon=False,
        title="Occupancy semantic legend",
        title_fontsize=10,
        bbox_to_anchor=(0.5, 0.02),
    )
    fig.tight_layout(rect=(0.0, 0.14, 1.0, 0.95))
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    torch.backends.cudnn.benchmark = True

    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = args.work_dir
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.work_dir) / "occ_map_plan_vis"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_grid_shape = tuple(int(x) for x in getattr(cfg, "grid_size", args.grid_shape))

    checkpoint_path = _checkpoint_path(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_instance = _load_model(cfg, checkpoint_path, device)
    if args.vis_index is not None and len(args.vis_index) > 0:
        # 显式索引会覆盖随机抽样，保持同 scene 帧连续（视频需要）
        cfg.val_dataset_config.update({"vis_indices": args.vis_index, "num_samples": 0})
    loader = _build_loader(cfg, args.split)

    pc_range = np.asarray(cfg.pc_range, dtype=np.float32)
    x_range = (float(pc_range[0]), float(pc_range[3]))
    y_range = (float(pc_range[1]), float(pc_range[4]))

    saved = 0
    with torch.no_grad():
        for batch_index, data in enumerate(loader):
            if batch_index < args.start_index:
                continue
            if args.num_samples > 0 and saved >= args.num_samples:
                break

            sample_token = _sample_token_from_batch(data)
            gt_map = _normalize_map_dict(_unwrap_data_container(data["gt_map"][0]))
            front_image = _load_camera_image(data, 0, "CAM_FRONT")
            back_image = _load_camera_image(data, 3, "CAM_BACK")

            for key in list(data.keys()):
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].to(device)

            input_imgs = data.pop("img")
            result_dict = model_instance(imgs=input_imgs, metas=data)

            pred_occ = result_dict["pred_occ"][-1][0].argmax(0)
            gt_occ = result_dict["sampled_label"][0]
            grid_shape = _infer_grid_shape(pred_occ, args.grid_shape, cfg_grid_shape)
            pred_occ = _reshape_occ(pred_occ, grid_shape)
            gt_occ = _reshape_occ(gt_occ, grid_shape)

            metas = result_dict.get("metas", data)
            ego_cmd = metas["ego_fut_cmd"]
            if isinstance(ego_cmd, torch.Tensor):
                command_idx = int(torch.argmax(ego_cmd[0]).item())
            else:
                command_idx = int(np.argmax(np.asarray(ego_cmd)[0]))

            gt_traj = _traj_to_polyline(metas["ego_fut_trajs"][0])
            pred_traj = _traj_to_polyline(result_dict["ego_fut_preds"][0, command_idx])
            pred_map = _prediction_map_dict(result_dict, args.score_thresh)

            save_path = out_dir / f"{saved:03d}_{sample_token}.png"
            _save_combined_figure(
                pred_occ,
                gt_occ,
                front_image,
                back_image,
                gt_map,
                pred_map,
                gt_traj,
                pred_traj,
                sample_token,
                save_path,
                x_range,
                y_range,
                command_idx,
                args.empty_label,
                args.dpi,
            )
            print(f"[OK] saved {save_path}")
            saved += 1

    if saved == 0:
        raise SystemExit("No samples were visualized. Check --start-index and dataset length.")

    print(f"[DONE] saved {saved} figure(s) to {out_dir}")


if __name__ == "__main__":
    main()
'''