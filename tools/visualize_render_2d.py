#!/usr/bin/env python3
"""
Visualize 2D Gaussian splatting renders vs pseudo labels.

Produces per-sample images showing:
  [pred_sem | gt_sem | pred_depth | gt_depth | input_img]
for all 6 cameras, stacked vertically.

Usage:
    CUDA_VISIBLE_DEVICES=0 python tools/visualize_render_2d.py \
        --py-config config/nuscenes_gs25600.py \
        --work-dir out/nuscenes_gs25600_nograd \
        --num-samples 10 --start-index 0 --split val
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from mmengine import Config
from mmseg.models import build_segmentor

# nuScenes occupancy 17-class palette (0-indexed: 0=barrier ... 16=free/empty)
_NUSC_PALETTE = np.array([
    [112, 128, 144],  # 0: barrier
    [220,  20,  60],  # 1: bicycle
    [255, 127,  80],  # 2: bus
    [255, 158,   0],  # 3: car
    [233, 150,  70],  # 4: construction_vehicle
    [255,  61,  99],  # 5: motorcycle
    [  0,   0, 230],  # 6: pedestrian
    [ 47,  79,  79],  # 7: traffic_cone
    [255, 140,   0],  # 8: trailer
    [255,  99,  71],  # 9: truck
    [  0, 207, 191],  # 10: driveable_surface
    [175,   0,  75],  # 11: other_flat
    [ 75,   0,  75],  # 12: sidewalk
    [112, 180,  60],  # 13: terrain
    [222, 184, 135],  # 14: manmade
    [  0, 175,   0],  # 15: vegetation
    [  0,   0,   0],  # 16: free/empty
], dtype=np.uint8)


def _colorize_sem(cls_map):
    """cls_map: (H, W) int, 0=invalid/noise, 1-16=classes → RGB (H, W, 3)"""
    rgb = np.full((*cls_map.shape, 3), 128, dtype=np.uint8)  # default gray
    valid = cls_map > 0
    rgb[valid] = _NUSC_PALETTE[np.clip(cls_map[valid] - 1, 0, 16)]
    return rgb


def _depth_to_rgb(depth_np, vmin=0.0, vmax=40.0):
    """depth_np: (H, W) float → RGB (H, W, 3)"""
    norm = np.clip((depth_np - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)
    r = np.clip(norm * 4 - 2, 0, 1)
    g = np.clip(np.minimum(norm * 4, 4 - norm * 4), 0, 1)
    b = np.clip(1 - norm * 4, 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    invalid = depth_np <= 0
    rgb[invalid] = 0.5
    return (rgb * 255).astype(np.uint8)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize 2D Gaussian splatting renders vs pseudo labels.")
    parser.add_argument("--py-config", required=True, help="Path to config file.")
    parser.add_argument("--work-dir", required=True, help="Experiment directory (contains latest.pth).")
    parser.add_argument("--resume-from", default="", help="Explicit checkpoint path.")
    parser.add_argument("--out-dir", default="", help="Output directory. Default: <work_dir>/render_2d_vis")
    parser.add_argument("--split", choices=("val", "train"), default="val")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dpi", type=int, default=1, help="Not used; kept for API compat.")
    return parser.parse_args()


def _checkpoint_path(args):
    if args.resume_from:
        return args.resume_from
    latest = os.path.join(args.work_dir, "latest.pth")
    if os.path.exists(latest):
        return latest
    raise FileNotFoundError(f"No checkpoint found at {latest}")


def _load_model(cfg, checkpoint_path, device):
    import model  # noqa: F401 — registers custom modules

    model_instance = build_segmentor(cfg.model)
    model_instance.init_weights()

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    try:
        load_msg = model_instance.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        model_state = model_instance.state_dict()
        filtered = {k: v for k, v in state_dict.items()
                    if k in model_state and hasattr(v, "shape") and model_state[k].shape == v.shape}
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

    # Ensure pseudo label config is present in val_dataset_config
    # (normally only train has it, but we need it for visualization)
    _pseudo_keys = [
        "metric3d_root", "grounded_sam_root",
        "pseudo_label_scale", "max_pseudo_depth", "pseudo_label_crop_top",
    ]
    for k in _pseudo_keys:
        if k not in cfg.val_dataset_config and k in cfg.train_dataset_config:
            cfg.val_dataset_config[k] = cfg.train_dataset_config[k]

    if split == "train":
        train_loader, _ = get_dataloader(
            cfg.train_dataset_config, cfg.val_dataset_config,
            cfg.train_loader, cfg.val_loader, dist=False, val_only=False,
        )
        return train_loader

    _, val_loader = get_dataloader(
        cfg.train_dataset_config, cfg.val_dataset_config,
        cfg.train_loader, cfg.val_loader, dist=False, val_only=True,
    )
    return val_loader


def _render_gaussians(model, gaussian, gs_extrins, gs_intrins):
    """
    Force-render Gaussians to 2D even in eval mode.
    Returns rendered_sem (1, nC, H, W, 17), rendered_depth (1, nC, H, W).
    """
    rasterizer = model.head.rasterizer_2d
    if rasterizer is None:
        raise RuntimeError("Model has no rasterizer_2d — is use_pseudo_label enabled in config?")
    with torch.no_grad():
        rendered_sem, rendered_depth = rasterizer(gaussian, gs_extrins, gs_intrins)
    return rendered_sem, rendered_depth


def _get_input_img(data, cam_idx):
    """Extract un-normalized camera image from input tensor for display."""
    input_imgs = data.get("img")
    if input_imgs is None:
        return None
    try:
        # input_imgs: (B, F, N, C, H, W)
        img_t = input_imgs[0, -1, cam_idx].detach().cpu().float().numpy()  # (C, H, W)
        img_t = img_t.transpose(1, 2, 0)  # (H, W, C)
        img_t = img_t * np.array([58.395, 57.12, 57.375], dtype=np.float32) \
                      + np.array([123.675, 116.28, 103.53], dtype=np.float32)
        img_t = np.clip(img_t, 0, 255).astype(np.uint8)
        # Un-flip if augmented
        aug_flip = data.get("aug_flip")
        if aug_flip is not None:
            flip_val = aug_flip.item() if hasattr(aug_flip, 'item') else bool(aug_flip)
            if flip_val:
                img_t = img_t[:, ::-1, :].copy()
        # Crop top to match pseudo label region
        H_img = img_t.shape[0]
        crop_start = int((318 - 36) / (900 - 36) * H_img)
        img_t = img_t[crop_start:, :, :]
        return img_t
    except Exception:
        return None


def _save_vis_image(rendered_sem, rendered_depth, pseudo_seg, pseudo_depth,
                    data, out_path):
    """
    Save a single visualization image.
    Layout: 6 cameras vertically, each row = [pred_sem | gt_sem | pred_depth | gt_depth | input_img]
    """
    B, nC, H, W, _ = rendered_sem.shape
    rows = []
    for cam in range(nC):
        # pred semantic
        pred_cls = rendered_sem[0, cam].detach().cpu().argmax(dim=-1).numpy()
        pred_sem_rgb = _colorize_sem(pred_cls)

        # gt semantic
        gt_cls = pseudo_seg[0, cam].detach().cpu().numpy().astype(np.int32)
        gt_sem_rgb = _colorize_sem(gt_cls)

        # pred depth
        pred_d = rendered_depth[0, cam].detach().cpu().numpy()
        pred_d_rgb = _depth_to_rgb(pred_d)

        # gt depth
        gt_d = pseudo_depth[0, cam].detach().cpu().numpy()
        gt_d_rgb = _depth_to_rgb(gt_d)

        # input image
        input_img = _get_input_img(data, cam)
        if input_img is not None:
            input_img_resized = np.array(Image.fromarray(input_img).resize((W, H), Image.BILINEAR))
        else:
            input_img_resized = None

        cols = [pred_sem_rgb, gt_sem_rgb, pred_d_rgb, gt_d_rgb]
        if input_img_resized is not None:
            cols.append(input_img_resized)
        row = np.concatenate(cols, axis=1)

        # separator
        sep = np.zeros((2, row.shape[1], 3), dtype=np.uint8)
        rows.append(sep)
        rows.append(row)

    combined = np.concatenate(rows, axis=0)
    Image.fromarray(combined).save(out_path, quality=92)


def main():
    args = parse_args()
    torch.backends.cudnn.benchmark = True

    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = args.work_dir
    out_dir = args.out_dir if args.out_dir else os.path.join(args.work_dir, "render_2d_vis")
    os.makedirs(out_dir, exist_ok=True)

    checkpoint_path = _checkpoint_path(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_instance = _load_model(cfg, checkpoint_path, device)
    loader = _build_loader(cfg, args.split)

    saved = 0
    with torch.no_grad():
        for batch_index, data in enumerate(loader):
            if batch_index < args.start_index:
                continue
            if saved >= args.num_samples:
                break

            # Check pseudo labels exist
            pseudo_seg = data.get("pseudo_seg")
            pseudo_depth = data.get("pseudo_depth")
            gs_extrins = data.get("gs_extrins")
            gs_intrins = data.get("gs_intrins")

            if pseudo_seg is None or gs_extrins is None:
                print(f"[SKIP] batch {batch_index}: no pseudo labels or gs params")
                continue

            # Move tensors to device
            for key in list(data.keys()):
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].to(device)

            input_imgs = data.pop("img")
            # Forward pass to get gaussian representation
            result_dict = model_instance(imgs=input_imgs, metas=data)

            # Get gaussian from result and force-render
            gaussian = result_dict.get("gaussian")
            if gaussian is None:
                print(f"[SKIP] batch {batch_index}: model did not return gaussian")
                continue

            gs_ext = data["gs_extrins"].unsqueeze(0) if data["gs_extrins"].dim() == 3 else data["gs_extrins"]
            gs_int = data["gs_intrins"].unsqueeze(0) if data["gs_intrins"].dim() == 3 else data["gs_intrins"]
            rendered_sem, rendered_depth = _render_gaussians(
                model_instance, gaussian, gs_ext, gs_int
            )

            # Pseudo labels
            ps = data["pseudo_seg"].unsqueeze(0) if data["pseudo_seg"].dim() == 3 else data["pseudo_seg"]
            pd = data["pseudo_depth"].unsqueeze(0) if data["pseudo_depth"].dim() == 3 else data["pseudo_depth"]

            # Save
            out_path = os.path.join(out_dir, f"sample_{batch_index:06d}.jpg")
            # Keep data on CPU for img extraction, put img back
            data["img"] = input_imgs
            _save_vis_image(rendered_sem, rendered_depth, ps, pd, data, out_path)
            print(f"[OK] saved {out_path}")
            saved += 1

    if saved == 0:
        raise SystemExit("No samples visualized. Check --start-index / pseudo label availability.")
    print(f"[DONE] saved {saved} figure(s) under {out_dir}")


if __name__ == "__main__":
    main()
