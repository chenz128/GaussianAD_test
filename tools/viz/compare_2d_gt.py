#!/usr/bin/env python3
"""
对比两种 2D GT 生成方式的可视化脚本:
  方法A: 当前方案 — 外部伪标签 (Metric3D 深度 + Grounded SAM 语义)
  方法B: VoxelSplat 方案 — 3D occ GT 体素通过 gsplat 在线渲染到 2D

对每个样本输出一张对比图:
  上方 6 相机: 方法A 的语义+深度
  下方 6 相机: 方法B 的语义+深度
  
用法 (在远程 H20 上运行):
  /data/chenz/conda_env/splatting/bin/python tools/viz/compare_2d_gt.py \
      --num-samples 3 --out-dir out/compare_2d_gt
"""

import argparse
import os
import sys
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from pyquaternion import Quaternion

# gsplat
import gsplat


# ─── nuScenes 17-class palette (class 1~17, index 0 = invalid/noise) ───
NUSC_PALETTE = np.array([
    [128, 128, 128],  # 0: noise/invalid → gray
    [255, 120, 50],   # 1: barrier → orange
    [255, 192, 203],  # 2: bicycle → pink
    [255, 255, 0],    # 3: bus → yellow
    [0, 150, 245],    # 4: car → blue
    [0, 255, 255],    # 5: construction_vehicle → cyan
    [255, 127, 80],   # 6: motorcycle → coral
    [200, 180, 0],    # 7: pedestrian → dark yellow
    [255, 0, 0],      # 8: traffic_cone → red
    [255, 240, 150],  # 9: trailer → light yellow
    [135, 60, 0],     # 10: truck → brown
    [160, 32, 240],   # 11: driveable_surface → purple
    [255, 0, 255],    # 12: other_flat → magenta
    [139, 137, 137],  # 13: sidewalk → dark gray
    [75, 0, 75],      # 14: terrain → dark purple
    [150, 240, 80],   # 15: manmade → light green
    [230, 230, 250],  # 16: vegetation → lavender
    [0, 0, 0],        # 17: empty/free → black (should not appear)
], dtype=np.uint8)


def colorize_sem(sem_map):
    """sem_map: (H, W) int, 0=invalid, 1-16=classes, 17=empty → (H, W, 3) RGB"""
    sem_map = np.clip(sem_map, 0, 17)
    return NUSC_PALETTE[sem_map]


def colorize_depth(depth_map, vmin=0.5, vmax=40.0):
    """depth_map: (H, W) float → (H, W, 3) RGB, invalid (<=0) → gray"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.cm as cm
    
    valid = depth_map > 0.1
    norm_d = np.clip((depth_map - vmin) / (vmax - vmin), 0, 1)
    colored = (cm.turbo(norm_d)[:, :, :3] * 255).astype(np.uint8)
    colored[~valid] = 128  # gray for invalid
    return colored


def load_pkl_info(pkl_path, sample_idx=0):
    """Load one sample's info from the PKL."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    scene_infos = data['infos']
    metadata = data['metadata']
    # get the sample_idx-th keyframe
    kf = sorted(metadata, key=lambda x: x[0] + "{:0>3}".format(str(x[1])))
    scene_token, idx = kf[sample_idx]
    info = scene_infos[scene_token][idx]
    return info


def get_cam_params(info, scale=0.44, crop_top=62):
    """Extract camera intrinsics and extrinsics (lidar2cam) for 6 cameras.
    
    Returns:
        intrins: (6, 3, 3) scaled intrinsics
        extrins: (6, 4, 4) lidar2cam matrices
    """
    sensor_types = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    
    lidar2ego = np.eye(4, dtype=np.float64)
    lidar2ego[:3, :3] = Quaternion(info['lidar2ego_rotation']).rotation_matrix
    lidar2ego[:3, 3] = np.asarray(info['lidar2ego_translation'])
    
    intrins_list = []
    extrins_list = []
    
    for cam_name in sensor_types:
        cam_info = info['cams'][cam_name]
        
        # sensor2ego (cam2ego)
        cam2ego = np.eye(4, dtype=np.float64)
        cam2ego[:3, :3] = Quaternion(cam_info['sensor2ego_rotation']).rotation_matrix
        cam2ego[:3, 3] = np.asarray(cam_info['sensor2ego_translation'])
        
        ego2cam = np.linalg.inv(cam2ego)
        lidar2cam = ego2cam @ lidar2ego  # (4, 4)
        
        # intrinsic
        K = np.array(cam_info['cam_intrinsic'], dtype=np.float64)  # (3, 3)
        K[0, 0] *= scale  # fx
        K[1, 1] *= scale  # fy
        K[0, 2] *= scale  # cx
        K[1, 2] *= scale  # cy
        K[1, 2] -= crop_top  # crop top adjustment
        
        intrins_list.append(K)
        extrins_list.append(lidar2cam)
    
    return np.array(intrins_list, dtype=np.float32), np.array(extrins_list, dtype=np.float32)


def load_occ_gt(occ_path):
    """Load SurroundOcc GT: sparse format (N, 4) → dense (120, 120, 8) semantic.
    
    Original 200x200x16 at 0.5m resolution centered at [-50,50]x[-50,50]x[-5,3].
    After crop to [-30,30]x[-30,30]x[-2,2]: inner 120x120x8 region.
    
    Returns:
        occ_sem: (120, 120, 8) int64, class 0-16 are valid, 17=empty
        occ_xyz: (120, 120, 8, 3) float32, voxel centers in LIDAR frame
    """
    label = np.load(occ_path)  # (N, 4): [x_idx, y_idx, z_idx, class]
    
    # Full grid is 200x200x16
    full_label = np.ones((200, 200, 16), dtype=np.int64) * 17  # 17 = empty
    full_label[label[:, 0], label[:, 1], label[:, 2]] = label[:, 3]
    
    # Crop to inner 120x120x8 (matching dataset code: [40:-40, 40:-40, 6:-2])
    occ_sem = full_label[40:-40, 40:-40, 6:-2]  # (120, 120, 8)
    
    # Generate xyz coordinates: pc_range=[-30, -30, -2, 30, 30, 2], grid=[120,120,8], reso=0.5
    reso = 0.5
    pc_range = [-30.0, -30.0, -2.0, 30.0, 30.0, 2.0]
    xx = np.arange(120) * reso + 0.5 * reso + pc_range[0]
    yy = np.arange(120) * reso + 0.5 * reso + pc_range[1]
    zz = np.arange(8) * reso + 0.5 * reso + pc_range[2]
    
    xxx, yyy, zzz = np.meshgrid(xx, yy, zz, indexing='ij')
    occ_xyz = np.stack([xxx, yyy, zzz], axis=-1).astype(np.float32)  # (120, 120, 8, 3)
    
    return occ_sem, occ_xyz


def render_occ_gt_gsplat(occ_sem, occ_xyz, intrins, extrins, render_h, render_w, 
                          num_classes=17, gaussian_size=0.2):
    """Render 3D occ GT to 2D using gsplat (VoxelSplat method).
    
    Args:
        occ_sem: (X, Y, Z) int64, semantic labels (0-16 valid, 17=empty)
        occ_xyz: (X, Y, Z, 3) float32, voxel centers in LIDAR frame
        intrins: (6, 3, 3) camera intrinsics
        extrins: (6, 4, 4) lidar2cam transforms
        render_h, render_w: output image size
        num_classes: number of semantic classes (excluding empty)
        gaussian_size: fixed scale of each gaussian
    
    Returns:
        rendered_sem: (6, H, W) int, argmax semantic class
        rendered_depth: (6, H, W) float, rendered depth
    """
    device = torch.device('cuda:0')
    
    # Filter non-empty voxels
    non_empty_mask = occ_sem < num_classes  # exclude 17 (empty) and anything invalid
    valid_mask = non_empty_mask & (occ_sem > 0)  # also exclude 0 (noise) for cleaner result
    
    pts = torch.from_numpy(occ_xyz[valid_mask]).to(device)      # (N, 3)
    labels = torch.from_numpy(occ_sem[valid_mask].astype(np.int64)).to(device)  # (N,)
    N = pts.shape[0]
    
    # Build gaussian properties
    # one-hot semantics as "colors"
    sem_onehot = F.one_hot(labels, num_classes=num_classes).float()  # (N, 17)
    
    # Fixed gaussian properties (like VoxelSplat)
    opacities = torch.ones(N, device=device)          # all opaque
    scales = torch.ones(N, 3, device=device) * gaussian_size
    rotations = torch.zeros(N, 4, device=device)
    rotations[:, 0] = 1.0  # identity quaternion (w,x,y,z)
    
    intrins_t = torch.from_numpy(intrins).to(device)   # (6, 3, 3)
    extrins_t = torch.from_numpy(extrins).to(device)   # (6, 4, 4)
    
    rendered_sems = []
    rendered_depths = []
    
    for cam_idx in range(6):
        viewmat = extrins_t[cam_idx:cam_idx+1]  # (1, 4, 4)
        K = intrins_t[cam_idx:cam_idx+1]         # (1, 3, 3)
        
        rendered, _, _ = gsplat.rasterization(
            means=pts,
            quats=rotations,
            scales=scales,
            opacities=opacities,
            colors=sem_onehot,
            viewmats=viewmat,
            Ks=K,
            width=render_w,
            height=render_h,
            render_mode='RGB+D',
            near_plane=0.1,
            far_plane=100.0,
        )
        # rendered: (1, H, W, 18) — first 17 are sem logits, last 1 is depth
        sem_logits = rendered[0, :, :, :num_classes]   # (H, W, 17)
        depth = rendered[0, :, :, num_classes]          # (H, W)
        
        sem_class = sem_logits.argmax(dim=-1).cpu().numpy() + 1  # shift to 1-16 (like GT labels)
        # where depth is 0, mark as invalid
        depth_np = depth.cpu().numpy()
        sem_class[depth_np < 0.1] = 0  # invalid
        
        rendered_sems.append(sem_class)
        rendered_depths.append(depth_np)
    
    return np.stack(rendered_sems), np.stack(rendered_depths)  # (6,H,W), (6,H,W)


def load_pseudo_labels(info, metric3d_root, grounded_sam_root, scale=0.44, crop_top=62, max_depth=40.0):
    """Load external pseudo labels (current method A).
    
    Returns:
        pseudo_seg: (6, H', W') int
        pseudo_depth: (6, H', W') float
    """
    from nuscenes import NuScenes
    
    # Camera reorder maps
    sensor_types = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    seg_cam_order = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_BACK_LEFT',
                     'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_FRONT_RIGHT']
    depth_cam_order = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                       'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    seg_reorder = [seg_cam_order.index(c) for c in sensor_types]
    depth_reorder = [depth_cam_order.index(c) for c in sensor_types]
    
    sample_token = info['token']
    scene_token = info['scene_token']
    
    # Need scene_name from scene_token
    nusc = NuScenes(version='v1.0-trainval', dataroot='data/nuscenes', verbose=False)
    scene_name = {s['token']: s['name'] for s in nusc.scene}[scene_token]
    
    # Load seg
    seg_path = os.path.join(grounded_sam_root, scene_name, f'{sample_token}.npy')
    pseudo_seg = torch.from_numpy(np.load(seg_path).astype(np.int64))  # (6, 900, 1600)
    pseudo_seg = pseudo_seg[seg_reorder]
    
    # Load depth
    depth_path = os.path.join(metric3d_root, scene_name, f'{sample_token}.npy')
    pseudo_depth = torch.from_numpy(np.load(depth_path).astype(np.float32))
    pseudo_depth = pseudo_depth[depth_reorder]
    
    # Downsample
    if scale != 1.0:
        pseudo_seg = F.interpolate(pseudo_seg[:, None].float(), scale_factor=scale, mode='nearest').squeeze(1).long()
        pseudo_depth = F.interpolate(pseudo_depth[:, None], scale_factor=scale, mode='bilinear', align_corners=False).squeeze(1)
    
    # Crop top
    if crop_top > 0:
        pseudo_seg = pseudo_seg[:, crop_top:]
        pseudo_depth = pseudo_depth[:, crop_top:]
    
    # Mask far
    far_mask = pseudo_depth > max_depth
    pseudo_seg[far_mask] = 0
    pseudo_depth[far_mask] = 0.0
    
    return pseudo_seg.numpy(), pseudo_depth.numpy()


def make_comparison_image(pseudo_seg, pseudo_depth, render_seg, render_depth, cam_names):
    """Create a side-by-side comparison image.
    
    Layout per camera: [Method_A_sem | Method_A_depth | Method_B_sem | Method_B_depth]
    All 6 cameras stacked vertically.
    """
    from PIL import Image, ImageDraw, ImageFont
    
    rows = []
    H = pseudo_seg.shape[1]
    W = pseudo_seg.shape[2]
    
    for cam_idx in range(6):
        # Method A (pseudo labels)
        sem_a = colorize_sem(pseudo_seg[cam_idx])
        depth_a = colorize_depth(pseudo_depth[cam_idx])
        
        # Method B (gsplat rendered GT)
        sem_b = colorize_sem(render_seg[cam_idx])
        depth_b = colorize_depth(render_depth[cam_idx])
        
        # Concatenate horizontally: [sem_A | depth_A | sem_B | depth_B]
        row = np.concatenate([sem_a, depth_a, sem_b, depth_b], axis=1)
        rows.append(row)
    
    # Stack all cameras vertically
    full_img = np.concatenate(rows, axis=0)
    
    # Add labels
    img = Image.fromarray(full_img)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Column headers
    col_w = W
    headers = ['Pseudo Sem (A)', 'Pseudo Depth (A)', 'Rendered Sem (B)', 'Rendered Depth (B)']
    for i, header in enumerate(headers):
        x = i * col_w + 10
        draw.text((x, 5), header, fill=(255, 255, 255), font=font)
    
    # Row labels (camera names)
    for cam_idx, cam_name in enumerate(cam_names):
        y = cam_idx * H + H // 2
        draw.text((5, y), cam_name.replace('CAM_', ''), fill=(255, 255, 255), font=font)
    
    return img


def main():
    parser = argparse.ArgumentParser(description='Compare 2D GT methods: pseudo labels vs gsplat rendered occ GT')
    parser.add_argument('--pkl', type=str, 
                        default='data/nuscenes_cam/nuscenes_infos_train_gaussian_ad_v4.pkl')
    parser.add_argument('--metric3d-root', type=str,
                        default='/data/chenz/Gaussianflowocc_test/data/metric_3d_nusc')
    parser.add_argument('--grounded-sam-root', type=str,
                        default='/data/chenz/Gaussianflowocc_test/data/grounded_sam_nusc')
    parser.add_argument('--scale', type=float, default=0.44)
    parser.add_argument('--crop-top', type=int, default=62,
                        help='Crop top pixels after downscale (default: 140*0.44≈62)')
    parser.add_argument('--max-depth', type=float, default=40.0)
    parser.add_argument('--gaussian-size', type=float, default=0.2,
                        help='Fixed gaussian scale for VoxelSplat rendering')
    parser.add_argument('--num-samples', type=int, default=3)
    parser.add_argument('--start-idx', type=int, default=100,
                        help='Start sample index in PKL')
    parser.add_argument('--out-dir', type=str, default='out/compare_2d_gt')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load PKL
    print(f"Loading PKL: {args.pkl}")
    with open(args.pkl, 'rb') as f:
        data = pickle.load(f)
    scene_infos = data['infos']
    metadata = data['metadata']
    keyframes = sorted(metadata, key=lambda x: x[0] + "{:0>3}".format(str(x[1])))
    
    # NuScenes for scene name lookup
    from nuscenes import NuScenes
    nusc = NuScenes(version='v1.0-trainval', dataroot='data/nuscenes', verbose=False)
    scene_token_to_name = {s['token']: s['name'] for s in nusc.scene}
    
    sensor_types = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    
    # Camera reorder for pseudo labels
    seg_cam_order = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_BACK_LEFT',
                     'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_FRONT_RIGHT']
    depth_cam_order = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                       'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    seg_reorder = [seg_cam_order.index(c) for c in sensor_types]
    depth_reorder = [depth_cam_order.index(c) for c in sensor_types]
    
    # Render resolution (after scale + crop)
    render_h = int(900 * args.scale) - args.crop_top  # 396 - 62 = 334
    render_w = int(1600 * args.scale)                  # 704
    print(f"Render resolution: {render_h} x {render_w}")
    
    for sample_i in range(args.num_samples):
        idx = args.start_idx + sample_i
        scene_token, frame_idx = keyframes[idx]
        info = scene_infos[scene_token][frame_idx]
        scene_name = scene_token_to_name[info['scene_token']]
        sample_token = info['token']
        
        print(f"\n[{sample_i+1}/{args.num_samples}] Scene: {scene_name}, Token: {sample_token[:12]}...")
        
        # ─── Method A: Pseudo labels ───
        print("  Loading pseudo labels (Method A)...")
        seg_path = os.path.join(args.grounded_sam_root, scene_name, f'{sample_token}.npy')
        depth_path = os.path.join(args.metric3d_root, scene_name, f'{sample_token}.npy')
        
        if not os.path.exists(seg_path) or not os.path.exists(depth_path):
            print(f"  SKIP: pseudo label files not found for {scene_name}/{sample_token}")
            continue
        
        pseudo_seg = torch.from_numpy(np.load(seg_path).astype(np.int64))[seg_reorder]
        pseudo_depth = torch.from_numpy(np.load(depth_path).astype(np.float32))[depth_reorder]
        
        if args.scale != 1.0:
            pseudo_seg = F.interpolate(pseudo_seg[:, None].float(), scale_factor=args.scale, mode='nearest').squeeze(1).long()
            pseudo_depth = F.interpolate(pseudo_depth[:, None], scale_factor=args.scale, mode='bilinear', align_corners=False).squeeze(1)
        if args.crop_top > 0:
            pseudo_seg = pseudo_seg[:, args.crop_top:]
            pseudo_depth = pseudo_depth[:, args.crop_top:]
        far_mask = pseudo_depth > args.max_depth
        pseudo_seg[far_mask] = 0
        pseudo_depth[far_mask] = 0.0
        
        pseudo_seg_np = pseudo_seg.numpy()
        pseudo_depth_np = pseudo_depth.numpy()
        
        # ─── Method B: gsplat rendered occ GT ───
        print("  Loading occ GT and rendering via gsplat (Method B)...")
        occ_path = info.get('occ_path', None)
        if occ_path is None:
            lidar_filename = info['lidar_path'].split('/')[-1]
            occ_path = os.path.join('data/surroundocc/train_samples', lidar_filename + '.npy')
        
        if not os.path.exists(occ_path):
            print(f"  SKIP: occ GT not found: {occ_path}")
            continue
        
        occ_sem, occ_xyz = load_occ_gt(occ_path)
        intrins, extrins = get_cam_params(info, scale=args.scale, crop_top=args.crop_top)
        
        with torch.no_grad():
            render_seg, render_depth = render_occ_gt_gsplat(
                occ_sem, occ_xyz, intrins, extrins,
                render_h, render_w,
                gaussian_size=args.gaussian_size
            )
        
        # ─── Make comparison image ───
        print("  Creating comparison image...")
        img = make_comparison_image(
            pseudo_seg_np, pseudo_depth_np,
            render_seg, render_depth,
            sensor_types
        )
        
        out_path = os.path.join(args.out_dir, f'compare_{idx:04d}_{scene_name}_{sample_token[:8]}.jpg')
        img.save(out_path, quality=90)
        print(f"  Saved: {out_path}")
    
    print(f"\nDone! Results in: {args.out_dir}")


if __name__ == '__main__':
    main()
