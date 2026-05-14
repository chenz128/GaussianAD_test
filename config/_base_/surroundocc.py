# ================== data ========================
data_root = "data/nuscenes/"
anno_root = "data/nuscenes_cam/"
train_occ_path = "data/surroundocc/train_samples"
val_occ_path = "data/surroundocc/val_samples"
input_shape = (704, 256)
batch_size = 1  # H20 96GB, full losses (occ+det+map, no plan)
grid_size=[200, 200, 16]
pc_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
map_classes = ['divider', 'ped_crossing','boundary']
fixed_ptsnum_per_gt_line = 20
num_frames = 4
map_eval_use_same_gt_sample_num_flag = True
meta_keys = [
    'img',
    'lidar2img',
    'cam_intrinsic',
    'ego2img',
    'image_wh'
]

collect_keys = [
    'img',
    'projection_mat',
    'lidar2img',
    'lidar2global',
    'image_wh',
    'occ_label',
    'occ_xyz',
    'occ_cam_mask',
    'gt_boxes',
    'gt_names',
    'gt_map',
    'gt_velocity',
    'num_lidar_pts',
    'ego_his_trajs',
    'ego_fut_trajs',
    'ego_fut_masks',
    'ego_fut_cmd',
    'ego_lcf_feat',
    'fut_valid_flag',
    'gt_agent_fut_trajs',
    'gt_agent_fut_masks',
    'gt_agent_fut_goal',
    'gt_agent_lcf_feat',
    'gt_agent_fut_yaw',
    "flow_info",
    "ego2global"
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type='LoadMultiViewImageFromMultiSweeps'),
    dict(type="LoadOccupancySurroundOcc", occ_path=train_occ_path, semantic=True, use_ego=False),
    dict(type="ResizeCropFlipImage"),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(type="NuScenesAdaptor", use_ego=False, num_cams=6, num_frames=num_frames, meta_keys=meta_keys),
    dict(type='CustomCollect3D', keys=collect_keys),
]

test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type='LoadMultiViewImageFromMultiSweeps'),
    dict(type="LoadOccupancySurroundOcc", occ_path=val_occ_path, semantic=True, use_ego=False),
    dict(type="ResizeCropFlipImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(type="NuScenesAdaptor", use_ego=False, num_cams=6, num_frames=num_frames, meta_keys=meta_keys),
    dict(type='CustomCollect3D', keys=collect_keys),
]

data_aug_conf = {
    "resize_lim": (0.40, 0.47),
    "final_dim": input_shape[::-1],
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (-5.4, 5.4),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
}

aux_seg_cfg = dict(
    use_aux_seg=False,
    bev_seg=False,
    pv_seg=False,
    seg_classes=1,
    feat_down_sample=32,
    pv_thickness=1,
)

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

train_dataset_config = dict(
    type='NuScenesDataset',
    data_root=data_root,
    imageset=anno_root + "nuscenes_infos_train_gaussian_ad_v4.pkl",
    data_aug_conf=data_aug_conf,
    pipeline=train_pipeline,
    phase='train',
    aux_seg=aux_seg_cfg,
    bev_size=(grid_size[1], grid_size[0]),
    pc_range=pc_range,
    map_classes=map_classes,
    fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
    padding_value=-10000,
    map_eval_use_same_gt_sample_num_flag=map_eval_use_same_gt_sample_num_flag,
    modality=input_modality,
)

val_dataset_config = dict(
    type='NuScenesDataset',
    data_root=data_root,
    imageset=anno_root + "nuscenes_infos_val_gaussian_ad_v4.pkl",
    data_aug_conf=data_aug_conf,
    pipeline=test_pipeline,
    phase='val',
    bev_size=(grid_size[1], grid_size[0]),
    pc_range=pc_range,
    map_classes=map_classes,
    fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
    padding_value=-10000,
    map_eval_use_same_gt_sample_num_flag=map_eval_use_same_gt_sample_num_flag,
    modality=input_modality,
)

train_loader = dict(
    batch_size=batch_size,
    num_workers=4,
    shuffle=True,
    persistent_workers=True,
    prefetch_factor=2,
)

val_loader = dict(
    batch_size=batch_size,
    num_workers=4,
    persistent_workers=True,
    prefetch_factor=2,
)
