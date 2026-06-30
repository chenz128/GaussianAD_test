"""
Plan A: Depth-initialized fixed learnable anchor.
- Use pre-computed xyz from depth maps as initial anchor positions.
- xyz is still nn.Parameter, fully learnable during training.
- Same loss/model as base config (occ + flow + det).
"""
_base_ = [
    './_base_/misc.py',
    './_base_/model.py',
    './_base_/surroundocc.py'
]
import os

# =========== data config ==============
input_shape = (1600, 864)
data_aug_conf = {
    "resize_lim": (1.0, 1.0),
    "final_dim": input_shape[::-1],
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
}
num_frames = 4
num_map_classes = len(_base_.map_classes)
pc_range = [-30.0, -30.0, -2.0, 30.0, 30.0, 2.0]
fixed_ptsnum_per_gt_line = 20
fixed_ptsnum_per_pred_line = 20

# =========== misc config ==============
lr = float(os.environ.get("LR", 2e-4))
optimizer = dict(
    optimizer = dict(
        type="AdamW", lr=lr, weight_decay=0.01,
    ),
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1)}
    )
)
grad_max_norm = 35
max_epochs = 15
frozen_modules = ['map_decoder', 'planner_head']
find_unused_parameters = False
backbone_fp16 = True

# ========= model config ===============
embed_dims = 128
num_decoder = 4
num_single_frame_decoder = 1
grid_size=[120, 120, 8]
scale_range = [0.08, 0.64]
xyz_coordinate = 'cartesian'
phi_activation = 'sigmoid'
include_opa = True
load_from = 'ckpts/r101_dcn_fcos3d_pretrain.pth'
semantics = True
semantic_dim = 17

offset = True
offset_dim = 2*6

voxel_size=[0.5, 0.5, 0.5]
det_config = dict(
    class_names=['car','truck', 'construction_vehicle', 'bus', 'trailer',
              'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'],
    num_point_features=28,
    grid_size=grid_size,
    voxel_size=voxel_size,
    point_cloud_range=pc_range,
    depth_downsample_factor=None,
)
_dim_ = 256
_pos_dim_ = _dim_//2

val_dataset_config = dict(
    imageset='data/nuscenes_cam/nuscenes_infos_val_gaussian_ad_v4.pkl',
    data_aug_conf=data_aug_conf,
    class_names=det_config['class_names'],
    pc_range=pc_range,
    num_frames=4
)
train_dataset_config = dict(
    imageset='data/nuscenes_cam/nuscenes_infos_train_gaussian_ad_v4.pkl',
    data_aug_conf=data_aug_conf,
    class_names=det_config['class_names'],
    pc_range=pc_range,
    num_frames=4,
)

model = dict(
    img_backbone_out_indices=[0, 1, 2, 3],
    history_no_grad=False,
    img_backbone=dict(
        _delete_=True,
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        with_cp = True,
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)),
    img_neck=dict(
        start_level=1),
    lifter=dict(
        type='GaussianLifter',
        num_anchor=25600,
        embed_dims=embed_dims,
        anchor_grad=True,
        semantic_dim=semantic_dim,
        include_opa=include_opa,
        offset=offset,
        offset_dim=offset_dim,
        init_xyz_path='data/depth_anchor_init_25600.npy',
        pc_range=pc_range,
    ),
    encoder=dict(
        type='GaussianOccEncoder',
        anchor_encoder=dict(
            type='SparseGaussian3DEncoder',
            embed_dims=embed_dims,
            include_opa=include_opa,
            semantics=semantics,
            semantic_dim=semantic_dim
        ),
        norm_layer=dict(type="LN", normalized_shape=embed_dims),
        ffn=dict(type="AsymmetricFFN", in_channels=embed_dims*2, pre_norm=dict(type="LN"), embed_dims=embed_dims),
        deformable_model=dict(
            type='DeformableFeatureAggregation',
            embed_dims=embed_dims,
            num_groups=4,
            num_levels=4,
            num_cams=6,
            attn_drop=0.15,
            use_deformable_func=True,
            use_camera_embed=True,
            residual_mode="cat",
            kps_generator=dict(
                type="SparseGaussian3DKeyPointsGenerator",
                num_learnable_pts=6,
                fix_scale=[
                    [0, 0, 0],
                    [0.45, 0, 0],
                    [-0.45, 0, 0],
                    [0, 0.45, 0],
                    [0, -0.45, 0],
                    [0, 0, 0.45],
                    [0, 0, -0.45],
                ],
                pc_range=pc_range,
                scale_range=scale_range),
        ),
        spconv_layer=dict(
            _delete_=True,
            type='SparseGaussian3DConv',
            embed_dims=embed_dims,
            pc_range=pc_range,
            grid_size=grid_size,
        ),
        refine_layer=dict(
            type='SparseGaussian3DRefinementModule',
            embed_dims=embed_dims,
            pc_range=pc_range,
            scale_range=scale_range,
            restrict_xyz=True,
            unit_xyz=[4.0, 4.0, 1.0],
            semantics=semantics,
            semantic_dim=semantic_dim,
            include_opa=include_opa,
            offset=offset,
            offset_dim=offset_dim,
        ),
        num_decoder=num_decoder,
        num_single_frame_decoder=num_single_frame_decoder,
        operation_order=[
            "spconv", "norm", "deformable", "norm", "ffn", "norm", "refine",
        ] * num_decoder,
    ),
    temporal_encoder=dict(
        type='GaussianTemporalEncoder',
        embed_dims=embed_dims,
        pc_range=pc_range,
        grid_size=grid_size,
        num_frames=4,
    ),
    head=dict(
        type='GaussianHead',
        pc_range=pc_range,
        num_anchor=25600,
        num_frames=num_frames,
        num_single_frame_decoder=num_single_frame_decoder,
        scale_range=scale_range,
        semantics=semantics,
        semantic_dim=semantic_dim,
        include_opa=include_opa,
        embed_dims=embed_dims,
        with_cp=False,
        xyz_coordinate=xyz_coordinate,
        phi_activation=phi_activation,
        grid_size=grid_size,
        offset=offset,
        offset_dim=offset_dim,
        empty_args=dict(
            mean=[0, 0, -1.0],
            scale=[60, 60, 4.0],
        ),
        with_empty=True,
        cuda_kwargs=dict(
            _delete_=True,
            scale_multiplier=5,
            H=120, W=120, D=8,
            pc_min=[-30.0, -30.0, -2.0],
            grid_size=0.5),
    ),
    decoder=dict(
        type='VoxelNeXt',
        num_class=10,
        det_config=det_config,
        model_cfg = dict(
            pts_voxel_layer=dict(
                max_num_points=32,
                point_cloud_range=pc_range,
                voxel_size=voxel_size,
                max_voxels=(25600, 25600),
                ),
            vfe=dict(type='MeanVFE'),
            backbone_3d=dict(type='VoxelResBackBone8xVoxelNeXt'),
            dense_head=dict(type='VoxelNeXtHead',
                            class_agnostic=False,
                            input_features=128,
                            class_names_each_head=[
                                ['car'],
                                ['truck', 'construction_vehicle'],
                                ['bus', 'trailer'],
                                ['barrier'],
                                ['motorcycle', 'bicycle'],
                                ['pedestrian', 'traffic_cone'],
                            ],
                            shared_conv_channel= 128,
                            kernel_size_head= 1,
                            predict_boxes_when_training=True,
                            use_bias_before_norm= 'true',
                            num_hm_conv= 2,
                            separate_head_cfg=dict(
                                head_order=['center', 'center_z', 'dim', 'rot', 'vel'],
                                head_dict={
                                    'center': {'out_channels': 2, 'num_conv': 2},
                                    'center_z': {'out_channels': 1, 'num_conv': 2},
                                    'dim': {'out_channels': 3, 'num_conv': 2},
                                    'rot': {'out_channels': 2, 'num_conv': 2},
                                    'vel': {'out_channels': 2, 'num_conv': 2},
                                }),
                            target_assigner_config=dict(
                                feature_map_stride=8,
                                num_max_objs=100,
                                gaussian_overlap=0.1,
                                min_radius=2,
                            ),
                            loss_config=dict(
                                loss_weights={
                                    'cls_weight': 1.0,
                                    'loc_weight': 0.25,
                                    'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 1.0, 1.0]
                                }),
                            post_processing=dict(
                                score_thresh=0,
                                post_center_limit_range=pc_range,
                                max_obj_per_sample=100,
                                nms_config=dict(
                                    nms_type='nms_gpu',
                                    nms_thresh=0.2,
                                    nms_pre_maxsize=1000,
                                    nms_post_maxsize=83,
                            ))),
            post_processing=dict(recall_thresh_list=[0.3, 0.5, 0.7])
        )
    ),
    map_decoder=dict(
        type='MapTRv2',
        use_grid_mask=True,
        video_test_mode=False,
        pts_bbox_head=dict(
            type='MapTRv2Head',
            bev_h=grid_size[1],
            bev_w=grid_size[0],
            num_query=900,
            num_vec_one2one=100,
            num_vec_one2many=600,
            k_one2many=6,
            num_pts_per_vec=fixed_ptsnum_per_pred_line,
            num_pts_per_gt_vec=fixed_ptsnum_per_gt_line,
            dir_interval=1,
            query_embed_type='instance',
            transform_method='minmax',
            gt_shift_pts_pattern='v2',
            num_classes=num_map_classes,
            in_channels=_dim_,
            sync_cls_avg_factor=True,
            with_box_refine=True,
            as_two_stage=False,
            code_size=2,
            code_weights=[1.0, 1.0, 1.0, 1.0],
            aux_seg=_base_.aux_seg_cfg,
            num_map_adapter_conv=2,
            map_adapter_input_channels=128,
            map_adapter_kernel_size=1,
            map_adapter_use_bias=True,
            bbox_coder=dict(
                type='MapTRNMSFreeCoder',
                post_center_range=[-20, -35, -20, -35, 20, 35, 20, 35],
                pc_range=pc_range,
                max_num=50,
                voxel_size=voxel_size,
                num_classes=num_map_classes),
            positional_encoding=dict(
                type='LearnedPositionalEncoding',
                num_feats=_pos_dim_,
                row_num_embed=grid_size[1],
                col_num_embed=grid_size[0],
                ),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=2.0),
            loss_bbox=dict(type='L1Loss', loss_weight=0.0),
            loss_iou=dict(type='GIoULoss', loss_weight=0.0),
            loss_pts=dict(type='PtsL1Loss', loss_weight=5.0),
            loss_dir=dict(type='PtsDirCosLoss', loss_weight=0.005),
            transformer=dict(
                type='MapTRPerceptionTransformer',
                rotate_prev_bev=True,
                use_shift=True,
                use_can_bus=True,
                embed_dims=_dim_,
                decoder=dict(
                    type='MapTRDecoder',
                    num_layers=6,
                    return_intermediate=True,
                    transformerlayers=dict(
                        type='MyCustomBaseTransformerLayer',
                        attn_cfgs=[
                            dict(
                                type='MultiheadAttention',
                                embed_dims=_dim_,
                                num_heads=8,
                                dropout=0.1),
                            dict(
                                type='CustomMSDeformableAttention',
                                embed_dims=_dim_,
                                num_levels=1),
                        ],
                        feedforward_channels=_dim_*2,
                        ffn_dropout=0.1,
                        operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                         'ffn', 'norm')))),
        )),
    planner_head=dict(
        type='PlannerHead',
        embed_dims=embed_dims,
        planning_steps=6,
        loss_planning=10.0,
        loss_collision=1.0,
        with_adapter=True,
        adapter_input_channels=128,
        use_col_optim=False,
        planning_eval='uniad',
        decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=dict(
                type='MyCustomBaseTransformerLayer',
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        embed_dims=embed_dims,
                        num_heads=8,
                        dropout=0.1),
                ],
                feedforward_channels=embed_dims*2,
                ffn_dropout=0.1,
                batch_first=False,
                operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
))

# ========= loss config ===============
loss = dict(
    type='MultiLoss',
    loss_cfgs=[
        dict(
            type='OccupancyLoss',
            weight=1.0,
            empty_label=17,
            num_classes=18,
            use_focal_loss=False,
            use_dice_loss=False,
            balance_cls_weight=True,
            multi_loss_weights=dict(
                loss_voxel_ce_weight=10.0,
                loss_voxel_lovasz_weight=1.0,
            ),
            input_dict={
                'voxel_semantics': 'sampled_label',
                'preds': 'pred_occ',
                'occ_mask': 'occ_mask',
                'sampled_xyz': 'sampled_xyz',
            },
        ),
        dict(
            type='OccupancyFlowLoss',
            weight=1.0,
            empty_label=17,
            num_classes=18,
            use_focal_loss=False,
            use_dice_loss=False,
            balance_cls_weight=True,
            multi_loss_weights=dict(
                loss_voxel_ce_weight=10.0,
                loss_voxel_lovasz_weight=1.0,
            ),
            input_dict={
                'preds': 'pred_occ',
                'flow_preds': 'pred_flow',
                'occ_mask': 'occ_mask',
                'sampled_xyz': 'sampled_xyz',
                'future_occ_xyz': 'future_occ_xyz',
                'future_occ_label': 'future_occ_label',
            },
        ),
        dict(
            type='DetectionLoss',
            weight=1.0,
            input_dict={
                'det_output': 'det_output',
                'gt_bboxes_3d': 'gt_bboxes_3d',
                'gt_labels_3d': 'gt_labels_3d',
            },
        ),
    ]
)

loss_input_convertion = dict(
    pred_occ='pred_occ',
    pred_flow='pred_flow',
    sampled_xyz='sampled_xyz',
    sampled_label='sampled_label',
    occ_mask='occ_mask',
    future_occ_xyz='future_occ_xyz',
    future_occ_label='future_occ_label',
    det_output='det_output',
    gt_bboxes_3d='gt_bboxes_3d',
    gt_labels_3d='gt_labels_3d',
)
