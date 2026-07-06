"""
nuscenes_gs25600_gtbox_oracle 实验配置
GT-box oracle（动静分离上界）：用 GT box(含速度) 作为干净的动静/刚体监督，
探测「最好的动静分离」下物理约束(PhysicsLoss)能达到的上限效果。
相对 dynamic_physics 的改动：
  - DynamicLoss / PhysicsLoss 均 use_gt_box=True（点在框内判定动静，绕开 2D 渲染与噪声伪标签）
  - 关闭 2D splatting 渲染（head.render_config=None）省算力
  - 关闭伪标签加载（metric3d_root/grounded_sam_root/dynamic_gt_root=None）
  - RenderLoss weight=0（保留但不回传，None-safe）
其余配置与 dynamic_physics 一致（轻量 3000 子集、冻结 map/plan 等）。
GPU：h20-new 后 4 张（4, 5, 6, 7）
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
max_epochs = 20

# ========= PCGrad 梯度手术 =========
# main = occ+flow+det (受保护的几何/未来任务，梯度一字不动)
# aux  = dynamic+physics (运动监督，与 main 冲突的梯度分量被正交投影删除)
# 关键：Flow 学 offset 的梯度在 main 组不受影响 -> 未来帧预测能力保留；
# 只砍掉 Physics/Dynamic 里伤当前帧几何的冲突分量。
use_pcgrad = True

# ========= loss config ================
loss = dict(
    type='MultiLoss',
    group_map=dict(
        OccupancyLoss='main',
        OccupancyFlowLoss='main',
        DetectionLoss='main',
        RenderLoss='main',
        DynamicLoss='aux',
        PhysicsLoss='aux',
    ),
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
                loss_voxel_lovasz_weight=1.0),
            use_sem_geo_scal_loss=False,
            use_lovasz_loss=True,
            lovasz_ignore=17,
            manual_class_weight=[
                1.01552756, 1.06897009, 1.30013094, 1.07253735, 0.94637502, 1.10087012,
                1.26960524, 1.06258364, 1.189019,   1.06217292, 1.00595144, 0.85706115,
                1.03923299, 0.90867526, 0.8936431,  0.85486129, 0.8527829,  0.5       ]),
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
                loss_voxel_lovasz_weight=1.0),
            use_sem_geo_scal_loss=False,
            use_lovasz_loss=True,
            lovasz_ignore=17,
            manual_class_weight=[
                1.01552756, 1.06897009, 1.30013094, 1.07253735, 0.94637502, 1.10087012,
                1.26960524, 1.06258364, 1.189019,   1.06217292, 1.00595144, 0.85706115,
                1.03923299, 0.90867526, 0.8936431,  0.85486129, 0.8527829,  0.5       ]),
        dict(
            type='DetectionLoss',
            weight=1.0,
            head_order=['center', 'center_z', 'dim', 'rot', 'vel'],
            loss_weights={
                'cls_weight': 1.0,
                'loc_weight': 0.25,
                'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 1.0, 1.0]
            }),
        dict(
            # weight=0: 语义/深度仍渲染但不回传梯度（不参与训练），仅保留可视化/诊断。
            type='RenderLoss',
            weight=0.0,
            sem_lw=5.0,
            depth_lw=0.5,
            vis_dir='out/nuscenes_gs25600_gtbox_oracle_v3/render_vis',
            vis_every=500,
        ),
        dict(
            type='DynamicLoss',
            weight=1.0,
            pos_weight=8.0,
            extra_weight=0.5,
            use_gt_box=True,
            v_thresh=0.5,
            # 地面高度门控：仅强制"高于框底不足 z_margin(0.2m)"的贴地高斯为静态，
            # 车身高度(0.3~1.5m)全部保留 -> 剔除移动框底部的静态地面，不误伤动态。
            z_margin=0.2,
            vis_dir='out/nuscenes_gs25600_gtbox_oracle_v3/dynamic_vis',
            vis_every=500,
        ),
        dict(
            type='PhysicsLoss',
            weight=1.0,
            static_w=5.0,
            smooth_w=20.0,
            rigid_w=10.0,
            warmup_epoch=2,
            use_gt_box=True,
            v_thresh=0.5,
            z_margin=0.2,
        ),
    ])

loss_input_convertion = dict(
    # occ loss inputs
    pred_occ='pred_occ',
    sampled_xyz='sampled_xyz',
    sampled_label='sampled_label',
    occ_mask='occ_mask',
    # occ flow loss inputs
    occ_flow='occ_flow',
    # det loss inputs
    pred_dicts='pred_dicts',
    target_dicts='target_dicts',
    batch_index='batch_index',
    voxel_indices='voxel_indices',
    # render loss inputs
    rendered_sem='rendered_sem',
    rendered_depth='rendered_depth',
    pseudo_seg='pseudo_seg',
    pseudo_depth='pseudo_depth',
    input_imgs='input_imgs',
    aug_flip='aug_flip',
    # dynamic loss inputs
    rendered_dynamic='rendered_dynamic',
    pseudo_dyn='pseudo_dyn',
    # multi-frame dynamic loss inputs
    rendered_extra_dynamic='rendered_extra_dynamic',
    extra_pseudo_dyn='extra_pseudo_dyn',
    extra_dyn_valid='extra_dyn_valid',
    # physics loss inputs
    offset='offset',
    dynamic_logits='dynamic_logits',
    current_epoch='current_epoch',
    # gt-box oracle for physics loss (dynamic gating + rigid membership)
    gaussian='gaussian',
    gt_boxes='gt_boxes',
)

frozen_modules = ['map_decoder', 'planner_head']
# temporal_encoder builds 3 refine modules but only the last one's dynamic head
# is rendered/supervised → the other 2 dynamic heads are unused → need True.
find_unused_parameters = True
# dynamic render path reuses last refine module's dynamic head inside the
# with_cp (checkpoint) region → a param is marked ready twice under DDP.
# graph is identical every train iter (vis/diag are outside autograd) → static OK.
static_graph = True
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

# ========= dataset config =============
val_dataset_config = dict(
    imageset='data/nuscenes_cam/nuscenes_infos_val_gaussian_ad_v4.pkl',
    data_aug_conf=data_aug_conf,
    class_names=det_config['class_names'],
    pc_range=pc_range,
    num_frames=4,
    num_samples=2000,
    subsample_seed=42,
)
train_dataset_config = dict(
    imageset='data/nuscenes_cam/nuscenes_infos_train_gaussian_ad_v4.pkl',
    data_aug_conf=data_aug_conf,
    class_names=det_config['class_names'],
    pc_range=pc_range,
    num_frames=4,
    # pseudo-label 已关闭：GT-box oracle 模式下不再使用伪标签（2D 渲染也已关）。
    # metric3d_root/grounded_sam_root=None → dataset.use_pseudo_label=False，
    # 跳过 pseudo_seg/pseudo_depth/pseudo_dyn/gs_extrins/gs_intrins/extra-frames 全部加载。
    metric3d_root=None,
    grounded_sam_root=None,
    pseudo_label_scale=0.44,
    max_pseudo_depth=40.0,
    pseudo_label_crop_top=140,
    dynamic_gt_root=None,
    # lightweight: reproducible 3000-sample subset (vs full ~28k) for fast iteration
    num_samples=3000,
    subsample_seed=42,
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
        with_cp=True,
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
        ffn=dict(
            type="AsymmetricFFN",
            in_channels=embed_dims * 2,
            embed_dims=embed_dims,
            feedforward_channels=embed_dims * 4,
        ),
        deformable_model=dict(
            embed_dims=embed_dims,
            kps_generator=dict(
                embed_dims=embed_dims,
                phi_activation=phi_activation,
                xyz_coordinate=xyz_coordinate,
                num_learnable_pts=6,
                learnable_scale=5,
                pc_range=pc_range,
                scale_range=scale_range
            ),
        ),
        refine_layer=dict(
            type='SparseGaussian3DRefinementModule',
            embed_dims=embed_dims,
            pc_range=pc_range,
            scale_range=scale_range,
            restrict_xyz=True,
            unit_xyz=[2.0, 2.0, 0.5],
            refine_manual=[0, 1, 2],
            phi_activation=phi_activation,
            semantics=semantics,
            semantic_dim=semantic_dim,
            include_opa=include_opa,
            xyz_coordinate=xyz_coordinate,
            semantics_activation='softplus',
        ),
        spconv_layer=dict(
            type='SparseConv3DBlock',
            _delete_=True,
            in_channels=embed_dims,
            embed_channels=embed_dims,
            pc_range=pc_range,
            use_out_proj=True,
            grid_size=[0.5]*3,
            kernel_size=[5, 5, 5],
            stride=[1, 1, 1],
            padding=[2, 2, 2],
            dilation=[1, 1, 1],
        ),
        num_decoder=num_decoder,
        num_single_frame_decoder=num_single_frame_decoder,
        with_cp=False,
        operation_order=[
            "deformable",
            "ffn",
            "norm",
            "refine",
        ] * num_single_frame_decoder + [
            "spconv",
            "norm",
            "deformable",
            "ffn",
            "norm",
            "refine",
        ] * (num_decoder - num_single_frame_decoder),
    ),
    temporal_encoder=dict(
        type='GaussianTemporalEncoder',
        pc_range=pc_range,
        num_anchor=25600,
        num_encoder=3,
        anchor_encoder=dict(
            type='SparseGaussian3DEncoder',
            embed_dims=embed_dims,
            include_opa=include_opa,
            semantics=semantics,
            semantic_dim=semantic_dim
        ),
        norm_layer=dict(type="LN", normalized_shape=embed_dims),
        ffn=dict(
            type='FFN',
            embed_dims=embed_dims,
            feedforward_channels=embed_dims*2,
            num_fcs=2,
            act_cfg=dict(type='ReLU', inplace=True),
        ),
        refine_layer=dict(
            type='SparseGaussian3DRefinementModule',
            embed_dims=embed_dims,
            pc_range=pc_range,
            scale_range=scale_range,
            restrict_xyz=True,
            unit_xyz=[2.0, 2.0, 0.5],
            refine_manual=[0, 1, 2],
            phi_activation=phi_activation,
            semantics=semantics,
            semantic_dim=semantic_dim,
            include_opa=include_opa,
            xyz_coordinate=xyz_coordinate,
            semantics_activation='softplus',
            use_dynamic=True,
        ),
        spconv_layer=dict(
            type='SparseConv4D',
            in_channels=embed_dims,
            embed_channels=embed_dims,
            pc_range=pc_range,
            use_out_proj=True,
            grid_size=[0.5]*3,
            kernel_size=[
                [5, 5, 5, 5],
                [5, 5, 5, 5],
                [5, 5, 5, 5]
            ],
            stride=[
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1]
            ],
            padding=[
                [2] + [2] * 3,
                [2] + [2] * 3,
                [2] + [2] * 3,
            ],
            dilation=[
                [1] + [1] * 3,
                [1] + [1] * 3,
                [1] + [1] * 3,
            ],
            spatial_shape=[num_frames, 120, 120, 8],
        ),
        operation_order=[
            "spconv",
            "refine",
        ] * 3,
        with_cp=False,
    ),
    head=dict(
        type='GaussianHead',
        apply_loss_type='random_1',
        num_classes=semantic_dim + 1,
        empty_args=dict(
            _delete_=True,
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
        # 2D 渲染已关闭（GT-box oracle 模式下 DynamicLoss/PhysicsLoss 走 use_gt_box，
        # 不再需要 2D splatting 渲染；RenderLoss weight=0 且 None-safe）。
        # render_config=None → self.rasterizer_2d=None，跳过语义/深度渲染和 dynamic 渲染，省算力。
        render_config=None,
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
                            shared_conv_channel=128,
                            kernel_size_head=1,
                            predict_boxes_when_training=True,
                            use_bias_before_norm='true',
                            num_hm_conv=2,
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
            loss_seg=dict(type='SimpleLoss', pos_weight=4.0, loss_weight=1.0),
            loss_pv_seg=dict(type='SimpleLoss', pos_weight=1.0, loss_weight=2.0),),
        train_cfg=dict(pts=dict(
            grid_size=grid_size,
            voxel_size=voxel_size,
            point_cloud_range=pc_range,
            out_size_factor=4,
            assigner=dict(
                type='MapTRAssigner',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBoxL1Cost', weight=0.0, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=0.0),
                pts_cost=dict(type='OrderedPtsL1Cost', weight=5),
                pc_range=pc_range)))
    ),
    planner_head=dict(
        type="VADHead",
        embed_dims=embed_dims,
        fut_ts=6,
        fut_mode=6,
        ego_fut_mode=3,
        ego_agent_decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=dict(
                type='MyCustomBaseTransformerLayer',
                attn_cfgs=[dict(
                    type='MultiheadAttention',
                    embed_dims=embed_dims,
                    num_heads=8,
                    dropout=0.1)],
                feedforward_channels=embed_dims*2,
                ffn_dropout=0.1,
                batch_first=False,
                operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
        ego_map_decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=dict(
                type='MyCustomBaseTransformerLayer',
                attn_cfgs=[dict(
                    type='MultiheadAttention',
                    embed_dims=embed_dims,
                    num_heads=8,
                    dropout=0.1)],
                feedforward_channels=embed_dims*2,
                ffn_dropout=0.1,
                batch_first=False,
                operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
        ego_gaussian_decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=dict(
                type='MyCustomBaseTransformerLayer',
                attn_cfgs=[dict(
                    type='MultiheadAttention',
                    embed_dims=embed_dims,
                    num_heads=8,
                    dropout=0.1)],
                feedforward_channels=embed_dims*2,
                ffn_dropout=0.1,
                batch_first=False,
                operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
    ))
