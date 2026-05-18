_base_ = [
    './_base_/misc.py',
    './_base_/model.py',
    './_base_/surroundocc.py'
]
import os

# =========== data config ==============
input_shape = (1600, 864)
data_aug_conf = {
    "resize_lim": (1.0, 1.0),#这是一个元组，定义了输入图像在数据增强过程中可能的缩放范围。resize_lim=(1.0, 1.0)表示输入图像将保持原始尺寸，不进行缩放。如果想要在训练过程中对输入图像进行随机缩放，可以将这个范围设置为一个大于1.0的值，例如resize_lim=(0.8, 1.2)，这样输入图像就会被随机缩放到原始尺寸的80%到120%之间。
    "final_dim": input_shape[::-1],
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
}
num_frames = 4   # TODO: dataset 改为4帧时序输入
num_map_classes = len(_base_.map_classes)
pc_range = [-30.0, -30.0, -2.0, 30.0, 30.0, 2.0]
fixed_ptsnum_per_gt_line = 20 # now only support fixed_pts > 0这
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
# ========= model config ===============
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
        # [SPLATTING] MapLoss re-enabled with gradient checkpointing
        dict(
            type='MapLoss',
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=2.0),
            loss_bbox=dict(type='L1Loss', loss_weight=0.0),
            loss_iou=dict(type='GIoULoss', loss_weight=0.0),
            loss_pts=dict(type='PtsL1Loss',
                        loss_weight=5.0),
            loss_dir=dict(type='PtsDirCosLoss', loss_weight=0.005),
            loss_seg=dict(type='SimpleLoss',
                pos_weight=4.0,
                loss_weight=1.0),
            loss_pv_seg=dict(type='SimpleLoss',
                        pos_weight=1.0,
                        loss_weight=2.0),
            assigner=dict(
                    type='MapTRAssigner',
                    cls_cost=dict(type='FocalLossCost', weight=2.0),
                    reg_cost=dict(type='BBoxL1Cost', weight=0.0, box_format='xywh'),
                    iou_cost=dict(type='IoUCost', iou_mode='giou', weight=0.0),
                    pts_cost=dict(type='OrderedPtsL1Cost', weight=5),
                    pc_range=pc_range),
            sync_cls_avg_factor=True,
            num_classes=num_map_classes,
            gt_shift_pts_pattern='v2',
            pc_range=pc_range,
            code_weights=[1.0, 1.0, 1.0, 1.0],
            aux_seg=_base_.aux_seg_cfg,
            num_pts_per_vec=fixed_ptsnum_per_pred_line,
            num_pts_per_gt_vec=fixed_ptsnum_per_gt_line,
            dir_interval=1,
            ),
        # RenderLoss: pseudo-label supervision via 2D Gaussian splatting
        dict(
            type='RenderLoss',
            weight=1.0,
            sem_lw=2.0,#这是语义损失的权重，控制语义损失在总损失中的重要性。较大的sem_lw会使模型更关注语义分割的准确性，而较小的sem_lw则会降低语义损失的影响力。
            depth_lw=0.05,#这是深度损失的权重，控制深度损失在总损失中的重要性。较大的depth_lw会使模型更关注深度预测的准确性，而较小的depth_lw则会降低深度损失的影响力。由于深度损失通常比语义损失更容易产生较大的数值，因此这里设置了一个较小的权重来平衡两者的影响。
            vis_dir='out/nuscenes_gs25600_splatting/render_vis',  # 渲染可视化输出目录
            vis_every=250,  # 每隔多少次 iter 保存一次可视化图片
            ),
        # PlanLoss re-enabled
        dict(
            type='PlanLoss',
            weight=10.0,
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
    # plan inputs
    ego_fut_preds='ego_fut_preds',
    ego_fut_gt='ego_fut_trajs',
    ego_fut_masks='ego_fut_masks',
    ego_fut_cmd='ego_fut_cmd',
    # map inputs
    all_cls_scores="all_cls_scores",
    all_bbox_preds="all_bbox_preds",
    all_pts_preds="all_pts_preds",
    # render loss inputs (from head output)
    rendered_sem='rendered_sem',
    rendered_depth='rendered_depth',
    # render loss inputs (from metas/data)
    pseudo_seg='pseudo_seg',
    pseudo_depth='pseudo_depth',
)#这是一个字典，定义了不同损失函数所需的输入数据在模型输出或数据加载过程中对应的键名。通过这个字典，模型在计算损失时可以根据键名从输入数据中提取相应的张量。例如，RenderLoss需要的输入包括'rendered_sem'、'rendered_depth'、'pseudo_seg'和'pseudo_depth'，这些键名会被映射到实际的数据张量上，以便在计算损失时使用。
# All modules trainable (map + plan + render all enabled)
frozen_modules = []
find_unused_parameters = False  # with_cp=True conflicts with find_unused_parameters=True in DDP; frozen modules don't need it
backbone_fp16 = True  # selective AMP: only backbone+neck run in fp16, rest stays fp32

# ========= model config ===============
embed_dims = 128#这是高斯编码器和解码器中使用的特征维度。较大的embed_dims可以提供更丰富的特征表示能力，但也会增加模型的计算复杂度和内存占用。根据实际需求和资源限制，可以调整这个值来平衡性能和效率。
num_decoder = 4
num_single_frame_decoder = 1
grid_size=[120, 120, 8]
scale_range = [0.08, 0.64]
xyz_coordinate = 'cartesian'#笛卡尔坐标系
phi_activation = 'sigmoid'#激活函数使用sigmoid，这意味着高斯点的特征会被压缩到0和1之间，适合表示概率或权重等信息。
include_opa = True#学习透明度信息
load_from = 'ckpts/r101_dcn_fcos3d_pretrain.pth'
semantics = True#学习语义信息
semantic_dim = 17#这是语义特征的维度，通常对应于数据集中不同类别的数量。在nuScenes数据集中，semantic_dim=17表示有17个不同的语义类别（不包括背景或无效类别）。这个参数用于定义高斯点云中每个点的语义特征维度，以便模型能够学习和区分不同的语义类别。

offset = True
offset_dim = 2*6

voxel_size=[0.5, 0.5, 0.5]#体素的分辨率，表示每个体素在x、y、z三个维度上的实际尺寸。较小的voxel_size可以提供更高的空间分辨率，但也会增加计算复杂度和内存占用。根据实际需求和资源限制，可以调整这个值来平衡性能和效率。
det_config = dict(
    class_names=['car','truck', 'construction_vehicle', 'bus', 'trailer',
              'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'],
    num_point_features=28, #TODO,
    grid_size=grid_size,
    voxel_size=voxel_size,
    point_cloud_range=pc_range,
    depth_downsample_factor=None,
)
_dim_ = 256#这是高斯点云中每个点的特征维度，通常用于定义高斯点云编码器和解码器中的特征表示能力。较大的_dim_可以提供更丰富的特征表示，但也会增加计算复杂度和内存占用。
_pos_dim_ = _dim_//2#这是位置特征的维度，通常是_dim_的一半，用于表示高斯点云中每个点的位置特征。

val_dataset_config = dict(
    imageset='data/nuscenes_cam/nuscenes_infos_val_gaussian_ad_v6.pkl',
    data_aug_conf=data_aug_conf,
    class_names=det_config['class_names'],
    pc_range=pc_range,
    num_frames=4
)
train_dataset_config = dict(
    imageset='data/nuscenes_cam/nuscenes_infos_train_gaussian_ad_v6.pkl',
    data_aug_conf=data_aug_conf,
    class_names=det_config['class_names'],
    pc_range=pc_range,
    num_frames=4,
    # pseudo label configs (splatting branch)
    metric3d_root='/data/chenz/Gaussianflowocc_test/data/metric_3d_nusc',
    grounded_sam_root='/data/chenz/Gaussianflowocc_test/data/grounded_sam_nusc',
    pseudo_label_scale=0.44,#这是一个缩放因子，用于调整根据点云生成的伪标签的尺度。由于点云数据和图像数据之间存在一定的尺度差异，直接使用点云生成的伪标签可能会导致与图像上的实际物体位置不匹配。通过设置pseudo_label_scale，可以对伪标签进行适当的缩放，使其更好地对齐图像上的物体，从而提高渲染损失的监督效果。具体的缩放因子需要根据数据集和模型的特点进行调整，以获得最佳性能。
    max_pseudo_depth=40.0,#这是一个阈值，用于过滤根据点云生成的伪标签中的深度值。由于点云数据中可能存在一些异常值或远距离的点，这些点在图像上可能对应于无效或不相关的区域。通过设置max_pseudo_depth，可以将伪标签中深度值超过该阈值的像素视为无效，从而在计算渲染损失时忽略这些像素。这有助于提高模型的训练稳定性和性能，特别是在处理具有较大深度范围的数据集时。
    pseudo_label_crop_top=140,#这是一个参数，用于在计算渲染损失时裁剪伪标签的顶部区域。由于图像的顶部区域通常包含天空或其他不相关的背景信息，这些区域的伪标签可能对训练没有帮助，甚至可能引入噪声。通过设置pseudo_label_crop_top，可以在计算渲染损失时忽略图像顶部的指定像素行，从而提高模型的训练效果。
)

model = dict(
    img_backbone_out_indices=[0, 1, 2, 3],
    # Run backbone + encoder for historical frames under torch.no_grad() to
    # save activation memory and backward time. Only the current (last) frame
    # contributes gradients. Temporal encoder still sees all frames.
    history_no_grad=True,
    img_backbone=dict(
        _delete_=True,
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,#在训练过程中，冻结BatchNorm层的统计信息，即不更新其均值和方差。这通常在使用预训练模型时进行，以保持预训练权重的稳定性。
        style='caffe',
        with_cp = True, # 这是一个布尔值，表示是否在ResNet的卷积层中使用checkpointing技术来节省内存。启用with_cp=True会在前向传播过程中保存一些中间激活值，并在反向传播时重新计算它们，以减少内存占用。这对于训练大型模型或使用较大批量大小时非常有用，但会增加一些计算开销。根据实际情况，可以选择是否启用这个选项来平衡内存使用和计算效率。
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False), # original DCNv2 will print log when perform load_state_dict
        stage_with_dcn=(False, False, True, True)),#这是一个元组，表示ResNet的每个阶段是否使用可变形卷积（Deformable Convolution）。在这个配置中，前两个阶段（stage 1和stage 2）不使用可变形卷积，而后两个阶段（stage 3和stage 4）使用可变形卷积。使用可变形卷积可以增强模型对几何变形的适应能力，从而提高特征提取的效果。根据实际需求，可以调整这个元组来选择在哪些阶段使用可变形卷积。
    img_neck=dict(
        start_level=1),#这是一个参数，表示特征金字塔网络（FPN）从ResNet的哪个阶段开始构建特征金字塔。在这个配置中，start_level=1表示FPN将从ResNet的第二个阶段（stage 2）的输出特征图开始构建特征金字塔，而忽略第一个阶段（stage 1）的输出特征图。根据实际需求，可以调整这个参数来选择从哪个阶段开始使用ResNet的特征图进行后续处理。
    lifter=dict(
        type='GaussianLifter',
        num_anchor=25600,
        embed_dims=embed_dims,
        anchor_grad=True,
        semantic_dim=semantic_dim,
        include_opa=include_opa,
        offset=offset,
        offset_dim=offset_dim,
    ),#这是一个字典，定义了高斯点云编码器（GaussianLifter）的配置参数。num_anchor=25600表示使用25600个锚点来表示场景中的物体和结构；embed_dims=128表示每个锚点的特征维度为128；anchor_grad=True表示在训练过程中对锚点进行梯度更新；semantic_dim=17表示语义特征的维度为17，对应于数据集中不同类别的数量；include_opa=True表示在编码器中包含透明度信息；offset=True表示使用偏移量来增强锚点的位置表达能力；offset_dim=12表示偏移量特征的维度为12。这些参数共同定义了高斯点云编码器的结构和功能，以便模型能够有效地从输入图像中提取空间和语义信息。
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
        ),
        spconv_layer = dict(
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
        apply_loss_type='random_1',#这是一个字符串参数，表示在训练过程中应用损失函数的方式。'random_1'表示在每个训练步骤中随机选择一个损失函数进行优化，而不是同时优化所有损失函数。这种方式可以帮助模型更好地平衡不同损失函数的影响，避免某个损失函数过度主导训练过程。根据实际需求，可以选择不同的应用方式，例如'sequential'（按顺序应用损失函数）或'all'（同时应用所有损失函数）。
        num_classes=semantic_dim + 1,
        empty_args=dict(
            _delete_=True,
            mean=[0, 0, -1.0],
            scale=[60, 60, 4.0],
        ),#这是一个字典，定义了高斯点云头部（GaussianHead）中用于表示空白区域的参数。mean=[0, 0, -1.0]表示空白区域的高斯分布的均值位置，通常设置在视图中心下方以覆盖地面区域；scale=[60, 60, 4.0]表示空白区域的高斯分布的尺度，较大的值可以使空白区域覆盖更广泛的范围，以确保模型能够正确地识别和处理空白区域。这些参数对于训练模型区分有物体存在的区域和没有物体的空白区域非常重要。
        with_empty=True,
        cuda_kwargs=dict(
            _delete_=True,
            scale_multiplier=5,
            H=120, W=120, D=8,
            pc_min=[-30.0, -30.0, -2.0],
            grid_size=0.5),
        render_config=dict(
            render_h=256,   # int(900 * 0.44) - 140 = ~256
            render_w=704,   # int(1600 * 0.44) = 704
            sem_lw=2.0,
            depth_lw=0.05,
        ),
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
                max_voxels=(25600, 25600), #TODO
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
            num_pts_per_vec=fixed_ptsnum_per_pred_line, # one bbox
            num_pts_per_gt_vec=fixed_ptsnum_per_gt_line,
            dir_interval=1,
            # query_embed_type='instance_pts',
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
            # z_cfg=z_cfg,
            num_map_adapter_conv=2,
            map_adapter_input_channels=128,
            map_adapter_kernel_size=1,
            map_adapter_use_bias=True,
            bbox_coder=dict(
                type='MapTRNMSFreeCoder',
                # post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
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
            loss_pts=dict(type='PtsL1Loss',
                        loss_weight=5.0),
            loss_dir=dict(type='PtsDirCosLoss', loss_weight=0.005),
            loss_seg=dict(type='SimpleLoss',
                pos_weight=4.0,
                loss_weight=1.0),
            loss_pv_seg=dict(type='SimpleLoss',
                        pos_weight=1.0,
                        loss_weight=2.0),),
        # model training and testing settings
        train_cfg=dict(pts=dict(
            grid_size=grid_size,
            voxel_size=voxel_size,
            point_cloud_range=pc_range,
            out_size_factor=4,
            assigner=dict(
                type='MapTRAssigner',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBoxL1Cost', weight=0.0, box_format='xywh'),
                # reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                # iou_cost=dict(type='IoUCost', weight=1.0), # Fake cost. This is just to make it compatible with DETR head.
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=0.0),
                pts_cost=dict(type='OrderedPtsL1Cost',
                        weight=5),
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
        ego_map_decoder=dict(
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
        ego_gaussian_decoder=dict(
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
