"""
nuscenes_gs25600_acc 实验配置（在 base_ft_depth 基础上加 A1 + A4）

继承 nuscenes_gs25600_base_ft_depth.py 的全部设置，仅做两处增量改动：

  A4（已在代码层完成，config 无需改）：
      gaussian_rasterizer 渲染模式 RGB+D → RGB+ED（期望深度，alpha 归一化），
      给高斯提供更精确的 z-placement 梯度，深度边界更锐利。

  A1（本 config 启用）：
      RenderLoss 新增 accumulation loss —— 对 gsplat 渲染的累积不透明度图 A
      施加 hinge 软约束：
        - 前景（pseudo_depth>0.5）        → A 趋向 1（A<tau_fg 才罚）
        - 天空/空区（seg==0 且 depth==0）→ A 趋向 0（A>tau_sky 才罚）
        - 中间地带（有语义无深度，如 >40m 远处）→ 不约束
      目的：清掉散落在空中的低 alpha floater，逼前景表面"实心化"，
      不碰 scale、不直接监督被遮挡高斯（其 T_i≈0，梯度≈0）。
      日志中以 RenderAccLoss 单独记录。

其余（occ/flow/det loss、optimizer、dataset、model、frozen_modules、
load_from=base_epoch_5.pth、num_samples、多帧深度等）与 base_ft_depth 完全一致。
"""

_base_ = ['./nuscenes_gs25600_base_ft_depth.py']

# ========= loss config（覆写：RenderLoss 启用 A1 accumulation）=========
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
        dict(
            type='RenderLoss',
            weight=1.0,
            sem_lw=0.0,          # 不做语义监督（与 base_ft_depth 一致）
            depth_lw=0.5,        # 当前帧深度
            extra_depth_lw=0.1,  # 多帧（历史2+未来2）深度
            # ── A1 accumulation loss ──
            acc_lw=0.5,          # accumulation hinge 权重（>0 才启用 A1）
            acc_tau_fg=0.8,      # 前景：A<0.8 才罚，允许停在 0.8+
            acc_tau_sky=0.2,     # 天空：A>0.2 才罚，允许残留 0.2-
            vis_dir='out/nuscenes_gs25600_acc/render_vis',
            vis_every=500,
        ),
    ])

# ========= loss input 映射（覆写：新增 rendered_acc）=========
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
    rendered_acc='rendered_acc',     # A1 accumulation map
    pseudo_seg='pseudo_seg',
    pseudo_depth='pseudo_depth',
    input_imgs='input_imgs',
    aug_flip='aug_flip',
    # multi-frame depth loss inputs
    rendered_extra_depth='rendered_extra_depth',
    extra_pseudo_depth='extra_pseudo_depth',
    extra_valid='extra_valid',
)
