"""
nuscenes_gs25600_pcgrad — 无 detach + PCGrad 梯度手术

目标：放开 scale/opacity 的 depth 梯度（detach_shape=False，回到 ft_depth 能涨的
工况，让高斯有"形状泄压阀"），同时用 PCGrad 在网络参数层面做正交投影手术：
    main = occ + flow + det  (受保护方，梯度一字不动)
    aux  = render (depth)     (让步方，与 occ 冲突的分量被投影删除)

数学保证：occ 在一阶上不被 depth 伤害。现实预期是 occ 守住 base_ep15(14.29) 上限，
对照 detach 实验（concentrate_new 9.42 / ft_ep15 摔到 8）验证保护机制是否生效。

与 ft_ep15_depth 的差异：
  1. detach_shape=True → False     （打开 scale/rot/opacity 的 depth 梯度）
  2. use_pcgrad=True               （启用梯度手术）
  3. loss 增加 group_map           （告诉 MultiLoss 谁是 main 谁是 aux）
  4. vis_dir 指向自己的 work-dir   （避免与 h20-old 上仍在跑的 ft_ep15 共享 NAS 冲突）

载入与训练时长沿用 ft_ep15_depth：load_from=base_epoch_15.pth, max_epochs=15。
"""

_base_ = ['./nuscenes_gs25600_ft_ep15_depth.py']

# ========= 打开 scale/rot/opacity 的 depth 梯度 =========
model = dict(
    head=dict(
        render_config=dict(
            detach_shape=False,
        ),
    ),
)

# ========= 启用 PCGrad 梯度手术 =========
use_pcgrad = True

# ========= loss：完整重声明，加 group_map + 改 vis_dir =========
loss = dict(
    type='MultiLoss',
    group_map=dict(
        OccupancyLoss='main',
        OccupancyFlowLoss='main',
        DetectionLoss='main',
        RenderLoss='aux',
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
            type='RenderLoss',
            weight=1.0,
            sem_lw=0.0,
            depth_lw=0.5,
            extra_depth_lw=0.1,
            concentration_lw=0.0,
            vis_dir='out/nuscenes_gs25600_pcgrad/render_vis',
            vis_every=500,
        ),
    ])
