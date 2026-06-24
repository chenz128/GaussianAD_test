"""
nuscenes_gs25600_ft_ep15_depth — 从成熟 base(ep15, 14.29) 出发的 detach+depth fine-tune

与 concentrate_new 完全相同的监督方式（detach_shape=True + depth_lw=0.5 +
extra_depth_lw=0.1 + concentration_lw=0，即纯 detach + 2D 深度监督），
仅两处不同：

  1. load_from = ckpts/base_epoch_15.pth  （base 训到 ep15=14.29 的成熟权重）
       —— concentrate_new 从 base_ep5(~10, 未成熟) 出发，2D 监督与未收敛模型纠缠 → 效果差
       —— 本实验从成熟几何出发，detach 保护 shape，depth 仅微调 means 位置
  2. max_epochs = 15  （load_from 只载权重不载 epoch 计数，从 0 起算 → 训 15 个新 epoch）

核心问题：从已收敛的 base(14.29) 出发，加 detach+depth 监督，能否突破 base 自己的上限(ep30=15.55)？
  能 → 2D 深度监督提供了 base 没有的正面信息
  不能 → 2D 深度监督对 occ 整体中性/有害
"""

_base_ = ['./nuscenes_gs25600_concentrate_new.py']

# ========= 从成熟 base(ep15) 出发，训 15 个新 epoch =========
load_from = 'ckpts/base_epoch_15.pth'
max_epochs = 15

# ========= loss：与 concentrate_new 一致，仅改 vis_dir =========
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
            sem_lw=0.0,
            depth_lw=0.5,
            extra_depth_lw=0.1,
            concentration_lw=0.0,
            vis_dir='out/nuscenes_gs25600_ft_ep15_depth/render_vis',
            vis_every=500,
        ),
    ])
