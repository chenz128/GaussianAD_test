"""
nuscenes_gs25600_concentrate_new — concentrate + detach_shape

在 concentrate 的基础上增加一项改动：
    detach_shape=True → 2D 渲染路径中 scales/rotations/opacities 被 detach，
    梯度不再流入这三个变量。它们只由 3D OccLoss 优化。

变量分工：
    means      ← depth loss (E[z]) + ConcLoss (Var[z]) + OccLoss
    scales     ← OccLoss only (detached from 2D)
    rotations  ← OccLoss only (detached from 2D)
    opacities  ← OccLoss only (detached from 2D)
    semantics  ← OccLoss only (sem_lw=0, no 2D semantic loss)

对照关系：
    base(14.29) → ft_depth(12.16) → concentrate(11.86) → concentrate_new(?)
    唯一新变量 = detach_shape（消除 2D depth 对 scale/rot/opa 的压薄/压低副作用）
"""

_base_ = ['./nuscenes_gs25600_concentrate.py']

# ========= model: render_config 加 detach_shape =========
model = dict(
    head=dict(
        render_config=dict(
            render_h=256,
            render_w=704,
            sem_lw=5.0,
            depth_lw=0.5,
            detach_shape=True,
        ),
    ),
)

# ========= work_dir / vis_dir =========
# 覆写 RenderLoss 的 vis_dir（loss_cfgs 是 list，需要整体覆写 RenderLoss 条目）
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
            concentration_lw=0.2,
            vis_dir='out/nuscenes_gs25600_concentrate_new/render_vis',
            vis_every=500,
        ),
    ])
