"""
nuscenes_gs25600_concentrate 实验配置（在 base_ft_depth 基础上加 ① 深度集中度损失）

继承 nuscenes_gs25600_base_ft_depth.py 的全部设置，仅做两处增量改动：

  代码层（已在 gaussian_rasterizer.py 完成，config 无需改）：
      A4 回退：渲染模式 RGB+ED → RGB+D（累积深度，非 alpha 归一化）。
      恢复深度损失对 opacity 的隐性地板（opa 低 → D 小 → 深度损失把 opa 顶上去），
      防止 acc 实验中出现的 opacity 坍塌（0.289 → 0.110）。

  ①（本 config 启用）：
      RenderLoss 新增 concentration loss —— 对 gsplat 渲染的每条前景光线
      （pseudo_depth>0.5）的深度方差 Var[z] = E[z²] - E[z]² 施加惩罚：
        逼迫沿光线的高斯坍缩到同一深度（锐利表面），消除"摊大饼"式的
        深度涂抹。期望深度 E[z] 监督无法区分"锐利表面"与"均值恰好对齐的雾团"，
        Var[z] 能。
      与 A1 的本质区别：A1 约束光线积分量 A，存在"堆叠半透明高斯"漏洞；
      ① 约束深度分布形态，唯一降低 Var 的方式是物理收紧高斯位置，无漏洞，
      且不直接监督 opacity（不会重蹈 A1 的 opacity 坍塌）。
      日志中以 RenderConcLoss 单独记录，诊断行打印 var_fg / std_fg。

其余（occ/flow/det loss、optimizer、dataset、model、frozen_modules、
load_from=base_epoch_5.pth、num_samples、多帧深度等）与 base_ft_depth 完全一致，
保证与 ft_depth(12.2) 的对照是干净的单变量消融（唯一差异 = ① + A4 回退）。
"""

_base_ = ['./nuscenes_gs25600_base_ft_depth.py']

# ========= loss config（覆写：RenderLoss 启用 ① concentration）=========
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
            sem_lw=0.0,            # 不做语义监督（与 base_ft_depth 一致）
            depth_lw=0.5,          # 当前帧深度（RGB+D 累积深度）
            extra_depth_lw=0.1,    # 多帧（历史2+未来2）深度
            # ── ① depth concentration loss ──
            concentration_lw=0.02, # 深度方差惩罚权重（>0 才启用 ①），首跑保守值；
                                   # 看诊断行 var_fg/std_fg/loss_conc 后再调
            vis_dir='out/nuscenes_gs25600_concentrate/render_vis',
            vis_every=500,
        ),
    ])

# ========= loss input 映射（覆写：新增 rendered_var）=========
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
    rendered_var='rendered_var',     # ① depth variance map
    pseudo_seg='pseudo_seg',
    pseudo_depth='pseudo_depth',
    input_imgs='input_imgs',
    aug_flip='aug_flip',
    # multi-frame depth loss inputs
    rendered_extra_depth='rendered_extra_depth',
    extra_pseudo_depth='extra_pseudo_depth',
    extra_valid='extra_valid',
)
