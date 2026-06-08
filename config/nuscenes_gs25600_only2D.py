"""
nuscenes_gs25600_only2D 实验配置（纯 2D 弱监督 / C' 方案）
目标：验证「仅靠 Grounded SAM + Metric3D 伪标签的 2D 渲染监督」能否学出有意义的 3D 占用。

继承自 nuscenes_gs25600_2D.py，差异：
  - 移除 OccupancyLoss / OccupancyFlowLoss / DetectionLoss（去掉所有 3D GT 监督）
  - 只保留 RenderLoss（sem_lw=5.0, depth_lw=0.5）
  - 新增 GaussianRegLoss（scale_lw=0.05 + opacity_lw=0.05），软约束让渲染更锐利：
      * scale_reg:   L1 惩罚高斯尺寸 → 投影边界更清晰
      * opacity_reg: 不透明度熵惩罚 → opacity 向 0/1 收敛，去除半透明雾糊
  - scale_range 保持 [0.08, 0.64] 不变（第一次实验只上软正则，隔离变量）
  - loss_input_convertion 只留 render 相关键 + 新增 gaussian
其余配置（backbone/encoder/temporal/head/scale_range/load_from/max_epochs/
backbone_fp16/history_no_grad/数据 pipeline）全部继承自 2D 配置。

起点：load_from = FCOS3D 预训练 ResNet101-DCN（backbone 命中，其余随机），
最纯净反映纯 2D 监督能力。
"""

_base_ = ['./nuscenes_gs25600_2D.py']

# ========= loss config（替换整张 loss_cfgs 列表）=========
loss = dict(
    type='MultiLoss',
    loss_cfgs=[
        dict(
            type='RenderLoss',
            weight=1.0,
            sem_lw=5.0,
            depth_lw=0.5,
            vis_dir='out/nuscenes_gs25600_only2D/render_vis',
            vis_every=500,
        ),
        dict(
            type='GaussianRegLoss',
            weight=1.0,
            scale_lw=0.05,
            opacity_lw=0.05,
        ),
    ])

# ========= loss 输入映射（_delete_ 丢弃 occ/det 键，只留 render + gaussian）=========
loss_input_convertion = dict(
    _delete_=True,
    # render loss inputs
    rendered_sem='rendered_sem',
    rendered_depth='rendered_depth',
    pseudo_seg='pseudo_seg',
    pseudo_depth='pseudo_depth',
    input_imgs='input_imgs',
    aug_flip='aug_flip',
    # gaussian reg loss input
    gaussian='gaussian',
)

# ========= 多帧时序渲染监督 =========
# 把当前帧的高斯渲染到 [当前帧, t-1, t-2] 共 3 个时刻的相机视角，各自用该时刻伪标签监督。
# 相机随 ego 运动产生视差，从而对深度形成多视角三角化约束——这是打破单帧深度退化
# （高斯排成水平条带）的核心机制，对应 GaussianFlowOcc 的多帧渲染思路。
# 历史帧会自动屏蔽动态类（车/人等），只用静态背景保证几何一致性。
train_dataset_config = dict(
    render_num_frames=3,
)
