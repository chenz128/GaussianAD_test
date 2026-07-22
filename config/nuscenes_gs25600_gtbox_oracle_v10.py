"""
nuscenes_gs25600_gtbox_oracle_v10 实验配置
v10 = v8（free offset 的权重基础）+ 运动条件化 + 有界 CTRA offset 头。
目标：当前帧不掉、未来帧提升、offset 幅值/方向/转弯都物理合理。

【相对 v8 的唯一自变量：offset 头的产生方式】
v8（free）：offset 头读 feat.detach() 直接输出 6×2=12 个自由数 → 幅值对但方向乱跳
            （总转弯角 254° vs 真值 8.4°）。
v10：
  1. 运动条件化 motion_cond=True：把每个高斯所属 GT 框的观测运动状态
     [vx, vy, heading] 拼接到 offset 头输入 → 头能区分“该转弯/直行/减速”。
  2. 有界 CTRA rollout：头只预测 [omega, accel]；omega = omega_max·tanh、
     accel = accel_max·tanh；初速度 v0 用观测速度 → 幅值/初始方向直接正确，
     omega 有界杜绝 v9 的打转（461°），accel 支持减速。
  3. 静态门控：观测速度 <motion_v_thresh 的高斯 offset 直接置 0 → 背景不飘
     （loss_static 变冗余，故 loss 权重沿用 v8 不再调整）。

【当前帧为何结构性不掉】
decouple_offset=True + offset_grad_scale=0.0（均从 v8 继承）：offset 头读
feat.detach()，新增的运动条件输入也只进 offset_layers，梯度永不回流 encoder。
当前帧占据走 means/semantics → LocalAggregator，与 offset 子图数学无关 → 当前帧 ≈ v8。

【推理】oracle 模式：测试同样用 GT 框的速度/朝向做条件（量上界）。落地时把 GT
换成检测头 vel/rot 预测值（另做，train/test gap 用 scheduled sampling 处理）。

【初始化】与 v8 相同，从 ImageNet/FCOS3D 预训练 backbone 从头训 15 epoch（继承 v8 的
load_from=pretrain，不从 v8 暖启——因为 offset_layers 形状变化，strict=False 无法跳过
同名不同形状的键，且从头训是与 v8 更公平的单变量对照）。虽从头训，但 v0=观测速度、
omega/accel 初始 tanh(0)=0，offset 从第 0 步就 ≈ 恒速外推（合理值），无冷启动塌缩。

其余（loss 权重 static_w=1/rigid_w=1/traj_w=4、GT-box 门控、3000 子集、冻结 map/plan、
decouple_offset=True/offset_grad_scale=0.0）与 v8 完全相同。
GPU：h20-new 后 4 张（4,5,6,7）或 h20-old 空闲卡。
"""

_base_ = ['./nuscenes_gs25600_gtbox_oracle_v8.py']

# 唯一自变量：temporal encoder 的 offset 头改为“运动条件化 + 有界 CTRA”。
# dict 深合并：decouple_offset=True / offset_grad_scale=0.0 从 v8 继承保留。
model = dict(
    temporal_encoder=dict(
        refine_layer=dict(
            offset_mode='kinematic',
            motion_cond=True,
            kin_dt=0.5,
            kin_omega_max=0.5,    # rad/s，3s 最多转 ~86°，覆盖真值 p90(38°)
            kin_accel_max=3.0,    # m/s²，支持刹车/加速
            motion_v_thresh=0.5,  # 观测速度 <0.5m/s 视为静止，offset 置 0
        ),
    ),
    # v10 的条件化 offset 头让每次迭代的计算图不再被 static_graph 认定为完全一致
    # （membership 依赖 means、条件 cat、CTRA rollout）→ DDP static_graph=True 会崩
    # （单卡不检查故冒烟能过，4 卡 iter1 空 spconv）。关掉 static_graph；随之必须把
    # backbone 的 with_cp 关掉，否则激活重计算会触发 'marked ready twice'。
    img_backbone=dict(with_cp=False),
)

static_graph = False
