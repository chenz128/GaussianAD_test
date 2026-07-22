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

【暖启】offset_layers 形状变化（输入 128→131、输出 12→2），load_from 以 strict=False
加载 v8 epoch_15：encoder/occ/det 全部暖启（当前帧从第 0 步即 ≈v8），仅 offset_layers
重新 init 训练。且因 v0=观测速度、omega/accel 初始 tanh(0)=0，offset 从第 0 步就 ≈
恒速外推（合理值），无冷启动塌缩。

其余（loss 权重 static_w=1/rigid_w=1/traj_w=4、GT-box 门控、3000 子集、冻结 map/plan、
decouple_offset=True/offset_grad_scale=0.0）与 v8 完全相同。
GPU：h20-new 后 4 张（4,5,6,7）或 h20-old 空闲卡。
"""

_base_ = ['./nuscenes_gs25600_gtbox_oracle_v8.py']

# 从 v8 最优 checkpoint 暖启（当前帧直接 ≈12.8；offset 头因形状变化 strict=False 重 init）
load_from = 'out/nuscenes_gs25600_gtbox_oracle_v8/checkpoints/epoch_15.pth'

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
)
