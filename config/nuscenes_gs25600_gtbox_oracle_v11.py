"""
nuscenes_gs25600_gtbox_oracle_v11 实验配置
v11c = v10（运动条件化 + 有界 CTRA offset 头）+ 运动交叉注意力（MotionCrossAttention）。

【相对 v10 的唯一自变量：offset 头的运动信息来源】
v10：offset 头读 feat.detach() 拼接 GT 观测运动状态 [vx, vy, heading]（3 个标量）。
     问题：feat 只为当前帧 occ 优化、且瞬时速度推不出未来曲率 → head 对"转弯"结构性盲，
     omega 塌成 0（实测转弯均值仅 0.6~1.7°，最大 ≤7.5°，几乎是直线）。
v11c：把那 3 个标量 concat 换成 MotionCrossAttention：
     当前帧高斯（Query）交叉注意力到历史帧高斯（Key/Value），几何关联（位置进 Q/K）。
     历史位置序列的曲率 = 真实转弯信号（omega 的来源）。v0 与静态门控仍取自 GT 观测速度。

【梯度策略】offset_grad_scale=s（从 v8 继承 0.0）：
     s=0 → Q/KV 的特征梯度都不回流 encoder → 当前帧 occ 可证明 == v10（干净验证转弯）。
     后续可把 s 逐档上调（0.1/0.3/…）观察当前帧 mIoU 是否随运动知识倒灌而提升。
     位置(means) 永远 detach（精确几何，不接受梯度）。

【为何几何关联而非按索引】lifter 把同一套 SPATIAL anchor 平铺到 4 帧；运动物体在不同帧
     占据不同 anchor 索引 → 按索引的时序注意力抓不到物体，必须用位置键的几何注意力。

【初始化】MotionCrossAttention 的输出投影 o_proj 零初始化 → 初始 attn=0 → motion_feat==feat；
     配合 offset_layers 末层零初始化（omega=accel=0）→ offset 从第 0 步就是 v0*t 恒速外推，
     无冷启动冲击。

【physics】只新增 turn_deg 诊断打印（量化 omega 是否复活）；不加转向监督 loss、不加曲率加权
     （loss_traj 的位置回归已隐式监督转弯，根因是输入信息不是监督强度）。

其余（GT-box oracle 动静门控、有界 CTRA、decouple_offset、decouple_dynamic、3000 子集、
冻结 map/plan、loss 权重）与 v10 完全相同。
GPU：h20-new 后 4 张（4,5,6,7）或 h20-old 空闲卡。
"""

_base_ = ['./nuscenes_gs25600_gtbox_oracle_v10.py']

# 唯一自变量：offset 头的运动信息来源改为"当前<-历史高斯交叉注意力"。
# dict 深合并：offset_mode='kinematic' / motion_cond=True / decouple_offset=True /
# offset_grad_scale=0.0 / decouple_dynamic=True 等全部从 v10/v8 继承保留。
model = dict(
    temporal_encoder=dict(
        refine_layer=dict(
            use_motion_attn=True,   # 启用 MotionCrossAttention
            motion_attn_heads=4,    # 256 / 4 = 64 head_dim
        ),
    ),
)

# 与 v10 相同：static_graph + backbone checkpoint 组合。MotionCrossAttention 每轮都
# 参与（训练恒有 4 帧历史），参与的参数集合每轮一致，static_graph 成立。
static_graph = True
