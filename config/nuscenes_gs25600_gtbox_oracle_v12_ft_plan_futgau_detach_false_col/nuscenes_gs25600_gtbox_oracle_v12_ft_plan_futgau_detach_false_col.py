"""
nuscenes_gs25600_gtbox_oracle_v12_ft_plan_futgau_detach_false_col
——— 在 futgau (detach=False) planner 基础上引入「碰撞规避损失」(P1)

【继承关系】
本 config 直接继承 futgau_detach_false，因此：
  - 模型结构（planner_head=VADHeadFutGaussian + head.use_plan_ego 等两处 delta）
    全部沿用 futgau_detach_false；
  - load_from / lr / optimizer / max_epochs 等训练超参沿用 base_plan；
  - 其余 4 个 loss（Occupancy / OccupancyFlow / Detection / Map）保持不变。

【唯一 delta —— 给 PlanLoss 打开碰撞损失】
PlanLoss 新增两个可选参数（默认 0，不改动任何既有实验）：
  - col_loss_weight：碰撞损失内部权重（相对 L1/bound/dir）。
    注意 PlanLoss.forward 会再整体乘以外层 weight=10.0，
    因此「有效碰撞权重 = 10.0 * col_loss_weight」。此处 0.2 -> 有效 2.0。
  - col_safe_margin：ego 与 agent 外接圆半径之和之外再加的安全间隙 (m)。

碰撞损失定义见 loss/plan_loss.py::PlanAgentCollisionLoss：对指令模态的 ego
预测轨迹按未来帧累积成 lidar 系绝对位置，与每个有效 GT agent 的未来足迹
（均以「均半展 0.25*(w+l)」的各向同性圆近似）做距离约束，仅在真实近碰撞时
产生梯度，直接对齐 plan_obj_box_col 指标；agent GT 为固定 target，梯度只回传
ego_fut_preds。

【mmengine 列表整体替换】
mmengine 对 list 变量做整体替换而非合并，故这里用 _base_.loss.loss_cfgs 取到
继承来的完整 loss 列表，仅把最后一项 PlanLoss 换成带碰撞参数的版本，其余四项
原样保留，避免重复誊写 OccupancyLoss/MapLoss 等超参。
"""

_base_ = [
    '../nuscenes_gs25600_gtbox_oracle_v12_ft_plan_futgau_detach_false/'
    'nuscenes_gs25600_gtbox_oracle_v12_ft_plan_futgau_detach_false.py'
]

# 复用继承来的前 4 个 loss，仅替换最后的 PlanLoss 为带碰撞损失的版本
loss = dict(
    type='MultiLoss',
    loss_cfgs=list(_base_.loss.loss_cfgs[:-1]) + [
        dict(
            type='PlanLoss',
            weight=10.0,
            col_loss_weight=0.2,    # 有效碰撞权重 = 10.0 * 0.2 = 2.0
            col_safe_margin=0.5,
        ),
    ],
)
