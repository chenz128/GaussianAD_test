"""
nuscenes_gs25600_gtbox_oracle_v12_futgu_satcol_loss
——— 在 futgau (detach=False) planner 基础上引入「基于分离轴定理 (SAT) 的碰撞规避损失」

【继承关系】
本 config 直接继承 canonical futgau_detach_false，因此：
  - 模型结构（planner_head=VADHeadFutGaussian + head.use_plan_ego 等两处 delta）
    全部沿用 futgau_detach_false；
  - load_from / lr / optimizer / max_epochs 等训练超参沿用 base_plan；
  - 其余 4 个 loss（Occupancy / OccupancyFlow / Detection / Map）保持不变。

【唯一 delta —— 给 PlanLoss 打开 SAT 碰撞损失】
PlanLoss 新增可选参数 col_sat（默认 False，不改动任何既有实验）：
  - col_loss_weight：碰撞损失内部权重（相对 L1/bound/dir）。
    注意 PlanLoss.forward 会再整体乘以外层 weight=10.0，
    因此「有效碰撞权重 = 10.0 * col_loss_weight」。此处 0.2 -> 有效 2.0。
  - col_safe_margin：SAT 重叠深度之外再加的安全间隙 (m)。
  - col_sat=True：使用 PlanAgentSATCollisionLoss（分离轴定理），
    否则回退到 PlanAgentCollisionLoss（外接圆）。

碰撞损失定义见 loss/plan_loss.py::PlanAgentSATCollisionLoss。

【与圆形近似的区别（为何用 SAT）】
圆形近似把 ego/agent 都当作各向同性圆（均半展 0.25*(w+l)），无法区分车身朝向，
对长车（如卡车）在高 yaw 偏差下会显著高估/低估碰撞。SAT 将 ego 视为轴对齐矩形
（yaw=0，与 plan_obj_box_col 指标一致），agent 视为带未来 yaw 的定向矩形，
在 4 条分离轴上（agent 两轴 + ego 两轴）计算穿透深度，梯度只沿真实穿透方向回传
ego 位置，直接对齐 plan_obj_box_col 语义。

【mmengine 列表整体替换】
mmengine 对 list 变量做整体替换而非合并，故这里用 _base_.loss.loss_cfgs 取到
继承来的完整 loss 列表，仅把最后一项 PlanLoss 换成带 SAT 碰撞参数的版本，其余四项
原样保留，避免重复誊写 OccupancyLoss/MapLoss 等超参。
"""

_base_ = [
    '../nuscenes_gs25600_gtbox_oracle_v12_ft_plan_futgau_detach_false/'
    'nuscenes_gs25600_gtbox_oracle_v12_ft_plan_futgau_detach_false.py'
]

# 复用继承来的前 4 个 loss，仅替换最后的 PlanLoss 为带 SAT 碰撞损失的版本
loss = dict(
    type='MultiLoss',
    loss_cfgs=list(_base_.loss.loss_cfgs[:-1]) + [
        dict(
            type='PlanLoss',
            weight=10.0,
            col_loss_weight=0.2,    # 有效碰撞权重 = 10.0 * 0.2 = 2.0
            col_safe_margin=0.5,
            col_sat=True,           # 使用分离轴定理 (SAT) 版本碰撞损失
        ),
    ],
)
