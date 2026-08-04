"""
nuscenes_gs25600_frontier_v1 -- 未来帧 frontier 高斯补全（S1）。

【问题】GaussianHead.forward_flow 对每个未来帧只做裁剪不做补充：出界的高斯被
torch.nonzero 索引丢弃，数量单调递减（t+3s 约剩 58%），而 ego 前进后新进入
occ 窗口的那条带子一个高斯都没有 —— 渲染出来必然是空的。但 flow_info[i] 的
GT 来自 t+i 时刻自己的 occ 标注（collect_flow_sweeps 取 keyframes[idx+i]），
那片区域是有真值的，只是没有可承接梯度的载体。

【本实验】把"裁剪"换成"slot 复用"：出界的 slot 不删除，而是重新放置到新进入
的条带内，属性由 FrontierGenerator 生成，直接受现有 OccupancyFlowLoss 监督。
每个未来帧的真实高斯数恒为 num_anchor，张量定长，DDP 安全。

【唯一自变量】head.type: GaussianHead -> GaussianHeadFrontier
其余（flow_grad_scale=0.0、GT-box oracle offset 头、loss 权重、3000 子集、
冻结 map/plan）与 v12 完全一致，构成干净的单变量对照。

【第一版范围】单候选、单步、无 V-JEPA、无 WTA。先验证"补齐高斯数量本身"值
多少分，作为后续条件生成的 baseline。

【预期】
  - 未来帧 mIoU：t+2.0s 之后的帧应有可见提升（frontier 占比最大）
  - current mIoU：flow_grad_scale=0.0 已切断未来->当前的梯度，应保持不变
  - 若 FutAvg 不涨，说明 prior 撒点质量不足，需要先做条件化而非加步数
"""

_base_ = ['../nuscenes_gs25600_gtbox_oracle_v12.py']

model = dict(
    head=dict(
        type='GaussianHeadFrontier',
        frontier_generator=dict(
            hidden_dims=256,
            scale_range=[0.08, 0.64],
            max_position_delta=1.0,
            min_band=0.5,
            init_scale=0.2,
            init_opacity=0.1,
        ),
    ),
)

# FrontierGenerator 的参数每个 iter 的 6 个未来帧都会用到，参与集合固定。
static_graph = True
