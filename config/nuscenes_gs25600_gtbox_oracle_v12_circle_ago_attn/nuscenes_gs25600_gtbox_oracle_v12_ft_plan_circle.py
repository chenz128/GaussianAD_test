"""
nuscenes_gs25600_gtbox_oracle_v12_ft_plan_circle —— 循环 (circle) 递进注意力 planner

【基础配置对齐 base_plan】
本 config 直接继承 nuscenes_gs25600_base_plan.py，因此 loss 权重与基础配置
与 base_plan 完全一致：
  - 5 个 loss：OccupancyLoss(1.0) / OccupancyFlowLoss(1.0) / DetectionLoss(1.0)
    / MapLoss / PlanLoss(10.0)；
  - lr=2e-4 + AdamW(wd=0.01, img_backbone lr_mult=0.1)，grad_max_norm=35；
  - max_epochs=15，frozen_modules=[]，其余模型结构均取自 base_plan。

【circle 的两处 delta（相对 base_plan）】
1) planner_head: VADHead -> VADHeadCircleGaussian
   把「ego↔agent -> ego↔map -> ego↔gaussian(当前) -> ego↔gaussian(未来)」这条
   4 路递进交叉注意力封装成一个 block，按同样的操作循环 num_circles 次
   （recurrent refinement）：每轮用上一轮融合后的 ego 重新再看一遍 4 路信息，
   逐轮迭代收敛，实现信息的充分融合，重点改善碰撞率 (obj_box_col)。
   最终用最后一轮的 4 路中间态拼接 (4D) 复用原 ego_fut_decoder 回归轨迹。
   详见 model/planner/planner_v5.py。
   - num_circles=3：循环次数 N（N=1 时退化为 planner_v3 单次融合，可调）。
   - share_circle_weights=False：每轮独立权重（stacked，容量更大）；
     置 True 则各轮共享权重（RNN 式，参数更少）。
   embed_dims/fut_ts/fut_mode/ego_fut_mode 及 ego_agent/map/gaussian 三路
   decoder 配置均从 base_plan 继承。
2) head: 打开 use_plan_ego + warmup=2，plan_ego_detach=False
   （occ_flow 未来帧 ego 补偿处梯度回传 planner），与 futgau_detach_false 对齐。

【load_from】
沿用 v12_fixempty epoch_15；circle 新增模块（fut_gaussian_fus_mlp /
ego_*_decoders 各 N 套 / 加宽后的 ego_fut_decoder）随机初始化 (strict=False)，
从 epoch 0 训 15 轮。
"""

_base_ = [
    '../nuscenes_gs25600_base_plan/nuscenes_gs25600_base_plan.py'
]

# base_plan 的 warm-start ckpt 本机不存在，改用已存在的 v12_fixempty epoch_15
load_from = 'exp/nuscenes_gs25600_v12_fixempty/checkpoints/epoch_15.pth'

# 循环次数 N（可调）：block「4 路递进注意力」重复次数
num_circles = 3

model = dict(
    # ---------- delta 1: planner_head VADHead -> VADHeadCircleGaussian ----------
    planner_head=dict(
        type='VADHeadCircleGaussian',
        num_circles=num_circles,
        share_circle_weights=False,
        # ego <-> 未来帧高斯 交叉注意力（第 4 路，循环内每轮复用其配置构建 N 套）
        ego_fut_gaussian_decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=dict(
                type='MyCustomBaseTransformerLayer',
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        embed_dims=128,
                        num_heads=8,
                        dropout=0.1),
                ],
                feedforward_channels=256,
                ffn_dropout=0.1,
                batch_first=False,
                operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
    ),
    # ---------- delta 2: 打开 plan_ego 机制（detach=False）----------
    head=dict(
        use_plan_ego=True,
        plan_ego_warmup_epochs=2,
        plan_ego_detach=False,
    ),
)
