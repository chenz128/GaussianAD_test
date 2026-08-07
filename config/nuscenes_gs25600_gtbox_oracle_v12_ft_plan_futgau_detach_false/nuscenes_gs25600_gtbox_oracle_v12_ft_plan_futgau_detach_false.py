"""
nuscenes_gs25600_gtbox_oracle_v12_ft_plan_futgau —— 未来帧高斯融合 planner (detach=False)

【基础配置对齐 base_plan】
本 config 直接继承 nuscenes_gs25600_base_plan.py，因此 loss 权重与基础配置
与 base_plan 完全一致：
  - 5 个 loss：OccupancyLoss(1.0) / OccupancyFlowLoss(1.0) / DetectionLoss(1.0)
    / MapLoss / PlanLoss(10.0)（无 Render/Dynamic/Physics）；
  - lr=2e-4 + AdamW(wd=0.01, img_backbone lr_mult=0.1)，grad_max_norm=35；
  - max_epochs=15，frozen_modules=[]，模型结构（lifter/encoder/temporal/decoder/
    map_decoder/planner_head 各子模块）均取自 base_plan。

【futgau 的两处 delta（相对 base_plan）】
1) planner_head: VADHead -> VADHeadFutGaussian
   让「未来帧高斯」作为与 agent/map/当前帧高斯完全对称的第 4 路 stream 融合进
   planner（ego↔未来帧高斯 交叉注意力），拼接 4D 后复用原 ego_fut_decoder 回归
   轨迹（增广输出头，非替换，无信息瓶颈）。详见 model/planner/planner_v3.py。
   embed_dims/fut_ts/fut_mode/ego_fut_mode 及原 3 路 decoder 均从 base_plan 继承。
2) head: 打开 use_plan_ego（base_plan 默认关闭）+ warmup=2，以便进行
   plan_ego_detach 对照实验。本 config = detach False（occ_flow 一致性梯度
   回传 planner）；对照组见 ..._futgau_detach（detach True）。

【load_from】
base_plan 原 load_from=out/nuscenes_gs25600_base/checkpoints/epoch_15.pth 在本机
不存在，故覆盖为已存在的 v12_fixempty epoch_15；futgau 新增模块
(fut_gaussian_fus_mlp / ego_fut_gaussian_decoder / 加宽后的 ego_fut_decoder)
随机初始化 (strict=False)，从 epoch 0 训 15 轮。
"""

_base_ = [
    '../nuscenes_gs25600_base_plan/nuscenes_gs25600_base_plan.py'
]

# base_plan 的 warm-start ckpt 本机不存在，改用已存在的 v12_fixempty epoch_15
load_from = 'exp/nuscenes_gs25600_v12_fixempty/checkpoints/epoch_15.pth'

model = dict(
    # ---------- delta 1: planner_head VADHead -> VADHeadFutGaussian ----------
    planner_head=dict(
        type='VADHeadFutGaussian',
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
    # ---------- delta 2: 打开 plan_ego 机制（detach 对照实验用）----------
    head=dict(
        use_plan_ego=True,
        plan_ego_warmup_epochs=2,
        plan_ego_detach=False,   # False: occ_flow 一致性梯度回传 planner
    ),
)
