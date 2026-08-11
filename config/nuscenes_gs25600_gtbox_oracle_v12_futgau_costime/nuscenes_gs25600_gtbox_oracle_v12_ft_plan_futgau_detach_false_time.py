"""
nuscenes_gs25600_gtbox_oracle_v12_ft_plan_futgau_detach_false_time ——
未来帧高斯融合 planner，并在「未来帧高斯 key/value」加入逐时间步位置编码。

【相对 futgau_detach_false 的唯一 delta】
planner_head: VADHeadFutGaussian -> VADHeadFutGaussianTime (model/planner/planner_v4.py)
  - 其余（当前帧 3 路、第 4 路 decoder、4D 输出头、4 loss 权重、plan_ego 机制）
    与 detach_false 完全一致。
  - 新增：fut_time_pos = nn.Embedding(fut_ts=6, embed_dims=128)，加到未来帧高斯
    特征 (fut_ts, G, D) 的逐时间步维度，使每个 key 明确携带「属于未来第几帧」。
  该新参数随机初始化 (strict=False)，从 epoch 0 训 15 轮。

【与对照实验】
  - futgau_detach_false   : 未来帧高斯无时间编码（baseline）
  - futgau_detach_false_time : 未来帧高斯 + 逐时间步位置编码（本实验）
"""

_base_ = [
    '../nuscenes_gs25600_base_plan/nuscenes_gs25600_base_plan.py'
]

# base_plan 的 warm-start ckpt 本机不存在，改用已存在的 v12_fixempty epoch_15
load_from = 'exp/nuscenes_gs25600_v12_fixempty/checkpoints/epoch_15.pth'

model = dict(
    # ---------- delta: planner_head VADHeadFutGaussian -> VADHeadFutGaussianTime ----------
    planner_head=dict(
        type='VADHeadFutGaussianTime',
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
    # ---------- 与 futgau_detach_false 一致：打开 plan_ego 机制 ----------
    head=dict(
        use_plan_ego=True,
        plan_ego_warmup_epochs=2,
        plan_ego_detach=False,
    ),
)
