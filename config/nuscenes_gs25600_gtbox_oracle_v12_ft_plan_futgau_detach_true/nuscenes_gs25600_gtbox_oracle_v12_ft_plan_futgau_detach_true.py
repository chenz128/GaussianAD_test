"""
nuscenes_gs25600_gtbox_oracle_v12_ft_plan_futgau_detach —— 未来帧高斯融合 planner (detach=True)

与 ..._futgau 完全相同（同样继承 base_plan、同样 VADHeadFutGaussian 第 4 路未来帧
高斯融合、同样打开 use_plan_ego+warmup=2），唯一区别：
  plan_ego_detach=True
    -> occ_flow 未来帧 ego 补偿 (means_fut - planner_res) 处切断梯度，
       OccFlowLoss 不再回传 planner，避免一致性损失污染 planner 训练。

这是与 ..._futgau (detach=False) 的对照实验。

【基础配置对齐 base_plan】loss 权重 / lr / optimizer / max_epochs / 模型结构
均通过 _base_ 继承 base_plan，与 ..._futgau 保持一致。
【load_from】同 futgau：base_plan 原 ckpt 本机不存在，改用已存在的 v12_fixempty
epoch_15；新增模块随机初始化 (strict=False)。
"""

_base_ = [
    '../nuscenes_gs25600_base_plan/nuscenes_gs25600_base_plan.py'
]

# base_plan 的 warm-start ckpt 本机不存在，改用已存在的 v12_fixempty epoch_15
load_from = 'exp/nuscenes_gs25600_v12_fixempty/checkpoints/epoch_15.pth'

model = dict(
    # ---------- delta 1: planner_head VADHead -> VADHeadFutGaussian（同 futgau）----------
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
    # ---------- delta 2: 打开 plan_ego 机制 + 唯一差异 detach=True ----------
    head=dict(
        use_plan_ego=True,
        plan_ego_warmup_epochs=2,
        plan_ego_detach=True,    # True: 切断 occ_flow->planner 梯度
    ),
)
