"""
nuscenes_gs25600_gtbox_oracle_v12_ft_plan_futattn —— ego 与未来帧高斯做注意力的
planner 实验。

在 v12_fixempty_ft_plan 基础上只改一件事：planner_head 的高斯交互方式。

【改动：planner_head VADHead -> VADHeadFutAttn】
  本项目未来帧高斯 = 当前帧高斯按模型预测的 flow 位移(offset)平移得到
  (未来第 t 帧高斯.xy = 当前高斯.xy + offset[...,t,:]，与 forward_flow 一致)。

  原 VADHead：ego_query 只与当前帧 agent/map/gaussian 交互 -> MLP 一次性回归 6 帧。
  新 VADHeadFutAttn：
    1) 当前帧分支不变（ego↔agent / ego↔map / ego↔gaussian(当前)），复用预训练权重；
    2) 新增未来帧高斯注意力：由 offset 平移当前高斯得到逐未来帧高斯，
       未来 ego token（当前 ego 特征初始化 + 逐帧位置编码 + 时间维 self-attn）
       与「该帧未来高斯」做交叉注意力(ego_fut_gaussian_decoder)；
    3) 逐时间步回归 -> [B, ego_fut_mode, fut_ts, 2]，PlanLoss / use_plan_ego 兼容。

  即 ego 分别与当前帧高斯、未来帧高斯做注意力，让 planner 显式感知高斯运动。
  其余（load_from、max_epochs、lr、frozen_modules、use_plan_ego、MapLoss/PlanLoss、
  loss_input_convertion）均继承 ft_plan，完全不变。

  新增参数（fut_gaussian_fus_mlp / ego_to_fut / fut_pos / fut_self_decoder /
  ego_fut_gaussian_decoder / fut_out_mlp）为随机初始化层，load_from 时 strict=False
  忽略缺失键，从头学习。
"""

_base_ = [
    '../nuscenes_gs25600_gtbox_oracle_v12_ft_plan/nuscenes_gs25600_gtbox_oracle_v12_ft_plan.py'
]

# ============ 未来帧高斯注意力：覆盖 planner_head ============
model = dict(
    planner_head=dict(
        # 新 head：继承 VADHead，新增 ego <-> 未来帧高斯 注意力
        type='VADHeadFutAttn',
        # ---- 未来 ego token 的时间维 self-attention（frames 在 seq 维）----
        fut_self_decoder=dict(
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
                operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
        # ---- ego(逐未来帧) <-> 未来帧高斯 交叉注意力（核心新增）----
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
)
