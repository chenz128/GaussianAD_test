"""
nuscenes_gs25600_gtbox_oracle_v12_ft_plan_futattn
planner

v12_fixempty_ft_plan planner_head

planner_head VADHead -> VADHeadFutAttn
  flow (offset)
  ( t .xy = .xy + offset[...,t,:] forward_flow )

  VADHead ego_query agent/map/gaussian -> MLP  6
  VADHeadFutAttn
    1) (ego agent / ego map / ego gaussian())
    2)  offset
       ego token ego +  +  self-attn
        ego_fut_gaussian_decoder
    3)  -> [B,ego_fut_mode,fut_ts,2] PlanLoss / use_plan_ego

  ego  planner
  load_from max_epochs lr frozen_modules use_plan_ego MapLoss/PlanLoss
  loss_input_convertion ft_plan

  fut_gaussian_fus_mlp / ego_to_fut / fut_pos / fut_self_decoder /
  ego_fut_gaussian_decoder / fut_out_mlp load_from  strict=False

"""

_base_ = [
    '../nuscenes_gs25600_gtbox_oracle_v12_ft_plan/nuscenes_gs25600_gtbox_oracle_v12_ft_plan.py'
]
import os

# ============  ============
lr = float(os.environ.get("LR", 2e-4))
optimizer = dict(
    optimizer=dict(type="AdamW", lr=lr, weight_decay=0.01),
    paramwise_cfg=dict(custom_keys={'img_backbone': dict(lr_mult=0.1)}),
)

# ============   planner_head ============
model = dict(
    planner_head=dict(
        type='VADHeadFutAttn',
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