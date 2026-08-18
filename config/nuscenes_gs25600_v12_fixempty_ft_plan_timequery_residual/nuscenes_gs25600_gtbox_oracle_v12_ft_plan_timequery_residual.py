"""ft_plan with hybrid temporal encoding and supervised time-query residuals.

This experiment inherits every optimization, schedule, freezing and head setting
from ft_plan. Only the planner type and one low-weight auxiliary loss are added.
"""

_base_ = ['../nuscenes_gs25600_gtbox_oracle_v12_ft_plan/nuscenes_gs25600_gtbox_oracle_v12_ft_plan.py']

model = dict(
    planner_head=dict(
        type='VADHeadTimeQueryResidual',
        time_interval=0.5,
        num_fourier_bands=8,
        ego_fut_gaussian_decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=dict(
                type='MyCustomBaseTransformerLayer',
                attn_cfgs=[dict(
                    type='MultiheadAttention',
                    embed_dims=128,
                    num_heads=8,
                    dropout=0.1)],
                feedforward_channels=256,
                ffn_dropout=0.1,
                batch_first=False,
                operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
        time_self_decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=dict(
                type='MyCustomBaseTransformerLayer',
                attn_cfgs=[dict(
                    type='MultiheadAttention',
                    embed_dims=128,
                    num_heads=8,
                    dropout=0.1)],
                feedforward_channels=256,
                ffn_dropout=0.1,
                batch_first=False,
                operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
        time_gaussian_decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=dict(
                type='MyCustomBaseTransformerLayer',
                attn_cfgs=[dict(
                    type='MultiheadAttention',
                    embed_dims=128,
                    num_heads=8,
                    dropout=0.1)],
                feedforward_channels=256,
                ffn_dropout=0.1,
                batch_first=False,
                operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
    ),
)

_time_query_plan_loss = dict(
    type='TimeQueryPlanLoss',
    weight=2.0,
    position_weight=1.0,
    beta=0.5,
)

loss = dict(
    type='MultiLoss',
    loss_cfgs=list(_base_.loss['loss_cfgs']) + [_time_query_plan_loss],
)

loss_input_convertion = dict(
    ego_fut_aux_preds='ego_fut_aux_preds',
)
