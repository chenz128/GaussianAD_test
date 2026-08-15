"""Exact costime global path plus scene-dependent per-frame residual planning.

The experiment keeps all ft_plan data, optimizer, schedule, loss and warm-start
settings unchanged. It intentionally adds neither an auxiliary imitation loss
nor a collision loss, so the comparison isolates the planner architecture.
"""

_base_ = [
    '../nuscenes_gs25600_gtbox_oracle_v12_ft_plan/'
    'nuscenes_gs25600_gtbox_oracle_v12_ft_plan.py'
]

_decoder_layer = dict(
    type='MyCustomBaseTransformerLayer',
    attn_cfgs=[dict(
        type='MultiheadAttention',
        embed_dims=128,
        num_heads=8,
        dropout=0.1)],
    feedforward_channels=256,
    ffn_dropout=0.1,
    batch_first=False,
)

model = dict(
    planner_head=dict(
        type='VADHeadDualTimeResidual',
        ego_fut_gaussian_decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=dict(
                **_decoder_layer,
                operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
        fut_self_decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=dict(
                **_decoder_layer,
                operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
        time_gaussian_decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=dict(
                **_decoder_layer,
                operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
    ),
)
