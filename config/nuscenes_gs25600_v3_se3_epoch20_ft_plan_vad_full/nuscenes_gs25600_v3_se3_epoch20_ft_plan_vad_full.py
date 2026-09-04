"""Stage-2: frozen joint-epoch20 V3-SE3 frontend + original VAD Planner.

Only ``planner_head`` is optimized.  The old FutAttn/Global/Residual Planner
and its predicted-Future-Gaussian adapter are excluded from the bootstrap
checkpoint before loading.
"""

from copy import deepcopy
import os


_base_ = [
    '../nuscenes_gs25600_v3_se3_full/nuscenes_gs25600_v3_se3_full.py'
]

custom_imports = deepcopy(_base_.custom_imports)
custom_imports['imports'] = list(custom_imports['imports']) + [
    'model.segmentor.bev_segmentor_v3_se3_frozen_vad_isolated',
]


# Exact full-data GaussianAD schedule.  This is a fresh Stage-2 schedule even
# though its frozen frontend comes from the completed joint epoch-20 model.
max_epochs = int(os.environ.get('MAX_EPOCHS', 20))
eval_every_epochs = int(os.environ.get('EVAL_EVERY_EPOCHS', 1))
train_dataset_config = dict(num_samples=0)
val_dataset_config = dict(num_samples=0)
train_loader = dict(batch_size=1)
val_loader = dict(batch_size=1)

load_from = os.environ.get(
    'FRONTEND_CKPT',
    'exp/nuscenes_gs25600_v3_se3_epoch20_ft_plan_vad_full/'
    'bootstrap/epoch_20_frontend_only.pth',
)

# The random-initialized VADHead uses the original full-data LR at global
# batch 8.  Frozen frontend parameters cannot be changed by this optimizer.
lr = float(os.environ.get('LR', 2e-4))
optimizer = dict(
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    paramwise_cfg=dict(),
)
grad_max_norm = 35
warmup_iters = 500

frozen_modules = [
    'img_backbone',
    'img_neck',
    'lifter',
    'encoder',
    'temporal_encoder',
    'decoder',
    'map_decoder',
    'head',
]
find_unused_parameters = False
static_graph = False


_attn = dict(
    type='MultiheadAttention',
    embed_dims=128,
    num_heads=8,
    dropout=0.1,
)
_vad_decoder = dict(
    type='CustomTransformerDecoder',
    num_layers=1,
    return_intermediate=False,
    transformerlayers=dict(
        type='MyCustomBaseTransformerLayer',
        attn_cfgs=[_attn],
        feedforward_channels=256,
        ffn_dropout=0.1,
        batch_first=False,
        operation_order=('cross_attn', 'norm', 'ffn', 'norm'),
    ),
)

model = dict(
    type='BEVSegmentorV3SE3FrozenVADIsolated',
    # The full frontend is already under no_grad, so the history split is both
    # unnecessary and less faithful to frozen eval-mode inference.
    history_no_grad=False,
    head=dict(type='GaussianHeadFrontierV3Isolated'),
    planner_head=dict(
        _delete_=True,
        type='VADHead',
        embed_dims=128,
        fut_ts=6,
        fut_mode=6,
        ego_fut_mode=3,
        ego_agent_decoder=deepcopy(_vad_decoder),
        ego_map_decoder=deepcopy(_vad_decoder),
        ego_gaussian_decoder=deepcopy(_vad_decoder),
    ),
)


# Exact original GaussianAD Planner supervision: trajectory L1 plus predicted
# map boundary/direction constraints.  No Future-Gaussian auxiliary loss and
# no explicit collision loss are added to this baseline experiment.
loss = dict(
    _delete_=True,
    type='MultiLoss',
    loss_cfgs=[dict(type='PlanLoss', weight=10.0)],
)
loss_input_convertion = dict(
    _delete_=True,
    ego_fut_preds='ego_fut_preds',
    ego_fut_gt='ego_fut_trajs',
    ego_fut_masks='ego_fut_masks',
    ego_fut_cmd='ego_fut_cmd',
    all_cls_scores='all_cls_scores',
    all_pts_preds='all_pts_preds',
)
