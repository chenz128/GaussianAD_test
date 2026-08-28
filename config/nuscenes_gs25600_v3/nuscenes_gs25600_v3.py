"""Isolated port of chenz/GaussianAD V3.

The effective Frontier-v2/fix overrides are flattened onto the identical
target v12 base.  All new implementations are loaded only by this config.
"""

from copy import deepcopy


custom_imports = dict(
    imports=[
        'model.encoder.temporal_encoder.gaussian_temporal_encoder_v3_isolated',
        'model.head.gaussian_head_frontier_v3_isolated',
        'loss.occupancy_loss_flow_v3_isolated',
        'dataset.future_lidar_pose_v3_isolated',
    ],
    allow_failed_imports=False,
)

_base_ = ['../nuscenes_gs25600_gtbox_oracle_v12.py']


# Source V3 globally changed OccupancyFlowLoss.  Select the isolated equivalent
# only for this experiment while preserving every other inherited loss entry.
loss = deepcopy(_base_.loss)
loss['loss_cfgs'][1]['type'] = 'OccupancyFlowLossV3Isolated'


# Reproduce the source dataset.py + surroundocc.py future-pose additions only
# inside this config.  The transform must run before multi-sweep processing,
# while lidar2global is still the current-frame 4x4 matrix.
collect_keys = list(_base_.collect_keys)
if 'future_lidar2global' not in collect_keys:
    collect_keys.append('future_lidar2global')

train_pipeline = deepcopy(_base_.train_pipeline)
train_pipeline.insert(0, dict(type='BuildFutureLidarPoseV3Isolated'))
train_pipeline[-1]['keys'] = collect_keys

test_pipeline = deepcopy(_base_.test_pipeline)
test_pipeline.insert(0, dict(type='BuildFutureLidarPoseV3Isolated'))
test_pipeline[-1]['keys'] = collect_keys

train_dataset_config = dict(pipeline=train_pipeline)
val_dataset_config = dict(pipeline=test_pipeline)


model = dict(
    temporal_encoder=dict(
        type='GaussianTemporalEncoderV3Isolated',
        current_frame_index=0,
    ),
    head=dict(
        type='GaussianHeadFrontierV3Isolated',
        target_num_gaussians=25600,
        flow_grad_scale=0.0,
        flow_include_empty=True,
        current_frame_index=0,
        min_current_gaussian_ratio=0.99,
        dynamic_class_multiplier=3.0,
        # Explicit source defaults keep the port stable against future changes.
        future_pose_mode='translation',
        strict_range_mask=False,
        range_mask_sigma=0.0,
        center_only_mask=False,
        # Retained for exact resolved-config parity with the source inheritance
        # chain. GaussianHeadFrontierV3 does not consume this legacy V2 field.
        frontier_context=dict(
            context_dims=64,
            image_in_dims=128,
            bev_size=30,
            local_radius=1,
            scale_range=[0.08, 0.64],
            min_band=0.5,
        ),
        direct_generator=dict(
            num_gaussians=12800,
            embed_dims=64,
            temporal_in_dims=128,
            image_in_dims=128,
            num_frames=4,
            current_frame_index=0,
            scale_range=[0.08, 0.64],
            min_band=0.5,
            front_fraction=0.75,
            responsibility_size=1.0,
            initial_opacity=0.02,
            detach_context=True,
        ),
    ),
)

max_epochs = 15
static_graph = True
