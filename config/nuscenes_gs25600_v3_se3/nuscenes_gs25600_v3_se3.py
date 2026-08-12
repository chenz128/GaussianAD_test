"""V3 with full current-to-future LiDAR SE(3) ego alignment."""

_base_ = ['../nuscenes_gs25600_v3/nuscenes_gs25600_v3.py']

model = dict(
    head=dict(
        future_pose_mode='se3',
        strict_range_mask=True,
        range_mask_sigma=3.0,
    ),
)

max_epochs = 15
static_graph = True