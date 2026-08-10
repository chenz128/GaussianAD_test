"""Frontier-v2 with the current-first temporal frame order handled correctly."""

_base_ = ['../nuscenes_gs25600_frontier_v2/nuscenes_gs25600_frontier_v2.py']

model = dict(
    temporal_encoder=dict(
        current_frame_index=0,
    ),
    head=dict(
        current_frame_index=0,
        min_current_gaussian_ratio=0.99,
    ),
)

max_epochs = 15
static_graph = True