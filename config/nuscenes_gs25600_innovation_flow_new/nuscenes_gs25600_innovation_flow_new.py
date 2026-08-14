"""Innovation latent flow matching (v2) with per-query image context + SE(3).

Key changes vs config/nuscenes_gs25600_innovation_flow:
1. ``future_pose_mode='se3'``: full current-to-future LiDAR SE(3) alignment.
2. ``center_only_mask=True``: old Gaussians are retained whenever their
   centres stay inside the future render volume (no scale*sigma margin), so
   the SE(3) rotation of the old bank across turns no longer drops most of
   the retained history.
3. The InnovationFlowGenerator now samples per-query multi-frame image
   context (mirroring the V3 direct generator) instead of a single global
   image pool, restoring the strongest supervision signal for the decoder.
"""

_base_ = ['../nuscenes_gs25600_innovation_flow/nuscenes_gs25600_innovation_flow.py']

model = dict(
    head=dict(
        future_pose_mode='se3',
        center_only_mask=True,
        innovation_flow=dict(
            num_gaussians=12800,
            latent_dims=64,
            query_dims=64,
            temporal_in_dims=128,
            image_in_dims=128,
            num_frames=4,
            current_frame_index=0,
            scale_range=[0.08, 0.64],
            min_band=0.5,
            front_fraction=0.75,
            responsibility_size=1.0,
            initial_opacity=0.02,
            num_flow_steps=4,
            match_radius=0.76,
            target_pose_mode='se3',
            detach_context=False,
        ),
    ),
)

max_epochs = 15
static_graph = True