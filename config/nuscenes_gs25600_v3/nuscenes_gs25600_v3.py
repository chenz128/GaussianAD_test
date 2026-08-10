"""V3: direct shared 3-second future Gaussian bank without attribute bases."""

_base_ = ['../nuscenes_gs25600_froniter_v2_fix/nuscenes_gs25600_froniter_v2_fix.py']

model = dict(
    temporal_encoder=dict(
        current_frame_index=0,
    ),
    head=dict(
        type='GaussianHeadFrontierV3',
        current_frame_index=0,
        min_current_gaussian_ratio=0.99,
        dynamic_class_multiplier=3.0,
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