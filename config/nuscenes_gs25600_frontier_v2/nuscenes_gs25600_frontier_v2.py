"""V2: exactly 25,600 future real Gaussians with 3D/image context."""

_base_ = ['../nuscenes_gs25600_gtbox_oracle_v12.py']

model = dict(
    head=dict(
        type='GaussianHeadFrontierV2',
        target_num_gaussians=25600,
        flow_grad_scale=0.0,
        flow_include_empty=True,
        frontier_context=dict(
            context_dims=64,
            image_in_dims=128,
            bev_size=30,
            local_radius=1,
            scale_range=[0.08, 0.64],
            min_band=0.5,
        ),
    ),
)

max_epochs = 15
static_graph = True