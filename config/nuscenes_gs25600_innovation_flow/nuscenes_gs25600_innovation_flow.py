"""Full-data innovation latent flow matching experiment.

The current/retained branch and all original GaussianAD task losses are
inherited from v3.  The direct future generator is replaced with conditional
flow matching over a 6x64x30x30 innovation OCC latent.  No v3 checkpoint is
loaded; the experiment keeps only the same backbone pretraining as its base.
"""

_base_ = ['../nuscenes_gs25600_v3/nuscenes_gs25600_v3.py']

model = dict(
    head=dict(
        type='GaussianHeadInnovationFlow',
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
            target_pose_mode='translation',
            detach_context=False,
        ),
    ),
)

loss = _base_.loss
loss.loss_cfgs.append(dict(type='FlowMatchingLoss', weight=1.0))
loss.loss_cfgs.append(dict(
    type='InnovationOccupancyLoss',
    weight=3.0,
    dynamic_multiplier=5.0,
    empty_label=17,
    num_classes=18,
))

loss_input_convertion = _base_.loss_input_convertion
loss_input_convertion.update(
    flow_matching_loss='flow_matching_loss',
)

train_dataset_config = dict(num_samples=0)
val_dataset_config = dict(num_samples=0)
max_epochs = 20
eval_every_epochs = 4
static_graph = True