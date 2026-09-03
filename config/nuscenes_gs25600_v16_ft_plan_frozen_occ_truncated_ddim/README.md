# v16 frozen-OCC truncated residual DDIM planner

## Fixed starting point

This experiment is a planner-only continuation of:

```text
exp/nuscenes_gs25600_v3_se3_ft_plan_futattn_global_residual_full/
checkpoints/epoch_16.pth
```

The checkpoint is loaded with `load_from` (weights only), never as an
optimizer/scheduler resume.  The launcher refuses to start if the v16 work
directory already contains `latest.pth` because `train.py` would otherwise
auto-resume it.

Epoch 16 is used because, under the corrected cumulative-position metric, it
is the safer audited full-data checkpoint.  Do not substitute a v14/v15 or
`gaussian_residual_dit` checkpoint.

## What is frozen

The image backbone/neck, lifter, Gaussian encoder, temporal encoder, decoder,
V3 future-Gaussian/OCC head, map decoder, and every inherited deterministic
planner child are frozen.  Planner inputs remain the GT-free whitelist exposed
by `BEVSegmentorV3FuturePlanIsolated`.  GT trajectories and boxes are visible
only to the loss after prediction.

A parameter-free segmentor guard inherits the original forward unchanged and
keeps all frozen frontend modules in eval mode after every `model.train()`
call.  This prevents BatchNorm buffers or dropout behavior from drifting even
though the repository's trainer re-enters train mode at each epoch.

## Planner design

The existing strong planner supplies three anchors:

1. the original fused main trajectory (candidate 0 and exact fallback);
2. the time-aligned per-frame trajectory;
3. the global low-L2 trajectory.

A six-token residual DiT produces four additional candidates.  It starts from
a small perturbation around candidate 0 (`t=0.25`) and uses two deterministic
DDIM updates at inference.  It never generates the full trajectory from pure
noise.

Every waypoint attends only to the matching future timestep in the exact
`planner_future_gaussians` bank.  Top-K selection uses route distance, opacity,
obstacle semantics, Gaussian scale and future uncertainty.  Collision risk is
computed over 15 samples covering the full 1.85 m x 4.084 m ego footprint,
including the +0.5 m offset used by the GaussianAD/VAD evaluator.  A bounded
analytic repulsion modifies generated candidates only.

The selector is deliberately asymmetric:

- candidate 0 is always available;
- an unsafe anchor may be replaced only by a materially lower-risk candidate;
- a safe anchor may be replaced only when risk is non-regressive and learned
  quality improves by a margin;
- residual size, acceleration and jerk remain hard constraints.

The learned quality score is trained against a GT-box oracle cost that combines
ADE/FDE and metric-aligned full-box collision.  A best-of-K trajectory loss,
differentiable SAT clearance, OCC risk, dynamics, diversity and trust-region
terms train only the new modules.

## Metric requirement

`dataset/metric_stp3.py` must convert the stored per-step displacements to
cumulative positions before L2 or collision evaluation.  `validate_v16.py`
fails if either conversion line is disabled.

## Validate and train

```bash
cd /data/xinyao/navsim_workspace/GaussianAD

export STRONG_OCC_CHECKPOINT=exp/nuscenes_gs25600_v3_se3_ft_plan_futattn_global_residual_full/checkpoints/epoch_16.pth
python config/nuscenes_gs25600_v16_ft_plan_frozen_occ_truncated_ddim/validate_v16.py --build-model

bash config/nuscenes_gs25600_v16_ft_plan_frozen_occ_truncated_ddim/train_frozen_occ_truncated_ddim.sh
```

To audit the complete launch contract without starting distributed training:

```bash
VALIDATE_ONLY=1 bash config/nuscenes_gs25600_v16_ft_plan_frozen_occ_truncated_ddim/train_frozen_occ_truncated_ddim.sh
```

The inherited full-data baseline settings remain unchanged: eight workers,
batch size one per GPU, AdamW learning rate `2e-4`, 20 epochs, and validation
every four epochs.  Only new planner parameters receive gradients.

## Test

```bash
EPOCH=20 \
bash config/nuscenes_gs25600_v16_ft_plan_frozen_occ_truncated_ddim/test_frozen_occ_truncated_ddim.sh
```

For the first ablation, report both cumulative-horizon keys
`plan_obj_box_col_{1,2,3}s` and endpoint keys
`plan_obj_box_col_stp3_{1,2,3}s`, together with candidate-0 selection rate.
