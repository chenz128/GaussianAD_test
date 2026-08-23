# v13 risk-aware global residual planner

This experiment is isolated from all existing planners and configs.  It adds
files only; `planner_v12.py`, package `__init__.py` files, existing configs,
checkpoints, and experiment directories are not changed.

## Unchanged training baseline

- Base config: `nuscenes_gs25600_gtbox_oracle_v12_ft_plan`
- Warm start: `exp/nuscenes_gs25600_v12_fixempty/checkpoints/epoch_15.pth`
- Epochs: 15
- Learning rate: `2e-4`
- Optimizer, scheduler, dataset, augmentation, decoder widths and existing loss
  weights: inherited without changes

## v13 deltas

- Mode-specific gate for the three navigation commands.
- Continuous risk sampled directly from future Gaussians (`mean + offset`, XY
  scale, opacity and the first ten obstacle semantic channels).
- Riskier global residuals are deterministically suppressed.
- Gate-ranking loss chooses the expert with lower cumulative error plus risk.
- Hard-negative SAT loss uses normalized log-sum-exp over agents and stronger
  late-horizon weights instead of averaging risk over every safe agent.
- Planner gradients into Gaussian parameters and offsets default to `1.0`,
  exactly preserving v12 training behavior.  `PLANNER_GRAD_SCALE=0.1` is kept
  only as an explicit later ablation.
- Quaternion rotation is used when measuring Gaussian risk: trajectory deltas
  are transformed into each Gaussian's local scale axes and the axis-aligned
  ego footprint is projected onto those axes.

## Commands

```bash
cd /data/xinyao/navsim_workspace/GaussianAD
bash config/nuscenes_gs25600_gtbox_oracle_v13_ft_plan_riskaware_global_residual/train_riskaware_global_residual.sh
```

Test epoch 15:

```bash
bash config/nuscenes_gs25600_gtbox_oracle_v13_ft_plan_riskaware_global_residual/test_riskaware_global_residual.sh
```

Optional ablations preserve the default values when omitted:

```bash
COL_W=0.1 GATE_W=0.1 AUX_W=2.0 PLANNER_GRAD_SCALE=1.0 \
  bash config/nuscenes_gs25600_gtbox_oracle_v13_ft_plan_riskaware_global_residual/train_riskaware_global_residual.sh
```

## Diagnostics to inspect before committing to 15 epochs

`RiskAwareGateLoss` writes the following values into the normal training log
without changing the loss or backward graph:

- `gate_selected_mean`, `gate_selected_abs_mean`
- `gate_positive_rate_0p1`, `gate_saturated_rate_0p8`
- `gate_mode_0_mean`, `gate_mode_1_mean`, `gate_mode_2_mean`
- `gate_prefer_global_rate`
- `risk_global_mean`, `risk_per_frame_mean`
- `risk_delta_mean`, `risk_delta_abs_mean`, `risk_delta_std`

The random numerical smoke range printed by `validate_v13.py` is not a measure
of risk discrimination.  The validator also constructs two trajectories in the
same scene and requires the collision trajectory risk to exceed the safe one by
at least 0.2.

Use a different `EXP_NAME` only by copying the launcher; the default launcher
never writes into the v12 best experiment directory.
