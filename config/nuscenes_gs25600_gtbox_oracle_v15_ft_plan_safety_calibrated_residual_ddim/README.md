# v15b：Collision-Guarded Residual DDIM Planner

## 设计结论

旧 v15 collision 上升的直接原因不是续训 checkpoint，而是未充分校准的 safety
score 以 `20×` 权重接管了候选排序，同时正常选择路径绕过了 v14 的 Gaussian
`risk_max < 0.45` 硬门限。旧 safety label 还只包含 vehicle，而正式 planner metric
使用 vehicle 与 pedestrian 的并集。

修订版保留 v14 residual-DDIM 生成器和 L2-aware cost，改成保守的
**Gaussian hard guard + baseline-relative Pareto + high-confidence veto**：

1. v14 的 Gaussian risk、residual trust region、acceleration、jerk 硬约束始终有效；
2. baseline 可行时，generated candidate 的 Gaussian mean/max risk 和 learned safety
   score 都不得比 baseline 更差；
3. 可行候选仍按 v14 analytic/quality cost 排序，避免用安全分数随意牺牲 L2；
4. 只有当前候选 `p >= 0.60`、替代候选 `p <= 0.30` 且 safety score 至少改善
   `0.15` 时，learned head 才允许强制 veto；
5. safety score spread 小于 `0.05` 时，逐元素回退到完整 v14 selector；
6. candidate zero 始终保留，所有新模块仍从 audited v12-fixempty 上随机初始化。

这些约束提高 collision 的预期鲁棒性，但最终是否同时提升 box collision 和 L2，
必须由候选级消融与正式 nuScenes 测试确认，不能由代码设计直接保证。

## 训练/推理候选对齐

旧设计训练 safety head 时只有 baseline + 1 条随机时刻 teacher 轨迹，推理却是
baseline + 4 条完整 DDIM 轨迹。修订版在训练时额外执行一次 detached 的正式
采样路径，直接复用相同固定噪声库和相同 NFE：

```text
train safety/rank: baseline + fixed-noise 4-NFE DDIM K=4 = 5 candidates
test selector    : baseline + fixed-noise 4-NFE DDIM K=4 = 5 candidates
```

四条 proposal 在每个 DDIM step 内合并为一个 batch，并在 `no_grad`、残差模块
临时 eval 和 `torch.random.fork_rng` 下生成，不会改变原 v14
diffusion/dropout 的随机流。原始 random-t teacher 及其
`x0 + position + FDE + Gaussian safety + dynamics` generator objective 保持不变；
完整多候选只训练 quality/safety scorer。代价是训练多执行 4 次无梯度去噪前向，
但它消除了旧 v15 最关键的候选分布错配。

## Safety supervision

正式 `compute_planner_metric_stp3` 使用 vehicle 与 pedestrian occupancy 的并集。
修订版 SAT label 因此包含：

- human category `2..8`；
- vehicle category `14..23`；
- ego footprint `1.85 m × 4.084 m` 与中心偏移 `(+0.5, 0)`；
- GT-ego 已碰撞 timestep 使用正式的无缓冲边界进行 mask；
- `0.25 m` conservative hard-collision buffer，用于覆盖正式 0.5 m raster metric
  的边界量化；该 buffer 只用于候选标签，不用于 GT mask；
- `0.75 m` near-miss soft target，提供比真实 collision 更密集的边界监督。

Safety head 的主损失现在是 hard-collision BCE；near-miss BCE 只是辅助项。正样本
权重为 12，远距离负样本降权，Brier 对 hard target 计算。候选 rank 改成只监督
target risk 差异超过 `0.05` 的 pair，安全相同的候选不再通过 `argmin` 自动偏向
candidate zero。

## 唯一允许的续训来源

```text
/data/xinyao/navsim_workspace/GaussianAD/exp/
  nuscenes_gs25600_v12_fixempty_ft_plan_futattn_global_residual/
  checkpoints/epoch_15.pth
```

配置使用 `load_from` 做 weights-only 初始化并强制 `resume_from=''`。不得恢复旧
optimizer、scheduler、scaler、epoch 或 global iteration。dataset、dataloader、
planner 以外模型、12 个 legacy losses、optimizer/LR/scheduler、grad clip、15 epochs
等继续严格继承 v12 baseline；只有 residual-DDIM 与本文件记录的 safety 参数是新增项。

## 验证与启动

```bash
cd /data/xinyao/navsim_workspace/GaussianAD
export VERIFIED_V12_CHECKPOINT=/data/xinyao/navsim_workspace/GaussianAD/exp/nuscenes_gs25600_v12_fixempty_ft_plan_futattn_global_residual/checkpoints/epoch_15.pth

/data/chenz/conda_env/splatting/bin/python \
  config/nuscenes_gs25600_gtbox_oracle_v15_ft_plan_safety_calibrated_residual_ddim/validate_v15.py

/data/chenz/conda_env/splatting/bin/python \
  config/nuscenes_gs25600_gtbox_oracle_v15_ft_plan_safety_calibrated_residual_ddim/validate_v15_data.py

DRY_RUN=1 VALIDATE_FIRST=0 \
  bash config/nuscenes_gs25600_gtbox_oracle_v15_ft_plan_safety_calibrated_residual_ddim/train_safety_calibrated_residual_ddim.sh

bash config/nuscenes_gs25600_gtbox_oracle_v15_ft_plan_safety_calibrated_residual_ddim/train_safety_calibrated_residual_ddim.sh
```

新训练默认写入独立目录，旧 v15 结果不会被覆盖：

```text
exp/nuscenes_gs25600_v15b_ft_plan_collision_guarded_residual_ddim
```

测试：

```bash
bash config/nuscenes_gs25600_gtbox_oracle_v15_ft_plan_safety_calibrated_residual_ddim/test_safety_calibrated_residual_ddim.sh
```

## 必须先报告的消融

同一个新 checkpoint、同一批候选至少比较：

1. candidate zero only；
2. exact v14 selector；
3. Gaussian no-regression selector（关闭 learned veto）；
4. 完整 v15b selector；
5. formal vehicle+pedestrian `box_col` oracle@5。

必须保存 selected index、legacy selected index、safety override、每候选 Gaussian
risk、safety probability 和正式 per-candidate collision。若 oracle@5 不优于 v14，
应优化 DDIM proposal diversity；若 oracle@5 明显更好但 selected 没有接近 oracle，
再校准 safety head/threshold，不能继续增大 safety score 权重。
