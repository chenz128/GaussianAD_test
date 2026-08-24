# v15：Safety-Calibrated Residual DDIM Planner

## 结论与选择

v14 的 DDIM 生成器不需要推倒重做：它已经取得当前最好的 L2，并且使用
`x0` prediction、zero-terminal-SNR、baseline residual 与 4-NFE DDIM，这些设计
方向是合理的。v15 只针对已证实的 collision 短板增加**评测对齐的安全校准与
safety-first 候选重排**，不引入 v13 的连续 gate，也不直接把 SAT penalty 加到
diffusion score 上。

正式名称：`Safety-Calibrated Residual DDIM (SCR-DDIM)`。

## v14 collision 短板的实际原因

| 现象 | 代码证据 | v15 修正 |
| --- | --- | --- |
| residual 轨迹没有直接的 box collision 标签 | v12 `PlanLoss(col_sat=True)` 只监督训练态 baseline；v14 residual loss 只有 Gaussian density proxy | 用 future GT vehicle box 生成 SAT safety target，监督独立安全头 |
| Gaussian risk 与 `box_col` 口径不一致 | v14 使用 Gaussian ellipse/opacity/dynamic probability；评测使用 0.5 m 栅格上的 vehicle rectangle | 安全 target 采用固定 ego footprint、`(+0.5, 0)` 中心偏移及 agent yaw/size |
| 旧 SAT 也不是完全 metric-aligned | 旧 loss 对所有有效 agent 求惩罚；`box_col` 只 rasterize vehicle，并剔除 GT ego 已碰撞时刻 | 只保留 category 14–23，并复现 GT-collision timestep mask |
| safety loss 量级弱且未校准 | v14 `safety_weight=0.1`，日志长期约 0.4–0.6；0.45 hard threshold 没有概率含义 | BCE + Brier 学习连续 near-miss target，输出有明确 0.5 m buffer 语义的 unsafe score |
| selector 允许 quality 与均值 risk 抵消尖峰风险 | v14 是一个加权和，主要风险项包含 horizon mean | 先看 worst-step + CVaR，再以 v14 cost 作有界 tie-breaker |
| 训练 fallback telemetry 不真实 | teacher path 把 `selected_index` 固定为 1，所以 baseline-selected rate 恒为 0 | v15 teacher 输出真实 safety selector index，仅作诊断，不替换 baseline 训练输出 |
| 训练 2 candidates、推理 5 candidates | v14 quality ranking 是集合分类，存在候选数量错位 | v15 safety head 对每条轨迹逐点打分，参数与 candidate count 无关 |

另一个必须澄清的结论：v14 的 residual reference 与 Gaussian context 都 detached，
训练态 `ego_fut_preds` 和 OccFlow 仍使用 v12 baseline。因此 mIoU 的提升不能直接
归因为“DDIM residual 反向帮助 occupancy”；它也可能来自 baseline 的继续训练或
实验方差。必须做同 seed、同 checkpoint 的控制实验后才能建立因果结论。

## 前沿方法中实际采用的部分

- [DiffusionDrive (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Liao_DiffusionDrive_Truncated_Diffusion_Model_for_End-to-End_Autonomous_Driving_CVPR_2025_paper.html)：保留 anchor/baseline 附近的少步扩散与候选置信度思想；v15 不回到纯噪声完整轨迹。
- [SparseDrive](https://arxiv.org/abs/2405.19620)：采用 hierarchical selection 与 collision-aware rescore，而不是把所有目标混成一个 planner logit。
- [SafeDrive (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/html/Kim_SafeDrive_Fine-Grained_Safety_Reasoning_for_End-to-End_Driving_in_a_Sparse_CVPR_2026_paper.html)：安全判断必须 trajectory-conditioned、agent-specific、time-specific。GaussianAD 的 future Gaussians 正好提供推理期 sparse-world proxy。
- [Hyper Diffusion Planner (2026)](https://arxiv.org/abs/2602.22801)：继续使用 `x0` prediction，并避免把依赖预测值的 collision auxiliary loss 直接混进 score matching；安全校准头的输入全部 detached。
- [BridgeDrive (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d4e8355bcc5ac0a8b30aaac05fccc1f6-Abstract-Conference.html)：支持 anchor-guided bridge 的理论方向，但 v15 不在已有强 v14 上同时改 forward process；bridge 留作单独消融，避免无法归因。

## Planner 数据流

```text
audited v12-fixempty epoch-15 (与 v14 相同起点)
    ├── v12 baseline candidate 0
    └── fresh residual DiT + 4-NFE DDIM -> candidates 1..4
                                      │
                         per-candidate/per-time features
          residual + position + velocity + acceleration + turning
          + trajectory-conditioned Gaussian mean/max tokens + risk + command
                                      │ (detach)
                  2-layer temporal safety encoder (hidden=96)
                                      │
                      p_unsafe[candidate, timestep]
                                      │
       structural feasibility -> p_max/CVaR safety -> v14 cost tie-break
                                      │
                           final selected trajectory
```

residual 输出层与安全头最后一层都零初始化。从 v12-fixempty 加载后，DDIM 生成
轨迹严格等于 baseline，所有候选的安全分数都是 `p=0.5`，该常量不会改变
候选相对顺序；只要安全分数尚不能区分候选，就显式保留 v14 的完整 selector
（包括 Gaussian-risk feasibility 与 baseline margin）。训练后安全分数形成有效差异，
才切换为安全优先排序。

推理的层级规则：

1. 保留 residual trust region、acceleration 和 jerk 的结构约束；
2. 优先考虑 `p_max <= threshold` 的候选；
3. 若存在安全候选，禁止不安全候选参与 tie-break；
4. 若全部不安全，选择 worst-step/CVaR 风险最低者，避免 arbitrary fallback；
5. v14 Gaussian/quality cost 归一化到 `[0,1]`，仅做小权重 tie-break；
6. baseline 永远是 candidate 0，并保留原 baseline margin 规则。

## Loss

v15 保留 v14 的完整 residual-DDIM objective：

```text
L_v14 = 1.0 L_x0
      + 0.5 L_position
      + 0.25 L_FDE
      + 0.1 L_gaussian_safe
      + 0.05 L_dynamics
      + 0.1 L_rank
```

新增项只训练 safety head：

```text
L_v15_extra = 0.25 (BCE_soft-SAT + 0.25 Brier)
            + 0.10 candidate_safety_rank
```

`soft-SAT target = sigmoid((0.5 m - signed_clearance) / 0.25 m)`，因此不仅真实
碰撞，near miss 也提供连续监督。正风险样本权重为 4。SAT label 构造在
`torch.no_grad()` 下完成，且 safety head 输入 detached；新增校准损失不会更新
candidate、v14 DiT、baseline、Gaussian 或 offset。

这里没有启用 v13 的 hard-negative SAT generator penalty。原因不是 SAT 无效，而是
它会直接改变 denoiser 的条件回归目标，可能用 collision 换 L2/生成质量。若 v15
证明 `oracle@5 box_col` 也没有改善空间，下一阶段再单独实现 HDP 风格的
reward-weighted post-training，不能和本次 selector 改动混在一起。

## 续训模型（唯一允许来源）

```text
/data/xinyao/navsim_workspace/GaussianAD/exp/
  nuscenes_gs25600_v12_fixempty_ft_plan_futattn_global_residual/
  checkpoints/epoch_15.pth
```

v15 使用 `load_from` 做 weights-only 初始化，并强制 `resume_from=''`。不会恢复
optimizer、scheduler、scaler、epoch 或 global iteration。checkpoint 必须严格加载
成 v12 planner；加载成 v15 时 missing key 只能来自 v14 residual-DDIM 模块及 v15
safety 模块。这样 v14 与 v15 使用完全相同的起始权重，指标差异才可归因于新设计。

## 与基线保持一致

v15 仍直接继承精确的 `v12_ft_plan_futattn_global_residual`，而不是继承可能继续
漂移的实验配置。以下项目解析后必须与 v12 完全一致：

- dataset、dataloader、augmentation；
- planner 以外的 model 全部模块；
- v12 planner 已有字段；
- 12 个 legacy losses 的顺序、权重与参数；
- optimizer、LR、scheduler、grad clip；
- `max_epochs=15`、evaluation interval、freeze 与 distributed 设置。

只允许改变 planner type、复刻 v14 residual 参数、增加 v15 safety 参数、追加一个
自包含 loss、增加输入映射；weights-only 来源固定为 audited v12-fixempty。特别是
`attr_labels_planner`、`gt_boxes`、`fut_valid_flag` 必须显式从 `metas` 映射给
safety loss；缺任一字段时直接报错，禁止把“缺标注”静默当成“无碰撞”。

## 验证与启动

先运行严格审计：

```bash
cd /data/xinyao/navsim_workspace/GaussianAD
export VERIFIED_V12_CHECKPOINT=/data/xinyao/navsim_workspace/GaussianAD/exp/nuscenes_gs25600_v12_fixempty_ft_plan_futattn_global_residual/checkpoints/epoch_15.pth
/data/chenz/conda_env/splatting/bin/python \
  config/nuscenes_gs25600_gtbox_oracle_v15_ft_plan_safety_calibrated_residual_ddim/validate_v15.py

# 读取一个真实 val batch，检查 box/trajectory/mask 映射与 SAT 解码
/data/chenz/conda_env/splatting/bin/python \
  config/nuscenes_gs25600_gtbox_oracle_v15_ft_plan_safety_calibrated_residual_ddim/validate_v15_data.py
```

检查命令但不启动：

```bash
VALIDATE_FIRST=0 DRY_RUN=1 \
  bash config/nuscenes_gs25600_gtbox_oracle_v15_ft_plan_safety_calibrated_residual_ddim/train_safety_calibrated_residual_ddim.sh
```

正式训练：

```bash
bash config/nuscenes_gs25600_gtbox_oracle_v15_ft_plan_safety_calibrated_residual_ddim/train_safety_calibrated_residual_ddim.sh
```

Server3 默认使用 `GPUS=4,5,6,7`，`NPROC` 自动等于 GPU 数量，torchrun 使用
standalone auto-port。它通过同一个解释器执行 `python -m torch.distributed.run`，
避免系统 Python、conda Python 与独立 `torchrun` 混用。脚本拒绝
`--resume-from`、GPU/NPROC 数量不一致、任何已存在的 work-dir，以及覆盖 v14
work-dir。可显式设置：

```bash
GPUS=0,1,2,3 SAFETY_PROB_THRESHOLD=0.5 DDIM_STEPS=4 GAUSSIAN_TOPK=128 \
  bash config/nuscenes_gs25600_gtbox_oracle_v15_ft_plan_safety_calibrated_residual_ddim/train_safety_calibrated_residual_ddim.sh
```

测试：

```bash
bash config/nuscenes_gs25600_gtbox_oracle_v15_ft_plan_safety_calibrated_residual_ddim/test_safety_calibrated_residual_ddim.sh
```

## 必须报告的消融

同一个 v15 checkpoint 至少测试 threshold `0.35/0.45/0.50/0.60`，并报告：

- selected、baseline 与 oracle@5 的 L2 / obj_col / box_col；
- safety probability 的 BCE、Brier、positive rate；
- baseline fallback rate、all-infeasible rate、candidate index histogram；
- `oracle@5 - selected` safety gap；
- mIoU/iou(geo) 0.0/1.0/2.0/3.0s；
- NFE、延迟和显存。

判断门槛：若 oracle@5 的 box collision 明显优于 v14 selected，而 v15 没有接近
oracle，说明 scorer/threshold 仍需校准；若 oracle@5 本身没有改善，则 K=4 生成器
缺少安全多样性，继续调 selector 没有意义，应进入独立的 reward-weighted generator
post-training。任何“SOTA”结论都必须等上述对照完成后再给出。
