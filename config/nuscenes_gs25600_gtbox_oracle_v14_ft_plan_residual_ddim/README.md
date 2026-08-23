# v14：基线条件残差 DiT + DDIM Planner

## 1. 续训模型（必须明确）

v14 唯一允许的初始化来源，是下面这个**精确架构**训练完成后的 audited
checkpoint：

```text
config/nuscenes_gs25600_gtbox_oracle_v12_ft_plan_futattn_global_residual/
  nuscenes_gs25600_gtbox_oracle_v12_ft_plan_futattn_global_residual.py
```

服务器上的预期权重路径为：

```text
/data/xinyao/navsim_workspace/GaussianAD/exp/
  nuscenes_gs25600_v12_fixempty_ft_plan_futattn_global_residual/
  checkpoints/epoch_15.pth
```

最终以审计通过的实际绝对路径为准。以下两类权重禁止使用：

- `exp/nuscenes_gs25600_v12_fixempty/checkpoints/epoch_15.pth`：这是更早的
  普通 v12 初始化，不等于训练完成的 `futattn_global_residual` 基线；
- 任意 `gaussian_residual_dit` 目录下的 checkpoint：其续训来源已经确认不可信。

v14 使用 `load_from` 做 **weights-only 初始化**，并强制
`resume_from=''`；不会恢复旧 optimizer、scheduler、epoch 或 scaler 状态。

推荐先执行：

```bash
cd /data/xinyao/navsim_workspace/GaussianAD
export VERIFIED_V12_CHECKPOINT=/data/xinyao/navsim_workspace/GaussianAD/exp/nuscenes_gs25600_v12_fixempty_ft_plan_futattn_global_residual/checkpoints/epoch_15.pth
python config/nuscenes_gs25600_gtbox_oracle_v14_ft_plan_residual_ddim/validate_v14.py
```

验证器会先把该 checkpoint 严格加载进原 v12 planner；任何 inherited key 的
missing、unexpected 或 shape mismatch 都会直接报错。然后再加载 v14，missing
只允许来自新增 residual-DDIM/selector 模块。

## 2. Planner 数据流

v14 不从纯噪声生成完整轨迹，而是在可靠 v12 结果附近生成小残差：

```text
v12 global-residual planner
        │
        ├── 6-step displacement Δp_b
        └── cumulative baseline p_b = cumsum(Δp_b)
                                      │
GT p_gt ──> normalized residual r* = (p_gt - stopgrad(p_b)) / scale
                                      │
fixed/train noise ε ──> z_t = α_t r* + σ_t ε
                                      │
       6 waypoint tokens + t/mode embedding
                                      │
       temporal self-attention (Micro-DiT)
                                      │
       time-aligned Gaussian Top-K cross-attention
                                      │
                    clean residual prediction r_hat
                                      │
              p_candidate = p_b + scale · r_hat
```

关键点：

- 轨迹只有 6 个规划点，因此使用 4-block、hidden=192 的 Micro-DiT，不照搬
  长轨迹的大型 DiT；
- diffusion 工作在**累计位置残差**，不是完整轨迹，也不是 step displacement；
- 训练采用截断 VP corruption，并直接预测 clean residual（`x0`）；
- 推理采用确定性 2-NFE DDIM、固定噪声库 `K=4`，不使用 CFG；
- 每个规划时刻只与同一未来时刻的 Gaussian 交互，按未来位置、旋转尺度、
  opacity、动态语义和 corridor 距离选择 Top-K（默认 128）；
- 第二次 DDIM evaluation 会围绕第一次预测的 clean path 重新选择 Gaussian；
- 不使用连续 gate，因此不存在 `tanh` 负 gate 导致的 residual 反向外推。

## 3. 推理安全选择

每个导航模态共有 5 个候选：

```text
candidate 0 = 原始 v12 baseline
candidate 1..4 = 固定噪声产生的 residual-DDIM 候选
```

选择器依次考虑：

1. future-Gaussian occupancy risk；
2. normalized residual trust region；
3. acceleration / jerk 限制；
4. learned candidate quality。

baseline 永远保留。如果 baseline 可行，且生成候选没有超过配置的 improvement
margin，则继续使用 baseline。新增 residual output 和 quality output 的末层均为
零初始化，所以权重加载后的初始 2-step DDIM 输出严格退化为 baseline。

## 4. 与基线保持一致的范围

v14 直接 `_base_` 继承精确的 `v12_ft_plan_futattn_global_residual` 配置。
以下内容不在 v14 中重新声明，因此解析后必须和基线逐项相同：

| 原样继承 | v14 允许新增/修改 |
| --- | --- |
| dataset、dataloader、augmentation | planner `type` |
| backbone/neck/lifter/encoder/decoder/map/head | residual Micro-DiT 参数 |
| v12 planner 的全部已有参数和 decoder | DDIM schedule、K、fixed noise |
| 全部原有 losses 及其顺序/权重 | Gaussian Top-K/risk/selector 参数 |
| optimizer、LR、scheduler、grad clip | 追加一个 `ResidualDDIMPlanLoss` |
| max_epochs、eval interval、freeze 设置 | `load_from` 与空 `resume_from` |

尤其注意：v14 不再覆盖 `max_epochs`、`lr`、`optimizer`、
`find_unused_parameters`、`frozen_modules` 或 backbone freeze flags。

训练时，继承的 v12 planner 保持原训练模式和原梯度尺度（Gaussian/offset 均为
1.0），并继续接受全部原基线 loss。新增 residual loss 使用 detached baseline
reference 和 detached Gaussian context，因此该**新增 loss**不会通过条件分支改写
旧 planner/感知参数；主 `PlanLoss` 仍通过 `ego_fut_preds` 保留原训练路径。

配置验证器会检查：

- 所有非白名单顶层配置与基线相同；
- 除 planner 以外的完整 `model` 配置相同；
- v12 planner 已有字段全部相同；
- 原 loss 列表逐项相同，且 v14 只在末尾追加一个 loss；
- 原 loss input mappings 全部保留；
- optimizer/LR/scheduler/epoch/freeze 等没有被暗改。

## 5. 新增 Loss

```text
L_new = 1.0 L_x0
      + 0.5 L_position
      + 0.25 L_FDE
      + 0.1 L_gaussian_safe
      + 0.05 L_dynamics
      + 0.1 L_rank
```

所有新增监督只作用于 GT command 对应模态，并严格应用未来帧 mask。原 v12 loss
列表完整保留；`L_new` 是追加项，不是替换项。

## 6. Residual scale

提交配置中的 identity scale 仅用于可执行 smoke test。正式实验前应在 train split
上运行 audited v12，并计算每个 horizon/axis 的 robust scale：

```text
r[h,d] = cumulative_gt[h,d] - cumulative_v12[h,d]
scale[h,d] = max(P90(abs(r[:,h,d])) / 1.645, 0.1)
```

不要使用 validation 数据，也不要使用单一 global min-max。输入顺序为
`x1,y1,...,x6,y6`：

```bash
export RESIDUAL_SCALE="0.30,0.20,0.45,0.28,0.60,0.35,0.75,0.43,0.90,0.50,1.05,0.58"
```

上面的数字只演示格式，不是 nuScenes 统计值。

## 7. 训练与验证

```bash
export VERIFIED_V12_CHECKPOINT=/absolute/path/to/audited_v12_global_residual.pth
bash config/nuscenes_gs25600_gtbox_oracle_v14_ft_plan_residual_ddim/train_residual_ddim.sh
```

launcher 会拒绝包含 `latest.pth` 的 work-dir，防止意外 resume。正式训练前还要用
同一个真实 fixed mini-batch 对比：

```text
max_abs(v14.ego_fut_base_preds - v12.ego_fut_preds) < 1e-6
```

必须记录 baseline/top-1/oracle@K 的 L2、collision、invalid@K、fallback rate、
NFE、延迟和显存。如果 A2（K=1 residual diffusion）不能稳定超过 deterministic
residual MLP，则停止 diffusion 分支，不再扩大 DiT。
