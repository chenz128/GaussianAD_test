#!/usr/bin/env bash
# ============================================================================
# v12_fixempty_ft_plan —— 在 v12_fixempty/epoch_15.pth 基础上续训，接入 planner
#
# 目标（config: nuscenes_gs25600_gtbox_oracle_v12_ft_plan.py）：
#   1. occ_flow 的 ego 运动补偿由 GT ego 切换为 planner 预测轨迹
#      (model.head.use_plan_ego=True, warmup=2 epoch 后切换)。
#   2. 解冻 map_decoder / planner_head (frozen_modules=[])，重新启用
#      MapLoss + PlanLoss(weight=1.0) 监督二者。
#   3. Gaussian 部分默认不冻结；flow_grad_scale 仍为 0.0（继承 v12）。
#
# 续训方式：config 里 load_from=exp/.../epoch_15.pth（从 epoch 0 计数训 15 轮，
#           optimizer/lr 全新，适配新解冻的参数与更小的 lr=2e-5）。
#   若想连续 epoch/optimizer，改为在下方 torchrun 追加：
#           --resume-from exp/nuscenes_gs25600_v12_fixempty/checkpoints/epoch_15.pth
#   （train.py 对 frozen_modules 变化导致的 optimizer state 不匹配会自动跳过）
#
# GPU：4,5,6,7（前 4 张严禁使用）
# 用法：tmux new -s train_v12_ft_plan
#       bash config/nuscenes_gs25600_gtbox_oracle_v12_ft_plan/train_ft_plan.sh
# 监控：epoch 2 起 planner 预测生效，关注 plan L2 / occ_flow FutAvg 是否稳定。
# ============================================================================
set -euo pipefail

SRC_CFG="nuscenes_gs25600_gtbox_oracle_v12_ft_plan"
EXP_NAME="nuscenes_gs25600_v12_fixempty_ft_plan"
REPO="/data/xinyao/navsim_workspace/GaussianAD"
PY="/data/chenz/conda_env/splatting/bin/torchrun"
CONFIG="config/${SRC_CFG}/${SRC_CFG}.py"
WORK_DIR="exp/${EXP_NAME}"
GPUS="4,5,6,7"
NPROC=4
MASTER_PORT=12482

cd "${REPO}"
mkdir -p "${WORK_DIR}"

{
  echo "=== launch $(date '+%Y-%m-%d %H:%M:%S') ==="
  echo "exp        : ${EXP_NAME}"
  echo "git branch : $(git rev-parse --abbrev-ref HEAD)"
  echo "git commit : $(git rev-parse HEAD)"
  echo "config     : ${CONFIG}"
  echo "init from  : exp/nuscenes_gs25600_v12_fixempty/checkpoints/epoch_15.pth (load_from)"
  echo "delta      : use_plan_ego=True + unfreeze map/plan + MapLoss/PlanLoss"
  echo "gpus       : ${GPUS}  (nproc=${NPROC}, port=${MASTER_PORT})"
} | tee -a "${WORK_DIR}/launch_history.log"

CUDA_VISIBLE_DEVICES="${GPUS}" "${PY}" \
    --nproc_per_node "${NPROC}" \
    --master_port "${MASTER_PORT}" \
    train.py \
    --py-config "${CONFIG}" \
    --work-dir "${WORK_DIR}" \
    --dataset nuscenes \
    2>&1 | tee -a "${WORK_DIR}/train_run.log"
