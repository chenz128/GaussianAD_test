#!/usr/bin/env bash
# ============================================================================
# nuscenes_gs25600_base_gt_ego —— 干净的 base 基线（当前代码 / GT ego 补偿）
#
# 目的：原 base 训练于 6/16-6/23，当时 forward_flow 用冻结 planner 的
#       ego_fut_preds（≈0）做 ego 补偿；commit 3a7ec54(7/10) 起改为 GT
#       ego_fut_trajs。用当前代码重跑 base，使其与 v10/v12 同公式，
#       给出可信的 current / future 目标线。
#
# 机器：h20-old  ssh -p 30300 root@8.130.174.55
# GPU ：4,5,6,7（0-3 常被他人占用，勿动）
# 用法：bash config/nuscenes_gs25600_base_gt_ego/train.sh
#       （建议在 tmux 里跑：tmux new -s train_base_gt_ego）
# ============================================================================
set -euo pipefail

EXP_NAME="nuscenes_gs25600_base_gt_ego"
REPO="/data/chenz/GaussianAD"
PY="/data/chenz/conda_env/splatting/bin/torchrun"
CONFIG="config/${EXP_NAME}/${EXP_NAME}.py"
WORK_DIR="out/${EXP_NAME}"
GPUS="4,5,6,7"
NPROC=4
MASTER_PORT=12470

cd "${REPO}"
mkdir -p "${WORK_DIR}"

# 留痕：记录本次启动的代码版本与配置
{
  echo "=== launch $(date '+%Y-%m-%d %H:%M:%S') ==="
  echo "exp        : ${EXP_NAME}"
  echo "git branch : $(git rev-parse --abbrev-ref HEAD)"
  echo "git commit : $(git rev-parse HEAD)"
  echo "config     : ${CONFIG}"
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
