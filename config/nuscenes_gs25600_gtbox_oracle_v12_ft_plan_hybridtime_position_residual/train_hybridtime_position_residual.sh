#!/usr/bin/env bash
set -euo pipefail

SRC_CFG="nuscenes_gs25600_gtbox_oracle_v12_ft_plan_hybridtime_position_residual"
EXP_NAME="nuscenes_gs25600_v12_fixempty_ft_plan_hybridtime_position_residual"
REPO="/data/xinyao/navsim_workspace/GaussianAD"
PY="/data/chenz/conda_env/splatting/bin/torchrun"
CONFIG="config/${SRC_CFG}/${SRC_CFG}.py"
WORK_DIR="exp/${EXP_NAME}"
GPUS="${GPUS:-4,5,6,7}"
NPROC="${NPROC:-4}"
MASTER_PORT="${MASTER_PORT:-12689}"

cd "${REPO}"
mkdir -p "${WORK_DIR}"

{
  echo "=== launch $(date '+%Y-%m-%d %H:%M:%S') ==="
  echo "exp        : ${EXP_NAME}"
  echo "git branch : $(git rev-parse --abbrev-ref HEAD)"
  echo "git commit : $(git rev-parse HEAD)"
  echo "config     : ${CONFIG}"
  echo "init from  : v7 dualtime residual epoch 15"
  echo "planner    : exact v7 anchor + hybrid-time bounded position residual"
  echo "auxiliary  : detached-anchor step/position + trust region"
  echo "collision  : disabled"
  echo "lr         : ${LR:-1e-4}"
  echo "gpus       : ${GPUS} (nproc=${NPROC}, port=${MASTER_PORT})"
} | tee -a "${WORK_DIR}/launch_history.log"

CUDA_VISIBLE_DEVICES="${GPUS}" "${PY}" \
    --nproc_per_node "${NPROC}" \
    --master_port "${MASTER_PORT}" \
    train.py \
    --py-config "${CONFIG}" \
    --work-dir "${WORK_DIR}" \
    --dataset nuscenes \
    2>&1 | tee -a "${WORK_DIR}/train_run.log"
