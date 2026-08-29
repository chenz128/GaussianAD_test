#!/usr/bin/env bash
set -euo pipefail

CFG_NAME="nuscenes_gs25600_v3_se3_ft_plan_futattn_global_residual_full"
REPO="/data/xinyao/navsim_workspace/GaussianAD"
PYTHON="/data/chenz/conda_env/splatting/bin/python"
CONFIG="config/${CFG_NAME}/${CFG_NAME}.py"
WORK_DIR="exp/${CFG_NAME}"

: "${GPUS:?Set GPUS explicitly, for example GPUS=0,1,2,3}"
EPOCH="${EPOCH:-20}"
CKPT="${CKPT:-${WORK_DIR}/checkpoints/epoch_${EPOCH}.pth}"
LOG_NAME="${LOG_NAME:-test_epoch${EPOCH}}"
MASTER_PORT="${MASTER_PORT:-21873}"

cd "${REPO}"
CUDA_VISIBLE_DEVICES="${GPUS}" MASTER_PORT="${MASTER_PORT}" "${PYTHON}" \
  test.py \
  --py-config "${CONFIG}" \
  --work-dir "${WORK_DIR}" \
  --resume-from "${CKPT}" \
  --log-name "${LOG_NAME}" \
  2>&1 | tee -a "${WORK_DIR}/${LOG_NAME}_console.log"
