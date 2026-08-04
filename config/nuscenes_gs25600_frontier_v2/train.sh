#!/usr/bin/env bash
set -euo pipefail

EXP_NAME="nuscenes_gs25600_frontier_v2"
REPO="/data/chenz/GaussianAD"
TORCHRUN="/data/chenz/conda_env/splatting/bin/torchrun"
CONFIG="config/${EXP_NAME}/${EXP_NAME}.py"
WORK_DIR="out/${EXP_NAME}"
GPUS="4,5,6,7"
NPROC=4
MASTER_PORT=12481

cd "${REPO}"
mkdir -p "${WORK_DIR}"
{
  echo "=== launch $(date '+%Y-%m-%d %H:%M:%S') ==="
  echo "exp        : ${EXP_NAME}"
  echo "git branch : $(git rev-parse --abbrev-ref HEAD)"
  echo "git commit : $(git rev-parse HEAD)"
  echo "config     : ${CONFIG}"
  echo "delta      : fixed 25600 real + 1 empty; Gaussian/image context frontier"
  echo "gpus       : ${GPUS} (nproc=${NPROC}, port=${MASTER_PORT})"
} | tee -a "${WORK_DIR}/launch_history.log"

CUDA_VISIBLE_DEVICES="${GPUS}" "${TORCHRUN}" \
  --nproc_per_node "${NPROC}" \
  --master_port "${MASTER_PORT}" \
  train.py \
  --py-config "${CONFIG}" \
  --work-dir "${WORK_DIR}" \
  --dataset nuscenes \
  2>&1 | tee -a "${WORK_DIR}/train_run.log"