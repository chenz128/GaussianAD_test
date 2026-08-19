#!/usr/bin/env bash
# ============================================================================
# nuscenes_gs25600_v3 —— 12,800 direct shared future Gaussians
# GPU: single-node 8 cards
# Usage: bash config/nuscenes_gs25600_v3/train.sh
# ============================================================================
set -euo pipefail

EXP_NAME="nuscenes_gs25600_v3"
REPO="/data/chenz/GaussianAD"
ENV_DIR="/data/chenz/conda_env/splatting"
PY="${ENV_DIR}/bin/torchrun"
CONFIG="config/${EXP_NAME}/${EXP_NAME}.py"
WORK_DIR="out/${EXP_NAME}"
GPUS="0,1,2,3,4,5,6,7"
NPROC=8
MASTER_PORT=12347

if [[ ! -x "${PY}" ]]; then
  echo "[FATAL] Training environment not found: ${ENV_DIR}" >&2
  exit 1
fi
export PATH="${ENV_DIR}/bin:${PATH}"

cd "${REPO}"
mkdir -p "${WORK_DIR}"
{
  echo "=== launch $(date '+%Y-%m-%d %H:%M:%S') ==="
  echo "exp        : ${EXP_NAME}"
  echo "git branch : $(git rev-parse --abbrev-ref HEAD)"
  echo "git commit : $(git rev-parse HEAD)"
  echo "config     : ${CONFIG}"
  echo "code       : direct 12800 shared bank; no attribute base/residual"
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