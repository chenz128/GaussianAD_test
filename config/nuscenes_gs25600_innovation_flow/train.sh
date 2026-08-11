#!/usr/bin/env bash
# ============================================================================
# nuscenes_gs25600_innovation_flow -- full-data private-cloud training
#
# Trains from scratch under the same backbone-pretraining convention as v3.
# This launcher does not load the v3 checkpoint and uses all eight cloud GPUs.
#
# Usage: tmux new -s train_innovation_flow
#        bash config/nuscenes_gs25600_innovation_flow/train.sh
# ============================================================================
set -euo pipefail

EXP_NAME="nuscenes_gs25600_innovation_flow"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
ENV_DIR="${ENV_DIR:-/data/chenz/conda_env/splatting}"
PY="${ENV_DIR}/bin/torchrun"
CONFIG="config/${EXP_NAME}/${EXP_NAME}.py"
WORK_DIR="${WORK_DIR:-out/${EXP_NAME}}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
NPROC="${NPROC:-8}"
MASTER_PORT="${MASTER_PORT:-12346}"

if [[ ! -x "${PY}" ]]; then
  echo "[FATAL] Training environment not found: ${ENV_DIR}" >&2
  exit 1
fi
export PATH="${ENV_DIR}/bin:${PATH}"

cd "${REPO}"
BRANCH="$(git branch --show-current)"
if [[ "${BRANCH}" != "splatting" ]]; then
  echo "[FATAL] Expected user branch 'splatting', got '${BRANCH}'." >&2
  exit 1
fi
mkdir -p "${WORK_DIR}"

{
  echo "=== launch $(date '+%Y-%m-%d %H:%M:%S') ==="
  echo "exp        : ${EXP_NAME}"
  echo "git branch : ${BRANCH}"
  echo "git commit : $(git rev-parse HEAD)"
  echo "config     : ${CONFIG}  (innovation latent FM, full data, 20 epochs)"
  echo "init       : from scratch except inherited backbone pretraining"
  echo "gpus       : ${GPUS}  (nproc=${NPROC}, port=${MASTER_PORT})"
} | tee -a "${WORK_DIR}/launch_history.log"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[DRY RUN] launcher validation complete; training was not started."
  exit 0
fi

CUDA_VISIBLE_DEVICES="${GPUS}" "${PY}" \
    --nproc_per_node "${NPROC}" \
    --master_port "${MASTER_PORT}" \
    train.py \
    --py-config "${CONFIG}" \
    --work-dir "${WORK_DIR}" \
    --dataset nuscenes \
    2>&1 | tee -a "${WORK_DIR}/train_run.log"