#!/usr/bin/env bash
# ============================================================================
# nuscenes_gs25600_innovation_flow -- v3-protocol private-cloud training
#
# Trains from scratch with v3's data subset, schedule, losses, and backbone
# pretraining. This launcher only follows v12-full's shell organization.
#
# Usage: tmux new -s train_innovation_flow
#        bash config/nuscenes_gs25600_innovation_flow/train.sh
# ============================================================================
set -euo pipefail

EXP_NAME="nuscenes_gs25600_innovation_flow"
REPO="${REPO:-/data/chenz/GaussianAD}"
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
  echo "[WARN] NAS checkout branch is '${BRANCH}', expected 'splatting'." >&2
  echo "[WARN] Continuing from the explicitly configured NAS path: ${REPO}" >&2
fi
mkdir -p "${WORK_DIR}"

{
  echo "=== launch $(date '+%Y-%m-%d %H:%M:%S') ==="
  echo "exp        : ${EXP_NAME}"
  echo "git branch : ${BRANCH}"
  echo "git commit : $(git rev-parse HEAD)"
  echo "config     : ${CONFIG}  (innovation latent FM, v3 data/schedule protocol)"
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