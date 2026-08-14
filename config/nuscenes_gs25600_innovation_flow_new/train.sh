#!/usr/bin/env bash
# ============================================================================
# nuscenes_gs25600_innovation_flow_new -- FM v2 (per-query image + SE3)
#
# Latent flow matching over innovation OCC with:
#   - SE(3) future alignment (future_pose_mode='se3')
#   - center-only retention of old Gaussians (center_only_mask=True)
#   - per-query multi-frame image context in the decoder
#
# Usage: tmux new -s train_innovation_flow_new
#        bash config/nuscenes_gs25600_innovation_flow_new/train.sh
# ============================================================================
set -euo pipefail

EXP_NAME="nuscenes_gs25600_innovation_flow_new"
REPO="${REPO:-/data/chenz/GaussianAD}"
ENV_DIR="${ENV_DIR:-/data/chenz/conda_env/faster}"
PY="${ENV_DIR}/bin/torchrun"
CONFIG="config/${EXP_NAME}/${EXP_NAME}.py"
WORK_DIR="${WORK_DIR:-out/${EXP_NAME}}"
GPUS="${GPUS:-0,1,2,3}"
NPROC="${NPROC:-4}"
MASTER_PORT="${MASTER_PORT:-12348}"

if [[ ! -x "${PY}" ]]; then
  echo "[FATAL] Training environment not found: ${ENV_DIR}" >&2
  exit 1
fi
export PATH="${ENV_DIR}/bin:${PATH}"

cd "${REPO}"
BRANCH="$(git branch --show-current)"
if [[ "${BRANCH}" != "splatting" ]]; then
  echo "[WARN] NAS checkout branch is '${BRANCH}', expected 'splatting'." >&2
fi
mkdir -p "${WORK_DIR}"

{
  echo "=== launch $(date '+%Y-%m-%d %H:%M:%S') ==="
  echo "exp        : ${EXP_NAME}"
  echo "git branch : ${BRANCH}"
  echo "git commit : $(git rev-parse HEAD)"
  echo "config     : ${CONFIG}  (FM v2: per-query image ctx + SE3 + center-only mask)"
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