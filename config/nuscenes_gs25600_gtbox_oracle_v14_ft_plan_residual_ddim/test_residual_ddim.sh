#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${VERIFIED_V12_CHECKPOINT:-}" ]]; then
  echo "VERIFIED_V12_CHECKPOINT is required to parse the v14 config." >&2
  exit 2
fi
if [[ ! -f "${VERIFIED_V12_CHECKPOINT}" ]]; then
  echo "Verified v12 checkpoint does not exist: ${VERIFIED_V12_CHECKPOINT}" >&2
  exit 2
fi
REPO="${REPO:-/data/xinyao/navsim_workspace/GaussianAD}"
PYTHON="${PYTHON:-/data/chenz/conda_env/splatting/bin/python}"
CONFIG="config/nuscenes_gs25600_gtbox_oracle_v14_ft_plan_residual_ddim/nuscenes_gs25600_gtbox_oracle_v14_ft_plan_residual_ddim.py"
WORK_DIR="${WORK_DIR:-exp/nuscenes_gs25600_v14_ft_plan_residual_ddim}"
EPOCH="${EPOCH:-15}"
CHECKPOINT="${CHECKPOINT:-${WORK_DIR}/checkpoints/epoch_${EPOCH}.pth}"
LOG_NAME="${LOG_NAME:-test_epoch${EPOCH}}"
GPUS="${GPUS:-0,1,2,3}"
MASTER_PORT="${MASTER_PORT:-20791}"

cd "${REPO}"
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint does not exist: ${CHECKPOINT}" >&2
  exit 2
fi
mkdir -p "${WORK_DIR}"

{
  echo "=== v14 residual DDIM test $(date '+%Y-%m-%d %H:%M:%S') ==="
  echo "config     : ${CONFIG}"
  echo "checkpoint : ${CHECKPOINT}"
  echo "gpus       : ${GPUS} (port=${MASTER_PORT})"
} | tee -a "${WORK_DIR}/launch_history.log"

CUDA_VISIBLE_DEVICES="${GPUS}" MASTER_PORT="${MASTER_PORT}" "${PYTHON}" test.py \
  --py-config "${CONFIG}" \
  --work-dir "${WORK_DIR}" \
  --resume-from "${CHECKPOINT}" \
  --log-name "${LOG_NAME}" "$@" \
  2>&1 | tee -a "${WORK_DIR}/${LOG_NAME}_console.log"
