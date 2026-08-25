#!/usr/bin/env bash
set -Eeuo pipefail

REPO="${REPO:-/data/xinyao/navsim_workspace/GaussianAD}"
PYTHON="${PYTHON:-/data/chenz/conda_env/splatting/bin/python}"
CONFIG="config/nuscenes_gs25600_gtbox_oracle_v15_ft_plan_safety_calibrated_residual_ddim/nuscenes_gs25600_gtbox_oracle_v15_ft_plan_safety_calibrated_residual_ddim.py"
VERIFIED_V12_CHECKPOINT="${VERIFIED_V12_CHECKPOINT:-${REPO}/exp/nuscenes_gs25600_v12_fixempty_ft_plan_futattn_global_residual/checkpoints/epoch_15.pth}"
WORK_DIR="${WORK_DIR:-exp/nuscenes_gs25600_v15b_ft_plan_collision_guarded_residual_ddim}"
EPOCH="${EPOCH:-15}"
CHECKPOINT="${CHECKPOINT:-${WORK_DIR}/checkpoints/epoch_${EPOCH}.pth}"
LOG_NAME="${LOG_NAME:-test_epoch${EPOCH}}"
GPUS="${GPUS:-4,5,6,7}"
MASTER_PORT="${MASTER_PORT:-20796}"
export VERIFIED_V12_CHECKPOINT

if [[ ! -d "${REPO}" || ! -x "${PYTHON}" ]]; then
  echo "Invalid REPO/PYTHON: ${REPO} / ${PYTHON}" >&2
  exit 2
fi
if [[ ! "${MASTER_PORT}" =~ ^[0-9]+$ \
      || "${MASTER_PORT}" -lt 1 || "${MASTER_PORT}" -gt 65535 ]]; then
  echo "MASTER_PORT must be an integer in [1, 65535]: ${MASTER_PORT}" >&2
  exit 2
fi
if [[ ! -f "${VERIFIED_V12_CHECKPOINT}" ]]; then
  echo "Verified v12-fixempty source checkpoint is missing: ${VERIFIED_V12_CHECKPOINT}" >&2
  exit 2
fi

cd "${REPO}"
if [[ ! -f "${CONFIG}" ]]; then
  echo "Config does not exist: ${CONFIG}" >&2
  exit 2
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint does not exist: ${CHECKPOINT}" >&2
  exit 2
fi
mkdir -p "${WORK_DIR}"

{
  echo "=== v15b collision-guarded residual DDIM test $(date '+%Y-%m-%d %H:%M:%S') ==="
  echo "config     : ${CONFIG}"
  echo "checkpoint : ${CHECKPOINT}"
  echo "thresholds : unsafe=${SAFETY_PROB_THRESHOLD:-0.60}, safe=${SAFETY_SAFE_PROB_THRESHOLD:-0.30}"
  echo "gpus       : ${GPUS} (port=${MASTER_PORT})"
} | tee -a "${WORK_DIR}/launch_history.log"

CUDA_VISIBLE_DEVICES="${GPUS}" MASTER_PORT="${MASTER_PORT}" "${PYTHON}" test.py \
  --py-config "${CONFIG}" \
  --work-dir "${WORK_DIR}" \
  --resume-from "${CHECKPOINT}" \
  --log-name "${LOG_NAME}" "$@" \
  2>&1 | tee -a "${WORK_DIR}/${LOG_NAME}_console.log"
