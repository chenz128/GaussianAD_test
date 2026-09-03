#!/usr/bin/env bash
set -euo pipefail

CFG_NAME="nuscenes_gs25600_v16_ft_plan_frozen_occ_truncated_ddim"
EXP_NAME="${EXP_NAME:-${CFG_NAME}}"
REPO="/data/xinyao/navsim_workspace/GaussianAD"
PYTHON="/data/chenz/conda_env/splatting/bin/python"
CONFIG="config/${CFG_NAME}/${CFG_NAME}.py"
WORK_DIR="exp/${EXP_NAME}"
BASE_INIT="exp/nuscenes_gs25600_v3_se3_ft_plan_futattn_global_residual_full/checkpoints/epoch_16.pth"
export STRONG_OCC_CHECKPOINT="${STRONG_OCC_CHECKPOINT:-${BASE_INIT}}"

GPUS="${GPUS:-0,1,2,3}"
MASTER_PORT="${MASTER_PORT:-21916}"
EPOCH="${EPOCH:-20}"
CKPT="${CKPT:-${WORK_DIR}/checkpoints/epoch_${EPOCH}.pth}"
LOG_NAME="${LOG_NAME:-test_epoch_${EPOCH}_protocol_vad}"

cd "${REPO}"
if [[ ! -f "${CKPT}" ]]; then
  echo "[FATAL] checkpoint not found: ${CKPT}" >&2
  exit 2
fi
"${PYTHON}" "config/${CFG_NAME}/validate_v16.py" \
  --checkpoint "${STRONG_OCC_CHECKPOINT}"

mkdir -p "${WORK_DIR}"
{
  echo "=== frozen-OCC v16 test $(date '+%Y-%m-%d %H:%M:%S') ==="
  echo "config     : ${CONFIG}"
  echo "checkpoint : ${CKPT}"
  echo "metric     : GaussianAD/VAD cumulative-position protocol"
  echo "gpus       : ${GPUS} (port=${MASTER_PORT})"
  echo "git commit : $(git rev-parse HEAD)"
} | tee -a "${WORK_DIR}/launch_history.log"

CUDA_VISIBLE_DEVICES="${GPUS}" MASTER_PORT="${MASTER_PORT}" "${PYTHON}" \
  test.py \
  --py-config "${CONFIG}" \
  --work-dir "${WORK_DIR}" \
  --resume-from "${CKPT}" \
  --log-name "${LOG_NAME}" \
  2>&1 | tee -a "${WORK_DIR}/${LOG_NAME}_console.log"
