#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${VERIFIED_V12_CHECKPOINT:-}" ]]; then
  echo "Set VERIFIED_V12_CHECKPOINT to the audited v12 weights." >&2
  exit 2
fi
if [[ ! -f "${VERIFIED_V12_CHECKPOINT}" ]]; then
  echo "Checkpoint does not exist: ${VERIFIED_V12_CHECKPOINT}" >&2
  exit 2
fi

REPO="${REPO:-/data/xinyao/navsim_workspace/GaussianAD}"
PYTHON="${PYTHON:-/data/chenz/conda_env/splatting/bin/python}"
TORCHRUN="${TORCHRUN:-/data/chenz/conda_env/splatting/bin/torchrun}"
CONFIG="config/nuscenes_gs25600_gtbox_oracle_v14_ft_plan_residual_ddim/nuscenes_gs25600_gtbox_oracle_v14_ft_plan_residual_ddim.py"
WORK_DIR="${WORK_DIR:-exp/nuscenes_gs25600_v14_ft_plan_residual_ddim}"
GPUS="${GPUS:-0,1,2,3}"
NPROC="${NPROC:-4}"
MASTER_PORT="${MASTER_PORT:-12741}"

cd "${REPO}"
if [[ -e "${WORK_DIR}/latest.pth" ]]; then
  echo "Refusing to resume ${WORK_DIR}; choose a fresh WORK_DIR." >&2
  exit 3
fi
mkdir -p "${WORK_DIR}"

if [[ "${VALIDATE_FIRST:-1}" == "1" ]]; then
  "${PYTHON}" \
    config/nuscenes_gs25600_gtbox_oracle_v14_ft_plan_residual_ddim/validate_v14.py
fi

{
  echo "=== v14 residual DDIM launch $(date '+%Y-%m-%d %H:%M:%S') ==="
  echo "git branch : $(git rev-parse --abbrev-ref HEAD)"
  echo "git commit : $(git rev-parse HEAD)"
  echo "config     : ${CONFIG}"
  echo "baseline   : v12_ft_plan_futattn_global_residual"
  echo "init       : ${VERIFIED_V12_CHECKPOINT} (weights only)"
  echo "work dir   : ${WORK_DIR}"
  echo "sigma/topk : ${SIGMA_MAX:-0.5} / ${GAUSSIAN_TOPK:-128}"
  echo "DDIM       : K=4, NFE=2, fixed noise, no CFG"
  echo "gpus       : ${GPUS} (nproc=${NPROC}, port=${MASTER_PORT})"
} | tee -a "${WORK_DIR}/launch_history.log"

CUDA_VISIBLE_DEVICES="${GPUS}" "${TORCHRUN}" \
  --nproc_per_node "${NPROC}" \
  --master_port "${MASTER_PORT}" \
  train.py \
  --py-config "${CONFIG}" \
  --work-dir "${WORK_DIR}" \
  --dataset nuscenes "$@" \
  2>&1 | tee -a "${WORK_DIR}/train_run.log"
