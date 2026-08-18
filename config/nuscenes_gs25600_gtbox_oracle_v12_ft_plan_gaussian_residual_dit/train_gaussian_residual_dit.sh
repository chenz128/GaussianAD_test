#!/usr/bin/env bash
set -euo pipefail

SRC_CFG="nuscenes_gs25600_gtbox_oracle_v12_ft_plan_gaussian_residual_dit"
EXP_NAME="nuscenes_gs25600_v12_fixempty_ft_plan_gaussian_residual_dit"
REPO="/data/xinyao/navsim_workspace/GaussianAD"
PY="/data/chenz/conda_env/splatting/bin/torchrun"
CONFIG="config/${SRC_CFG}/${SRC_CFG}.py"
WORK_DIR="exp/${EXP_NAME}"
GPUS="${GPUS:-0,1,2,3}"
NPROC="${NPROC:-4}"
MASTER_PORT="${MASTER_PORT:-12589}"

cd "${REPO}"
mkdir -p "${WORK_DIR}"

{
  echo "=== launch $(date '+%Y-%m-%d %H:%M:%S') ==="
  echo "exp        : ${EXP_NAME}"
  echo "git branch : $(git rev-parse --abbrev-ref HEAD)"
  echo "git commit : $(git rev-parse HEAD)"
  echo "config     : ${CONFIG}"
  echo "init from  : exp/nuscenes_gs25600_v12_fixempty_ft_plan_dualtime_residual/checkpoints/epoch_15.pth"
  echo "planner    : dualtime chain + deeper DiT(4L, adaLN-Zero) + 10-step DDIM train / 8-step infer"
  echo "condition  : agent/map/current Gaussian/future Gaussian"
  echo "collision  : disabled (architecture-only comparison)"
  echo "gpus       : ${GPUS} (nproc=${NPROC}, port=${MASTER_PORT})"
} | tee -a "${WORK_DIR}/launch_history.log"

CUDA_VISIBLE_DEVICES="${GPUS}" "${PY}" \
    --nproc_per_node "${NPROC}" \
    --master_port "${MASTER_PORT}" \
    train.py \
    --py-config "${CONFIG}" \
    --work-dir "${WORK_DIR}" \
    --dataset nuscenes \
    2>&1 | tee -a "${WORK_DIR}/train_run.log"
