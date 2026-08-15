#!/usr/bin/env bash
set -euo pipefail

SRC_CFG="nuscenes_gs25600_gtbox_oracle_v12_ft_plan_dualtime_residual"
EXP_NAME="nuscenes_gs25600_v12_fixempty_ft_plan_dualtime_residual"
REPO="/data/xinyao/navsim_workspace/GaussianAD"
PY="/data/chenz/conda_env/splatting/bin/torchrun"
CONFIG="config/nuscenes_gs25600_gtbox_oracle_v12_ft_plan_dualtime_residual/${SRC_CFG}.py"
WORK_DIR="exp/${EXP_NAME}"
GPUS="${GPUS:-4,5,6,7}"
NPROC="${NPROC:-4}"
MASTER_PORT="${MASTER_PORT:-12489}"

cd "${REPO}"
mkdir -p "${WORK_DIR}"

{
  echo "=== launch $(date '+%Y-%m-%d %H:%M:%S') ==="
  echo "exp        : ${EXP_NAME}"
  echo "git branch : $(git rev-parse --abbrev-ref HEAD)"
  echo "git commit : $(git rev-parse HEAD)"
  echo "config     : ${CONFIG}"
  echo "init from  : exp/nuscenes_gs25600_v12_fixempty/checkpoints/epoch_15.pth"
  echo "delta      : exact costime K/V + per-frame query + relative-time residual gate"
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
