#!/usr/bin/env bash
set -euo pipefail

SRC_CFG="nuscenes_gs25600_gtbox_oracle_v13_ft_plan_riskaware_global_residual"
EXP_NAME="nuscenes_gs25600_v12_fixempty_ft_plan_riskaware_global_residual"
REPO="/data/xinyao/navsim_workspace/GaussianAD"
PY="/data/chenz/conda_env/splatting/bin/torchrun"
CONFIG="config/${SRC_CFG}/${SRC_CFG}.py"
WORK_DIR="exp/${EXP_NAME}"
GPUS="${GPUS:-0,1,2,3}"
NPROC="${NPROC:-4}"
MASTER_PORT="${MASTER_PORT:-12733}"

cd "${REPO}"
mkdir -p "${WORK_DIR}"

{
  echo "=== launch $(date '+%Y-%m-%d %H:%M:%S') ==="
  echo "exp        : ${EXP_NAME}"
  echo "git branch : $(git rev-parse --abbrev-ref HEAD)"
  echo "git commit : $(git rev-parse HEAD)"
  echo "config     : ${CONFIG}"
  echo "init from  : v12_fixempty epoch 15 (planner from scratch, same as futattn/timequery)"
  echo "planner    : v13 mode-specific Gaussian risk-aware global residual"
  echo "risk       : top-k=32, margin=0.5, uncertainty_growth=0.15"
  echo "plan grad  : ${PLANNER_GRAD_SCALE:-1.0} (1.0 keeps v12 baseline behavior)"
  echo "loss       : v12 losses + gate rank(${GATE_W:-0.1}) + hard SAT(${COL_W:-0.1})"
  echo "epochs     : 15"
  echo "lr         : ${LR:-2e-4}"
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
