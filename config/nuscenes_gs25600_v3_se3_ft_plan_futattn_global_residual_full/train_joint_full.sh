#!/usr/bin/env bash
set -euo pipefail

CFG_NAME="nuscenes_gs25600_v3_se3_ft_plan_futattn_global_residual_full"
EXP_NAME="${CFG_NAME}"
REPO="/data/xinyao/navsim_workspace/GaussianAD"
TORCHRUN="/data/chenz/conda_env/splatting/bin/torchrun"
CONFIG="config/${CFG_NAME}/${CFG_NAME}.py"
WORK_DIR="exp/${EXP_NAME}"

# Require an explicit GPU selection so this isolated experiment cannot
# accidentally occupy cards used by another job.
: "${GPUS:?Set GPUS explicitly, for example GPUS=0,1,2,3}"
NPROC="${NPROC:-4}"
MASTER_PORT="${MASTER_PORT:-12873}"
export LR="${LR:-2e-4}"
export AUX_W="${AUX_W:-2.0}"
export COL_W="${COL_W:-0.1}"

cd "${REPO}"
mkdir -p "${WORK_DIR}"

{
  echo "=== joint full-data launch $(date '+%Y-%m-%d %H:%M:%S') ==="
  echo "experiment : ${EXP_NAME}"
  echo "config     : ${CONFIG}"
  echo "init       : V3-SE3 epoch15 frontend + planner epoch15 map/planner"
  echo "data       : full train + full val"
  echo "schedule   : 20 epochs, evaluate every 4 epochs"
  echo "lr         : ${LR}"
  echo "loss       : V3 losses + map/plan + aux(${AUX_W}) + SAT col(${COL_W})"
  echo "gpus       : ${GPUS} (nproc=${NPROC}, port=${MASTER_PORT})"
  echo "git branch : $(git rev-parse --abbrev-ref HEAD)"
  echo "git commit : $(git rev-parse HEAD)"
} | tee -a "${WORK_DIR}/launch_history.log"

CUDA_VISIBLE_DEVICES="${GPUS}" "${TORCHRUN}" \
  --nproc_per_node "${NPROC}" \
  --master_port "${MASTER_PORT}" \
  train.py \
  --py-config "${CONFIG}" \
  --work-dir "${WORK_DIR}" \
  --dataset nuscenes \
  2>&1 | tee -a "${WORK_DIR}/train_run.log"
