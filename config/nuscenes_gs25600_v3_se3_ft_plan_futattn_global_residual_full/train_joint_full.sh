#!/usr/bin/env bash
set -euo pipefail

CFG_NAME="nuscenes_gs25600_v3_se3_ft_plan_futattn_global_residual_full"
EXP_NAME="${CFG_NAME}"
REPO="/data/xinyao/navsim_workspace/GaussianAD"
TORCHRUN="/data/chenz/conda_env/splatting/bin/torchrun"
CONFIG="config/${CFG_NAME}/${CFG_NAME}.py"
WORK_DIR="exp/${EXP_NAME}"

# V3-SE3 requires batch_size=1 per GPU.  Eight DDP workers therefore give
# global_batch=8 without changing the model-side batch dimension.
NPROC="${NPROC:-${NPROC_PER_NODE:-8}}"
GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((NPROC - 1)))}}"
MASTER_PORT="${MASTER_PORT:-12873}"
PER_GPU_BATCH=1
GLOBAL_BATCH=$((NPROC * PER_GPU_BATCH))

if [[ "${NNODES:-1}" != "1" ]]; then
  echo "[FATAL] train_joint_full.sh supports single-node training only" >&2
  exit 2
fi
IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
if (( ${#GPU_LIST[@]} != NPROC )); then
  echo "[FATAL] GPUS=${GPUS} contains ${#GPU_LIST[@]} devices, NPROC=${NPROC}" >&2
  exit 2
fi

# Match the v12_full eight-GPU AdamW reference.  Override LR explicitly only
# for a separate learning-rate ablation.
export LR="${LR:-2e-4}"
export AUX_W="${AUX_W:-2.0}"
export COL_W="${COL_W:-0.1}"

cd "${REPO}"
mkdir -p "${WORK_DIR}"

{
  echo "=== joint full-data launch $(date '+%Y-%m-%d %H:%M:%S') ==="
  echo "experiment : ${EXP_NAME}"
  echo "config     : ${CONFIG}"
  echo "init       : R101 backbone pretrain; joint OCC/map/planner from epoch 0"
  echo "data       : full train + full val"
  echo "schedule   : 20 epochs, evaluate every 4 epochs"
  echo "batch      : per_gpu=${PER_GPU_BATCH}, global=${GLOBAL_BATCH}"
  echo "lr         : ${LR} (v12_full 8-GPU reference)"
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
