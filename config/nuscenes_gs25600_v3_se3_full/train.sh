#!/usr/bin/env bash
set -euo pipefail

CFG_NAME="nuscenes_gs25600_v3_se3_full"
EXP_NAME="${CFG_NAME}"
REPO="/data/xinyao/navsim_workspace/GaussianAD"
TORCHRUN="/data/chenz/conda_env/splatting/bin/torchrun"
PYTHON="/data/chenz/conda_env/splatting/bin/python"
CONFIG="config/${CFG_NAME}/${CFG_NAME}.py"
WORK_DIR="exp/${EXP_NAME}"

# Accept both the standard variables and the PET_* variables injected by the
# cloud platform. This Stage-1 configuration is intentionally fixed to one
# eight-GPU node: batch_size=1 per GPU and global_batch=8.
NNODES="${NNODES:-${PET_NNODES:-1}}"
NODE_RANK="${NODE_RANK:-${PET_NODE_RANK:-0}}"
NPROC="${NPROC:-${NPROC_PER_NODE:-${PET_NPROC_PER_NODE:-8}}}"
MASTER_ADDR="${MASTER_ADDR:-${PET_MASTER_ADDR:-127.0.0.1}}"
MASTER_PORT="${MASTER_PORT:-${PET_MASTER_PORT:-12874}}"
export LR="${LR:-2e-4}"

if [[ "${NNODES}" != "1" || "${NODE_RANK}" != "0" || "${NPROC}" != "8" ]]; then
  echo "[FATAL] expected one 8-GPU node: NNODES=${NNODES}, NODE_RANK=${NODE_RANK}, NPROC=${NPROC}" >&2
  exit 2
fi

GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((NPROC - 1)))}}"
IFS=',' read -r -a GPU_IDS <<< "${GPUS}"
if (( ${#GPU_IDS[@]} != 8 )); then
  echo "[FATAL] expected 8 GPU ids: GPUS=${GPUS}" >&2
  exit 2
fi

cd "${REPO}"
mkdir -p "${WORK_DIR}"

VISIBLE_GPUS=$(CUDA_VISIBLE_DEVICES="${GPUS}" "${PYTHON}" -c 'import torch; print(torch.cuda.device_count())')
if [[ "${VISIBLE_GPUS}" != "8" ]]; then
  echo "[FATAL] PyTorch sees ${VISIBLE_GPUS} GPUs, but this experiment requires 8" >&2
  exit 2
fi

# torchrun owns these process-level variables; stale platform values can break
# local rank assignment when the launcher creates its eight workers.
unset RANK WORLD_SIZE LOCAL_RANK

{
  echo "=== Stage-1 full-data launch $(date '+%Y-%m-%d %H:%M:%S') ==="
  echo "experiment : ${EXP_NAME}"
  echo "config     : ${CONFIG}"
  echo "data       : full nuScenes train + full val"
  echo "schedule   : 20 epochs, evaluate every 4 epochs"
  echo "model      : V3-SE3 perception/Future OCC, no Planner"
  echo "batch      : per_gpu=1, global=8"
  echo "lr         : ${LR}"
  echo "distributed: nnodes=${NNODES}, node_rank=${NODE_RANK}, nproc=${NPROC}"
  echo "rendezvous : ${MASTER_ADDR}:${MASTER_PORT}"
  echo "gpus       : ${GPUS}"
  echo "git branch : $(git rev-parse --abbrev-ref HEAD)"
  echo "git commit : $(git rev-parse HEAD)"
} | tee -a "${WORK_DIR}/launch_history.log"

CUDA_VISIBLE_DEVICES="${GPUS}" PYTHONUNBUFFERED=1 "${TORCHRUN}" \
  --nnodes "${NNODES}" \
  --node_rank "${NODE_RANK}" \
  --nproc_per_node "${NPROC}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  train.py \
  --py-config "${CONFIG}" \
  --work-dir "${WORK_DIR}" \
  --dataset nuscenes \
  2>&1 | tee -a "${WORK_DIR}/train_run.log"
