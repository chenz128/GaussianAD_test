#!/usr/bin/env bash
set -euo pipefail

CFG="nuscenes_gs25600_v3_se3_epoch20_ft_plan_vad_full"
REPO="/data/xinyao/navsim_workspace/GaussianAD"
PYTHON="/data/chenz/conda_env/splatting/bin/python"
TORCHRUN="/data/chenz/conda_env/splatting/bin/torchrun"
CONFIG="config/${CFG}/${CFG}.py"
WORK_DIR="exp/${CFG}"
SOURCE_CKPT="${SOURCE_CKPT:-exp/nuscenes_gs25600_v3_se3_ft_plan_futattn_global_residual_full/checkpoints/epoch_20.pth}"
FRONTEND_CKPT="${FRONTEND_CKPT:-${WORK_DIR}/bootstrap/epoch_20_frontend_only.pth}"

GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
IFS=',' read -r -a GPU_IDS <<< "${GPUS}"
NPROC="${NPROC:-${NPROC_PER_NODE:-${#GPU_IDS[@]}}}"
MASTER_PORT="${MASTER_PORT:-12875}"
MAX_EPOCHS="${MAX_EPOCHS:-20}"
EVAL_EVERY_EPOCHS="${EVAL_EVERY_EPOCHS:-1}"

if (( ${#GPU_IDS[@]} != NPROC )); then
  echo "[FATAL] GPUS=${GPUS} has ${#GPU_IDS[@]} devices, NPROC=${NPROC}" >&2
  exit 2
fi
if [[ -z "${LR:-}" ]]; then
  case "${NPROC}" in
    8) LR="2e-4" ;;
    4) LR="1e-4" ;;
    2) LR="5e-5" ;;
    1) LR="2.5e-5" ;;
    *) echo "[FATAL] set LR explicitly for NPROC=${NPROC}" >&2; exit 2 ;;
  esac
fi

cd "${REPO}"
mkdir -p "${WORK_DIR}/bootstrap"
if [[ ! -f "${FRONTEND_CKPT}" ]]; then
  "${PYTHON}" "config/${CFG}/prepare_frontend_checkpoint.py" \
    --input "${SOURCE_CKPT}" \
    --output "${FRONTEND_CKPT}"
fi

export FRONTEND_CKPT LR MAX_EPOCHS EVAL_EVERY_EPOCHS
echo "[Stage-2 VAD] full nuScenes, frozen epoch20 frontend, epochs=${MAX_EPOCHS}, GPUs=${GPUS}, LR=${LR}" |
  tee -a "${WORK_DIR}/launch_history.log"

CUDA_VISIBLE_DEVICES="${GPUS}" "${TORCHRUN}" \
  --nproc_per_node "${NPROC}" \
  --master_port "${MASTER_PORT}" \
  train.py \
  --py-config "${CONFIG}" \
  --work-dir "${WORK_DIR}" \
  --dataset nuscenes \
  2>&1 | tee -a "${WORK_DIR}/train_run.log"
