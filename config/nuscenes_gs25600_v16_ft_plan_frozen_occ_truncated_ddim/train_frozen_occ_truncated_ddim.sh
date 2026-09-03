#!/usr/bin/env bash
set -euo pipefail

CFG_NAME="nuscenes_gs25600_v16_ft_plan_frozen_occ_truncated_ddim"
EXP_NAME="${EXP_NAME:-${CFG_NAME}}"
REPO="/data/xinyao/navsim_workspace/GaussianAD"
TORCHRUN="/data/chenz/conda_env/splatting/bin/torchrun"
PYTHON="/data/chenz/conda_env/splatting/bin/python"
CONFIG="config/${CFG_NAME}/${CFG_NAME}.py"
WORK_DIR="exp/${EXP_NAME}"
DEFAULT_INIT="exp/nuscenes_gs25600_v3_se3_ft_plan_futattn_global_residual_full/checkpoints/epoch_16.pth"
export STRONG_OCC_CHECKPOINT="${STRONG_OCC_CHECKPOINT:-${DEFAULT_INIT}}"

NPROC="${NPROC:-${NPROC_PER_NODE:-8}}"
GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((NPROC - 1)))}}"
MASTER_PORT="${MASTER_PORT:-12916}"
PER_GPU_BATCH=1
GLOBAL_BATCH=$((NPROC * PER_GPU_BATCH))

if [[ "${NNODES:-1}" != "1" ]]; then
  echo "[FATAL] this launcher supports one node only" >&2
  exit 2
fi
IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
if (( ${#GPU_LIST[@]} != NPROC )); then
  echo "[FATAL] GPUS=${GPUS} has ${#GPU_LIST[@]} entries; expected ${NPROC}" >&2
  exit 2
fi

cd "${REPO}"
if [[ ! -f "${STRONG_OCC_CHECKPOINT}" ]]; then
  echo "[FATAL] audited strong-OCC checkpoint is missing:" >&2
  echo "        ${STRONG_OCC_CHECKPOINT}" >&2
  exit 2
fi
if [[ -e "${WORK_DIR}/latest.pth" ]] || \
   compgen -G "${WORK_DIR}/checkpoints/*.pth" >/dev/null; then
  echo "[FATAL] ${WORK_DIR} already contains a training checkpoint." >&2
  echo "        train.py may resume state or overwrite an existing epoch." >&2
  echo "        Use a fresh EXP_NAME/work directory for a clean ablation." >&2
  exit 2
fi

# Preserve the strong baseline's full-data schedule and optimizer settings.
export LR="${LR:-2e-4}"
export AUX_W="${AUX_W:-2.0}"
export COL_W="${COL_W:-0.1}"
export PLAN_DIRECT_BUDGET="${PLAN_DIRECT_BUDGET:-6400}"
export PLAN_FUTURE_GRAD_SCALE="0.0"
export DDIM_STEPS="${DDIM_STEPS:-2}"
export DDIM_SAMPLES="${DDIM_SAMPLES:-4}"
export DDIM_START_T="${DDIM_START_T:-0.25}"
export GAUSSIAN_TOPK="${GAUSSIAN_TOPK:-128}"

"${PYTHON}" \
  "config/${CFG_NAME}/validate_v16.py" \
  --checkpoint "${STRONG_OCC_CHECKPOINT}" \
  --build-model

if [[ "${VALIDATE_ONLY:-0}" == "1" ]]; then
  echo "[OK] VALIDATE_ONLY=1; launch contract passed, training not started."
  exit 0
fi

mkdir -p "${WORK_DIR}"
{
  echo "=== frozen-OCC v16 launch $(date '+%Y-%m-%d %H:%M:%S') ==="
  echo "experiment : ${EXP_NAME}"
  echo "config     : ${CONFIG}"
  echo "load_from  : ${STRONG_OCC_CHECKPOINT} (weights only)"
  echo "frontend   : fully frozen; deterministic planner anchor frozen"
  echo "schedule   : inherited full baseline (20 epochs, full train/val)"
  echo "batch      : per_gpu=${PER_GPU_BATCH}, global=${GLOBAL_BATCH}"
  echo "lr         : ${LR}"
  echo "diffusion  : residual truncated DDIM, K=${DDIM_SAMPLES}, NFE=${DDIM_STEPS}, t0=${DDIM_START_T}"
  echo "selector   : main/per-frame/global anchors + OCC-safe proposals"
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
