#!/usr/bin/env bash
set -Eeuo pipefail

REPO="${REPO:-/data/xinyao/navsim_workspace/GaussianAD}"
PYTHON="${PYTHON:-/data/chenz/conda_env/splatting/bin/python}"
TORCHRUN="${TORCHRUN:-/data/chenz/conda_env/splatting/bin/torchrun}"
CONFIG="config/nuscenes_gs25600_gtbox_oracle_v14_ft_plan_residual_ddim/nuscenes_gs25600_gtbox_oracle_v14_ft_plan_residual_ddim.py"
BASELINE_CHECKPOINT="exp/nuscenes_gs25600_v12_fixempty_ft_plan_futattn_global_residual/checkpoints/epoch_15.pth"
WORK_DIR="${WORK_DIR:-exp/nuscenes_gs25600_v14_ft_plan_residual_ddim}"
GPUS="${GPUS:-0,1,2,3}"
VALIDATE_FIRST="${VALIDATE_FIRST:-1}"
DRY_RUN="${DRY_RUN:-0}"

# The audited server checkpoint is the default. An explicit environment value
# is still supported for a copied/renamed artifact, and is exported because
# both the config parser and validator read it in child processes.
VERIFIED_V12_CHECKPOINT="${VERIFIED_V12_CHECKPOINT:-${REPO}/${BASELINE_CHECKPOINT}}"
export VERIFIED_V12_CHECKPOINT

if [[ ! -d "${REPO}" ]]; then
  echo "Repository does not exist: ${REPO}" >&2
  exit 2
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "Python is not executable: ${PYTHON}" >&2
  exit 2
fi
if [[ ! -x "${TORCHRUN}" ]]; then
  echo "torchrun is not executable: ${TORCHRUN}" >&2
  exit 2
fi
if [[ ! -f "${VERIFIED_V12_CHECKPOINT}" ]]; then
  echo "Audited baseline checkpoint does not exist: ${VERIFIED_V12_CHECKPOINT}" >&2
  exit 2
fi

IFS=',' read -r -a V14_GPU_IDS <<< "${GPUS}"
GPU_COUNT="${#V14_GPU_IDS[@]}"
NPROC="${NPROC:-${GPU_COUNT}}"
if [[ ! "${NPROC}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NPROC must be a positive integer: ${NPROC}" >&2
  exit 2
fi
if [[ "${NPROC}" -ne "${GPU_COUNT}" ]]; then
  echo "NPROC=${NPROC} does not match GPUS=${GPUS} (${GPU_COUNT} devices)." >&2
  exit 2
fi
for V14_GPU_ID in "${V14_GPU_IDS[@]}"; do
  if [[ ! "${V14_GPU_ID}" =~ ^[0-9]+$ ]]; then
    echo "GPUS must be comma-separated numeric device ids: ${GPUS}" >&2
    exit 2
  fi
done

for V14_ARG in "$@"; do
  case "${V14_ARG}" in
    --resume-from|--resume-from=*|--iter-resume)
      echo "v14 is weights-only; --resume-from is forbidden. Use a fresh WORK_DIR." >&2
      exit 3
      ;;
    --work-dir|--work-dir=*|--py-config|--py-config=*|--dataset|--dataset=*)
      echo "Do not override config/work-dir/dataset through extra arguments." >&2
      echo "Use WORK_DIR for a fresh v14 output directory." >&2
      exit 3
      ;;
  esac
done

cd "${REPO}"
if [[ ! -f "${CONFIG}" ]]; then
  echo "Config does not exist: ${REPO}/${CONFIG}" >&2
  exit 2
fi
if [[ -e "${WORK_DIR}/latest.pth" || -d "${WORK_DIR}/checkpoints" ]]; then
  echo "Refusing a used work directory: ${WORK_DIR}" >&2
  echo "Choose a fresh WORK_DIR so optimizer/scheduler state cannot resume." >&2
  exit 3
fi

if [[ "${VALIDATE_FIRST}" == "1" ]]; then
  "${PYTHON}" \
    config/nuscenes_gs25600_gtbox_oracle_v14_ft_plan_residual_ddim/validate_v14.py
elif [[ "${VALIDATE_FIRST}" != "0" ]]; then
  echo "VALIDATE_FIRST must be 0 or 1: ${VALIDATE_FIRST}" >&2
  exit 2
fi

TORCHRUN_ARGS=(--nproc_per_node "${NPROC}")
if [[ -n "${MASTER_PORT:-}" ]]; then
  if [[ ! "${MASTER_PORT}" =~ ^[0-9]+$ \
        || "${MASTER_PORT}" -lt 1 || "${MASTER_PORT}" -gt 65535 ]]; then
    echo "MASTER_PORT must be an integer in [1, 65535]: ${MASTER_PORT}" >&2
    exit 2
  fi
  TORCHRUN_ARGS+=(--master_port "${MASTER_PORT}")
  RENDEZVOUS="port=${MASTER_PORT}"
else
  # Single-node standalone mode asks torchrun for an available rendezvous port,
  # avoiding collisions between concurrent experiments on this 8-GPU host.
  TORCHRUN_ARGS+=(--standalone --nnodes 1)
  RENDEZVOUS="standalone(auto-port)"
fi

TRAIN_ARGS=(
  train.py
  --py-config "${CONFIG}"
  --work-dir "${WORK_DIR}"
  --dataset nuscenes
)
TRAIN_ARGS+=("$@")

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'CUDA_VISIBLE_DEVICES=%q PYTHONUNBUFFERED=1 ' "${GPUS}"
  printf '%q ' "${TORCHRUN}" "${TORCHRUN_ARGS[@]}" "${TRAIN_ARGS[@]}"
  printf '\n'
  exit 0
elif [[ "${DRY_RUN}" != "0" ]]; then
  echo "DRY_RUN must be 0 or 1: ${DRY_RUN}" >&2
  exit 2
fi

mkdir -p "${WORK_DIR}"
{
  echo "=== v14 residual DDIM launch $(date '+%Y-%m-%d %H:%M:%S') ==="
  echo "git branch : $(git rev-parse --abbrev-ref HEAD)"
  echo "git commit : $(git rev-parse HEAD)"
  echo "config     : ${CONFIG}"
  echo "baseline   : v12_ft_plan_futattn_global_residual"
  echo "init       : ${VERIFIED_V12_CHECKPOINT} (weights only)"
  echo "work dir   : ${WORK_DIR}"
  echo "schedule   : cosine zero-terminal-SNR"
  echo "DDIM       : K=4, NFE=${DDIM_STEPS:-4}, fixed noise, no CFG"
  echo "Gaussian K : ${GAUSSIAN_TOPK:-128}"
  echo "gpus       : ${GPUS} (nproc=${NPROC}, ${RENDEZVOUS})"
  if [[ -z "${RESIDUAL_SCALE:-}" ]]; then
    echo "res scale  : identity (smoke only; set train-split robust statistics)"
  else
    echo "res scale  : provided by RESIDUAL_SCALE"
  fi
} | tee -a "${WORK_DIR}/launch_history.log"

CUDA_VISIBLE_DEVICES="${GPUS}" PYTHONUNBUFFERED=1 "${TORCHRUN}" \
  "${TORCHRUN_ARGS[@]}" \
  "${TRAIN_ARGS[@]}" \
  2>&1 | tee -a "${WORK_DIR}/train_run.log"
