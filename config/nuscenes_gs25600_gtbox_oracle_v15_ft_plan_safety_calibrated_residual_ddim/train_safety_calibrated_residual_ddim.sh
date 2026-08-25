#!/usr/bin/env bash
set -Eeuo pipefail

REPO="${REPO:-/data/xinyao/navsim_workspace/GaussianAD}"
PYTHON="${PYTHON:-/data/chenz/conda_env/splatting/bin/python}"
CONFIG="config/nuscenes_gs25600_gtbox_oracle_v15_ft_plan_safety_calibrated_residual_ddim/nuscenes_gs25600_gtbox_oracle_v15_ft_plan_safety_calibrated_residual_ddim.py"
V12_CHECKPOINT="exp/nuscenes_gs25600_v12_fixempty_ft_plan_futattn_global_residual/checkpoints/epoch_15.pth"
WORK_DIR="${WORK_DIR:-exp/nuscenes_gs25600_v15b_ft_plan_collision_guarded_residual_ddim}"
# Server3's current experiment allocation uses the upper four GPUs. Override
# GPUS explicitly if the scheduler assigns a different set.
GPUS="${GPUS:-4,5,6,7}"
VALIDATE_FIRST="${VALIDATE_FIRST:-1}"
DRY_RUN="${DRY_RUN:-0}"

VERIFIED_V12_CHECKPOINT="${VERIFIED_V12_CHECKPOINT:-${REPO}/${V12_CHECKPOINT}}"
export VERIFIED_V12_CHECKPOINT

if [[ ! -d "${REPO}" ]]; then
  echo "Repository does not exist: ${REPO}" >&2
  exit 2
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "Python is not executable: ${PYTHON}" >&2
  exit 2
fi
if [[ "${VERIFIED_V12_CHECKPOINT}" != /* ]]; then
  echo "VERIFIED_V12_CHECKPOINT must be absolute: ${VERIFIED_V12_CHECKPOINT}" >&2
  exit 2
fi
if [[ ! -f "${VERIFIED_V12_CHECKPOINT}" ]]; then
  echo "Audited v12-fixempty checkpoint does not exist: ${VERIFIED_V12_CHECKPOINT}" >&2
  exit 2
fi
case "${VERIFIED_V12_CHECKPOINT,,}" in
  */exp/nuscenes_gs25600_v12_fixempty_ft_plan_futattn_global_residual/checkpoints/epoch_15.pth)
    ;;
  *)
    echo "Invalid continuation checkpoint: ${VERIFIED_V12_CHECKPOINT}" >&2
    echo "v15 requires the exact v12-fixempty epoch-15 baseline." >&2
    exit 3
    ;;
esac

IFS=',' read -r -a V15_GPU_IDS <<< "${GPUS}"
GPU_COUNT="${#V15_GPU_IDS[@]}"
NPROC="${NPROC:-${GPU_COUNT}}"
if [[ ! "${NPROC}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NPROC must be a positive integer: ${NPROC}" >&2
  exit 2
fi
if [[ "${NPROC}" -ne "${GPU_COUNT}" ]]; then
  echo "NPROC=${NPROC} does not match GPUS=${GPUS} (${GPU_COUNT} devices)." >&2
  exit 2
fi
for V15_GPU_ID in "${V15_GPU_IDS[@]}"; do
  if [[ ! "${V15_GPU_ID}" =~ ^[0-9]+$ ]]; then
    echo "GPUS must be comma-separated numeric device ids: ${GPUS}" >&2
    exit 2
  fi
done

for V15_ARG in "$@"; do
  case "${V15_ARG}" in
    --resume-from|--resume-from=*|--iter-resume)
      echo "v15 is weights-only; optimizer/scheduler resume is forbidden." >&2
      exit 3
      ;;
    --work-dir|--work-dir=*|--py-config|--py-config=*|--dataset|--dataset=*)
      echo "Do not override config/work-dir/dataset through extra arguments." >&2
      echo "Use WORK_DIR for a fresh v15 output directory." >&2
      exit 3
      ;;
  esac
done

cd "${REPO}"
if [[ ! -f "${CONFIG}" ]]; then
  echo "Config does not exist: ${REPO}/${CONFIG}" >&2
  exit 2
fi
if [[ "${WORK_DIR}" == "exp/nuscenes_gs25600_v14_ft_plan_residual_ddim" ]]; then
  echo "Refusing to overwrite the v14 experiment directory." >&2
  exit 3
fi
if [[ -e "${WORK_DIR}" ]]; then
  echo "Refusing an existing work directory: ${WORK_DIR}" >&2
  echo "Choose a new WORK_DIR; this launcher never resumes or reuses a run." >&2
  exit 3
fi

if [[ "${VALIDATE_FIRST}" == "1" ]]; then
  "${PYTHON}" \
    config/nuscenes_gs25600_gtbox_oracle_v15_ft_plan_safety_calibrated_residual_ddim/validate_v15.py
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
  printf '%q ' "${PYTHON}" -m torch.distributed.run \
    "${TORCHRUN_ARGS[@]}" "${TRAIN_ARGS[@]}"
  printf '\n'
  exit 0
elif [[ "${DRY_RUN}" != "0" ]]; then
  echo "DRY_RUN must be 0 or 1: ${DRY_RUN}" >&2
  exit 2
fi

mkdir -p "${WORK_DIR}"
{
  echo "=== v15b collision-guarded residual DDIM launch $(date '+%Y-%m-%d %H:%M:%S') ==="
  echo "git branch : $(git rev-parse --abbrev-ref HEAD)"
  echo "git commit : $(git rev-parse HEAD)"
  echo "config     : ${CONFIG}"
  echo "python     : ${PYTHON}"
  echo "generator  : v14 residual DDIM + inference-matched five-candidate calibration"
  echo "init       : ${VERIFIED_V12_CHECKPOINT} (v12-fixempty, weights only)"
  echo "work dir   : ${WORK_DIR}"
  echo "DDIM       : K=4, NFE=${DDIM_STEPS:-4}, fixed noise, no CFG"
  echo "safety     : v14 hard guard + Gaussian no-regression + vehicle/human SAT veto"
  echo "thresholds : unsafe=${SAFETY_PROB_THRESHOLD:-0.60}, safe=${SAFETY_SAFE_PROB_THRESHOLD:-0.30}"
  echo "Gaussian K : ${GAUSSIAN_TOPK:-128}"
  echo "gpus       : ${GPUS} (nproc=${NPROC}, ${RENDEZVOUS})"
} | tee -a "${WORK_DIR}/launch_history.log"

CUDA_VISIBLE_DEVICES="${GPUS}" PYTHONUNBUFFERED=1 "${PYTHON}" \
  -m torch.distributed.run \
  "${TORCHRUN_ARGS[@]}" \
  "${TRAIN_ARGS[@]}" \
  2>&1 | tee -a "${WORK_DIR}/train_run.log"
