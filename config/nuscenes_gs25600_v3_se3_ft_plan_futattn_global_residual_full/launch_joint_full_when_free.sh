#!/usr/bin/env bash
set -euo pipefail

CFG_NAME="nuscenes_gs25600_v3_se3_ft_plan_futattn_global_residual_full"
REPO="/data/xinyao/navsim_workspace/GaussianAD"
CFG_DIR="${REPO}/config/${CFG_NAME}"
WORK_DIR="${REPO}/exp/${CFG_NAME}"

GPUS="${GPUS:-4,5,6,7}"
MAX_USED_MIB="${MAX_USED_MIB:-5000}"
STABLE_CHECKS="${STABLE_CHECKS:-3}"
CHECK_INTERVAL="${CHECK_INTERVAL:-60}"
MASTER_PORT="${MASTER_PORT:-12873}"
LR="${LR:-2e-4}"
AUX_W="${AUX_W:-2.0}"
COL_W="${COL_W:-0.1}"

mkdir -p "${WORK_DIR}"
exec 9>"${WORK_DIR}/launch_guard.lock"
if ! flock -n 9; then
  echo "another guarded launcher/training process already holds the lock"
  exit 2
fi

IFS=',' read -r -a GPU_IDS <<< "${GPUS}"
NPROC="${#GPU_IDS[@]}"
stable=0

echo "[$(date '+%Y-%m-%d %H:%M:%S')] waiting for GPUs ${GPUS}; threshold=${MAX_USED_MIB} MiB, stable_checks=${STABLE_CHECKS}"
while true; do
  mapfile -t used_memory < <(
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
  )
  ready=1
  snapshot=()
  for gpu_id in "${GPU_IDS[@]}"; do
    used="${used_memory[${gpu_id}]}"
    snapshot+=("gpu${gpu_id}=${used}MiB")
    if (( used > MAX_USED_MIB )); then
      ready=0
    fi
  done

  if (( ready )); then
    stable=$((stable + 1))
  else
    stable=0
  fi
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${snapshot[*]} stable=${stable}/${STABLE_CHECKS}"

  if (( stable >= STABLE_CHECKS )); then
    break
  fi
  sleep "${CHECK_INTERVAL}"
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPUs are safely free; starting joint training"
cd "${REPO}"
exec env \
  GPUS="${GPUS}" \
  NPROC="${NPROC}" \
  MASTER_PORT="${MASTER_PORT}" \
  LR="${LR}" \
  AUX_W="${AUX_W}" \
  COL_W="${COL_W}" \
  "${CFG_DIR}/train_joint_full.sh"
