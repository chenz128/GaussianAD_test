#!/usr/bin/env bash
# ============================================================================
# nuscenes_gs25600_froniter_v2_fix —— frontier-v2 时序帧索引修复实验
#
# 相对 nuscenes_gs25600_frontier_v2：
#   temporal encoder 与 frontier 图像条件均使用 current_frame_index=0；
#   强制当前帧至少保留 99% 高斯，防止历史帧硬裁切形成整齐空白切面。
#   训练/验证数据与原 frontier-v2 相同，max_epochs=15。
#
# GPU ：单节点 8 卡
# 用法：tmux new -s train_froniter_v2_fix
#       bash config/nuscenes_gs25600_froniter_v2_fix/train.sh
# ============================================================================
set -euo pipefail

EXP_NAME="nuscenes_gs25600_froniter_v2_fix"
REPO="/data/chenz/GaussianAD"
ENV_DIR="/data/chenz/conda_env/splatting"
PY="${ENV_DIR}/bin/torchrun"
CONFIG="config/${EXP_NAME}/${EXP_NAME}.py"
WORK_DIR="out/${EXP_NAME}"
GPUS="0,1,2,3,4,5,6,7"
NPROC=8
MASTER_PORT=12346

if [[ ! -x "${PY}" ]]; then
  echo "[FATAL] Training environment not found: ${ENV_DIR}" >&2
  exit 1
fi
export PATH="${ENV_DIR}/bin:${PATH}"

cd "${REPO}"
mkdir -p "${WORK_DIR}"

{
  echo "=== launch $(date '+%Y-%m-%d %H:%M:%S') ==="
  echo "exp        : ${EXP_NAME}"
  echo "git branch : $(git rev-parse --abbrev-ref HEAD)"
  echo "git commit : $(git rev-parse HEAD)"
  echo "config     : ${CONFIG}  (frontier-v2 + temporal current-index fix)"
  echo "code       : current_frame_index=0 + >=99% current Gaussian assertion"
  echo "gpus       : ${GPUS}  (nproc=${NPROC}, port=${MASTER_PORT})"
} | tee -a "${WORK_DIR}/launch_history.log"

CUDA_VISIBLE_DEVICES="${GPUS}" "${PY}" \
    --nproc_per_node "${NPROC}" \
    --master_port "${MASTER_PORT}" \
    train.py \
    --py-config "${CONFIG}" \
    --work-dir "${WORK_DIR}" \
    --dataset nuscenes \
    2>&1 | tee -a "${WORK_DIR}/train_run.log"