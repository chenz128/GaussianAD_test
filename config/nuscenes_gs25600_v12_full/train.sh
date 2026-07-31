#!/usr/bin/env bash
# ============================================================================
# nuscenes_gs25600_v12_full —— v12（含 empty-gaussian bugfix）全量数据训练
#
# 与 config/nuscenes_gs25600_gtbox_oracle_v12/train_fixempty.sh 的差别只有数据：
#   train 3000 子集 → 全量、val 2000 子集 → 全量、max_epochs 15 → 20
#   （20 epoch 对齐 out/nuscenes_gs25600_4gpu_v4 那次 4 卡全量实验）
#
# GPU ：单节点 8 卡
# 用法：tmux new -s train_v12_full
#       bash config/nuscenes_gs25600_v12_full/train.sh
# ============================================================================
set -euo pipefail

EXP_NAME="nuscenes_gs25600_v12_full"
REPO="/data/chenz/GaussianAD"
ENV_DIR="/data/chenz/conda_env/splatting"
PY="${ENV_DIR}/bin/torchrun"
CONFIG="config/${EXP_NAME}/${EXP_NAME}.py"
WORK_DIR="out/${EXP_NAME}"
GPUS="0,1,2,3,4,5,6,7"
NPROC=8
MASTER_PORT=12345

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
  echo "config     : ${CONFIG}  (v12 + full train/val, max_epochs=20)"
  echo "code       : includes empty-gaussian bugfix cfb1356"
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
