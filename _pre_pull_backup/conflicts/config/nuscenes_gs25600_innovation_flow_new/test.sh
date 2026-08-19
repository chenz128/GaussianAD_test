#!/usr/bin/env bash
# ============================================================================
# nuscenes_gs25600_innovation_flow_new —— 测试脚本
#   FM v2 (per-query image + SE3)，测试 epoch_15（训练已完成）
#
# 机器：h20-new  ssh -p 32344 root@8.130.174.55
# GPU ：6,7（该机只允许使用 4-7，这里用后两张卡 6,7）
# 用法：bash config/nuscenes_gs25600_innovation_flow_new/test.sh
#       （建议 tmux：tmux new -s test_innovation_flow_new）
# ============================================================================
set -euo pipefail

EXP_NAME="nuscenes_gs25600_innovation_flow_new"
REPO="/data/chenz/GaussianAD"
PY="/data/chenz/conda_env/faster/bin/python"
CONFIG="config/${EXP_NAME}/${EXP_NAME}.py"
WORK_DIR="out/${EXP_NAME}"
CKPT="${WORK_DIR}/checkpoints/epoch_15.pth"
LOG_NAME="test_ep15"
GPUS="6,7"

cd "${REPO}"

{
  echo "=== test launch $(date '+%Y-%m-%d %H:%M:%S') ==="
  echo "exp        : ${EXP_NAME}"
  echo "git branch : $(git rev-parse --abbrev-ref HEAD)"
  echo "git commit : $(git rev-parse HEAD)"
  echo "config     : ${CONFIG}"
  echo "ckpt       : ${CKPT}"
  echo "gpus       : ${GPUS}"
} | tee -a "${WORK_DIR}/launch_history.log"

CUDA_VISIBLE_DEVICES="${GPUS}" "${PY}" test.py \
    --py-config "${CONFIG}" \
    --work-dir "${WORK_DIR}" \
    --resume-from "${CKPT}" \
    --log-name "${LOG_NAME}" \
    2>&1 | tee -a "${WORK_DIR}/${LOG_NAME}_console.log"
