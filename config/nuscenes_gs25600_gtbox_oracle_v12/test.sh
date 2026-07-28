#!/usr/bin/env bash
# ============================================================================
# v12 (gtbox_oracle_v12) 测试脚本
#   v12 = v10fix + head.flow_grad_scale=0.0（切断未来帧对当前帧高斯的梯度）
#
# 目的：训练期 val mIoU 已达 17.45（v10fix 12.84 / base15 14.15 / base30 15.45），
#       但训练期不计算未来占据。本测试用于拿到 future occ 指标（FutAvg），
#       验证 P1 的核心权衡：当前帧大涨的同时，未来帧是否守住 6.57。
#
# 机器：h20-new  ssh -p 32344 root@8.130.174.55
# GPU ：4,5,6,7（该机只允许使用后四张）
# 用法：bash config/nuscenes_gs25600_gtbox_oracle_v12/test.sh
#       （建议 tmux：tmux new -s test_v12）
#
# 注：v12 的 config 仍在扁平路径 config/nuscenes_gs25600_gtbox_oracle_v12.py
#     （训练时即以该路径记录在 work-dir 中，不移动以免破坏可追溯性）。
# ============================================================================
set -euo pipefail

EXP_NAME="nuscenes_gs25600_gtbox_oracle_v12"
REPO="/data/chenz/GaussianAD"
PY="/data/chenz/conda_env/splatting/bin/python"
CONFIG="config/${EXP_NAME}.py"
WORK_DIR="out/${EXP_NAME}"
CKPT="${WORK_DIR}/checkpoints/epoch_15.pth"
LOG_NAME="test_ep15"
GPUS="4,5,6,7"

cd "${REPO}"

# 留痕
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
