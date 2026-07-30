#!/usr/bin/env bash
# ============================================================================
# v12_fixempty —— 测试脚本
#   权重来自 config/nuscenes_gs25600_gtbox_oracle_v12/train_fixempty.sh
#   （config 与原 v12 完全相同，差别是训练代码含 empty-gaussian bugfix cfb1356）
#
# 目的：拿到 bugfix 后重训的 current mIoU + future occ（FutAvg）指标。
#       对照：
#         v12 旧权重 + 修复后推理  → current 17.451 / FutAvg 10.71
#       预期：FutAvg ≥ 10.71（offset 头这次用健康的 flow loss 训练）
#
# 注：推理路径同样走 bugfix 代码（flow_include_empty 默认 True），
#     即"用修改过 bug 后的代码测试"。
#
# 机器：h20-new  ssh -p 32344 root@8.130.174.55
# GPU ：4,5,6,7（前 4 张严禁使用）
# 用法：tmux new -s test_v12_fixempty
#       bash config/nuscenes_gs25600_gtbox_oracle_v12/test_fixempty.sh
# ============================================================================
set -euo pipefail

SRC_CFG="nuscenes_gs25600_gtbox_oracle_v12"
EXP_NAME="nuscenes_gs25600_v12_fixempty"
REPO="/data/chenz/GaussianAD"
PY="/data/chenz/conda_env/splatting/bin/python"
CONFIG="config/${SRC_CFG}.py"
WORK_DIR="out/${EXP_NAME}"
CKPT="${WORK_DIR}/checkpoints/epoch_15.pth"
LOG_NAME="test_ep15_fixempty"
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
