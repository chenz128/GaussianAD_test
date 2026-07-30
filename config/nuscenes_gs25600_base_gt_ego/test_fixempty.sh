#!/usr/bin/env bash
# ============================================================================
# base_gt_ego_fixempty —— 测试脚本
#   权重来自 config/nuscenes_gs25600_base_gt_ego/train_fixempty.sh
#   （config 与 base_gt_ego 完全相同，差别是训练代码含 empty-gaussian
#     bugfix cfb1356，且 flow_grad_scale=1.0 —— B 路梯度全开）
#
# 目的：回答"修复 bug 后放开未来帧梯度是否更好"。
#       对照：
#         base_gt_ego（bug 期训练）           → 未来帧非空预测率 0.0%
#         v12_fixempty（flow_grad_scale=0.0） → current 17.577 / FutAvg 10.92
#
# 注：推理路径同样走 bugfix 代码（flow_include_empty 默认 True）。
#
# 机器：h20-new  ssh -p 32344 root@8.130.174.55
# GPU ：4,5,6,7（前 4 张严禁使用）
# 用法：tmux new -s test_base_fixempty
#       bash config/nuscenes_gs25600_base_gt_ego/test_fixempty.sh
# ============================================================================
set -euo pipefail

SRC_EXP="nuscenes_gs25600_base_gt_ego"
EXP_NAME="nuscenes_gs25600_base_gt_ego_fixempty"
REPO="/data/chenz/GaussianAD"
PY="/data/chenz/conda_env/splatting/bin/python"
CONFIG="config/${SRC_EXP}/${SRC_EXP}.py"
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
