#!/usr/bin/env bash
# ============================================================================
# base_plan 测试脚本
#   base 模型基础上接入 planner（MapLoss + PlanLoss）
#
# 目的：测试 base_plan 模型的 future occ 指标（FutAvg），作为 planner 接入的
#       基线对照，用于对比 v12_fixempty_ft_plan 等后续变体。
#
# 机器：h20-new  ssh -p 30300 root@8.130.174.55
# GPU ：4,5,6,7（该机只允许使用后四张）
# 用法：bash config/nuscenes_gs25600_base_plan/test.sh
#       （建议 tmux：tmux new -s test_base_plan）
# ============================================================================
set -euo pipefail

SRC_CFG="nuscenes_gs25600_base_plan"
EXP_NAME="nuscenes_gs25600_base_plan"
REPO="/data/xinyao/navsim_workspace/GaussianAD"
PY="/data/chenz/conda_env/splatting/bin/python"
CONFIG="config/${SRC_CFG}/${SRC_CFG}.py"
WORK_DIR="exp/${EXP_NAME}"
CKPT="${WORK_DIR}/checkpoints/epoch_9.pth"
LOG_NAME="test_epoch09"
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