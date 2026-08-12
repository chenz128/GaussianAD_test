#!/usr/bin/env bash
# ============================================================================
# v12_fixempty_ft_plan 测试脚本
#   v12_fixempty 基础上续训，接入 planner（MapLoss + PlanLoss + use_plan_ego）
#
# 目的：测试 ft_plan 模型的 future occ 指标（FutAvg），验证 planner 接入后
#       当前帧与未来帧的权衡效果。
#
# 机器：h20-new  ssh -p 30300 root@8.130.174.55
# GPU ：4,5,6,7（该机只允许使用后四张）
# 用法：bash config/nuscenes_gs25600_gtbox_oracle_v12_ft_plan/test.sh
#       （建议 tmux：tmux new -s test_v12_ft_plan）
# ============================================================================
set -euo pipefail

SRC_CFG="nuscenes_gs25600_gtbox_oracle_v12_ft_plan"
EXP_NAME="nuscenes_gs25600_v12_fixempty_ft_plan"
REPO="/data/xinyao/navsim_workspace/GaussianAD"
PY="/data/chenz/conda_env/splatting/bin/python"
CONFIG="config/${SRC_CFG}/${SRC_CFG}.py"
WORK_DIR="exp/${EXP_NAME}"
CKPT="${WORK_DIR}/checkpoints/epoch_15.pth"
LOG_NAME="test_epoch15_v2"
GPUS="4,5,6,7"
PORT="${MASTER_PORT:-20520}"   # 手动改这里，或用 MASTER_PORT=xxxxx bash ... 覆盖

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
  echo "port       : ${PORT}"
} | tee -a "${WORK_DIR}/launch_history.log"

CUDA_VISIBLE_DEVICES="${GPUS}" MASTER_PORT="${PORT}" "${PY}" test.py \
    --py-config "${CONFIG}" \
    --work-dir "${WORK_DIR}" \
    --resume-from "${CKPT}" \
    --log-name "${LOG_NAME}" \
    2>&1 | tee -a "${WORK_DIR}/${LOG_NAME}_console.log"
