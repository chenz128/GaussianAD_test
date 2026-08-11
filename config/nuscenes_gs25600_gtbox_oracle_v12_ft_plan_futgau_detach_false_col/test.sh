#!/usr/bin/env bash
# ============================================================================
# v12_fixempty_ft_plan_futgau_detach_false_col 测试脚本
#   futgau (detach=False) + 碰撞规避损失 的 planner 指标评测。
#
# 目的：对比 futgau_detach_false，验证碰撞损失是否降低 plan_obj_box_col
#       而不损害 plan_L2。
#
# GPU ：4,5,6,7   端口：20510
# 用法：bash config/nuscenes_gs25600_gtbox_oracle_v12_ft_plan_futgau_detach_false_col/test.sh
#       （建议 tmux：tmux new -s test_v12_futgau_col）
# ============================================================================
set -euo pipefail

SRC_CFG="nuscenes_gs25600_gtbox_oracle_v12_ft_plan_futgau_detach_false_col"
EXP_NAME="nuscenes_gs25600_v12_fixempty_ft_plan_futgau_detach_false_col"
REPO="/data/xinyao/navsim_workspace/GaussianAD"
PY="/data/chenz/conda_env/splatting/bin/python"
CONFIG="config/${SRC_CFG}/${SRC_CFG}.py"
WORK_DIR="exp/${EXP_NAME}"
CKPT="${WORK_DIR}/checkpoints/epoch_15.pth"
LOG_NAME="test_epoch15"
GPUS="4,5,6,7"
PORT="${MASTER_PORT:-20510}"

cd "${REPO}"

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
