#!/usr/bin/env bash
# ============================================================================
# v12_fixempty_ft_plan_frozen_futgau_detach_false 测试脚本
#   frozen Gaussian stack + futgau (detach=False) planner 的指标评测。
#
# 目的：与 futgau_detach_false 对照，验证固定高斯输入后 planner 单独学习
#       「未来帧高斯融合」的效果：plan_L2 / plan_obj_box_col / occ_flow FutAvg。
#
# GPU ：0,1,2,3   端口：20512
# 用法：bash config/nuscenes_gs25600_gtbox_oracle_v12_ft_plan_frozen_futgau_detach_false/test_frozen.sh
#       （建议 tmux：tmux new -s test_frozen_futgau）
# ============================================================================
set -euo pipefail

SRC_CFG="nuscenes_gs25600_gtbox_oracle_v12_ft_plan_frozen_futgau_detach_false"
EXP_NAME="nuscenes_gs25600_v12_fixempty_ft_plan_frozen_futgau_detach_false"
REPO="/data/xinyao/navsim_workspace/GaussianAD"
PY="/data/chenz/conda_env/splatting/bin/python"
CONFIG="config/nuscenes_gs25600_gtbox_oracle_v12_ft_plan_frozen_futgau_detach_false/${SRC_CFG}.py"
WORK_DIR="exp/${EXP_NAME}"
CKPT="${WORK_DIR}/checkpoints/epoch_15.pth"
LOG_NAME="test_epoch15_v2"
GPUS="4,5,6,7"
PORT="${MASTER_PORT:-20512}"

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
