#!/usr/bin/env bash
# ============================================================================
# v12_fixempty_ft_plan_futgau_time 测试脚本
#   在 VADHeadFutGaussian 基础上，未来帧高斯加入逐时间步位置编码 (planner_v4)。
# 目的：对比 futgau_detach_false (无时间编码)，验证时间编码是否改善
#       跨帧区分能力（FutAvg / plan L2 / obj_box_col）。
#
# GPU ：0,1,2,3
# 用法：bash config/nuscenes_gs25600_gtbox_oracle_v12_futgau_costime/test_futgau_time.sh
#       （建议 tmux：tmux new -s test_v12_futgau_time）
# ============================================================================
set -euo pipefail

SRC_CFG="nuscenes_gs25600_gtbox_oracle_v12_ft_plan_futgau_detach_false_time"
EXP_NAME="nuscenes_gs25600_v12_fixempty_ft_plan_futgau_detach_false_time"
REPO="/data/xinyao/navsim_workspace/GaussianAD"
PY="/data/chenz/conda_env/splatting/bin/python"
CONFIG="config/nuscenes_gs25600_gtbox_oracle_v12_futgau_costime/${SRC_CFG}.py"
WORK_DIR="exp/${EXP_NAME}"
CKPT="${WORK_DIR}/checkpoints/epoch_15.pth"
LOG_NAME="test_epoch15_v2"
GPUS="0,1,2,3"
PORT="${MASTER_PORT:-20510}"   # 手动改这里，或用 MASTER_PORT=xxxxx bash ... 覆盖

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
