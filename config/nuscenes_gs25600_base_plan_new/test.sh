#!/usr/bin/env bash
# ============================================================================
# base_plan_new 测试脚本
#   权重来自 config/nuscenes_gs25600_base_plan_new/train.sh
#   （base_gt_ego_fixempty 基础上接入 planner：use_plan_ego=True + map/plan）
#
# 目的：测试闭环口径下的 future occ 指标（FutAvg）+ current mIoU。
#       对照：
#         base_gt_ego_fixempty（GT ego）→ current 15.103 / FutAvg 8.33
#         v12_fixempty_ft_plan（plan ego, fgs=0.0）→ current 18.97 / FutAvg 10.66
#
# 注：测试时 current_epoch 不传 → warmup 不生效 → 直接用 planner 预测轨迹，
#     与"部署闭环"语义一致。
#
# GPU：h20-new  ssh -p 32344 root@8.130.174.55（前 4 张严禁使用）
# 用法：tmux new -s test_base_plan_new
#       bash config/nuscenes_gs25600_base_plan_new/test.sh
# ============================================================================
set -euo pipefail

EXP_NAME="nuscenes_gs25600_base_plan_new"
REPO="/data/chenz/GaussianAD"
PY="/data/chenz/conda_env/splatting/bin/python"
CONFIG="config/${EXP_NAME}/${EXP_NAME}.py"
WORK_DIR="out/${EXP_NAME}"
CKPT="${WORK_DIR}/checkpoints/epoch_15.pth"
LOG_NAME="test_ep15_plan"
GPUS="4,5,6,7"
PORT="${MASTER_PORT:-20521}"

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