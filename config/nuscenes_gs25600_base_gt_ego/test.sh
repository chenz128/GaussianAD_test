#!/usr/bin/env bash
# ============================================================================
# nuscenes_gs25600_base_gt_ego —— 测试脚本
#   base_gt_ego = 干净 base 基线（当前代码 / GT ego 补偿 / occ+flow+det）
#
# 目的：拿到与 v10fix / v12 同一套 ego 补偿公式下的 current + future 指标，
#       作为 oracle 系列真正可对标的基线。
#       （旧 base15=14.15 训练于 6/16-6/23，用冻结 planner 的 ego_fut_preds≈0，
#         未来帧口径与 v4+ 不可比。）
#
# 机器：h20-new  ssh -p 32344 root@8.130.174.55
# GPU ：4,5,6,7（该机只允许使用后四张，0-3 严禁使用）
# 用法：bash config/nuscenes_gs25600_base_gt_ego/test.sh
#       （建议 tmux：tmux new -s test_base_gt_ego）
# ============================================================================
set -euo pipefail

EXP_NAME="nuscenes_gs25600_base_gt_ego"
REPO="/data/chenz/GaussianAD"
PY="/data/chenz/conda_env/splatting/bin/python"
CONFIG="config/${EXP_NAME}/${EXP_NAME}.py"
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
