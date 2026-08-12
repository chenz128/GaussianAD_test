#!/usr/bin/env bash
# ============================================================================
# v12_fixempty_ft_plan_futgau 测试脚本
#   ft_plan 基础上把 planner_head 换成 VADHeadFutGaussian
#   （ego 与「未来帧高斯」做交叉注意力，作为对称的第 4 路 stream 融合）
#
# 目的：测试 futgau 模型的 future occ 指标（FutAvg）与 planner 指标（L2/col），
#       验证「未来帧高斯正常融合」是否在不损害 planner 的前提下改善未来帧占用。
#
# 机器：h20-new  ssh -p 30300 root@8.130.174.55
# GPU ：0,1,2,3
# 用法：bash config/nuscenes_gs25600_gtbox_oracle_v12_ft_plan_futgau/test.sh
#       （建议 tmux：tmux new -s test_v12_futgau）
# ============================================================================
set -euo pipefail

SRC_CFG="nuscenes_gs25600_gtbox_oracle_v12_ft_plan_circle"
CFG_DIR="nuscenes_gs25600_gtbox_oracle_v12_circle_ago_attn"
EXP_NAME="nuscenes_gs25600_v12_fixempty_ft_plan_circle"
REPO="/data/xinyao/navsim_workspace/GaussianAD"
PY="/data/chenz/conda_env/splatting/bin/python"
CONFIG="config/${CFG_DIR}/${SRC_CFG}.py"
WORK_DIR="exp/${EXP_NAME}"
CKPT="${WORK_DIR}/checkpoints/epoch_15.pth"
LOG_NAME="test_epoch15"
GPUS="4,5,6,7"
PORT="${MASTER_PORT:-20509}"   # 手动改这里，或用 MASTER_PORT=xxxxx bash ... 覆盖

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
