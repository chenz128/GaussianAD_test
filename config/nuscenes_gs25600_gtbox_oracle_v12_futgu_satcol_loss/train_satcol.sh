#!/usr/bin/env bash
# ============================================================================
# v12_ft_plan_futgau_detach_false_satcol —— 基于分离轴定理 (SAT) 的碰撞规避损失
#
#   在 futgau (detach=False) planner 基础上，把 PlanLoss 的碰撞损失从
#   「外接圆近似」换成「SAT 定向矩形穿透深度」：
#     - ego：轴对齐矩形 (yaw=0)，与 plan_obj_box_col 指标一致
#     - agent：带未来 yaw 的定向矩形
#     - 4 条分离轴 (agent x/y + ego x/y) 取最小分离度 -> 穿透深度
#     - 梯度只沿真实穿透方向回传 ego 位置
#
#   唯一 delta：PlanLoss 打开 col_sat=True / col_loss_weight=0.2 /
#   col_safe_margin=0.5（有效碰撞权重 = 外层 weight 10.0 * 0.2 = 2.0），
#   其余结构/超参与 futgau_detach_false 完全一致。实现见
#   loss/plan_loss.py::PlanAgentSATCollisionLoss。
#
# GPU：0,1,2,3   端口：12486
# 用法：tmux new -s train_satcol
#       bash config/nuscenes_gs25600_gtbox_oracle_v12_futgu_satcol_loss/train_satcol.sh
# 监控：关注 plan_L2 / plan_obj_box_col（碰撞率应下降且不显著抬高 L2）。
# ============================================================================
set -euo pipefail

SRC_CFG="nuscenes_gs25600_gtbox_oracle_v12_ft_plan_futgau_detach_false_satcol"
EXP_NAME="nuscenes_gs25600_v12_ft_plan_futgau_detach_false_satcol"
REPO="/data/xinyao/navsim_workspace/GaussianAD"
PY="/data/chenz/conda_env/splatting/bin/torchrun"
CONFIG="config/nuscenes_gs25600_gtbox_oracle_v12_futgu_satcol_loss/${SRC_CFG}.py"
WORK_DIR="exp/${EXP_NAME}"
GPUS="0,1,2,3"
NPROC=4
MASTER_PORT=12486

cd "${REPO}"
mkdir -p "${WORK_DIR}"

{
  echo "=== launch $(date '+%Y-%m-%d %H:%M:%S') ==="
  echo "exp            : ${EXP_NAME}"
  echo "git branch     : $(git rev-parse --abbrev-ref HEAD)"
  echo "git commit     : $(git rev-parse HEAD)"
  echo "config         : ${CONFIG}"
  echo "delta          : PlanLoss col_sat=True (SAT 定向矩形碰撞损失)"
  echo "col_loss_weight: 0.2 (有效 2.0)   col_safe_margin: 0.5"
  echo "gpus           : ${GPUS}  (nproc=${NPROC}, port=${MASTER_PORT})"
} | tee -a "${WORK_DIR}/launch_history.log"

CUDA_VISIBLE_DEVICES="${GPUS}" "${PY}" \
    --nproc_per_node "${NPROC}" \
    --master_port "${MASTER_PORT}" \
    train.py \
    --py-config "${CONFIG}" \
    --work-dir "${WORK_DIR}" \
    --dataset nuscenes \
    2>&1 | tee -a "${WORK_DIR}/train_run.log"
