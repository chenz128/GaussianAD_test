#!/usr/bin/env bash
# ============================================================================
# v12_fixempty_ft_plan_futgau_detach_false_col
#   在 futgau (detach=False) planner 基础上引入「碰撞规避损失」(P1)
#
#   唯一 delta：PlanLoss 打开 col_loss_weight=0.2 / col_safe_margin=0.5
#   （有效碰撞权重 = 外层 weight 10.0 * 0.2 = 2.0），其余结构/超参与
#   futgau_detach_false 完全一致。碰撞损失见
#   loss/plan_loss.py::PlanAgentCollisionLoss。
#
# GPU：4,5,6,7   端口：12485
# 用法：tmux new -s train_futgau_col
#       bash config/nuscenes_gs25600_gtbox_oracle_v12_ft_plan_futgau_detach_false_col/train.sh
# 监控：关注 plan_L2 / plan_obj_box_col（碰撞率应下降且不显著抬高 L2）。
# ============================================================================
set -euo pipefail

SRC_CFG="nuscenes_gs25600_gtbox_oracle_v12_ft_plan_futgau_detach_false_col"
EXP_NAME="nuscenes_gs25600_v12_fixempty_ft_plan_futgau_detach_false_col"
REPO="/data/xinyao/navsim_workspace/GaussianAD"
PY="/data/chenz/conda_env/splatting/bin/torchrun"
CONFIG="config/${SRC_CFG}/${SRC_CFG}.py"
WORK_DIR="exp/${EXP_NAME}"
GPUS="0,1,2,3"
NPROC=4
MASTER_PORT=12485

cd "${REPO}"
mkdir -p "${WORK_DIR}"

{
  echo "=== launch $(date '+%Y-%m-%d %H:%M:%S') ==="
  echo "exp            : ${EXP_NAME}"
  echo "git branch     : $(git rev-parse --abbrev-ref HEAD)"
  echo "git commit     : $(git rev-parse HEAD)"
  echo "config         : ${CONFIG}"
  echo "init from      : exp/nuscenes_gs25600_v12_fixempty/checkpoints/epoch_15.pth (load_from)"
  echo "delta          : PlanLoss col_loss_weight=0.2 col_safe_margin=0.5 (collision-avoidance)"
  echo "base planner   : VADHeadFutGaussian (futgau, detach=False)"
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
