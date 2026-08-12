#!/usr/bin/env bash
# ============================================================================
# v12_fixempty_ft_plan_futgau_time —— 未来帧高斯融合 planner + 逐时间步位置编码
#
# planner_head = VADHeadFutGaussianTime (model/planner/planner_v4.py)：
#   在 VADHeadFutGaussian 基础上，给未来帧高斯 key/value (fut_ts, G, D) 逐时间步
#   加入可学习时间位置编码 fut_time_pos = nn.Embedding(fut_ts, embed_dims)，
#   使跨帧注意力能明确区分「未来第几帧」的高斯。
#
# 对照：futgau_detach_false（未来帧高斯无时间编码）为 baseline。
# plan_ego_detach=False：occ_flow 一致性梯度回传 planner。
#
# GPU：0,1,2,3   端口：12485
# 用法：tmux new -s train_futgau_time
#       bash config/nuscenes_gs25600_gtbox_oracle_v12_futgau_costime/train_futgau_time.sh
# ============================================================================
set -euo pipefail

SRC_CFG="nuscenes_gs25600_gtbox_oracle_v12_ft_plan_futgau_detach_false_time"
EXP_NAME="nuscenes_gs25600_v12_fixempty_ft_plan_futgau_detach_false_time"
REPO="/data/xinyao/navsim_workspace/GaussianAD"
PY="/data/chenz/conda_env/splatting/bin/torchrun"
CONFIG="config/nuscenes_gs25600_gtbox_oracle_v12_futgau_costime/${SRC_CFG}.py"
WORK_DIR="exp/${EXP_NAME}"
GPUS="4,5,6,7"
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
  echo "delta          : VADHeadFutGaussian -> VADHeadFutGaussianTime (future-gaussian + time-pos-enc)"
  echo "plan_ego_detach: False (occ_flow 梯度回传 planner)"
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
