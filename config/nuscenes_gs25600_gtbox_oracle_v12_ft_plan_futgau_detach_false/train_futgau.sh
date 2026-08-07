#!/usr/bin/env bash
# ============================================================================
# v12_fixempty_ft_plan_futgau —— 未来帧高斯融合 planner 实验 (plan_ego_detach=False)
#
# planner_head = VADHeadFutGaussian：让「未来帧高斯」作为与 agent/map/当前帧高斯
#   对称的第 4 路 stream 融合进 planner（ego↔未来帧高斯 交叉注意力），拼接 4D 后
#   复用原 ego_fut_decoder 回归轨迹（增广输出头，非替换，无信息瓶颈）。
#
# plan_ego_detach=False：occ_flow 未来帧 ego 补偿 (means_fut - planner_res) 处
#   梯度回传 planner（OccFlowLoss 一致性梯度可影响 planner）。
#
# 对照实验见同目录 train_futgau_detach.sh (plan_ego_detach=True)。
#
# GPU：0,1,2,3   端口：12484
# 用法：tmux new -s train_futgau
#       bash config/nuscenes_gs25600_gtbox_oracle_v12_ft_plan_futgau/train_futgau.sh
# 监控：epoch 2 起 planner 预测生效，关注 plan_L2 / obj_box_col / occ_flow FutAvg。
# ============================================================================
set -euo pipefail

SRC_CFG="nuscenes_gs25600_gtbox_oracle_v12_ft_plan_futgau_detach_false"
EXP_NAME="nuscenes_gs25600_v12_fixempty_ft_plan_futgau_detach_false"
REPO="/data/xinyao/navsim_workspace/GaussianAD"
PY="/data/chenz/conda_env/splatting/bin/torchrun"
CONFIG="config/${SRC_CFG}/${SRC_CFG}.py"
WORK_DIR="exp/${EXP_NAME}"
GPUS="4,5,6,7"
NPROC=4
MASTER_PORT=12484

cd "${REPO}"
mkdir -p "${WORK_DIR}"

{
  echo "=== launch $(date '+%Y-%m-%d %H:%M:%S') ==="
  echo "exp            : ${EXP_NAME}"
  echo "git branch     : $(git rev-parse --abbrev-ref HEAD)"
  echo "git commit     : $(git rev-parse HEAD)"
  echo "config         : ${CONFIG}"
  echo "init from      : exp/nuscenes_gs25600_v12_fixempty/checkpoints/epoch_15.pth (load_from)"
  echo "delta          : planner_head VADHead -> VADHeadFutGaussian (future-gaussian 4th stream)"
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
