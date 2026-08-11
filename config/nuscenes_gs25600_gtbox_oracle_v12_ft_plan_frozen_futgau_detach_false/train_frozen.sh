#!/usr/bin/env bash
# ============================================================================
# v12_fixempty_ft_plan_frozen_futgau_detach_false —— 对照实验
#
# 基于 futgau_detach_false，新增 frozen_modules 冻结「高斯预测模型」：
#   ['lifter','encoder','temporal_encoder','decoder','head']
# 高斯的位置/形状/语义/offset 全部固定；planner_head(VADHeadFutGaussian)
# 与 map_decoder 保持可训练，use_plan_ego=True / detach=False 不变。
#
# 目的：验证 planner 单独学习「未来帧高斯融合」能否成立（固定高斯输入）。
#
# GPU：4,5,6,7   端口：12486
# 用法：tmux new -s train_frozen_futgau
#       bash config/nuscenes_gs25600_gtbox_oracle_v12_ft_plan_frozen_futgau_detach_false/train_frozen.sh
# 监控：epoch 2 起 planner 预测生效，关注 plan_L2 / obj_box_col / occ_flow FutAvg。
# ============================================================================
set -euo pipefail

SRC_CFG="nuscenes_gs25600_gtbox_oracle_v12_ft_plan_frozen_futgau_detach_false"
EXP_NAME="nuscenes_gs25600_v12_fixempty_ft_plan_frozen_futgau_detach_false"
REPO="/data/xinyao/navsim_workspace/GaussianAD"
PY="/data/chenz/conda_env/splatting/bin/torchrun"
CONFIG="config/${SRC_CFG}/${SRC_CFG}.py"
WORK_DIR="exp/${EXP_NAME}"
GPUS="4,5,6,7"
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
  echo "init from      : exp/nuscenes_gs25600_v12_fixempty/checkpoints/epoch_15.pth (load_from)"
  echo "delta          : frozen Gaussian stack = lifter,encoder,temporal_encoder,decoder,head"
  echo "planner        : VADHeadFutGaussian (trainable) + plan_ego_detach=False"
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
