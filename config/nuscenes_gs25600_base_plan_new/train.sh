#!/usr/bin/env bash
# ============================================================================
# base_plan_new —— base_gt_ego_fixempty 基础上接入 planner 闭环（对比实验）
#
# 目标（config: nuscenes_gs25600_base_plan_new.py）：
#   1. occ_flow 的 ego 运动补偿由 GT ego 切换为 planner 预测轨迹
#      (model.head.use_plan_ego=True, warmup=2 epoch 后切换)。
#   2. 解冻 map_decoder / planner_head (frozen_modules=[])，重新启用
#      MapLoss + PlanLoss(weight=10.0) 监督二者。
#   3. flow_grad_scale=1.0 保持不变（与 base_gt_ego_fixempty 完全一致）。
#
# 续训方式：config 里 load_from=out/nuscenes_gs25600_base_gt_ego_fixempty/
#           checkpoints/epoch_15.pth（从 epoch 0 计数训 15 轮，
#           optimizer/lr 全新，适配新解冻的参数）。
#   不要 --resume-from（optimizer state 不匹配，train.py 会自动跳过）。
#
# 相对 v12_fixempty_ft_plan 的差异：flow_grad_scale=1.0（参照为 0.0），
# 保持与自身基线 base_gt_ego_fixempty 的单变量对比。
#
# GPU：h20-new  后 4 张（4,5,6,7，前 4 张严禁使用）
# 用法：tmux new -s train_base_plan_new
#       bash config/nuscenes_gs25600_base_plan_new/train.sh
# 监控：epoch 2 起 planner 预测生效，关注 plan L2 / occ_flow FutAvg 是否稳定。
# ============================================================================
set -euo pipefail

EXP_NAME="nuscenes_gs25600_base_plan_new"
REPO="/data/chenz/GaussianAD"
PY="/data/chenz/conda_env/splatting/bin/torchrun"
CONFIG="config/${EXP_NAME}/${EXP_NAME}.py"
WORK_DIR="out/${EXP_NAME}"
INIT_FROM="out/nuscenes_gs25600_base_gt_ego_fixempty/checkpoints/epoch_15.pth"
GPUS="4,5,6,7"
NPROC=4
MASTER_PORT=12483

cd "${REPO}"
mkdir -p "${WORK_DIR}"

{
  echo "=== launch $(date '+%Y-%m-%d %H:%M:%S') ==="
  echo "exp        : ${EXP_NAME}"
  echo "git branch : $(git rev-parse --abbrev-ref HEAD)"
  echo "git commit : $(git rev-parse HEAD)"
  echo "config     : ${CONFIG}"
  echo "init from  : ${INIT_FROM} (load_from)"
  echo "delta      : use_plan_ego=True + unfreeze map/plan + MapLoss/PlanLoss"
  echo "gpus       : ${GPUS}  (nproc=${NPROC}, port=${MASTER_PORT})"
} | tee -a "${WORK_DIR}/launch_history.log"

CUDA_VISIBLE_DEVICES="${GPUS}" "${PY}" \
    --nproc_per_node "${NPROC}" \
    --master_port "${MASTER_PORT}" \
    train.py \
    --py-config "${CONFIG}" \
    --work-dir "${WORK_DIR}" \
    --dataset nuscenes \
    2>&1 | tee -a "${WORK_DIR}/train_run.log"