#!/usr/bin/env bash
# ============================================================================
# v12_fixempty_ft_plan_futattn —— 未来帧轨迹 attention 化 planner 实验
#
# 目标（config: nuscenes_gs25600_gtbox_oracle_v12_ft_plan_futattn.py）：
#   在 v12_fixempty_ft_plan 基础上，仅把 planner_head 从 VADHead 换成
#   VADHeadFutAttn：未来帧轨迹同样走 attention 机制
#     - fut_query 时间步 token + 时间维 self-attention（时序连贯性）
#     - fut_query 与 agent/map/gaussian 上下文交叉注意力（与当前帧对称）
#     - 逐时间步回归 ego_fut_mode*2 -> [B, ego_fut_mode, fut_ts, 2]
#   其余（load_from、use_plan_ego、MapLoss/PlanLoss、lr 等）与 ft_plan 完全一致。
#
# 续训方式：config 里 load_from=exp/nuscenes_gs25600_v12_fixempty/checkpoints/
#           epoch_15.pth。新增 fut_*_decoder 层随机初始化，从 epoch 0 训 15 轮。
#   若想连续 epoch/optimizer，改为在下方 torchrun 追加：
#           --resume-from exp/nuscenes_gs25600_v12_fixempty/checkpoints/epoch_15
.pth
#
# GPU：4,5,6,7（前 4 张严禁使用）
# 用法：tmux new -s train_v12_ft_plan_futattn
#       bash config/nuscenes_gs25600_gtbox_oracle_v12_ft_plan_futattn/train_futa
ttn.sh
# 监控：epoch 2 起 planner 预测生效，关注 plan L2 / occ_flow FutAvg 是否稳定。
# ============================================================================
set -euo pipefail

SRC_CFG="nuscenes_gs25600_gtbox_oracle_v12_ft_plan_futattn"
EXP_NAME="nuscenes_gs25600_v12_fixempty_ft_plan_futattn"
REPO="/data/xinyao/navsim_workspace/GaussianAD"
PY="/data/chenz/conda_env/splatting/bin/torchrun"
CONFIG="config/${SRC_CFG}/${SRC_CFG}.py"
WORK_DIR="exp/${EXP_NAME}"
GPUS="0,1,2,3"
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
  echo "init from  : exp/nuscenes_gs25600_v12_fixempty/checkpoints/epoch_15.pth
(load_from)"
  echo "delta      : planner_head VADHead -> VADHeadFutAttn (future-frame attn)"
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
