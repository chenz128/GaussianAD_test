#!/usr/bin/env bash
# ============================================================================
# v12_fixempty —— 修复 empty-gaussian bug 后重训 v12
#
# 背景：commit cfb1356 修复了 forward_flow 中未来帧渲染丢失 empty gaussian 的
#       缺陷。旧 v12 权重在修复后的推理路径下（未重训，仅换推理）：
#           current mIoU  17.451 → 17.451  （完全不变，验证修复只碰未来帧路径）
#           FutAvg         7.24  → 10.71   （+48%）
#           iou(geo)      恒定值 → 27.74→15.36 单调衰减（复活）
#       v12 之所以能幸免，是因为 flow_grad_scale=0 切断了病态梯度；同口径下
#       base 的未来帧非空预测率为 0.0%，而 v12 守住了 9.0%（GT=14.18%）。
#
# 本实验：config 与原 v12 完全相同（flow_grad_scale=0.0 保持梯度全断），
#         唯一差别是代码含 bugfix。offset 头这次将用健康的 flow loss 训练，
#         预期 FutAvg ≥ 10.71。
#
#         保持 flow_grad_scale=0 是低风险选择——它已有 10.71 的实证兜底。
#         "修复后放开 B 路是否更好"由并行的 base_gt_ego_fixempty
#         （flow_grad_scale=1.0）回答，无需在主力实验上冒险。
#
# 机器：h20-new  ssh -p 32344 root@8.130.174.55
# GPU ：4,5,6,7（前 4 张严禁使用）
# 用法：tmux new -s train_v12_fixempty
#       bash config/nuscenes_gs25600_gtbox_oracle_v12/train_fixempty.sh
#
# 监控：epoch 2 / 5 查未来帧 pred-nonempty（GT=14.18%），应向 14% 收敛。
# ============================================================================
set -euo pipefail

SRC_CFG="nuscenes_gs25600_gtbox_oracle_v12"
EXP_NAME="nuscenes_gs25600_v12_fixempty"
REPO="/data/chenz/GaussianAD"
PY="/data/chenz/conda_env/splatting/bin/torchrun"
CONFIG="config/${SRC_CFG}.py"
WORK_DIR="out/${EXP_NAME}"
GPUS="4,5,6,7"
NPROC=4
MASTER_PORT=12472

cd "${REPO}"
mkdir -p "${WORK_DIR}"

{
  echo "=== launch $(date '+%Y-%m-%d %H:%M:%S') ==="
  echo "exp        : ${EXP_NAME}"
  echo "git branch : $(git rev-parse --abbrev-ref HEAD)"
  echo "git commit : $(git rev-parse HEAD)"
  echo "config     : ${CONFIG}  (identical to ${SRC_CFG})"
  echo "delta      : code-level bugfix cfb1356 (flow_include_empty=True)"
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
