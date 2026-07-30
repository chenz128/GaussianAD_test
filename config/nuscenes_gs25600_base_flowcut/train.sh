#!/usr/bin/env bash
# ============================================================================
# base_flowcut —— 消融实验训练脚本
#   base_flowcut = base_gt_ego + head.flow_grad_scale=0.0（严格单变量）
#
# 目的：量出 v12 当前帧 +2.47 mIoU 中，究竟有多少来自"切断 OccFlowLoss 的 B 路
#       梯度"。详见 config 文件头部注释。
#
# 对照组（三者同为 bugfix 后代码，同数据子集，同 15 epoch 调度）：
#   base_gt_ego_fixempty  (flow_grad_scale=1.0, 无 Dyn/Phys)  current 15.103 / FutAvg 8.33
#   base_flowcut          (flow_grad_scale=0.0, 无 Dyn/Phys)  ← 本实验
#   v12_fixempty          (flow_grad_scale=0.0, 有 Dyn/Phys + gtbox)  current 17.577 / FutAvg 10.92
#
# 机器：h20-new  ssh -p 32344 root@8.130.174.55
# GPU ：4,5,6,7（前 4 张严禁使用）
# 用法：tmux new -s train_base_flowcut
#       bash config/nuscenes_gs25600_base_flowcut/train.sh
# ============================================================================
set -euo pipefail

EXP_NAME="nuscenes_gs25600_base_flowcut"
REPO="/data/chenz/GaussianAD"
PY="/data/chenz/conda_env/splatting/bin/torchrun"
CONFIG="config/${EXP_NAME}/${EXP_NAME}.py"
WORK_DIR="out/${EXP_NAME}"
GPUS="4,5,6,7"
NPROC=4
MASTER_PORT=12473

cd "${REPO}"
mkdir -p "${WORK_DIR}"

{
  echo "=== launch $(date '+%Y-%m-%d %H:%M:%S') ==="
  echo "exp        : ${EXP_NAME}"
  echo "git branch : $(git rev-parse --abbrev-ref HEAD)"
  echo "git commit : $(git rev-parse HEAD)"
  echo "config     : ${CONFIG}"
  echo "delta      : base_gt_ego + head.flow_grad_scale=0.0 (single variable)"
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
