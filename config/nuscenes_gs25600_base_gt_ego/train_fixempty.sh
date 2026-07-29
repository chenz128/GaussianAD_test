#!/usr/bin/env bash
# ============================================================================
# base_gt_ego_fixempty —— 修复 empty-gaussian bug 后的合法基线
#
# 背景：commit cfb1356 修复了 forward_flow 中未来帧渲染丢失 empty gaussian 的
#       缺陷（mask ⊂ [0,G) 永远选不到第 G 个 empty）。修复前未来帧 ch17 恒为 0，
#       而 GT 中 86-92% 是 empty，导致 CE 的唯一下降方向是"把所有非空 logit
#       压低"——约 53% 的训练信号在做无差别抑制。
#
#       实测后果：旧 base_gt_ego 权重在修复后的推理路径下，未来帧非空预测率
#       为 0.0%（完全打不过 empty gaussian），FutAvg 崩到 0.04。其高斯表征
#       已被这条病态梯度摧毁，不能作为基线使用，必须重训。
#
# 本实验：config 与原 base_gt_ego 完全相同（flow_grad_scale=1.0 梯度全通、
#         decouple_offset=False），唯一差别是代码含 bugfix。因此它同时给出：
#           1) 汇报所需的合法基线
#           2) "修复后 flow 梯度全通是否仍有害"的答案
#
# 机器：h20-old  ssh -p 30300 root@8.130.174.55
# GPU ：4,5,6,7（0-3 常被他人占用，勿动）
# 用法：tmux new -s train_base_fixempty
#       bash config/nuscenes_gs25600_base_gt_ego/train_fixempty.sh
#
# 监控：epoch 2 / 5 查未来帧 pred-nonempty（GT=14.18%）。向 14% 收敛为健康；
#       若往 0 掉说明高斯又被压制，可早停。
# ============================================================================
set -euo pipefail

SRC_EXP="nuscenes_gs25600_base_gt_ego"
EXP_NAME="nuscenes_gs25600_base_gt_ego_fixempty"
REPO="/data/chenz/GaussianAD"
PY="/data/chenz/conda_env/splatting/bin/torchrun"
CONFIG="config/${SRC_EXP}/${SRC_EXP}.py"
WORK_DIR="out/${EXP_NAME}"
GPUS="4,5,6,7"
NPROC=4
MASTER_PORT=12471

cd "${REPO}"
mkdir -p "${WORK_DIR}"

{
  echo "=== launch $(date '+%Y-%m-%d %H:%M:%S') ==="
  echo "exp        : ${EXP_NAME}"
  echo "git branch : $(git rev-parse --abbrev-ref HEAD)"
  echo "git commit : $(git rev-parse HEAD)"
  echo "config     : ${CONFIG}  (identical to ${SRC_EXP})"
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
