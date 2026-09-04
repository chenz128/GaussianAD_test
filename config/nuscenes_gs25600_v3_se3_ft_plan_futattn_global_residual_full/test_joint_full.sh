#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# 测评脚本（evaluation）: nuscenes_gs25600_v3_se3_ft_plan_futattn_global_residual_full
#
# 用法示例：
#   bash test_joint_full.sh                      # 自动挑最新 epoch checkpoint，GPU 0,1,2,3
#   EPOCH=16 bash test_joint_full.sh             # 指定评估第 16 epoch
#   CKPT=/path/to/xxx.pth bash test_joint_full.sh   # 指定任意 checkpoint
#   GPUS=4,5,6,7 bash test_joint_full.sh         # 指定 GPU
#   LOG_NAME=test_ep16 bash test_joint_full.sh   # 自定义日志名
#
# 说明：
#   - 默认 EPOCH=latest：自动选取 checkpoints/ 下编号最大的 epoch_*.pth
#   - 日志输出到 ${WORK_DIR}/${LOG_NAME}_console.log（tee 追加）
#   - 评估结果（plan_L2 / collision / mIoU / iou(geo)）打印在 console log 尾部
# ============================================================================

CFG_NAME="nuscenes_gs25600_v3_se3_ft_plan_futattn_global_residual_full"
REPO="/data/xinyao/navsim_workspace/GaussianAD"
PYTHON="/data/chenz/conda_env/splatting/bin/python"
CONFIG="config/${CFG_NAME}/${CFG_NAME}.py"
WORK_DIR="exp/${CFG_NAME}"

# ---- 运行资源（均可环境变量覆盖）----
GPUS="${GPUS:-0,1,2,3}"
MASTER_PORT="${MASTER_PORT:-21873}"

# ---- checkpoint 选择 ----
EPOCH="${EPOCH:-10}"
if [[ "${EPOCH}" == "latest" ]]; then
  CKPT_DIR="${WORK_DIR}/checkpoints"
  CKPT_FILE="$(
    ls "${CKPT_DIR}"/epoch_*.pth 2>/dev/null |
      sed -E 's/.*epoch_([0-9]+)\.pth/\1 &/' | sort -n | tail -1 | cut -d' ' -f2-
  )"
  if [[ -z "${CKPT_FILE}" ]]; then
    echo "[FATAL] no epoch_*.pth found under ${CKPT_DIR}" >&2
    exit 2
  fi
  CKPT="${CKPT_FILE}"
  EPOCH_TAG="latest_$(basename "${CKPT_FILE}" .pth)"
else
  CKPT="${CKPT:-${WORK_DIR}/checkpoints/epoch_${EPOCH}.pth}"
  EPOCH_TAG="epoch_${EPOCH}"
fi
LOG_NAME="${LOG_NAME:-test_${EPOCH_TAG}_v1}"

if [[ ! -f "${CKPT}" ]]; then
  echo "[FATAL] checkpoint not found: ${CKPT}" >&2
  exit 2
fi

mkdir -p "${WORK_DIR}"
cd "${REPO}"

{
  echo "=== test launch $(date '+%Y-%m-%d %H:%M:%S') ==="
  echo "exp        : ${CFG_NAME}"
  echo "config     : ${CONFIG}"
  echo "ckpt       : ${CKPT}"
  echo "gpus       : ${GPUS}  port: ${MASTER_PORT}"
  echo "git branch : $(git rev-parse --abbrev-ref HEAD)"
  echo "git commit : $(git rev-parse HEAD)"
} | tee -a "${WORK_DIR}/launch_history.log"

CUDA_VISIBLE_DEVICES="${GPUS}" MASTER_PORT="${MASTER_PORT}" "${PYTHON}" \
  test.py \
  --py-config "${CONFIG}" \
  --work-dir "${WORK_DIR}" \
  --resume-from "${CKPT}" \
  --log-name "${LOG_NAME}" \
  2>&1 | tee -a "${WORK_DIR}/${LOG_NAME}_console.log"
