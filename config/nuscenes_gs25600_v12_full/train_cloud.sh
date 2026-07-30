#!/usr/bin/env bash
# ============================================================================
# nuscenes_gs25600_v12_full —— 云平台（PyTorch Elastic / PET_*）启动脚本
#
# 与 train.sh 的差别：不写死 GPU 与端口，改为读平台注入的 PET_* 拓扑变量，
# 支持单机多卡与多机多卡；本地/H20 直接跑也可（PET_* 缺省时退化为单机全卡）。
#
# 平台「训练配置」里填：
#   cd /data/chenz/GaussianAD && bash config/nuscenes_gs25600_v12_full/train_cloud.sh
#
# 可用环境变量覆盖：REPO / PY / DATA_ROOT / CKPT_ROOT
# ============================================================================
set -euo pipefail

# ── 1. 分布式拓扑：平台的 PET_* → torchrun 参数 ────────────────────────────
NNODES=${PET_NNODES:-1}
NODE_RANK=${PET_NODE_RANK:-0}
NPROC_PER_NODE=${PET_NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}
MASTER_ADDR=${PET_MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${PET_MASTER_PORT:-12481}

# RANK / WORLD_SIZE 必须由 torchrun 按进程写入（进程级语义）。若平台把节点级的值
# 提前 export 了，这里清掉，避免 train.py 的 initialize() 读到错的全局 rank。
unset RANK WORLD_SIZE LOCAL_RANK NODE_NUM GPU_COUNT || true

# ── 2. CPU 内存软水位：压到硬限的 80%，让内核回收而不是 OOM-kill ──────────
CGROUP_REL=$(awk -F: '/^0::/{print $3}' /proc/self/cgroup 2>/dev/null || echo "")
CGROUP_BASE="/sys/fs/cgroup${CGROUP_REL}"
if [[ -n "${CGROUP_REL}" && -r "${CGROUP_BASE}/memory.max" && -w "${CGROUP_BASE}/memory.high" ]]; then
    MEM_LIMIT=$(cat "${CGROUP_BASE}/memory.max")
    if [[ "${MEM_LIMIT}" != "max" ]]; then
        echo $(( MEM_LIMIT * 80 / 100 )) > "${CGROUP_BASE}/memory.high"
    fi
fi

# ── 3. 路径 ────────────────────────────────────────────────────────────────
EXP_NAME="nuscenes_gs25600_v12_full"
REPO="${REPO:-/data/chenz/GaussianAD}"
PY="${PY:-/data/chenz/conda_env/splatting/bin/torchrun}"
CONFIG="config/${EXP_NAME}/${EXP_NAME}.py"
WORK_DIR="out/${EXP_NAME}"

cd "${REPO}"
mkdir -p "${WORK_DIR}"

# 平台挂载点与仓库内相对路径不一致时，用软链对齐（config 里全是相对路径）
[[ -n "${DATA_ROOT:-}" && ! -e data ]] && ln -s "${DATA_ROOT}" data
[[ -n "${CKPT_ROOT:-}" && ! -e ckpts ]] && ln -s "${CKPT_ROOT}" ckpts

# ── 4. 启动前自检（多机最常见的两个坑）────────────────────────────────────
VISIBLE_GPUS=$(python3 -c "import torch;print(torch.cuda.device_count())")
if [[ "${VISIBLE_GPUS}" -lt 2 ]]; then
    echo "[FATAL] 可见 GPU=${VISIBLE_GPUS}；train.py 以 torch.cuda.device_count()>1 为开 DDP 的条件，单卡会退化成非分布式。" >&2
    exit 1
fi
if [[ "${VISIBLE_GPUS}" -ne "${NPROC_PER_NODE}" ]]; then
    echo "[FATAL] 可见 GPU=${VISIBLE_GPUS} != nproc_per_node=${NPROC_PER_NODE}；train.py 断言 rank%device_count==local_rank 会失败。" >&2
    exit 1
fi

# ── 5. 留痕（只在 node 0 记一次）──────────────────────────────────────────
if [[ "${NODE_RANK}" == "0" ]]; then
{
  echo "=== launch $(date '+%Y-%m-%d %H:%M:%S') ==="
  echo "exp        : ${EXP_NAME}"
  echo "git branch : $(git rev-parse --abbrev-ref HEAD)"
  echo "git commit : $(git rev-parse HEAD)"
  echo "config     : ${CONFIG}  (v12 + full train/val, max_epochs=20)"
  echo "code       : includes empty-gaussian bugfix cfb1356"
  echo "topology   : nnodes=${NNODES} nproc=${NPROC_PER_NODE} master=${MASTER_ADDR}:${MASTER_PORT}"
  echo "world_size : $(( NNODES * NPROC_PER_NODE ))  (bs=1/GPU -> effective batch)"
} | tee -a "${WORK_DIR}/launch_history.log"
fi

# ── 6. 启动（train.py 自动检测 work_dir/latest.pth 续训）──────────────────
"${PY}" \
    --nnodes "${NNODES}" \
    --node_rank "${NODE_RANK}" \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    train.py \
    --py-config "${CONFIG}" \
    --work-dir "${WORK_DIR}" \
    --dataset nuscenes \
    2>&1 | tee -a "${WORK_DIR}/train_run_node${NODE_RANK}.log"
