sh -c 'echo "MP_STATUS_CHECK_INTERVAL = 1.0" >> /usr/local/lib/python3.8/dist-packages/torch/utils/data/_utils/__init__.py'
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

if [ ${DEBUG_MODE}_D == "1_D" ]
then
    export NCCL_DEBUG=INFO
    export NCCL_DEBUG_SUBSYS=ALL
    export TORCH_CPP_LOG_LEVEL=INFO
    export TORCH_DISTRIBUTED_DEBUG=DETAIL
    export TORCH_SHOW_CPP_STACKTRACES=1
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
fi

if [ -n "${MAX_RESTART}" ]
then
    MAX_RESTART="--max_restarts $MAX_RESTART"
else
    MAX_RESTART="--max_restarts 3"
fi

DISTRIBUTED_ARGS="--nproc_per_node $GPU_NUM --nnodes $NODE_NUM --node_rank $RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT";

if [ ${PROFILE_ENABLE}_D == "1_D" ]
then
    nsys profile \
            -t cuda,osrt,nvtx \
            -c cudaProfilerApi \
            --capture-range-end=stop \
            -o ${LPAI_MODEL_DIR}/profile_%q{RANK}_%p.nsys-rep \
            torchrun $DISTRIBUTED_ARGS train.py $@;
else
    echo "command = [torchrun $MAX_RESTART $DISTRIBUTED_ARGS train.py $@]"
    torchrun $MAX_RESTART $DISTRIBUTED_ARGS train.py $@
fi
