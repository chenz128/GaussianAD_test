#!/bin/bash
set -x
set -e

export PYTHONPATH=./src:$PYTHONPATH

export CUDA_VISIBLE_DEVICES=1,2,3
NUM_GPUS=${NUM_GPUS:-3}
random_port=$(( ( RANDOM % 1000 )  + 10000 ))

torchrun --nproc_per_node $NUM_GPUS --master_port $random_port train.py $@