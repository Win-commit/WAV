#!/usr/bin/bash

export WANDB_BASE_URL="https://api.bandw.top"
API_KEY=bf924aa39303a0d8808787e3777696c3626d4850
wandb login $API_KEY

script_path=${1}
echo $script_path

config_path=${2}
echo $config_path

MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-23456}

if [ -z $WORLD_SIZE ]; then
NGPU=`nvidia-smi --list-gpus | wc -l`
echo "Training on 1 Nodes, $NGPU GPUs"
torchrun --nnodes=1 \
    --nproc_per_node=$NGPU \
    --node_rank=0 \
    --master-addr $MASTER_ADDR \
    --master-port $MASTER_PORT \
    $script_path \
    --config_file $config_path
else
echo "Training on $WORLD_SIZE Nodes, 8 GPU per Node"
NGPU=`nvidia-smi --list-gpus | wc -l`
torchrun --nnodes=$WORLD_SIZE \
    --nproc_per_node=$NGPU \
    --node_rank=$RANK \
    --master-addr $MASTER_ADDR \
    --master-port $MASTER_PORT \
    $script_path \
    --config_file $config_path
fi
