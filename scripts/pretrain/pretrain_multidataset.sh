#!/bin/bash
set -x

# Usage:
#   Single GPU:  sh scripts/pretrain_catok_ddt_intern.sh single 0 [config_name]
#   Multi GPU:   sh scripts/pretrain_catok_ddt_intern.sh multi 0,1,2,3 [config_name]

MODE=${1:-"single"}  # "single" or "multi"
GPU_IDS=${2:-"0"}    # GPU IDs (e.g., "0" or "0,1,2,3")
CONFIG_NAME=${3:-"vanilla_diffusion"}  # Config name (default: "config")
CONFIG_PATH=${4:-"../conf/config_libero_bridge"}

echo "Mode: $MODE"
echo "GPU IDs: $GPU_IDS"
echo "Config name: $CONFIG_NAME"

# Set environment variables
export CUDA_VISIBLE_DEVICES=$GPU_IDS
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export HYDRA_FULL_ERROR=1

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=SYS


# Master address and random port
MASTER_ADDR="127.0.0.1"
MASTER_PORT=$(shuf -i 20000-65535 -n 1)
echo "Using master port: $MASTER_PORT"

MODEL_SCRIPT="scripts/pretrain_multidataset.py"

if [ "$MODE" = "single" ]; then
    echo "=== Running in Single GPU Mode ==="
    
    # Single GPU mode - use torchrun with 1 process
    torchrun --nproc_per_node=1 \
             --nnodes=1 \
             --node_rank=0 \
             --master_addr=$MASTER_ADDR \
             --master_port=$MASTER_PORT \
             $MODEL_SCRIPT --config-path=$CONFIG_PATH --config-name=$CONFIG_NAME +use_ddp=False

elif [ "$MODE" = "multi" ]; then
    echo "=== Running in Multi GPU Mode ==="
    
    # Count number of GPUs
    NUM_GPUS=$(echo $GPU_IDS | tr ',' '\n' | wc -l)
    echo "Number of GPUs: $NUM_GPUS"
    
    # Multi GPU mode - use torchrun with DDP
    torchrun --nproc_per_node=$NUM_GPUS \
             --nnodes=1 \
             --node_rank=0 \
             --master_addr=$MASTER_ADDR \
             --master_port=$MASTER_PORT \
             $MODEL_SCRIPT --config-path=$CONFIG_PATH --config-name=$CONFIG_NAME +use_ddp=True 

else
    echo "Unknown mode: $MODE"
    echo "Usage: sh scripts/pretrain_catok_ddt_intern.sh [single|multi] [gpu_ids] [config_name]"
    exit 1
fi
