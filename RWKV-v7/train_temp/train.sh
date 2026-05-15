#!/bin/bash
set -e

# 1. Workspace Setup
# Adjust this to your B200 cluster's specific path if different from /work
cd /work

# 2. Clone the RWKV repository and your Blackwell branch
if [ ! -d "RWKV-v7" ]; then
    echo "Cloning RWKV-v7 repository..."
    # Replace with your actual RWKV repo URL
    git clone -b mono https://github.com/SW10-Cryptanalysis/RWKV-LM.git 
fi

cd RWKV-LM/RWKV-v7/train_temp
mkdir -p logs

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Training Job started on $(hostname) at $(date) with $NUM_GPUS GPU(s)"

export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))

export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

export OMP_NUM_THREADS=32 

if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

uv sync

# This environment variable tells PyTorch which architecture to compile for
export TORCH_CUDA_ARCH_LIST="10.0"

MASTER_PORT=$((10000 + $RANDOM % 20000))

# 8. Launch RWKV-7 Training
# Note: Ensure the path to your training script matches your project structure
echo "Launching RWKV training with $NUM_GPUS B200 processes..."
uv run torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    rwkv7_train_cipher.py \

echo "Training Job finished at $(date)"