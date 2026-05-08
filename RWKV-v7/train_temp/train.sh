#!/bin/bash
set -e

# 1. Workspace Setup
# Adjust this to your B200 cluster's specific path if different from /work
cd /ceph/project/SW10-CausalLM/RWKV-LM/

# 2. Clone the RWKV repository and your Blackwell branch
if [ ! -d "RWKV-v7" ]; then
    echo "Cloning RWKV-v7 repository..."
    # Replace with your actual RWKV repo URL
    git clone -b ucloud https://github.com/SW10-Cryptanalysis/RWKV-LM.git 
fi

cd RWKV-v7
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

# ---------------------------------------------------------
# 5. Virtual Environment & Blackwell Dependencies
# ---------------------------------------------------------
echo "Setting up Blackwell-compatible environment..."
uv venv
source .venv/bin/activate

uv pip install --upgrade pip
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 # Or cu128 if available

uv pip install ninja wandb weave datasets tqdm

uv pip install -e .

# This environment variable tells PyTorch which architecture to compile for
export TORCH_CUDA_ARCH_LIST="10.0"

# 7. W&B Login (Optional: if your API key isn't in env)
# uv run wandb login [YOUR_KEY_HERE]

MASTER_PORT=$((10000 + $RANDOM % 20000))

# 8. Launch RWKV-7 Training
# Note: Ensure the path to your training script matches your project structure
echo "Launching RWKV training with $NUM_GPUS B200 processes..."
uv run torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    train.py \
   # --without-spaces

echo "Training Job finished at $(date)"