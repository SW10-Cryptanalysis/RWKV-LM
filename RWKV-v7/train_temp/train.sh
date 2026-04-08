#!/bin/bash
set -e

# 1. Navigate to your mounted workspace
cd /work

# 2. Repository Setup
if [ ! -d "RWKV-LM" ]; then
    echo "Cloning repository..."
    git clone -b dev https://github.com/SW10-Cryptanalysis/RWKV-LM
fi

cd RWKV-LM/RWKV-v7/train_temp  # Navigating to the specific train dir used in your SLURM script
mkdir -p logs
export HF_HUB_ENABLE_HF_TRANSFER=1

# 3. GPU Detection
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Training Job started on $(hostname) at $(date) with $NUM_GPUS GPU(s)"

# 4. uv & Dependency Management
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# We use 'uv pip install' within a managed environment for consistency
echo "Setting up dependencies..."
uv venv .venv
source .venv/bin/activate
uv pip install --upgrade pip
uv pip install ninja  # Explicitly needed for RWKV-7 kernels
uv pip install -r requirements.txt

# 5. Performance & NCCL Optimizations
export OMP_NUM_THREADS=16
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# H100/A100 optimizations
if [ "$NUM_GPUS" -gt 1 ]; then
    export NCCL_P2P_DISABLE=0
    export NCCL_IB_DISABLE=0
    LAUNCHER="torchrun --nproc_per_node=$NUM_GPUS"
else
    LAUNCHER="python"
fi

# 6. Launch Training
echo "Launching training with $LAUNCHER..."

# Using 'python' or 'torchrun' directly since we activated the venv
$LAUNCHER rwkv7_train_cipher.py 2>&1 | tee -a logs/train_live_standalone.log

echo "Training Job finished at $(date)"