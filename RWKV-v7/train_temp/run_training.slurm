#!/bin/bash
#SBATCH --job-name=rwkv7_cipher
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=l4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:l4:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

set -e
set -o pipefail

# 1. PATH SETUP
PROJECT_DIR="/ceph/project/SW10-CausalLM/RWKV-LM/RWKV-v7/train_temp"
cd $PROJECT_DIR || exit 1
mkdir -p logs

# 2. TERMINATION CHECK
# If the final model exists, we stop the chain.
FINAL_MODEL="rwkv7_cipher_final.pth"
if [ -f "$FINAL_MODEL" ]; then
    echo "Success: $FINAL_MODEL found. Training complete. Breaking chain."
    exit 0
fi

# 3. SELF-RESUBMISSION
# We queue the next job immediately. 'afterany' ensures that even if this 
# job times out or fails, the next one starts and attempts to resume from checkpoint.
if [ -n "$SLURM_JOB_ID" ]; then
    echo "Queuing continuation job..."
    sbatch --dependency=afterany:$SLURM_JOB_ID $0
fi

# 4. ENVIRONMENT SETUP
export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=offline
export TORCH_CUDA_ARCH_LIST="8.9" # Explicitly target L4 Ada architecture

export CC=$(which gcc)
export CXX=$(which g++)
export NVCC_CCBIN=$CC
export TORCH_EXTENSIONS_DIR="$PROJECT_DIR/torch_extensions_cache"
mkdir -p $TORCH_EXTENSIONS_DIR

export PYTORCH_ALLOC_CONF="expandable_segments:True"

export USE_NINJA=1

# 5. DEPENDENCY CHECK (Using uv)
# Syncing ensures the environment is consistent across different nodes
if [ ! -d ".venv" ]; then
    uv sync
fi

#uv pip install nvidia-cuda-nvcc-cu12 nvidia-cuda-runtime-cu12 ninja

#export CUDA_HOME=$(python -c "import nvidia.cuda_nvcc; print(nvidia.cuda_nvcc.__path__[0])")
#export PATH="$CUDA_HOME/bin:$PATH"

echo "Job started on $(hostname) at $(date)" | tee -a "logs/train_live_${SLURM_JOB_ID}.log"
nvidia-smi | tee -a "logs/train_live_${SLURM_JOB_ID}.log"

export MASTER_PORT=$(shuf -i 10000-65000 -n 1)
echo "Using Master Port: $MASTER_PORT"

# 6. EXECUTION
# We use 'timeout' to stop the script 5 minutes before the Slurm walltime expires.
# This gives the Python script a chance to catch the signal and the system to clean up.
# Slurm time is 4h (14400s). We stop at 14100s.
timeout 42900s uv run torchrun --nproc_per_node=4 --master_port=$MASTER_PORT rwkv7_train_cipher.py 2>&1 | tee -a logs/train_live_${SLURM_JOB_ID}.log

echo "Job cycle finished at $(date)" | tee -a "logs/train_live_${SLURM_JOB_ID}.log"