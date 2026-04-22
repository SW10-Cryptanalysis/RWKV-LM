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
#SBATCH --dependency=singleton

set -e
set -o pipefail

# 1. PATH & DIR SETUP
PROJECT_DIR="/ceph/project/SW10-CausalLM/RWKV-LM/RWKV-v7/train_temp"
cd $PROJECT_DIR || exit 1
mkdir -p logs
mkdir -p tmp  # For the redirected /tmp files

# 2. PRE-RUN TERMINATION CHECK
FINAL_MODEL="rwkv7_cipher_final.pth"
if [ -f "$FINAL_MODEL" ]; then
    echo "Success: $FINAL_MODEL found. Training complete."
    exit 0
fi

# 3. ENVIRONMENT SETUP (Everything restored + TMPDIR fix)
export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=offline
export TORCH_CUDA_ARCH_LIST="8.9" 

# Fix for "No space left on device" during compilation
export TMPDIR="$PROJECT_DIR/tmp"
export TEMP="$PROJECT_DIR/tmp"
export TMP="$PROJECT_DIR/tmp"

export CC=$(which gcc)
export CXX=$(which g++)
export NVCC_CCBIN=$CC
export TORCH_EXTENSIONS_DIR="$PROJECT_DIR/torch_extensions_cache"
mkdir -p $TORCH_EXTENSIONS_DIR

export PYTORCH_ALLOC_CONF="expandable_segments:True"
export USE_NINJA=1

# 4. DEPENDENCY CHECK
if [ ! -d ".venv" ]; then
    uv sync
fi

# 5. LOGGING START
echo "Job started on $(hostname) at $(date)" | tee -a "logs/train_live_${SLURM_JOB_ID}.log"
nvidia-smi | tee -a "logs/train_live_${SLURM_JOB_ID}.log"

export MASTER_PORT=$(shuf -i 10000-65000 -n 1)
echo "Using Master Port: $MASTER_PORT"

# 6. EXECUTION
# Slurm is 4h (14400s). We stop at 3h 50m (13800s) to save/cleanup.
timeout 13800s uv run torchrun \
    --nproc_per_node=4 \
    --master_port=$MASTER_PORT \
    rwkv7_train_cipher.py 2>&1 | tee -a logs/train_live_${SLURM_JOB_ID}.log

# 7. POST-RUN RESUBMISSION
# We move this to the BOTTOM. It only runs if the job finishes its cycle 
# and the final model still doesn't exist.
if [ ! -f "$FINAL_MODEL" ]; then
    echo "Cycle complete. Queuing continuation job..."
    sbatch $0
fi

echo "Job cycle finished at $(date)" | tee -a "logs/train_live_${SLURM_JOB_ID}.log"