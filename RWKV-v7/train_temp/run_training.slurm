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
# Ensure only one job with this name runs at a time
#SBATCH --dependency=singleton

set -e
set -o pipefail

PROJECT_DIR="/ceph/project/SW10-CausalLM/RWKV-LM/RWKV-v7/train_temp"
cd $PROJECT_DIR || exit 1

# 1. PRE-RUN CHECK
FINAL_MODEL="rwkv7_cipher_final.pth"
if [ -f "$FINAL_MODEL" ]; then
    echo "Success: $FINAL_MODEL found. Breaking chain."
    exit 0
fi

# 2. ENVIRONMENT SETUP
export WANDB_MODE=offline
export TORCH_EXTENSIONS_DIR="$PROJECT_DIR/torch_extensions_cache"
export MASTER_PORT=$(shuf -i 10000-65000 -n 1)

# 3. EXECUTION 
# Slurm walltime is 14400s (4h). 
# We stop at 13800s (3h 50m) to allow 10 mins for saving/cleanup.
echo "Job started at $(date)" | tee -a "logs/train_live_${SLURM_JOB_ID}.log"

timeout 13800s uv run torchrun \
    --nproc_per_node=4 \
    --master_port=$MASTER_PORT \
    rwkv7_train_cipher.py 2>&1 | tee -a logs/train_live_${SLURM_JOB_ID}.log

# 4. POST-RUN RESUBMISSION
# Only resubmit if the job finished naturally (timed out) 
# and the final model still hasn't been created.
if [ ! -f "$FINAL_MODEL" ]; then
    echo "Cycle complete. Queuing next job..."
    sbatch $0
fi