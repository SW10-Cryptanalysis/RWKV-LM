#!/bin/bash
#SBATCH --job-name=rwkv7_goose_demo
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=l4
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:l4:1  # We only need 1 GPU for the simplified demo

set -e

# Path setup based on your cluster structure
PROJECT_DIR=/ceph/project/SW10-CausalLM/RWKV-LM/RWKV-v7/train_temp
cd $PROJECT_DIR

mkdir -p logs

echo "Job started on $(hostname) at $(date)" | tee logs/train_live_${SLURM_JOB_ID}.log

# Environment setup
export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=offline  # Keeps it from hanging on login prompts

# Sync dependencies using uv
# Ninja is strictly required for RWKV-7's custom CUDA kernels
uv pip install -r requirements.txt

# Run the simplified demo
# We use standard python because the simplified script isn't built for torchrun/DDP
uv run python rwkv7_train_cipher.py 2>&1 | tee -a logs/train_live_${SLURM_JOB_ID}.log

echo "Job finished at $(date)" | tee -a logs/train_live_${SLURM_JOB_ID}.log