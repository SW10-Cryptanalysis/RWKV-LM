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
# Ensure only one instance of this job name runs at a time
#SBATCH --dependency=singleton

# Exit on any error UNLESS we are explicitly handling it
set -e
set -o pipefail

# --- 1. PATH & DIR SETUP ---
PROJECT_DIR="/ceph/project/SW10-CausalLM/RWKV-LM/RWKV-v7/train_temp"
cd $PROJECT_DIR || exit 1
mkdir -p logs
mkdir -p tmp 

# --- 2. PRE-RUN TERMINATION CHECK ---
FINAL_MODEL="rwkv7_cipher_final.pth"
if [ -f "$FINAL_MODEL" ]; then
    echo "Success: $FINAL_MODEL found. Training complete. Exiting."
    exit 0
fi

# --- 3. ENVIRONMENT SETUP ---
export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=offline
export TORCH_CUDA_ARCH_LIST="8.9" 

# Redirect temp files to Ceph to avoid "No space left on device"
export TMPDIR="$PROJECT_DIR/tmp"
export TEMP="$PROJECT_DIR/tmp"
export TMP="$PROJECT_DIR/tmp"

# Compiler handles for RWKV CUDA Kernels
export CC=$(which gcc)
export CXX=$(which g++)
export NVCC_CCBIN=$CC
export TORCH_EXTENSIONS_DIR="$PROJECT_DIR/torch_extensions_cache"
mkdir -p $TORCH_EXTENSIONS_DIR

export PYTORCH_ALLOC_CONF="expandable_segments:True"
export USE_NINJA=1

# --- 4. DEPENDENCY CHECK ---
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Syncing with uv..."
    uv sync
fi

# --- 5. LOGGING START ---
echo "Job started on $(hostname) at $(date)" | tee -a "logs/train_live_${SLURM_JOB_ID}.log"
nvidia-smi | tee -a "logs/train_live_${SLURM_JOB_ID}.log"

export MASTER_PORT=$(shuf -i 10000-65000 -n 1)
echo "Using Master Port: $MASTER_PORT"

# --- 6. EXECUTION WITH TIMEOUT HANDLING ---
# We stop at 3h 50m (13800s) to give Python time to shut down.
# We temporarily disable 'set -e' so the script doesn't exit when 'timeout' hits.
set +e 
timeout 13800s uv run torchrun \
    --nproc_per_node=4 \
    --master_port=$MASTER_PORT \
    rwkv7_train_cipher.py 2>&1 | tee -a logs/train_live_${SLURM_JOB_ID}.log

exit_code=$?
set -e # Re-enable exit on error

# --- 7. POST-RUN RESUBMISSION LOGIC ---
echo "Process exited with code: $exit_code" | tee -a "logs/train_live_${SLURM_JOB_ID}.log"

# Exit code 124 = Timed out (Expected behavior)
# Exit code 0   = Finished naturally (e.g. reached max epochs)
if [ $exit_code -eq 124 ] || [ $exit_code -eq 0 ]; then
    if [ ! -f "$FINAL_MODEL" ]; then
        echo "Cycle complete. Queuing next job in the chain..." | tee -a "logs/train_live_${SLURM_JOB_ID}.log"
        sbatch $0
    else
        echo "Training reached natural conclusion. Final model found." | tee -a "logs/train_live_${SLURM_JOB_ID}.log"
    fi
else
    echo "ERROR: Training crashed with exit code $exit_code." | tee -a "logs/train_live_${SLURM_JOB_ID}.log"
    echo "Chain broken. Please check the logs/slurm_${SLURM_JOB_ID}.err for details."
    exit $exit_code
fi

echo "Job cycle finished at $(date)" | tee -a "logs/train_live_${SLURM_JOB_ID}.log"