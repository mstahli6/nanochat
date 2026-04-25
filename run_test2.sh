#!/bin/bash
#SBATCH -p ams-part-1
#SBATCH --gres=gpu:a6000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:15:00          # 2 hours for training + 15m buffer
#SBATCH --array=0-3              
#SBATCH -o logs/test_%A_%a.out
#SBATCH -e logs/test_%A_%a.err

# 1. LOAD MODULES
module load libs/cuda/12.2

# 2. PATH HACKING
VENV_SITE_PACKAGES="$HOME/nanochat/.venv/lib/python3.10/site-packages"
ALL_NVIDIA_LIBS=$(find $VENV_SITE_PACKAGES/nvidia -type d -name "lib" 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${ALL_NVIDIA_LIBS}${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
export PATH="$HOME/.local/bin:$PATH"

# 3. DIRECTORY SETUP
export NANOCHAT_BASE_DIR="$HOME/scratch/nanochat_tests"
export WANDB_MODE="offline"

OPTIMIZERS=("adamw" "muon" "soap" "kl-shampoo")
OPTIM=${OPTIMIZERS[$SLURM_ARRAY_TASK_ID]}

echo "Job starting on node: $SLURM_NODELIST"
echo "Running Optimizer: $OPTIM (Medium Scale / medtest1)"

# 4. EXECUTION (Auto-Scaling Mode)
srun .venv/bin/python -m scripts.base_train \
    --depth=12 \
    --device-batch-size=32 \
    --total-batch-size=-1 \
    --target-param-data-ratio=12 \
    --max-seq-len=512 \
    --optimizer=$OPTIM \
    --run=medtest1_run_${OPTIM} \
    --model-tag=medtest1_${OPTIM} \
    --save-every=1000 \
    --eval-every=1000