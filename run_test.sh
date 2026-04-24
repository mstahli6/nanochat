#!/bin/bash
#SBATCH -p student-gpu
#SBATCH --account=math440_fall2025_eileenrmartin
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00             # 15 min limit for a 10 min test
#SBATCH --array=0-3                 # 0=AdamW, 1=Muon, 2=SOAP, 3=KL-Shampoo
#SBATCH -o logs/test_%A_%a.out
#SBATCH -e logs/test_%A_%a.err

# 1. Setup the optimizer mapping
OPTIMIZERS=("adamw" "muon" "soap" "kl-shampoo")
OPTIM=${OPTIMIZERS[$SLURM_ARRAY_TASK_ID]}

# 2. Redirect output to Wendian Scratch (IMPORTANT: Ensure this path exists)
export NANOCHAT_BASE_DIR="/scratch/$USER/nanochat_tests"
mkdir -p $NANOCHAT_BASE_DIR
mkdir -p logs

echo "Running 10-minute test for Optimizer: $OPTIM"

# 3. The Test Execution
# - depth 4: much faster forward/backward pass
# - iterations 200: enough to trigger two evaluations and two saves
# - model-tag: ensures each optimizer gets its own folder in scratch
srun uv run python -m scripts.base_train \
    --depth=4 \
    --device-batch-size=64 \
    --total-batch-size=131072 \
    --max-seq-len=512 \
    --num-iterations=200 \
    --window-pattern=L \
    --optimizer=$OPTIM \
    --run=test_run_${OPTIM} \
    --model-tag=test_${OPTIM} \
    --save-every=100 \
    --eval-every=100

echo "Test for $OPTIM complete."