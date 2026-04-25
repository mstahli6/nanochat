#!/bin/bash
#SBATCH --job-name=optimizer_benchmarks
#SBATCH --partition=gpu          # Adjust if your student allotment uses a specific partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1             # Single A6000 request
#SBATCH --mem=64G
#SBATCH --time=72:00:00          # Total time for 4 runs (~16h each) + overhead
#SBATCH --output=logs/%j_benchmarks.out

# -----------------------------------------------------------------------------
# 1. Environment Setup
# -----------------------------------------------------------------------------
module load cuda/12.1            # Match the version used for your JAX/PyTorch build
source .venv/bin/activate

# -----------------------------------------------------------------------------
# 2. Configuration Toggle
# -----------------------------------------------------------------------------
MODE="TEST"
# MODE="FULL"

if [ "$MODE" == "TEST" ]; then
    echo "RUNNING IN TEST MODE (10 MINUTES PER OPTIMIZER)"
    ITER_LIMIT="--num-iterations=100"
    SAVE_FREQ="--save-every=50"
    EVAL_FREQ="--eval-every=20"
else
    echo "RUNNING IN FULL MODE (16 HOUR MARATHON)"
    ITER_LIMIT=""                # Let target-param-data-ratio determine the steps
    SAVE_FREQ="--save-every=2000"
    EVAL_FREQ="--eval-every=200"
fi

# Point 5: Scaled architecture for 1x A6000
COMMON_ARGS="--depth=12 \
             --device-batch-size=4 \
             --total-batch-size=524288 \
             --target-param-data-ratio=8 \
             $ITER_LIMIT \
             $SAVE_FREQ \
             $EVAL_FREQ"

# -----------------------------------------------------------------------------
# 3. Sequential Execution Logic
# -----------------------------------------------------------------------------
OPTIMIZERS=("adamw" "muon" "soap" "kl-shampoo")

for OPT in "${OPTIMIZERS[@]}"
do
    echo "----------------------------------------------------------------"
    echo "STARTING RUN: $OPT at $(date)"
    echo "----------------------------------------------------------------"
    
    # We use a unique model-tag for each run to separate checkpoints and CSV logs
    RUN_NAME="${OPT}_16h_$(date +%Y%m%d)"
    
    python -m scripts.base_train \
        $COMMON_ARGS \
        --optimizer=$OPT \
        --run=$RUN_NAME \
        --model-tag=$RUN_NAME

    echo "COMPLETED RUN: $OPT at $(date)"
    
    # Optional: Clear peak memory stats or temp cache between runs
    # rm -rf ~/.cache/torch_extensions (Only if you suspect compiler bloat)
done

echo "ALL BENCHMARKS COMPLETE."