#!/bin/bash
#SBATCH --job-name=srgnn_ssl_retailrocket
#SBATCH --array=0-3

# Submit through: bash scripts/slurm/slurm_common.sh [sbatch options] scripts/slurm/run_all.slurm
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/slurm/slurm_common.sh" || exit 1
setup_slurm_environment

TRAIN_CONFIGS=(
    "configs/srgnn_ssl_recency_retailrocket_budget020.toml"
    "configs/srgnn_ssl_uniform_retailrocket_budget020_masked.toml"
    "configs/srgnn_ssl_recency_retailrocket_budget020_masked.toml"
    "configs/srgnn_ssl_recency_retailrocket_substitution_budget020_masked.toml"
)

EVAL_CONFIGS=(
    "configs/eval_robustness_srgnn_ssl_recency_retailrocket_budget020.toml"
    "configs/eval_robustness_srgnn_ssl_uniform_retailrocket_budget020_masked.toml"
    "configs/eval_robustness_srgnn_ssl_recency_retailrocket_budget020_masked.toml"
    "configs/eval_robustness_srgnn_ssl_recency_retailrocket_substitution_budget020_masked.toml"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID must be set}"
case "${TASK_ID}" in
    0|1|2|3) ;;
    *) echo "Invalid array task ID: ${TASK_ID} (expected 0-3)" >&2; exit 1 ;;
esac
TRAIN_CONFIG="${TRAIN_CONFIGS[$TASK_ID]}"
EVAL_CONFIG="${EVAL_CONFIGS[$TASK_ID]}"

echo "Job ID: ${SLURM_ARRAY_JOB_ID:-local}_${TASK_ID}"
echo "Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Train config: ${TRAIN_CONFIG}"
echo "Eval config:  ${EVAL_CONFIG}"

# Train, then evaluate robustness across all 10 conditions (set -e aborts eval if train fails).
uv run --frozen --no-sync python scripts/train.py --config "${TRAIN_CONFIG}"
uv run --frozen --no-sync python scripts/evaluate_robustness.py --config "${EVAL_CONFIG}"

echo "Done: ${TRAIN_CONFIG} + ${EVAL_CONFIG}"
