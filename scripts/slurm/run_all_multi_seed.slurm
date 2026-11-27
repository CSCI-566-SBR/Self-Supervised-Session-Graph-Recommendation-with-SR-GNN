#!/bin/bash
#SBATCH --job-name=srgnn_multi_seed
#SBATCH --array=0-31

# Submit through: bash scripts/slurm/slurm_common.sh [sbatch options] scripts/slurm/run_all_multi_seed.slurm
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/slurm/slurm_common.sh" || exit 1
setup_slurm_environment

# Four consecutive tasks per config: seeds 2024, 2025, 2026, 2027.
TRAIN_CONFIGS=(
    "configs/srgnn_diginetica_global_recursive.toml"
    "configs/srgnn_ssl_recency_diginetica_budget020.toml"
    "configs/srgnn_ssl_diginetica_final_recency_masked.toml"
    "configs/srgnn_retailrocket_global_recursive.toml"
    "configs/srgnn_ssl_recency_retailrocket_budget020.toml"
    "configs/srgnn_ssl_recency_retailrocket_budget020_masked.toml"
    "configs/srgnn_ssl_uniform_diginetica_budget020_masked.toml"
    "configs/srgnn_ssl_uniform_retailrocket_budget020_masked.toml"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID must be set}"
case "${TASK_ID}" in
    [0-9]|1[0-9]|2[0-9]|3[0-1]) ;;
    *) echo "Invalid array task ID: ${TASK_ID} (expected 0-31)" >&2; exit 1 ;;
esac
CONFIG_INDEX=$((TASK_ID / 4))
SEED=$((2024 + TASK_ID % 4))
TRAIN_CONFIG="${TRAIN_CONFIGS[$CONFIG_INDEX]}"
CONFIG_STEM="${TRAIN_CONFIG##*/}"
CONFIG_STEM="${CONFIG_STEM%.toml}"
OUTPUT_DIR="experiments/${CONFIG_STEM}_seed${SEED}"

echo "Job ID: ${SLURM_ARRAY_JOB_ID:-local}_${TASK_ID}"
echo "Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Train/eval config: ${TRAIN_CONFIG} | Seed: ${SEED}"
echo "Output directory: ${OUTPUT_DIR}"

uv run --frozen --no-sync python scripts/train.py \
    --config "${TRAIN_CONFIG}" --seed "${SEED}" --output-dir "${OUTPUT_DIR}"
uv run --frozen --no-sync python scripts/evaluate_robustness.py \
    --config "${TRAIN_CONFIG}" --output-dir "${OUTPUT_DIR}"

echo "Done: ${TRAIN_CONFIG} | Seed: ${SEED}"
