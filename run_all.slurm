#!/bin/bash
#SBATCH --job-name=srgnn_ssl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

TRAIN_CONFIGS=(
    "configs/srgnn_ssl_uniform_diginetica_budget010.toml"
    "configs/srgnn_ssl_uniform_diginetica_budget010_stochastic.toml"
    "configs/srgnn_ssl_uniform_diginetica_budget020.toml"
    "configs/srgnn_ssl_uniform_diginetica_budget020_stochastic.toml"
    "configs/srgnn_ssl_recency_diginetica_budget010.toml"
    "configs/srgnn_ssl_recency_diginetica_budget010_stochastic.toml"
    "configs/srgnn_ssl_recency_diginetica_budget020.toml"
    "configs/srgnn_ssl_recency_diginetica_budget020_stochastic.toml"
)

EVAL_CONFIGS=(
    "configs/eval_robustness_srgnn_ssl_uniform_diginetica_budget010.toml"
    "configs/eval_robustness_srgnn_ssl_uniform_diginetica_budget010_stochastic.toml"
    "configs/eval_robustness_srgnn_ssl_uniform_diginetica_budget020.toml"
    "configs/eval_robustness_srgnn_ssl_uniform_diginetica_budget020_stochastic.toml"
    "configs/eval_robustness_srgnn_ssl_recency_diginetica_budget010.toml"
    "configs/eval_robustness_srgnn_ssl_recency_diginetica_budget010_stochastic.toml"
    "configs/eval_robustness_srgnn_ssl_recency_diginetica_budget020.toml"
    "configs/eval_robustness_srgnn_ssl_recency_diginetica_budget020_stochastic.toml"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID must be set}"
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
