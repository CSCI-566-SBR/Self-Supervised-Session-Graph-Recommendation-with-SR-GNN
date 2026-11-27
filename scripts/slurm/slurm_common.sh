#!/bin/bash
# Execute to submit either launcher; source to share its runtime setup.
set -euo pipefail

setup_slurm_environment() {
    module load cuda
    cd "${SLURM_SUBMIT_DIR:-$PWD}"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    if [[ "$#" -eq 0 ]]; then
        echo "Usage: bash scripts/slurm/slurm_common.sh [sbatch options] launcher-file" >&2
        exit 2
    fi
    mkdir -p logs
    exec sbatch \
        --partition=gpu \
        --gres=gpu:p100:1 \
        --cpus-per-task=6 \
        --mem=8G \
        --time=01:00:00 \
        --output='logs/%x_%A_%a.out' \
        --error='logs/%x_%A_%a.err' \
        "$@"
fi
