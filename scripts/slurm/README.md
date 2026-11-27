# CARC Slurm launchers

This directory contains both array launchers and their shared submission/runtime helper. Run every command below on CARC from the **repository root**.

## Which launcher to use

| Launcher | Purpose | Array |
|---|---|---|
| `run_all.slurm` | Four RetailRocket SSL comparisons at seed 2028 | `0–3` |
| `run_all_multi_seed.slurm` | Four model variants on both datasets, using additional seeds 2024–2027 | `0–31` |

Each task trains one model, selects its best checkpoint by validation MRR@20, then evaluates clean data and all nine corrupted conditions. Training failure stops the task before robustness evaluation.

## Shared setup

`slurm_common.sh` submits the chosen launcher with these defaults:

| Resource | Default |
|---|---|
| Partition | `gpu` |
| GPU | One Tesla P100 |
| CPUs per task | 6 |
| Memory per task | 8G |
| Time per task | One hour |
| Standard output | `logs/%x_%A_%a.out` |
| Standard error | `logs/%x_%A_%a.err` |

The helper creates `logs/` before submission. Both launchers source it inside their allocated jobs to enable strict Bash error handling, load CUDA, and enter the submission directory.

**Submit through the helper**, rather than calling `sbatch` directly: resource options live in the helper, while each launcher defines its own job name and array size. Additional Slurm options passed to the helper override its defaults.

Before launching concurrent tasks, prepare the environment and datasets once:

```bash
module load cuda
uv sync --frozen
uv run --frozen --no-sync bash scripts/download_data.sh all
```

The tasks use `uv run --frozen --no-sync`, so they reuse the prepared environment. Pre-downloading avoids concurrent dataset downloads.

## RetailRocket single-seed comparisons

```bash
bash scripts/slurm/slurm_common.sh --partition=debug,gpu scripts/slurm/run_all.slurm
```

| Task | Variant |
|---:|---|
| 0 | Unmasked Recency 20%, nearest |
| 1 | Masked Uniform 20%, nearest |
| 2 | Masked Recency 20%, nearest |
| 3 | Masked recency-aware substitution 20%, nearest |

This launcher uses the existing training/evaluation TOML pairs. Their seed-2028 output directories remain defined in those configs. Existing output directories are refused, so completed comparisons do not need to be resubmitted.

## Multi-seed comparisons

```bash
bash scripts/slurm/slurm_common.sh --partition=debug,gpu scripts/slurm/run_all_multi_seed.slurm
```

| Tasks | Dataset | Model |
|---|---|---|
| 0–3 | Diginetica | Plain SR-GNN |
| 4–7 | Diginetica | Unmasked Recency 20%, nearest |
| 8–11 | Diginetica | Masked Recency 20%, nearest |
| 12–15 | RetailRocket | Plain SR-GNN |
| 16–19 | RetailRocket | Unmasked Recency 20%, nearest |
| 20–23 | RetailRocket | Masked Recency 20%, nearest |
| 24–27 | Diginetica | Masked Uniform 20%, nearest |
| 28–31 | RetailRocket | Masked Uniform 20%, nearest |

Within each group, consecutive task IDs use seeds 2024, 2025, 2026, and 2027. The mapping is `config_index = task_id / 4` with integer division and `seed = 2024 + task_id % 4`.

All 32 tasks are independently schedulable. CARC determines how many can run concurrently based on GPU availability and account limits.

The launcher reuses eight existing training TOMLs. It passes seed/output overrides to training and the same output directory to robustness evaluation. No per-seed TOMLs or separate evaluation configs are required.

Results retain the flat experiment layout:

```text
experiments/srgnn_diginetica_global_recursive_seed2024/
experiments/srgnn_diginetica_global_recursive_seed2025/
experiments/srgnn_ssl_recency_retailrocket_budget020_masked_seed2024/
```

Every directory contains `best_model.pt`, `results.json`, and `robustness_results.json`. The new tasks never target seed 2028. Existing seed-2028 results stay where they are; no consolidation or aggregation occurs during submission.

## Monitoring and selective submission

```bash
squeue -u "$USER"
# Replace 123456 with the submitted array job ID.
sacct -j 123456 --format=JobID,State,Elapsed,ExitCode
tail -f logs/srgnn_multi_seed_123456_0.out
```

To submit only selected tasks, pass an array override before the launcher path:

```bash
bash scripts/slurm/slurm_common.sh --partition=debug,gpu --array=24-31 scripts/slurm/run_all_multi_seed.slurm
```

This example submits only the additional Masked Uniform comparisons. For a retry, inspect the task's existing output directory first: training refuses to overwrite any existing directory, including a partial failed run.
