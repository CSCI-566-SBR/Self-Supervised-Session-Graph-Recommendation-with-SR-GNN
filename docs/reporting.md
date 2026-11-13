# Reporting Scripts

Standalone tools for generating reproducible CSV tables and charts from saved
experiment results and preprocessing metadata.

These scripts do not train models, run preprocessing, or modify input files.

## Requirements

- Python 3.12 or newer, matching the project requirement.
- CSV exporters use only the Python standard library.
- Plotting requires Matplotlib >= 3.8.

Use your project environment if it already includes Matplotlib. Otherwise,
create an isolated reporting environment:

```powershell
python -m venv .venv-reporting
.\.venv-reporting\Scripts\python.exe -m pip install "matplotlib>=3.8"
```

Verify installation:

```powershell
.\.venv-reporting\Scripts\python.exe -c "import matplotlib; print(matplotlib.__version__)"
```

The commands below assume Windows and are run from the repository root.
On Linux/macOS, the reporting interpreter is `.venv-reporting/bin/python`.

The isolated environment is a local setup option. It does not add Matplotlib
to the project's dependency declaration or update its lockfile.

## Files

```text
scripts/reporting/
    collect_results.py
    generate_dataset_summary.py
    plot_model_comparison.py
    plot_system_metrics.py
    README.md
```

## Input and Output Safety

Keep these entries in the repository's `.gitignore`:

```gitignore
/reports/generated/
/.venv-reporting/
```

All scripts:

- Read input files without modifying them.
- Refuse to overwrite existing output files.
- Require a new output filename for an updated export.
- Run locally without uploading files to GitHub.

They do not protect training outputs from changes made by other programs.
Each training run should have its own experiment directory.

Recommended generated-output location:

```text
reports/generated/
```

Downloaded metadata used for local reporting checks can be kept under:

```text
reports/generated/inputs/
```

These paths are ignored by Git, not deleted.

## 1. Collect Experiment Results

Script:

```text
scripts/reporting/collect_results.py
```

Recursively finds files named `results.json` and writes one CSV row per
distinct resolved source path.

Different models normally use the same filename in different directories:

```text
experiments/
    sasrec_diginetica_global_recursive_seed2028/
        results.json
    srgnn_diginetica_global_recursive_seed2028/
        results.json
```

The collector reads model identity from the file's metadata.

### Collect All Runs

```powershell
python scripts/reporting/collect_results.py --results-root experiments --output reports/generated/all_results.csv
```

### Filter Models, Seeds, and Datasets

```powershell
python scripts/reporting/collect_results.py --results-root experiments --model srgnn sasrec --seed 2028 2029 --dataset diginetica --output reports/generated/diginetica_results.csv
```

Within one filter, any listed value may match. Across different filters,
all conditions must match.

Model and dataset matching is case-insensitive.

### Collect One Exact Run

```powershell
python scripts/reporting/collect_results.py --results-root experiments/srgnn_diginetica_global_recursive_seed2028 --output reports/generated/srgnn_diginetica_2028.csv
```

### Expected Input

Each result file must contain a non-empty `test` object.

Supported metadata includes:

- Experiment name, dataset, model type, and seed.
- Final test and best-validation metrics.
- Configuration and model parameters.
- Dataset checksums and artifact provenance.
- Runtime, parameter count, and peak GPU allocated memory.
- Software versions and Git commit.
- A `noise` object, if present in the result file.

Legacy SR-GNN files without `model_type` are identified using
`config.gnn_steps`. This inference is recorded in `model_identification`.
Otherwise, an unrecorded model is labelled `unknown`.

### Behavior

- Missing values remain blank, not zero.
- Ranking metrics remain fractions, not percentages.
- Separate runs and seeds are not averaged.
- Distinct result files with identical contents remain separate rows.
- Repeated references to the same resolved path are collected once.
- Nested fields become dotted CSV columns, such as `test.mrr@20`.
- Per-epoch history is not exported; its entry count is recorded.
- Invalid discovered input files stop collection, even when a filter would
  otherwise exclude them.
- No matching results causes an error without creating a CSV.

Runs with the same model, dataset, and seed can still have different
hyperparameters. Keeping each run separate is intentional.

### Robustness Results Are Separate

This collector reads `results.json`, not `robustness_results.json`.

A robustness file contains metrics indexed by clean/noisy evaluation
conditions and uses a different schema. Use the existing robustness plotting
utilities described later in this README.

## 2. Export Dataset Statistics

Script:

```text
scripts/reporting/generate_dataset_summary.py
```

Reads clean `preprocessing_summary.json` metadata and exports one row per
distinct resolved input path.

### From Processed Dataset Directories

```powershell
python scripts/reporting/generate_dataset_summary.py --summaries data/processed/diginetica/clean/preprocessing_summary.json data/processed/retailrocket/clean/preprocessing_summary.json --output reports/generated/dataset_summary.csv
```

### From Downloaded Metadata

The filenames can differ, provided their contents use the expected schema:

```powershell
python scripts/reporting/generate_dataset_summary.py --summaries reports/generated/inputs/preprocessing_summary_digi.json reports/generated/inputs/preprocessing_summary_retail.json --output reports/generated/dataset_summary.csv
```

### Exported Information

- Dataset and protocol version.
- Training-vocabulary size.
- Session counts for training, validation, and test.
- Prefix/target example counts for each split.
- Minimum item frequency and session length.
- Split strategy, window definitions, and cutoff timestamps.
- Filtering, vocabulary, and prefix-generation policies.
- Source-summary path and SHA-256 fingerprint.

### Limitations

The script expects the current preprocessing schema, including nested
`counts` and `split_strategy` objects.

- Noisy summaries are rejected.
- Noise manifests are not inputs to this script.
- Repeated references to the same resolved file are exported once.
- Separate files remain separate rows, even for the same dataset.
- Missing required statistics cause an error rather than fabricated values.
- Statistics are read from metadata, not recalculated or verified against
  raw datasets or pickle artifacts.
- The output is CSV, not a formatted LaTeX table.

Real metadata must be supplied to generate real dataset statistics.

## 3. Plot Final Test Metrics

Script:

```text
scripts/reporting/plot_model_comparison.py
```

Reads the collector's CSV and plots one panel per requested metric.

```powershell
.\.venv-reporting\Scripts\python.exe scripts/reporting/plot_model_comparison.py --csv reports/generated/all_results.csv --dataset diginetica --metrics recall@20 mrr@20 ndcg@20 --output reports/generated/model_comparison.png
```

### Filter Models and Seeds

```powershell
.\.venv-reporting\Scripts\python.exe scripts/reporting/plot_model_comparison.py --csv reports/generated/all_results.csv --dataset diginetica --model srgnn sasrec --seed 2028 --metrics recall@10 recall@20 mrr@20 ndcg@20 --output reports/generated/selected_comparison.png
```

Request metrics recorded in at least one selected run.

### Behavior

- Each bar represents one saved run.
- Runs and seeds are not averaged.
- Supports recorded Recall@K, MRR@K, and NDCG@K.
- Scores are displayed as fractions.
- Missing measurements are labelled N/A, not zero.
- A metric missing from every selected run causes an error.
- Duplicate source-file rows are rejected.
- Multi-run charts require matching test and item-mapping checksums.
- One dataset is selected per chart.
- Output format is PNG.

### Comparison Limitations

Matching fingerprints establish a shared test artifact and item mapping.
They do not establish equal training data, tuning budgets, hardware, or
training settings.

Inspect configuration columns before drawing conclusions. The script does
not automatically select a winner, calculate multi-seed averages, or perform
statistical significance tests.

These charts compare final test metrics. They are not training curves.

## 4. Plot System Metrics

Script:

```text
scripts/reporting/plot_system_metrics.py
```

Produces two panels:

1. Recorded run runtime in seconds.
2. Peak GPU allocated memory in MiB.

```powershell
.\.venv-reporting\Scripts\python.exe scripts/reporting/plot_system_metrics.py --csv reports/generated/all_results.csv --dataset diginetica --output reports/generated/system_metrics.png
```

### Filter Models and Seeds

```powershell
.\.venv-reporting\Scripts\python.exe scripts/reporting/plot_system_metrics.py --csv reports/generated/all_results.csv --dataset diginetica --model srgnn sasrec --seed 2028 --output reports/generated/selected_system_metrics.png
```

### Behavior

- Each bar represents one run; runs are not averaged.
- The recorded device is shown when available.
- Runtime is read from `runtime_seconds`.
- Memory is read from `peak_gpu_memory_bytes`.
- Memory conversion uses 1 MiB = 1,048,576 bytes.
- Missing measurements appear as N/A, not zero.
- Negative, non-finite, or non-numeric measurements are rejected.
- If all selected runs lack both measurements, plotting fails.
- Duplicate source-file rows are rejected.
- Output format is PNG.

### Interpretation Limitations

Hardware, epoch counts, batch sizes, and run settings may differ. These
charts are descriptive, not hardware-normalized efficiency rankings.

- CPU/CUDA labels do not identify exact hardware.
- Runtime reflects the runner's measured interval, not necessarily
  training-only time or inference latency.
- Peak GPU allocated memory is not total GPU usage or system RAM.
- Missing GPU memory for CPU execution does not mean zero memory usage.
- Measurements are read from saved results, not measured on the machine
  generating the chart.

`eval_time_seconds` in robustness results measures evaluation time. It must
not be treated as equivalent to the experiment's `runtime_seconds`.

## Existing Training and Robustness Plotting Utilities

The SASRec PR adds or extends these separate tools:

```text
scripts/plot_results.py
scripts/evaluate_robustness.py
scripts/plot_robustness.py
src/recsys/evaluation/plotting.py
```

They become available on `main` when the corresponding PR is merged.

- `plot_results.py` plots training/validation histories.
- `evaluate_robustness.py` evaluates saved checkpoints on clean/noisy test
  conditions. It requires model dependencies, checkpoints, and processed data.
- `plot_robustness.py` plots saved `robustness_results.json` files.

Consult `scripts/README.md` on the branch containing those utilities for
their commands and requirements.

The scripts in `scripts/reporting/` complement these tools without importing
model implementations or running robustness evaluation.

Overwrite and validation behavior documented here applies to the four
reporting scripts, not automatically to those separate utilities.

## Tests

The four test files currently contain 55 tests.

Use the environment containing Matplotlib:

```powershell
.\.venv-reporting\Scripts\python.exe -m unittest discover -s tests -p "test_reporting_scripts.py" -v
```

```powershell
.\.venv-reporting\Scripts\python.exe -m unittest discover -s tests -p "test_dataset_summary_reporting.py" -v
```

```powershell
.\.venv-reporting\Scripts\python.exe -m unittest discover -s tests -p "test_model_comparison_reporting.py" -v
```

```powershell
.\.venv-reporting\Scripts\python.exe -m unittest discover -s tests -p "test_system_metrics_reporting.py" -v
```

Tests use temporary files, not actual experiment artifacts. They cover:

- Filtering and metadata extraction.
- Missing values and invalid inputs.
- Duplicate source-path handling.
- Source-file preservation.
- Comparison fingerprint checks.
- PNG generation.
- Existing-output protection.

Passing tests does not replace checking the real experiment artifacts and
visually reviewing generated figures.

## Current Scope

Implemented here:

- Experiment CSV collection with model, seed, and dataset filters.
- Clean dataset-statistics CSV export.
- Final test-metric bar charts.
- Runtime and peak GPU allocated-memory charts.
- Automated tests for all four scripts.

Handled by the separate robustness utilities:

- Saved-checkpoint evaluation on clean/noisy test conditions.
- Robustness line charts and grouped bar charts.

Not implemented by this reporting package:

- Collection of `robustness_results.json` into CSV.
- Multi-seed means, uncertainty intervals, or statistical tests.
- Hardware-normalized efficiency comparisons.

Use the actual approved experiment results and frozen preprocessing metadata
when generating figures and tables for the report.