# Reporting Scripts

Standalone utilities for collecting experiment results and preprocessing
metadata into CSV, plus an optional LaTeX draft-table helper.

These scripts do not train models, run preprocessing, or modify input files.

## Requirements

- Use the project's supported Python version and standard uv workflow.
- The three utilities documented here use only Python's standard library.

Run all commands from the repository root:

```powershell
uv sync --locked
```

Use module execution: `uv run python -m scripts.reporting.<module>`.
No separate reporting environment or requirements file is needed.


## Files

```text
docs/reporting.md

scripts/reporting/
    _output.py  # Shared output-safety helper
    collect_results.py
    generate_dataset_summary.py
    export_latex_table.py  # Optional draft-table helper
```

## Input and Output Safety

Keep these entries in the repository's `.gitignore`:

```gitignore
/reports/generated/
```

All scripts:

- Read input files without modifying them.
- Existing outputs are protected by default.
- Use `--overwrite` or `-f` to intentionally replace an output.
- Replacement occurs only after successful writing.
- Input files remain protected, including when overwrite is enabled.
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
uv run python -m scripts.reporting.collect_results --results-root experiments --output reports/generated/all_results.csv
```

### Filter Models, Seeds, and Datasets

```powershell
uv run python -m scripts.reporting.collect_results --results-root experiments --model srgnn sasrec --seed 2028 2029 --dataset diginetica --output reports/generated/diginetica_results.csv
```

Within one filter, any listed value may match. Across different filters,
all conditions must match.

Model and dataset matching is case-insensitive.

### Collect One Exact Run

```powershell
uv run python -m scripts.reporting.collect_results --results-root experiments/srgnn_diginetica_global_recursive_seed2028 --output reports/generated/srgnn_diginetica_2028.csv
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

A robustness file contains metrics indexed by clean/noisy evaluation
conditions and uses a different schema. Robustness evaluation and plotting are handled by the shared utilities documented in `scripts/README.md`. This reporting package does not export robustness results to CSV.

## 2. Export Dataset Statistics

Script:

```text
scripts/reporting/generate_dataset_summary.py
```

Reads clean `preprocessing_summary.json` metadata and exports one row per
distinct resolved input path.

### From Processed Dataset Directories

```powershell
uv run python -m scripts.reporting.generate_dataset_summary --summaries data/processed/diginetica/clean/preprocessing_summary.json data/processed/retailrocket/clean/preprocessing_summary.json --output reports/generated/dataset_summary.csv
```

### From Downloaded Metadata

The filenames can differ, provided their contents use the expected schema:

```powershell
uv run python -m scripts.reporting.generate_dataset_summary --summaries reports/generated/inputs/preprocessing_summary_digi.json reports/generated/inputs/preprocessing_summary_retail.json --output reports/generated/dataset_summary.csv
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



## 3. Export LaTeX Tables - Optional Draft Helper

Optionally converts selected CSV columns into a LaTeX draft-table fragment.
It is not a required step in training, evaluation, or report generation.
Review the generated numbers, formatting, and compiled layout before use.

```powershell
uv run python -m scripts.reporting.export_latex_table --csv reports/generated/all_results.csv --columns model seed test.recall@20 test.mrr@20 --headers Model Seed Recall@20 MRR@20 --round-columns test.recall@20 test.mrr@20 --decimals 4 --caption "Diginetica test results; ranking scores are fractions." --label tab:diginetica-results --output reports/generated/model_results.tex
```

### Behavior and Limitations

- Selected column names must exist in the input CSV.
- Custom headers must match the number of selected columns.
- `--round-columns` controls which columns receive numeric formatting.
- Blank cells become N/A.
- Text, headers, and captions are escaped by default; --raw-headers disables escaping only for explicitly supplied headers.
- Use `--raw-headers` with explicit `--headers` to allow trusted LaTeX formatting in headers. CSV cells and captions remain escaped.
- In PowerShell, single-quote mathematical headers containing `$`, for example: '$\mathrm{Recall}@20$'.
- Fractions remain unchanged by default.
- Use `--percent-columns` to display selected fraction columns as percentages, e.g. 0.4951 becomes 49.51% with `--decimals 2`.
- Selected nonblank percentage values must be numeric fractions in [0, 1].
- Source CSV values are never changed.
- The document requires `\usepackage{booktabs}`.
- Existing output files require explicit --overwrite or -f to replace.

Copy the generated fragment into the report or include it using `\input`.
The exporter does not compile LaTeX or verify the scientific interpretation
of the table.


## Shared Evaluation and Plotting Utilities

Use the existing shared utilities for training curves, robustness evaluation,
and comparison/system-metric figures:

- `scripts/plot_results.py`
- `scripts/evaluate_robustness.py`
- `scripts/plot_robustness.py`
- `scripts/plot_system_metrics.py`
- `scripts/plot_stratified.py`

Consult `scripts/README.md` and each command's `--help` for supported inputs
and options. These tools do not necessarily consume the collector's CSV.

The three utilities in `scripts/reporting/` complement those tools without
running model training or robustness evaluation.

The output-safety guarantees documented here apply only to the three
reporting commands, not automatically to the shared utilities.

## Tests

Four test files cover the three retained commands and shared output helper.
The expected total is 58 tests: 12 collector, 13 dataset-summary,
23 LaTeX-export, and 10 overwrite tests.

Run from the repository root:

```powershell
uv run python -m unittest discover -s tests -p "test_reporting_scripts.py" -v
uv run python -m unittest discover -s tests -p "test_dataset_summary_reporting.py" -v
uv run python -m unittest discover -s tests -p "test_export_latex_table.py" -v
uv run python -m unittest discover -s tests -p "test_reporting_overwrite.py" -v
```

Tests use temporary files and cover:

- Filtering, metadata extraction, and missing values.
- Invalid inputs and duplicate source-path handling.
- Source-file preservation and input/output alias rejection.
- Default output protection and explicit overwrite.
- Failed-write cleanup and preservation of existing outputs.
- LaTeX escaping, numeric formatting, and percentage conversion.

Passing tests does not replace checking real source artifacts or reviewing
the compiled layout of exported LaTeX tables.

## Current Scope

Implemented here:

- Experiment CSV collection with model, seed, and dataset filters.
- Clean dataset-statistics CSV export.
- Optional LaTeX draft-table export from selected CSV columns.
- Shared output protection and automated tests.

Handled separately by shared project utilities:

- Saved-checkpoint evaluation on clean/noisy test conditions.
- Training, robustness, comparison, and system-metric plotting.

Not implemented by this reporting package:

- Robustness-result CSV export or noise-degradation summary tables.
- Multi-seed means, uncertainty intervals, or statistical tests.
- Hardware-normalized efficiency comparisons.

Use approved experiment results and frozen preprocessing metadata when
generating report tables. The exporters do not establish that runs are
scientifically comparable.