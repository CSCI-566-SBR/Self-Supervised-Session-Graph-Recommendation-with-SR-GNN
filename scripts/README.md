# Experiment & Utility Scripts Guide

Comprehensive execution reference for dataset downloads, baseline training, multi-seed aggregation, and metric visualization.

---

## 1. Plot Training Curves & Comparisons (`plot_results.py`)
Parses `history` from one or more `results.json` files and renders a publication-ready 2x2 multi-panel PNG containing:
* **Subplot 1:** Cross-Entropy Loss Progression (Train vs. Validation Loss)
* **Subplot 2:** Validation Recall@K Progression
* **Subplot 3:** Validation MRR@K Progression
* **Subplot 4:** Validation NDCG@K Progression

> **Note:** The script handles missing metrics or older experiment schemas (e.g., absent `val_loss` or `ndcg`) without crashing.

### Single Experiment Curves
```bash
uv run python scripts/plot_results.py \
    experiments/sasrec_diginetica_global_recursive_seed2028/results.json \
    --output experiments/sasrec_diginetica_global_recursive_seed2028/curves.png
```

### Comparative Multi-Model Overlay (e.g. SR-GNN vs. SASRec)
Pass multiple `results.json` files to overlay their trajectories onto the same axes:
```bash
uv run python scripts/plot_results.py \
    experiments/srgnn_diginetica_global_recursive_seed2028/results.json \
    experiments/sasrec_diginetica_global_recursive_seed2028/results.json \
    --output experiments/comparison_srgnn_vs_sasrec.png \
    --title "Diginetica Baseline Comparison: SR-GNN vs. SASRec"
```

### Command-Line Options
* `--cutoffs` / `-k`: One or more metric cutoffs to display simultaneously (e.g. `--cutoffs 10 20`, default: `10 20`).
* `--title` / `-t`: Optional custom plot title.
* `--output` / `-o`: Output image path (default: `experiment_curves.png`).

---

## 2. Train & Evaluate Models (`train.py`)
Executes training for any registered architecture (SR-GNN, SASRec, etc.) from a TOML configuration, tracks validation MRR@20 early stopping, evaluates test ranking once, and outputs `best_model.pt` and `results.json`.

### Interactive Run
```bash
uv run python scripts/train.py --config configs/sasrec_diginetica_global_recursive.toml
```

### Headless / Background Execution
```bash
nohup uv run python scripts/train.py --config configs/sasrec_diginetica_global_recursive.toml > train_sasrec.log 2>&1 &
```

---

## 3. Multi-Seed Statistical Evaluation (`run_seeds.py`)
Runs an experiment across multiple random seeds, evaluates stability, and exports mean ± std aggregate metrics.

```bash
uv run python scripts/run_seeds.py \
    --config configs/sasrec_diginetica_global_recursive.toml \
    --seeds 2024 2025 2026 2027 2028 \
    --output-root experiments/sasrec_multiseed
```

---

## 4. Download Processed Datasets (`download_data.sh`)
Downloads frozen snapshot artifacts for Diginetica and RetailRocket from Hugging Face Hub.

```bash
# Set HF token if downloading gated or private repositories
export HF_TOKEN="hf_..."

# Execute download pipeline
bash scripts/download_data.sh
```

---

## 5. Evaluate Noise Robustness (`evaluate_robustness.py`)
Evaluates a trained model checkpoint (`best_model.pt` or fallback `model.pt`) across the clean test set and all 9 synthetic noise conditions (`deletion`, `mixed`, `substitution` at 5%, 10%, 20%), auto-downloading datasets from Hugging Face if missing, and exports `robustness_results.json`.

Accepts either a dedicated evaluation config from `configs/` or directly reuses a training config:

```bash
# Evaluate SASRec robustness
uv run python scripts/evaluate_robustness.py --config configs/eval_robustness_sasrec_diginetica.toml

# Evaluate SR-GNN robustness
uv run python scripts/evaluate_robustness.py --config configs/eval_robustness_srgnn_diginetica.toml
```

---

## 6. Plot Robustness Comparisons (`plot_robustness.py`)
Generates publication-ready degradation curves across corruption ratios (Deletion, Substitution, Mixed) or side-by-side grouped bar charts from one or more `robustness_results.json` files.

```bash
# Compare SASRec vs SR-GNN degradation curves across noise ratios
uv run python scripts/plot_robustness.py \
    experiments/sasrec_diginetica_global_recursive_seed2028 \
    experiments/srgnn_diginetica_global_recursive_seed2028 \
    --output experiments/robustness_comparison.png

# Grouped bar chart comparison
uv run python scripts/plot_robustness.py \
    experiments/sasrec_diginetica_global_recursive_seed2028 \
    experiments/srgnn_diginetica_global_recursive_seed2028 \
    --style bars \
    --output experiments/robustness_bars.png
```
