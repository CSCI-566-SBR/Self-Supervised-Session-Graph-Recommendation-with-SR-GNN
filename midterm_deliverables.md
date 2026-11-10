# Midterm Deliverables: Self-Supervised Session Graph Recommendation with SR-GNN

This document outlines the current project status for the midterm milestone, detailing completed components, active in-progress tasks, and the technical roadmap following the midterm report.

---

## 1. Completed Work

### Core Repository & Environment Architecture
* **Environment Setup:** Python 3.12 managed via `uv` with virtual environment isolation and lockfile pinning (`uv.lock`).
* **Modular Codebase Layout:** Standardized namespaces established:
  * `recsys.data`: Dataset ingestion, graph transformations, perturbation, and validation.
  * `recsys.evaluation`: Top-$K$ ranking metric computation and report aggregation.
  * `recsys.models`: Baseline and experimental neural recommendation architectures.
  * `recsys.training`: Model-agnostic training loops, configuration schemas, and execution runners.
* **Tooling & Standards:** Strict linting, type-checking, and test runner configurations configured in `pyproject.toml`.

### Evaluation Infrastructure (`main` branch)
* **Status:** Merged to `main` via PR #3 (`3110052`).
* **Implementation:** `src/recsys/evaluation/metrics.py`, `src/recsys/evaluation/reporting.py`.
* **Key Components:**
  * **Top-$K$ Ranking Metrics:** Vectorized calculations for **Recall@K**, **MRR@K**, and **NDCG@K** supporting standard evaluation cutoffs ($K \in \{10, 20\}$).
  * **Tie-Breaking & Missing Target Policy:** Strictly resolves score ties in favor of smaller item IDs; treats targets outside top-$K$ as zero-score predictions.
  * **Batch Aggregation:** `RankingSummary` utility providing example-weighted batch evaluation without memory overhead.
  * **Test Harness:** Comprehensive unit tests in `tests/test_evaluation.py` validating exact ranking mathematics, bounds, and boundary edge cases.

### Data Preprocessing & Ingestion Engine (`data_perturbation` branch)
* **Status:** Implemented in `origin/data_perturbation` (`1ab492b`).
* **Implementation:** `src/recsys/data/preprocess.py`, `pipeline.py`, `processed.py`, `validate.py`, `hub.py`.
* **Key Components:**
  * **Standardized 5-File Contract:** Every model-facing dataset directory produces/consumes `train.txt`, `validation.txt`, `test.txt`, `item_mapping.pkl`, and `preprocessing_summary.json`.
  * **Deterministic Clean Ingestion:** Preprocessing workflows for **Diginetica** and **RetailRocket** with contiguous 1-based indexing (0 reserved for model padding).
  * **Filtering Logic:** Iterative pruning removing sessions shorter than 2 interactions and items appearing fewer than 5 times.
  * **Temporal Windows:** Chronological evaluation splits (last 7 days validation and last 7 days test for Diginetica; 14 days validation and 14 days test for RetailRocket) with prefix expansion.
  * **Integrity Validation & Remote Hub:** SHA-256 artifact verification in `validate.py` and automated dataset snapshot management with Hugging Face Hub (`hub.py`).

### Offline Robustness Perturbation Framework (`data_perturbation` branch)
* **Status:** Implemented in `origin/data_perturbation`.
* **Implementation:** `src/recsys/data/perturb.py`.
* **Key Components:**
  * **Controlled Corruption Sets:** Deterministic generation of noisy test splits at corruption ratios $p \in \{0.05, 0.10, 0.20\}$ with fixed seed `2026`.
  * **Three Perturbation Operators:** Item deletion, random vocabulary substitution, and mixed corruption.
  * **Invariance Guarantees:** Strict protection ensuring held-out target items and sequence boundaries ($\ge 2$) are never corrupted.

### Stochastic Graph Augmentation Pipeline (`data_perturbation` branch)
* **Status:** Implemented in `origin/data_perturbation`.
* **Implementation:** `src/recsys/data/graph_augmentations.py`.
* **Key Components:**
  * **Dual-View Generator:** In-memory view generator (`make_two_views`) producing paired stochastic graph views for contrastive learning.
  * **Augmentation Operators:**
    1. *Uniform Edge Dropout:* Uniform edge deletion with fallback preventing graph collapse.
    2. *Recency-Aware Perturbation:* Dropout probability decaying chronologically via $p \cdot (1 - t/L)^\gamma$, strictly preserving the transition to the final observed item.
    3. *Uniform Node Identity Dropout:* Random candidate item removal with fallback safeguarding minimum session length $\ge 2$ and valid edge transitions.

### SR-GNN Baseline Implementation (`feature/srgnn-baseline` branch)
* **Status:** Implemented in `origin/feature/srgnn-baseline` (`a9d742f`).
* **Implementation:** `src/recsys/models/baselines/srgnn.py`, `src/recsys/data/session_graph.py`, `src/recsys/training/runner.py`, `baseline.py`, `config.py`.
* **Key Components:**
  * **Modern PyTorch SR-GNN:** Faithfully recreates the original gated graph neural network (GGNN) update step with separate incoming and outgoing normalized connection weights, followed by hybrid session representation readout (last-item local preference + attention-weighted global preference).
  * **Graph Construction & Batch Collator:** Dynamic construction of directed graph tensors $[B, N, 2N]$ with node sorting and sequence alias preservation.
  * **Training & Experiment Runner:** End-to-end training loop with multiclass cross-entropy, Adam optimization, StepLR decay, early stopping on validation MRR@20, single test evaluation pass, and experiment metadata logging (`results.json`).
  * **Verification Tests:** Unit and smoke tests in `tests/test_srgnn.py`, `tests/test_session_graph.py`, `tests/test_experiment_runner.py`, and `tests/test_training_smoke.py`.

---

## 2. In-Progress Work

1. **Branch Reconciliation & Pipeline Unification:**
   * Reconciling the slight interface divergence in `src/recsys/data/processed.py` between `origin/data_perturbation` and `origin/feature/srgnn-baseline`.
   * Merging both feature branches into `main` to create a unified execution pipeline from raw data to model training.
2. **Clean Baseline Metric Reproduction:**
   * Executing full training runs on clean Diginetica data to replicate published literature baselines (target: Recall@20 $\approx 50.7\%$, MRR@20 $\approx 17.6\%$).
   * Profiling baseline runtime, epoch convergence rates, parameter counts, and peak CUDA VRAM consumption.
3. **Sequential Baseline (SASRec) Architecture:**
   * Implementing the non-graph Self-Attentive Sequential Recommendation baseline (`src/recsys/models/baselines/sasrec.py`) to provide an attention-based sequence modeling benchmark.

---

## 3. Post-Midterm Roadmap

1. **Self-Supervised Contrastive Objective Integration ($\mathcal{L} = \mathcal{L}_{rec} + \lambda \mathcal{L}_{SSL}$):**
   * Implement InfoNCE / contrastive session loss comparing dual augmented session representations ($z_1, z_2$) across mini-batches.
   * Integrate dynamic view generation (`graph_augmentations.py`) into the PyTorch DataLoader during training passes.
2. **Hyperparameter Tuning & Augmentation Ablation Suite:**
   * Grid search over contrastive loss weight $\lambda \in [0.001, 0.1]$, temperature $\tau \in [0.05, 0.5]$, perturbation rate $p \in [0.05, 0.25]$, and recency decay exponent $\gamma \in [1.0, 3.0]$.
   * Systematic ablation comparing Uniform Edge Dropout vs. Recency-Aware Perturbation vs. Uniform Node Dropout vs. Unaugmented SR-GNN.
3. **Controlled-Noise Robustness Evaluation:**
   * Evaluate clean baseline SR-GNN, SASRec, and SSL-augmented SR-GNN models against fixed noisy test sets (5%, 10%, and 20% deletion, substitution, and mixed noise).
   * Measure absolute Recall@20/MRR@20 retention curves and relative degradation from clean baselines.
4. **Cross-Dataset Validation on RetailRocket:**
   * Execute preprocessing on raw RetailRocket `events.csv`.
   * Benchmark the optimal SSL configuration on RetailRocket to verify generalizability across varying session length distributions.
5. **Computational Cost & Trade-Off Profiling:**
   * Quantify training time per epoch, convergence time, peak GPU memory overhead, and parameter count between baseline SR-GNN and dual-view contrastive SR-GNN.
6. **Final Report & Deliverables Synthesis:**
   * Synthesize final experimental figures, performance tables, ablation analyses, and presentation slides.
