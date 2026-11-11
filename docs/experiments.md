# Experiment records

This document records reviewed experiment outcomes. The configuration files and
machine-readable `results.json` files are the detailed sources of truth; this
page summarizes the research purpose, result, and interpretation of each run.

## SR-GNN on Diginetica: seed 2028

**Status:** successful preliminary baseline run (one seed)

**Run date:** 2026-11-06

**Purpose:** verify that the frozen processed-data artifact and the modernized
SR-GNN implementation train and evaluate end to end, and check whether the
result is reasonably close to the published SR-GNN result.

### Reproducibility record

| Field | Value |
|---|---|
| Configuration | [`configs/srgnn_diginetica_global_recursive.toml`](../configs/srgnn_diginetica_global_recursive.toml) |
| Machine-readable result | [`results.json`](../experiments/srgnn_diginetica_global_recursive_seed2028/results.json) |
| Code commit used for the run | `0a5f9a3d545df77b0342fbac3b1f5cef00c5b2de` |
| Dataset repository | `CSCI-566-SB/srgnn-processed-diginetica` |
| Dataset revision | `ac9f069fb5305de9885a566301ddcc8f9317669e` |
| Data protocol | `diginetica_clean_v1` |
| Seed | `2028` |
| Device | CPU |
| Python | 3.12.15 |
| PyTorch | 2.15.0 |
| NumPy | 2.5.4 |

The result file also records SHA-256 checksums for every processed artifact.
Those checksums identify the exact train, validation, test, item-mapping, and
preprocessing-summary files used by the run.

### Model and training settings

| Setting | Value |
|---|---:|
| Hidden/item embedding dimension | 100 |
| GNN propagation steps | 1 |
| Session representation | Hybrid local + attention-weighted global |
| Batch size | 100 |
| Optimizer | Adam |
| Initial learning rate | 0.001 |
| Weight decay | 0.00001 |
| Learning-rate schedule | Multiply by 0.1 every 3 epochs |
| Maximum epochs | 30 |
| Early-stopping patience | 10 epochs |
| Checkpoint selection | Highest validation MRR@20 |
| Evaluation cutoff | 20 |
| PyTorch deterministic algorithms | Disabled |

### Dataset and model scale

| Quantity | Value |
|---|---:|
| Training prefix/target examples | 679,395 |
| Validation prefix/target examples | 44,638 |
| Test prefix/target examples | 60,411 |
| Candidate items | 42,068 |
| Trainable parameters | 4,358,300 |

### Result

Training stopped after epoch 14 because validation MRR@20 had not improved for
10 epochs. Epoch 4 was selected, its checkpoint was restored, and the test set
was evaluated once.

| Split | Recall@20 | MRR@20 |
|---|---:|---:|
| Validation at selected epoch | 50.69% | 17.53% |
| Test using selected checkpoint | 49.55% | 17.11% |

The run took 1,931.99 seconds (about 32 minutes) on CPU. Training loss continued
to fall after epoch 4 while validation metrics declined, so checkpoint selection
and early stopping prevented the final, more overfit model from being reported.

### Comparison with the published SR-GNN result

The SR-GNN paper reports 50.73% P@20 and 17.59% MRR@20 on Diginetica. With one
held-out target per example, the paper's P@20 is equivalent to the Recall@20
reported by this project.

| Result | Recall/P@20 | MRR@20 |
|---|---:|---:|
| Published SR-GNN | 50.73% | 17.59% |
| This run | 49.55% | 17.11% |
| Absolute difference | -1.18 points | -0.48 points |

This is close enough to support the correctness of the baseline, but it is not
an exact reproduction of the paper's dataset protocol. The project artifact
uses recursive frequency filtering and a chronological validation window. The
legacy preprocessing uses a single filtering pass, reports a slightly different
item vocabulary and example counts, and describes a random 10% validation
subset. The project protocol is intentionally frozen so that SR-GNN and later
SSL models can be compared on exactly the same examples.

This run also represents only one random seed, and deterministic PyTorch
algorithms were not forced. It should therefore be treated as a successful
baseline validation run rather than the project's final aggregate result.

After this preliminary run, deterministic PyTorch algorithms were enabled in
the linked configuration for the final repeated-seed experiment. The exact
configuration used by this earlier run remains embedded in its `results.json`.

Original references:

- [SR-GNN paper](https://arxiv.org/abs/1811.00855)
- [Official SR-GNN implementation](https://github.com/CRIPAC-DIG/SR-GNN)

### Remaining work before freezing the baseline

- Run the team's agreed additional seeds with the same frozen configuration and
  processed artifact.
- Report the mean and standard deviation of Recall@20 and MRR@20.
- Record hardware and runtime for the final comparison environment.
- Freeze the validated baseline commit before adding SSL functionality.
