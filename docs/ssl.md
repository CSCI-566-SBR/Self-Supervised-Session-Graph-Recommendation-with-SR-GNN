# SR-GNN Self-Supervised Extension

## Implementation

The `srgnn_ssl` model reuses the existing SR-GNN encoder for clean
next-item prediction and two augmented session-graph views.

Model and loss: `src/recsys/models/srgnn_ssl.py`.
CPU-side view generation and collation: `src/recsys/data/ssl_graph.py`.

Training uses a dedicated collator to generate both views before GPU
transfer. With multiple DataLoader workers, augmentation runs in those
workers; with zero workers, it runs on the main CPU process.
Validation and test use the ordinary clean graph collator.

Training minimizes:

`total_loss = recommendation_loss + ssl_weight * contrastive_loss`

The contrastive objective is one-way, batch-mean InfoNCE over
L2-normalized session embeddings. Matching rows across the two views
are positives; other batch examples are negatives. Same-session
negative masking is not applied.

Inference uses the ordinary SR-GNN forward pass without augmentation.
Setting `ssl_weight = 0` bypasses augmented-view generation and the
contrastive objective.

## Budget-Matched Augmentation

Implemented in `src/recsys/data/graph_augmentations.py`.

For each observed prefix:

- `M` is the original number of transition occurrences.
- Prefixes containing at most two items remain unchanged.
- The final observed transition is protected in both policies.
- The remaining transitions are eligible for deletion.
- Both policies remove exactly
  `k = min(round(deletion_budget * M), eligible_count)` occurrences.
- Python's `round()` uses ties-to-even.

Uniform-edge augmentation samples without replacement using equal
weights. Recency-aware augmentation samples without replacement using:

`weight_i = ((M - i) / (M - 1)) ** recency_gamma`

Here `i` is the one-based transition position among the original
transitions, excluding the protected final transition. Older transitions
receive greater deletion weight. Weights are normalized over the
remaining candidates at each sampling step.

Observed item identities and their sequence order remain unchanged.
Held-out targets are never included in augmented views.

Budgets count transition occurrences, not unique binary graph edges.
A repeated graph edge remains present if at least one occurrence survives.
Matched occurrence counts therefore do not guarantee equal numbers of
removed unique graph edges.

## Initial Configurations

| Configuration | Policy | Deletion budget |
|---|---|---|
| `configs/srgnn_ssl_uniform_diginetica_budget010.toml` | Uniform edge | 10% |
| `configs/srgnn_ssl_uniform_diginetica_budget020.toml` | Uniform edge | 20% |
| `configs/srgnn_ssl_recency_diginetica_budget010.toml` | Recency-aware | 10% |
| `configs/srgnn_ssl_recency_diginetica_budget020.toml` | Recency-aware | 20% |

These retain the existing SR-GNN Diginetica configuration's frozen
data, artifact checksums, encoder settings, and training settings.
Each configuration has a distinct output directory.

Initial SSL settings:

- `ssl_weight = 0.01`
- `temperature = 0.2`
- `recency_gamma = 2.0`
- `augmentation_seed = 2026`

These are starting settings, not validated optimal hyperparameters.
The existing training seed remains `2028`.

Without DataLoader workers, augmentation uses a persistent RNG seeded
by `augmentation_seed`. With workers, each worker uses a separate
persistent RNG derived from `augmentation_seed` and its PyTorch worker
seed. Reproducing augmentation streams requires the same seeds,
worker count, and data-loading settings. Changing worker count can
change sampled views.


## Diginetica Budget Diagnostics

Measured on the frozen training artifacts at revision
`ac9f069fb5305de9885a566301ddcc8f9317669e`, containing
679,395 training prefixes and 2,126,242 transition occurrences per view.

| Requested budget | Realized deletion rate | Zero-deletion prefixes |
|---|---|---|
| 10% | 6.685692% | 81.418909% |
| 20% | 18.506689% | 58.149824% |

These counts apply equally to uniform-edge and recency-aware policies.
They were calculated from prefix lengths without model training.
Current per-prefix rounding remains unchanged pending team discussion.


## Execution and Outputs

For later execution from the repository root:

```bash
uv run python scripts/train.py --config configs/srgnn_ssl_uniform_diginetica_budget010.toml
```

Use the corresponding configuration path for other policies or budgets.

The existing runner selects the best checkpoint using validation
MRR@20 and saves `best_model.pt` and `results.json`.

Each SSL epoch records model-defined diagnostics under
`history[*]["training_metrics"]`:

- `recommendation_loss`: example-weighted mean recommendation loss.
- `ssl_loss`: example-weighted mean unweighted contrastive loss.
- `augmentation_original_transitions`: original occurrence count across both views.
- `augmentation_deleted_transitions`: deleted occurrence count across both views.
- `augmentation_prefixes`: number of original training prefixes processed.
- `augmentation_zero_deletion_prefixes`: prefixes receiving zero deletions.
- `realized_deletion_rate`: total deleted occurrences divided by total original occurrences.
- `zero_deletion_prefix_percentage`: 100 times zero-deletion prefixes divided by processed prefixes.

Prefix counts are not doubled for the two views. Both policies delete
the same count per prefix, so zero-deletion status is shared by its views.
Rates are calculated from epoch-wide totals, not averaged batch rates.
Realized deletion rate is null when there are no original transitions.

The ordinary SR-GNN and SASRec recommendation-loss training paths remain
in place. SSL does not change shared preprocessing or evaluation metrics.

## Tests and Current Status

Tests are provided in:

- `tests/test_ssl_augmentations.py`
- `tests/test_srgnn_ssl.py`
- `tests/test_ssl_training.py`

They cover deletion budgets, protection rules, legacy augmentation
compatibility, contrastive loss, encoder reuse, shared graph collation,
worker-loading reproducibility, baseline training compatibility,
configuration values, checkpoint vocabulary validation, zero-deletion
diagnostics, and synthetic runner/checkpoint integration.

After the review fixes, the full repository test suite passed all
188 tests, including 34 SSL-specific tests and synthetic CPU training
and checkpoint save/reload checks.
Full-dataset SSL experiments and CARC GPU validation remain pending.
No SSL effectiveness claims are made.

## Scope and Next Experiments

This implementation covers uniform-edge and recency-aware SSL.
Uniform node dropout, unprotected-uniform ablations, and same-session
negative masking are not included.

Compare the initial configurations on Diginetica using validation
results for selection, then validate the strongest configuration
on RetailRocket. Hyperparameter trial count and subsequent multi-seed
experiments remain to be scheduled.