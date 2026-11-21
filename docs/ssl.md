# SR-GNN Self-Supervised Extension

## Implementation

The `srgnn_ssl` model reuses the existing SR-GNN encoder for clean
next-item prediction and two augmented session-graph views.

Implementation: `src/recsys/models/srgnn_ssl.py`.

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
The existing training seed remains `2028`. Augmentation uses a separate,
persistent random-number generator.

## Execution and Outputs

For later execution from the repository root:

```bash
uv run python scripts/train.py --config configs/srgnn_ssl_uniform_diginetica.toml
```

Use the corresponding configuration path for other policies or budgets.

The existing runner selects the best checkpoint using validation
MRR@20 and saves `best_model.pt` and `results.json`.

Each SSL epoch additionally records:

- Recommendation loss and unweighted contrastive loss.
- Original and deleted transition-occurrence counts across both views.
- Realized deletion rate after rounding and short-prefix safeguards.

The ordinary SR-GNN and SASRec recommendation-loss training paths remain
in place. SSL does not change shared preprocessing or evaluation metrics.

## Tests and Current Status

Tests are provided in:

- `tests/test_ssl_augmentations.py`
- `tests/test_srgnn_ssl.py`
- `tests/test_ssl_training.py`

They cover deletion budgets, protection rules, contrastive loss,
encoder reuse, baseline training compatibility, and synthetic
runner/checkpoint integration.

Test execution and full-dataset SSL experiments are pending.
No SSL effectiveness claims are made.

## Scope and Next Experiments

This implementation covers uniform-edge and recency-aware SSL.
Uniform node dropout, unprotected-uniform ablations, and same-session
negative masking are not included.

Compare the initial configurations on Diginetica using validation
results for selection, then validate the strongest configuration
on RetailRocket. Hyperparameter trial count and subsequent multi-seed
experiments remain to be scheduled.