# SR-GNN Self-Supervised Extension

## Implementation

The `srgnn_ssl` model reuses the existing SR-GNN encoder architecture
for clean next-item prediction and two augmented session-graph views.
All three passes share the same trainable encoder parameters.

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
- A deletion count is calculated using the configured rounding rule.
- Both augmented views use that same deletion count.

The default, `budget_rounding = "nearest"`, uses:

`k = min(round(deletion_budget * M), eligible_count)`

Python's `round()` uses ties-to-even.

The optional `budget_rounding = "stochastic"` uses:

`x = deletion_budget * M`

`k = min(floor(x) + Bernoulli(x - floor(x)), eligible_count)`

For example, if `x = 0.6`, the count is one with probability 0.6
and zero otherwise. The count is sampled once per prefix and shared
between its two views. Short-prefix safeguards and final-transition
protection apply under both rounding rules.

Uniform-edge augmentation samples exactly `k` eligible occurrences
without replacement using equal weights. Recency-aware augmentation
samples exactly `k` eligible occurrences without replacement using:

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

## Configurations

The original configurations explicitly use nearest rounding:

| Configuration | Policy | Deletion budget |
|---|---|---|
| `configs/srgnn_ssl_uniform_diginetica_budget010.toml` | Uniform edge | 10% |
| `configs/srgnn_ssl_uniform_diginetica_budget020.toml` | Uniform edge | 20% |
| `configs/srgnn_ssl_recency_diginetica_budget010.toml` | Recency-aware | 10% |
| `configs/srgnn_ssl_recency_diginetica_budget020.toml` | Recency-aware | 20% |

The corresponding stochastic-rounding configurations are:

- `configs/srgnn_ssl_uniform_diginetica_budget010_stochastic.toml`
- `configs/srgnn_ssl_uniform_diginetica_budget020_stochastic.toml`
- `configs/srgnn_ssl_recency_diginetica_budget010_stochastic.toml`
- `configs/srgnn_ssl_recency_diginetica_budget020_stochastic.toml`

These retain the existing SR-GNN Diginetica configuration's frozen
data, artifact checksums, encoder settings, and training settings.
Each configuration has a distinct experiment name and output directory.
Corresponding nearest and stochastic configurations differ only in
rounding rule and experiment identity/output location.

Initial SSL settings:

- `ssl_weight = 0.01`
- `temperature = 0.2`
- `recency_gamma = 2.0`
- `augmentation_seed = 2026`

These are starting settings, not validated optimal hyperparameters.
The existing training seed remains `2028`.

Without DataLoader workers, augmentation uses persistent RNGs seeded
from `augmentation_seed`. With workers, each worker uses persistent
RNGs derived from `augmentation_seed` and its PyTorch worker seed.
Reproducing augmentation streams requires the same seeds, worker count,
input ordering, and data-loading settings. Changing worker count can
change sampled views.

Stochastic deletion counts use a separate RNG from edge selection.
This prevents differences in edge-selection RNG consumption between
uniform and recency-aware policies from changing their deletion counts.
Matched stochastic counts across policies require identical seeds,
input ordering, worker counts, and data-loading settings.

Nearest rounding does not consume random draws for deletion counts.

## Diginetica Budget Diagnostics

Measured on the frozen training artifacts at revision
`ac9f069fb5305de9885a566301ddcc8f9317669e`, containing
679,395 training prefixes and 2,126,242 transition occurrences per view.

### Nearest rounding

| Requested budget | Realized deletion rate | Zero-deletion prefixes |
|---|---|---|
| 10% | 6.685692% | 81.418909% |
| 20% | 18.506689% | 58.149824% |

These counts apply equally to uniform-edge and recency-aware policies.
They were calculated from prefix lengths without model training.

### Stochastic rounding: analytical expectations

| Requested budget | Expected deletion rate | Expected zero-deletion prefixes |
|---|---|---|
| 10% | 9.405985% | 73.202128% |
| 20% | 18.811970% | 58.246793% |

These are analytical expectations, not observed training measurements.
Actual stochastic counts vary between sampled passes.

Short-prefix safeguards and final-transition protection can prevent
the overall deletion rate from reaching the nominal requested budget,
even with stochastic rounding.

Nearest rounding remains the default. Stochastic rounding is an
experimental alternative; neither has demonstrated superior
recommendation performance yet.

## Execution and Outputs

For later execution from the repository root:

```bash
uv run python scripts/train.py --config configs/srgnn_ssl_uniform_diginetica_budget010.toml
```

For the corresponding stochastic-rounding experiment:

```bash
uv run python scripts/train.py --config configs/srgnn_ssl_uniform_diginetica_budget010_stochastic.toml
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

Prefix counts are not doubled for the two views. Both views share the
same deletion count, so zero-deletion status is shared by a prefix's views.
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

Additional stochastic-rounding tests cover rounding thresholds,
shared deletion counts across views and policies, preservation of
default nearest-rounding behavior, worker reproducibility, distinct
configuration outputs, and synthetic runner integration.

After adding stochastic rounding, all 195 repository tests passed,
including 41 SSL-specific tests and synthetic CPU runner/checkpoint
smoke checks.
Full-dataset SSL experiments and CARC GPU validation remain pending.
No SSL effectiveness claims are made.

## Scope and Next Experiments

This implementation covers uniform-edge and recency-aware SSL, with
nearest rounding as the default and stochastic rounding as an option.
Uniform node dropout, unprotected-uniform ablations, and same-session
negative masking are not included.

Initially compare both policies and both rounding rules at the 10%
budget on Diginetica, keeping the other hyperparameters fixed.
Use validation results for configuration selection; do not select
configurations using test performance.

The 20% configurations are prepared for subsequent experiments.
A rounding rule that performs better at 10% is not guaranteed to
perform better at 20%.

Subsequently validate the strongest configuration on RetailRocket.
Further hyperparameter trials and multi-seed experiments remain
to be scheduled.