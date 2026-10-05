# Data and perturbation pipeline

## Scope

This package owns two perturbation layers:

1. fixed offline click deletion/substitution files for controlled evaluation;
2. stochastic in-memory graph views for self-supervised training.

InfoNCE, SR-GNN, and SASRec remain model/training responsibilities.

## Raw datasets

### Diginetica

- Original source: CIKM Cup 2016 Track 2.
- Required file: `train-item-views.csv`.
- Local development cache: `~/.rs_datasets/diginetica`.
- Verified: 1,235,380 views, 310,324 view sessions, 122,993 viewed items.
- Raw SHA-256: `0b9edf1b7960bb855a2248972f8ed593f18721f321b976b2077c9f782fb43e38`.
- Diginetica: 7 validation days and 7 test days.

The Google Drive URL in `rs_datasets 0.5.1` no longer downloads. The development
copy came from Kaggle dataset `profalbusdumbledore/diginetica-dataset`, whose
manifest contains the expected CIKM files. Its Kaggle license field is unknown,
so raw data must not be committed.

### RetailRocket

- Source: Kaggle dataset `retailrocket/ecommerce-dataset`.
- Required file: `events.csv`.
- Local development cache: `~/.rs_datasets/retail_rocket`.
- Verified: 2,756,101 total events, 1,407,580 visitors, 235,061 event items.
- Raw SHA-256: `3745aa83238b1e6d44d8fda209807899f420084398f94ddf745f3cbcfecbf9e7`.
- RetailRocket: 14 validation days and 14 test days.


`rs_datasets 0.5.1` downloads RetailRocket but later calls the removed Pandas
`DataFrame.append` method. This package reads the valid raw CSV directly.

## Preprocessing protocol

`recsys.data.preprocess` performs:

1. view-event filtering for RetailRocket;
2. supplied Diginetica sessions or 30-minute RetailRocket inactivity sessions;
3. recursive removal of sessions shorter than two and items occurring fewer
   than five times;
4. chronological train/validation/test splitting using each
   complete session's final timestamp;
5. by default, the seven days preceding the test period are
   validation and the final seven days are test;
6. prefix/next-item expansion is performed separately inside
   each split;
8. one-indexed contiguous item mapping with zero reserved for padding;
9. timestamped logging and a JSON summary.

Outputs are `train.txt`, `validation.txt`, `test.txt`, `all_train_seq.txt`, and
`item_mapping.pkl`. Despite the `.txt` suffix, the first three are binary pickle
files following the original SR-GNN convention.

## Filtering and vocabulary protocol

The pipeline uses a benchmark-compatible SR-GNN filtering protocol.

1. Sessions shorter than two interactions and items occurring fewer
   than five times are iteratively removed from the complete dataset.
2. Complete filtered sessions are chronologically divided into
   training, validation, and test sets.
3. The item vocabulary is constructed only from items appearing in
   training.
4. Validation and test interactions containing items absent from the
   training vocabulary are removed.
5. Validation and test sessions that become shorter than two
   interactions are removed.

This preserves comparability with common SR-GNN preprocessing while
ensuring that the model embedding table contains only training items.
The same artifacts are used by SR-GNN and SGL+SR-GNN.

## Fixed evaluation perturbations

`recsys.data.perturb` creates separate deletion-only,
substitution-only, and mixed-corruption test files at noise ratios
0.05, 0.10, and 0.20 using perturbation seed 2026.
Targets and example counts remain unchanged. Substitutions cannot equal
the original item or the prediction target. Length-one clean prefixes
remain unchanged, and deletion never produces an empty prefix.

`recsys.data.validate` independently checks target equality, population,
vocabulary range, length constraints, and file hashes.

## Training graph augmentations

`recsys.data.graph_augmentations` exposes:

```python
import random

augmentation_rng = random.Random(2027)  # create once for the training run
view_one, view_two = make_two_views(
    sequence,
    mode="recency",  # uniform_edge | uniform_node | recency
    probability=0.2,
    gamma=1.0,
    rng=augmentation_rng,
)
```

- Uniform edge dropout independently removes chronological transitions.
- Uniform node dropout protects the final item and removes incident edges
  without inventing shortcut edges.
- Recency-aware dropout uses
  `p(e_t) = p_base * (1 - t/k) ** gamma`, protects the final incoming edge,
  and skips sessions of length two or less.

`GraphView.edges` must be used for adjacency construction. Reconstructing edges
from the filtered node-drop sequence could incorrectly invent a shortcut edge.

## Zero-edge augmentation policy

For training prefixes containing at least two items, an augmented graph
view must retain at least one original transition edge.

- Uniform edge dropout randomly restores one original edge if all edges
  were dropped.
- Item-identity dropout restores recently observed dropped item
  identities until at least one original edge survives.
- Recency-aware edge dropout already protects the edge entering the
  final observed item.
- One-item prefixes naturally contain no edges and are left unchanged.

The augmentation process never creates shortcut edges that were absent
from the original session graph.

## Random seed policy

Different random processes use separate seeds:

| Random process | Default seed |
|---|---:|
| Offline test corruption | 2026 |
| Dynamic training graph augmentation | 2027 |
| Model initialization | 2028 |
| Training batch shuffling | 2029 |

The test-corruption seed remains fixed across model runs so every model
is evaluated using identical noisy test data.

Within test corruption, the same ordered click selection is deliberately
used across corruption types and noise ratios where possible. This
creates controlled, comparable evaluation conditions.

The graph-augmentation random-number generator is created once at the
beginning of training and reused throughout the training run. It is not
reset for every session, view, batch, or epoch.

## Reproduction commands

```bash
uv run python -m recsys.data.preprocess --dataset diginetica \
  --input data/diginetica/raw/train-item-views.csv \
  --output-dir data/diginetica
uv run python -m recsys.data.perturb --data-dir data/diginetica
uv run python -m recsys.data.validate --data-dir data/diginetica

uv run python -m recsys.data.preprocess --dataset retailrocket \
  --input data/retailrocket/raw/events.csv \
  --output-dir data/retailrocket
uv run python -m recsys.data.perturb --data-dir data/retailrocket
uv run python -m recsys.data.validate --data-dir data/retailrocket
```
