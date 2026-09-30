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

`rs_datasets 0.5.1` downloads RetailRocket but later calls the removed Pandas
`DataFrame.append` method. This package reads the valid raw CSV directly.

## Preprocessing protocol

`recsys.data.preprocess` performs:

1. view-event filtering for RetailRocket;
2. supplied Diginetica sessions or 30-minute RetailRocket inactivity sessions;
3. recursive removal of sessions shorter than two and items occurring fewer
   than five times;
4. chronological split using the final 7 Diginetica or 14 RetailRocket days;
5. prefix/next-item expansion;
6. one-indexed contiguous item mapping with zero reserved for padding;
7. timestamped logging and a JSON summary.

Outputs are `train.txt`, `test.txt`, `all_train_seq.txt`, and
`item_mapping.pkl`. Despite the `.txt` suffix, the first three are binary pickle
files following the original SR-GNN convention.

## Fixed evaluation perturbations

`recsys.data.perturb` writes `test_noise_0.05.txt`,
`test_noise_0.10.txt`, and `test_noise_0.20.txt` using seed 2026. It uses one
global click budget per ratio, randomly deletes or substitutes selected clicks,
never changes targets, never deletes below two clicks, and preserves naturally
occurring one-click prefixes unchanged.

`recsys.data.validate` independently checks target equality, population,
vocabulary range, length constraints, and file hashes.

## Training graph augmentations

`recsys.data.graph_augmentations` exposes:

```python
import random

augmentation_rng = random.Random(2026)  # create once for the training run
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
