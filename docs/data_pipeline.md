# Data pipeline

## Ownership and boundaries

The data layer produces frozen examples for every model. The exact five-file
contract is documented in
[`processed_data_interface.md`](processed_data_interface.md). It has four separate
responsibilities:

1. `preprocess.py` creates only the clean train, validation, and test data.
2. `perturb.py` derives fixed noisy evaluation datasets from the clean data.
3. `pipeline.py` runs both stages and writes the complete directory tree.
4. `processed.py` is the single model-facing ingestion and validation API.

InfoNCE, SR-GNN, and SASRec are not dataset preprocessing. Dynamic uniform
edge, uniform node, and recency-aware graph views remain in
`graph_augmentations.py` because training must resample those views each epoch.

## Raw sources

| Dataset | Required raw file | Local `rs_datasets` cache | Split windows |
|---|---|---|---|
| Diginetica | `train-item-views.csv` | `~/.rs_datasets/diginetica/` | 7 validation days, 7 test days |
| RetailRocket | `events.csv` | `~/.rs_datasets/retail_rocket/` | 14 validation days, 14 test days |

RetailRocket preprocessing retains only `view` events and creates a new session
after more than 30 minutes of inactivity. Diginetica uses its supplied session
IDs. Raw files are never committed to Git.

## Clean preprocessing

The clean stage performs the following deterministic operations:

1. Normalize raw view rows to session, item, timestamp, and original row order.
2. Recursively remove sessions shorter than two and items occurring fewer than
   five times until no rows change.
3. Split complete sessions by final timestamp before creating examples.
4. Build a contiguous one-based vocabulary from the training split only.
5. Remove unknown validation/test events and then remove sessions that become
   shorter than two.
6. Generate all non-empty prefix/next-item pairs independently within each
   split.
7. Write and immediately reload the five-file contract to validate it.

The final test window is the last 7 days for Diginetica and 14 days for
RetailRocket. Validation is the equally sized window immediately before test;
everything earlier is training. The exact cutoffs, filtering counts, source
hash, unknown-item counts, and final session/example counts are recorded in
`preprocessing_summary.json` and `preprocessing.log`.

## Output layout

One complete run has this structure:

```text
data/processed/diginetica/
├── pipeline_manifest.json
├── clean/
│   ├── train.txt
│   ├── validation.txt
│   ├── test.txt
│   ├── item_mapping.pkl
│   ├── preprocessing_summary.json
│   └── preprocessing.log
└── noisy/
    ├── deletion_0.05/
    │   └── <the same five required model files>
    ├── substitution_0.05/
    │   └── <the same five required model files>
    ├── mixed_0.05/
    │   └── <the same five required model files>
    └── ... 0.10 and 0.20 variants
```

Every directory under `clean/` or `noisy/` is independently model-ready. A
model changes evaluation conditions only by changing the directory supplied to
`load_processed_dataset`; it does not need a noise-specific parser.

## Model-facing contract

Each model-ready directory requires exactly these names (additional files are
ignored):

- `train.txt`
- `validation.txt`
- `test.txt`
- `item_mapping.pkl`
- `preprocessing_summary.json`

The `.txt` files are trusted binary pickle files containing
`(prefixes, targets)`. Model ID `0` is reserved for padding. Real IDs are
contiguous Python integers in `1..num_items`. Prefixes never include the
held-out final event merely because it is the target; naturally repeated item
events are preserved.

Load any condition with the same call:

```python
from recsys.data.processed import load_processed_dataset

dataset = load_processed_dataset("data/processed/diginetica/clean")
print(dataset.num_items)
print(dataset.train[0])
print(dataset.checksums)
```

The loader converts each example to `((item_id, ...), target_item_id)` and
validates all five files. Pickle can execute code while loading, so never use
this loader on an untrusted artifact directory.

## Offline noise datasets

Noise generation reads only the clean dataset. It never reparses raw data.
Each fixed-seed variant copies clean train, validation, and mapping artifacts,
and changes only test prefixes. Targets and test example order remain exactly
unchanged.

- Ratios: `0.05`, `0.10`, and `0.20`
- Seed: `2026`
- Operators: deletion-only, substitution-only, and mixed
- Length-one prefixes: unchanged
- Prefixes originally of length at least two: never shortened below two
- Substitutions: sampled from the training vocabulary and cannot equal the
  replaced item or held-out target

The noisy summary contains the clean source checksums, requested ratio,
operator counts, skipped operations, and target-invariance declaration.

## Run the complete pipeline

```bash
uv run python -m recsys.data.pipeline \
  --dataset diginetica \
  --input ~/.rs_datasets/diginetica/train-item-views.csv \
  --output-root data/processed/diginetica

uv run python -m recsys.data.pipeline \
  --dataset retailrocket \
  --input ~/.rs_datasets/retail_rocket/events.csv \
  --output-root data/processed/retailrocket
```

Validate a whole tree or one model-ready directory:

```bash
uv run python -m recsys.data.validate \
  --data-dir data/processed/diginetica

uv run python -m recsys.data.validate \
  --data-dir data/processed/diginetica/clean
```

## Hugging Face artifact reuse

Generated artifacts are intentionally ignored by Git. They can instead be
stored in a private Hugging Face dataset repository after the team confirms the
source-dataset redistribution terms.

```bash
uv sync --extra hub
uv run hf auth login

uv run python -m recsys.data.hub upload \
  --source data/processed/diginetica \
  --repo-id CSCI-566-SBR/srgnn-processed-diginetica

uv run python -m recsys.data.hub download \
  --repo-id CSCI-566-SBR/srgnn-processed-diginetica \
  --destination data/processed/diginetica
```

Uploads are private by default. The upload command validates all artifact
directories and excludes logs; the download command validates the fetched
snapshot. Pin `--revision` to a Hub tag or full commit hash in experiment
scripts so the exact bytes do not change between runs.

## Random seeds

| Process | Default seed |
|---|---:|
| Offline test noise | 2026 |
| Dynamic training graph augmentation | 2027 |
| Model initialization | 2028 |
| Training shuffle | 2029 |

Graph augmentation should receive one persistent `random.Random(2027)` object
for a run. It should not reset the seed per session or per epoch.
