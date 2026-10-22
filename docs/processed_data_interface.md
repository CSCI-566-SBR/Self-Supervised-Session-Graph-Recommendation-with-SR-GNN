# Processed data interface

This is the frozen dataset contract for SR-GNN and every subsequent model.
Raw parsing, filtering, splitting, and item remapping happen before this
interface. Models must call `recsys.data.processed.load_processed_dataset` and
must not introduce another artifact format without a versioned team decision.

## Required files

One model-ready directory contains:

```text
train.txt
validation.txt
test.txt
item_mapping.pkl
preprocessing_summary.json
```

Additional files are ignored. The `.txt` split files are binary Python pickle
files, not text. Only load trusted project artifacts because pickle is unsafe
for untrusted content.

## IDs and item mapping

`item_mapping.pkl` is a non-empty pickled dictionary from raw item IDs to model
item IDs. Mapping values must be built-in Python `int` objects containing every
integer in `1..num_items` exactly once. ID `0` is reserved for model padding and
must not occur in an artifact.

Train, validation, and test share the training-only mapping. Unknown
validation/test events are removed, after which sessions shorter than two are
removed. Every stored prefix item and target must therefore be in the mapping.

## Split files

Each split stores exactly one two-element tuple:

```python
(prefixes, targets)
```

Both elements may be lists or tuples, must be non-empty, and must have the same
length. Each prefix is a non-empty sequence of built-in Python integers and
each target is one built-in Python integer. Entry `prefixes[i]` predicts
`targets[i]`.

Splitting happens at session level before expansion. For a mapped session
`[1, 2, 3, 4]`, expansion produces:

```text
[1]       -> 2
[1, 2]    -> 3
[1, 2, 3] -> 4
```

Repeated events are preserved. For example, `[2, 4] -> 2` is valid when the
original session ends in `[2, 4, 2]`; the earlier `2` is observed history, not
the held-out final event.

## Preprocessing summary

`preprocessing_summary.json` is a UTF-8 JSON object. It records at minimum the
dataset, protocol version, `num_items`, split rules, filtering rules, thresholds,
mapping scope, unknown-item policy, prefix expansion rule, and final session and
example counts. The loader requires any reported `num_items` (or legacy
`vocab_size`) to match `item_mapping.pkl`.

Noise variants additionally record clean-source checksums, corruption type,
ratio, seed, operation counts, target invariance, and minimum-length policy.

## In-memory interface

```python
from recsys.data.processed import load_processed_dataset

dataset = load_processed_dataset("data/processed/diginetica/clean")
print(dataset.num_items)
print(dataset.train[0])
print(dataset.checksums)
```

Every example becomes:

```python
((item_id, ...), target_item_id)
```

The returned object exposes `train`, `validation`, `test`, `item_mapping`,
`summary`, `num_items`, and a SHA-256 checksum for every required file.

The loader verifies structure and ranges. Preprocessing tests separately verify
methodology: deterministic ordering, chronological session separation,
filtering, training-only vocabulary, unknown handling, prefix expansion,
target separation, repeat preservation, and reported counts.
