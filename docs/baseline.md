# SR-GNN baseline

## Scope

The baseline starts from processed next-item examples:

```text
observed item prefix -> held-out next-item target
```

Raw dataset parsing, temporal splitting, item-frequency filtering, graph
augmentation, robustness corruption, and self-supervised losses are outside the
baseline implementation.

## Processed-data contract

- Item ID `0` is reserved for batch padding.
- Real item IDs are contiguous integers from `1` through `num_items`.
- A prefix contains at least one observed item.
- A target is the held-out next event and has an ID in `1..num_items`.
- The model has `num_items + 1` embedding rows but produces `num_items` scores.
- Target item ID `t` corresponds to score column and cross-entropy class `t - 1`.

## Session graphs

Each observed prefix becomes its own directed graph. Repeated item IDs share a
node, while alias indices preserve their original sequence positions. Graph edges
are binary, matching the official SR-GNN PyTorch implementation: observing the
same directed transition more than once does not increase its edge weight.

The adjacency tensor concatenates separately normalized incoming and outgoing
connections and has shape `[num_nodes, 2 * num_nodes]`.

## Batch contract

The collator pads only to the largest graph and sequence in the current batch:

```text
node_ids:       [batch, max_nodes]
adjacency:      [batch, max_nodes, 2 * max_nodes]
alias_indices:  [batch, max_sequence_length]
sequence_mask:  [batch, max_sequence_length]
targets:        [batch]
```

## Model and objective

The model follows the official hybrid SR-GNN behavior:

```text
item embeddings
  -> gated incoming/outgoing graph propagation
  -> ordered sequence states recovered through aliases
  -> attention-based global preference + last-item local preference
  -> hybrid session embedding
  -> dot products against every real item embedding
```

Training uses multiclass cross-entropy. The optimizer and scheduler remain
outside the model so experiment settings are explicit.

The implementation modernizes the official PyTorch 0.4 code without changing
its equations. Padding row zero is fixed to the zero vector, device movement is
explicit, and optimizers, schedulers, training, and evaluation are not stored
inside the model.

## Current validation

Synthetic tests cover graph construction, repeated items, binary transitions,
adjacency normalization, batch padding, forward shapes, finite gradients,
deterministic initialization, Recall/MRR calculations, and an end-to-end
training smoke test. Real-data integration waits for the preprocessing protocol
and artifact format to be frozen.

## Processed artifacts and training entry point

The model-facing loader expects one frozen artifact directory containing:

```text
train.txt
validation.txt
test.txt
item_mapping.pkl
preprocessing_summary.json
```

Each split is a trusted pickle containing `(prefixes, targets)`. The loader
validates the shared one-based vocabulary contract and records SHA-256 checksums.
It preserves the preprocessing summary, including whichever filtering protocol
the data team selects, without implementing that protocol in baseline code.

Run an experiment with:

```bash
uv run python scripts/train.py --config configs/srgnn_diginetica.toml
```

Validation MRR@20 selects one checkpoint. Recall@20 is also recorded, and the
test split is evaluated once after restoring the selected checkpoint. The run
directory records configuration, seed, data checksums, per-epoch history,
runtime, parameter count, software versions, and Git commit. Existing run
directories are never overwritten.
