# Self-Supervised-Session-Graph-Recommendation-with-SR-GNN

## Prerequisites

This project uses Python 3.12 and
[`uv`](https://docs.astral.sh/uv/getting-started/installation/) for Python and
dependency management.

Install `uv` on macOS or Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows, run the following in PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Confirm that it is available:

```bash
uv --version
```

## Setup

Clone the repository, enter its directory, and create the project environment:

```bash
uv sync
```

`uv` reads `.python-version` and `uv.lock`, installs the required Python version
when necessary, and creates a local `.venv` environment.

## Common commands

Run Python inside the project environment:

```bash
uv run python
```

Verify that the project package can be imported:

```bash
uv run python -c "import recsys; print('Hello from recsys!')"
```

Run all tests:

```bash
uv run python -m unittest discover -s tests -v
```

Add or remove a dependency:

```bash
uv add <package-name>
uv remove <package-name>
```

Refresh the lockfile after manually editing `pyproject.toml`:

```bash
uv lock
```

## Evaluation metrics

Calculate the correct next item's rank once, then reuse it for whichever metrics
and cutoffs you need. Each example has one correct target. Score columns correspond
to `item_ids`; larger scores rank first, with ties resolved by smaller item ID.
Item IDs do not need to be consecutive or start at zero.

This complete example uses K=3 to make the results easy to check. The project's
evaluation cutoffs are K=10 and K=20; each metric defaults to K=20.

```python
from recsys.evaluation import (
    format_results,
    get_target_ranks,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)

item_ids = [40, 10, 30, 20]
scores = [
    [0.1, 0.9, 0.3, 0.5],
    [0.8, 0.2, 0.4, 0.6],
    [0.9, 0.8, 0.7, 0.6],
]
targets = [10, 30, 20]

ranks = get_target_ranks(scores, targets, item_ids)  # [1, 3, 4]
results = {
    "Recall@3": recall_at_k(ranks, k=3),
    "MRR@3": mrr_at_k(ranks, k=3),
    "NDCG@3": ndcg_at_k(ranks, k=3),
}
print(format_results(results))
```

Output:

```text
Recall@3: 66.67%
MRR@3: 44.44%
NDCG@3: 50.00%
```

Call only the metrics you need. Reuse `ranks` with `k=10` or `k=20` without
processing the scores again. Metric functions return fractions between 0 and 1;
`format_results` displays only the supplied values as percentages, in their given
order. It returns text, leaving printing or logging to the caller.

For batched predictions, extend one ranks list with each batch's
`get_target_ranks(...)` result, then calculate metrics from that list. This counts
each example equally, including misses and examples in a smaller final batch.
Do not average batch metrics without weighting them by their example counts.

Pass reusable sequences of numeric scores and integer IDs. Convert framework
tensors to CPU lists before calling these utilities. All compared models must use
the same test examples and candidate items. Exclude padding candidates and their
score columns; previously observed items remain eligible. Targets must be present
among candidates. Empty inputs, inconsistent dimensions, invalid ranks or cutoffs,
and non-finite scores raise `ValueError`.
