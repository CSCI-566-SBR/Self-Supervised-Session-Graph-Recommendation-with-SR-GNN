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

## Dataset and perturbation pipeline

The data package supports Diginetica and RetailRocket preprocessing, fixed
target-preserving noisy test sets, artifact validation, and model-independent
training graph augmentations. Full provenance, schemas, and commands are in
[`docs/data_pipeline.md`](docs/data_pipeline.md).

```bash
uv run python -m recsys.data.preprocess --dataset diginetica \
  --input data/diginetica/raw/train-item-views.csv \
  --output-dir data/diginetica

uv run python -m recsys.data.perturb --data-dir data/diginetica
uv run python -m recsys.data.validate --data-dir data/diginetica
```

Training-time uniform edge, uniform node, and recency-aware graph views are
generated in memory by `recsys.data.graph_augmentations`. They are intentionally
not saved as static files, allowing fresh stochastic views each epoch.
