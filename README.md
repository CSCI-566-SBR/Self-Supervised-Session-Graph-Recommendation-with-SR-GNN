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
