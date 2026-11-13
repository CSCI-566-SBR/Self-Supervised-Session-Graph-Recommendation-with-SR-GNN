"""Collect saved experiment results into a CSV without modifying source files.

Optional filters select models, seeds, and datasets.
Each distinct results.json file produces one row.
Existing CSV files are never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def require_object(value: Any, location: str) -> dict[str, Any]:
    """Require a JSON object instead of silently accepting malformed data."""
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be a JSON object")
    return value


def flatten(
    value: dict[str, Any],
    prefix: str,
) -> dict[str, Any]:
    """Convert nested fields into CSV columns such as test.mrr@20."""
    output: dict[str, Any] = {}

    for key, item in sorted(value.items()):
        column = f"{prefix}.{key}" if prefix else key

        if isinstance(item, dict):
            output.update(flatten(item, column))
        elif isinstance(item, list):
            output[column] = json.dumps(item, sort_keys=True, allow_nan=False)
        elif item is None or isinstance(item, (str, int, float, bool)):
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError(f"{column} contains a non-finite number")
            output[column] = item
        else:
            raise ValueError(f"unsupported value in {column}")

    return output


def read_result(path: Path) -> dict[str, Any]:
    """Read one experiment without changing it."""
    with path.open(encoding="utf-8") as handle:
        data = require_object(json.load(handle), str(path))

    config = require_object(data.get("config", {}), f"{path}: config")
    artifact = require_object(
        data.get("dataset_artifact", {}),
        f"{path}: dataset_artifact",
    )
    test = require_object(data.get("test"), f"{path}: test")

    if not test:
        raise ValueError(f"{path}: test results are empty")

    history = data.get("history", [])
    if not isinstance(history, list):
        raise ValueError(f"{path}: history must be a list")

    model = data.get("model_type") or config.get("model_type")
    model_source = "recorded"

    # Support the existing legacy SR-GNN result, which lacks model_type.
    if not model:
        if "gnn_steps" in config:
            model = "srgnn"
            model_source = "legacy_config.gnn_steps"
        else:
            model = "unknown"
            model_source = "unavailable"

    row: dict[str, Any] = {
        "source_file": path.resolve().as_posix(),
        "experiment": data.get("experiment") or path.parent.name,
        "dataset": data.get("dataset") or config.get("dataset"),
        "model": model,
        "model_identification": model_source,
        "seed": data.get("seed", config.get("seed")),
        "device": data.get("device"),
        "best_epoch": data.get("best_epoch"),
        "epochs_recorded": len(history),
        "checkpoint_metric": data.get("checkpoint_metric"),
        "runtime_seconds": data.get("runtime_seconds"),
        "peak_gpu_memory_bytes": data.get("peak_gpu_memory_bytes"),
        "parameter_count": data.get("parameter_count"),
        "git_commit": data.get("git_commit"),
        "protocol_version": artifact.get("protocol_version"),
        "artifact_repository": artifact.get("repository"),
        "artifact_revision": artifact.get("revision"),
        "filtering_protocol": data.get("filtering_protocol"),
    }

    # Preserve available metrics and metadata. Missing values stay blank.
    for section in (
        "test",
        "best_validation",
        "config",
        "dataset_artifact",
        "dataset_checksums",
        "software",
        "noise",
        "model_params",
    ):
        if section in data:
            content = require_object(data[section], f"{path}: {section}")
            row.update(flatten(content, section))

    return row


def collect_results(
    results_root: Path,
    output_path: Path,
    *,
    models: list[str] | None = None,
    seeds: list[int] | None = None,
    datasets: list[str] | None = None,
) -> int:
    """Select saved experiments and write one row per distinct result file."""
    if not results_root.is_dir():
        raise ValueError(f"results directory not found: {results_root}")

    paths = sorted({
        path.resolve()
        for path in results_root.rglob("results.json")
    })
    if not paths:
        raise ValueError(f"no results.json files found in {results_root}")

    if output_path.suffix.lower() != ".csv":
        raise ValueError("output filename must end in .csv")

    if output_path.exists():
        raise FileExistsError(
            f"output already exists: {output_path}; "
            "choose a different CSV filename"
        )

    selected_models = (
        {model.strip().casefold() for model in models}
        if models else None
    )
    selected_seeds = set(seeds) if seeds else None
    selected_datasets = (
        {dataset.strip().casefold() for dataset in datasets}
        if datasets else None
    )

    if selected_seeds and any(seed < 0 for seed in selected_seeds):
        raise ValueError("seeds must be non-negative integers")

    rows: list[dict[str, Any]] = []

    # Validate inputs before creating the CSV.
    for path in paths:
        try:
            row = read_result(path)
        except (OSError, ValueError) as error:
            raise ValueError(f"could not read {path}: {error}") from error

        if (
            selected_models is not None
            and str(row.get("model", "")).strip().casefold()
            not in selected_models
        ):
            continue

        if (
            selected_seeds is not None
            and row.get("seed") not in selected_seeds
        ):
            continue

        if (
            selected_datasets is not None
            and str(row.get("dataset", "")).strip().casefold()
            not in selected_datasets
        ):
            continue

        rows.append(row)

    if not rows:
        raise ValueError(
            "no experiments match the requested filters; "
            "check the model, seed, dataset, and results directory"
        )

    preferred_columns = [
        "source_file",
        "experiment",
        "dataset",
        "model",
        "model_identification",
        "seed",
        "device",
        "protocol_version",
        "test.examples",
        "test.loss",
        "test.recall@10",
        "test.recall@20",
        "test.mrr@10",
        "test.mrr@20",
        "test.ndcg@10",
        "test.ndcg@20",
        "runtime_seconds",
        "peak_gpu_memory_bytes",
        "parameter_count",
        "best_epoch",
        "epochs_recorded",
    ]

    available_columns = {key for row in rows for key in row}
    columns = preferred_columns + sorted(
        available_columns - set(preferred_columns)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # "x" means exclusive creation: an existing file cannot be overwritten.
    with output_path.open("x", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("experiments"),
        help="Folder to search recursively for results.json files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/generated/all_results.csv"),
        help="New CSV filename; existing files are never overwritten.",
    )
    parser.add_argument(
        "--model",
        nargs="+",
        dest="models",
        help="Include these models, e.g. srgnn sasrec.",
    )
    parser.add_argument(
        "--seed",
        nargs="+",
        type=int,
        dest="seeds",
        help="Include these seeds, e.g. 2028 2029.",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        dest="datasets",
        help="Include these datasets, e.g. diginetica retailrocket.",
    )

    args = parser.parse_args()

    try:
        count = collect_results(
            args.results_root,
            args.output,
            models=args.models,
            seeds=args.seeds,
            datasets=args.datasets,
        )
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")

    print(f"Collected {count} experiment(s).")
    print(f"CSV saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()