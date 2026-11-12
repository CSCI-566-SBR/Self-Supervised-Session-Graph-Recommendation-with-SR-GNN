"""Run and summarize repeated experiments that differ only by random seed."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

from recsys.training.config import ExperimentConfig
from recsys.training.runner import run_experiment


def _validate_seeds(seeds: list[int]) -> None:
    if len(seeds) < 2:
        raise ValueError("a multi-seed experiment requires at least two seeds")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    if any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        for seed in seeds
    ):
        raise ValueError("seeds must be non-negative integers")


def _metric_summary(
    results: list[dict[str, Any]], split: str, metric: str
) -> dict[str, Any]:
    values = [float(result[split][metric]) for result in results]
    return {
        "mean": fmean(values),
        "sample_standard_deviation": stdev(values),
        "values_by_seed": {
            str(result["seed"]): value for result, value in zip(results, values)
        },
    }


def _require_shared_value(results: list[dict[str, Any]], field: str) -> Any:
    value = results[0][field]
    if any(result[field] != value for result in results[1:]):
        raise ValueError(f"seed results disagree on shared field: {field}")
    return value


def aggregate_seed_results(
    results: list[dict[str, Any]], metric_k: int
) -> dict[str, Any]:
    """Return mean and sample standard deviation for comparable seed runs."""

    if len(results) < 2:
        raise ValueError("at least two completed seed results are required")

    seeds = [result["seed"] for result in results]
    _validate_seeds(seeds)
    recall_name = f"recall@{metric_k}"
    mrr_name = f"mrr@{metric_k}"

    shared = {
        field: _require_shared_value(results, field)
        for field in (
            "dataset",
            "dataset_artifact",
            "dataset_checksums",
            "git_commit",
            "parameter_count",
        )
    }
    runs = [
        {
            "seed": result["seed"],
            "experiment": result["experiment"],
            "output_dir": result["config"]["output_dir"],
            "best_epoch": result["best_epoch"],
            "best_validation": result["best_validation"],
            "test": result["test"],
            "runtime_seconds": result["runtime_seconds"],
        }
        for result in results
    ]

    return {
        "experiment": results[0]["experiment"].rsplit("_seed", 1)[0],
        "seeds": seeds,
        "number_of_runs": len(results),
        "metric_k": metric_k,
        "standard_deviation_definition": "sample standard deviation (n - 1)",
        "shared": shared,
        "aggregate": {
            "best_validation": {
                recall_name: _metric_summary(
                    results, "best_validation", recall_name
                ),
                mrr_name: _metric_summary(results, "best_validation", mrr_name),
            },
            "test": {
                recall_name: _metric_summary(results, "test", recall_name),
                mrr_name: _metric_summary(results, "test", mrr_name),
            },
        },
        "total_runtime_seconds": sum(
            float(result["runtime_seconds"]) for result in results
        ),
        "runs": runs,
    }


def run_seed_experiments(
    base_config: ExperimentConfig,
    seeds: list[int],
    output_root: Path,
) -> dict[str, Any]:
    """Run one experiment per seed and save an aggregate summary."""

    _validate_seeds(seeds)
    if output_root.exists():
        raise FileExistsError(
            f"output root already exists; refusing to overwrite: {output_root}"
        )
    output_root.mkdir(parents=True)

    results = []
    for index, seed in enumerate(seeds, start=1):
        seed_config = replace(
            base_config,
            name=f"{base_config.name}_seed{seed}",
            output_dir=output_root / f"seed_{seed}",
            seed=seed,
        )
        print(f"Running seed {seed} ({index}/{len(seeds)})", flush=True)
        results.append(run_experiment(seed_config))

    summary = aggregate_seed_results(results, base_config.metric_k)
    summary["base_config"] = base_config.as_serializable_dict()
    summary_path = output_root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary
