"""Command-line script to plot stratified session length accuracy charts from existing results.

Loads stratified_length_results.json directly without requiring checkpoint re-evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from recsys.evaluation.plotting import plot_stratified_results


def load_stratified_json(path: Path) -> tuple[str, dict[str, Any]]:
    """Load stratified results JSON and determine the model label."""
    if path.is_dir():
        exp_dir = path
        json_path = path / "stratified_length_results.json"
    else:
        json_path = path
        exp_dir = path.parent

    if not json_path.is_file():
        raise FileNotFoundError(f"Stratified results not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "deletion_aug" in exp_dir.name:
        label = "SR-GNN-DA"
    elif "sasrec" in exp_dir.name:
        label = "SASRec"
    elif "srgnn" in exp_dir.name:
        label = "SR-GNN"
    else:
        label = exp_dir.name

    return label, data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiments",
        nargs="+",
        type=Path,
        help="Experiment directories containing stratified_length_results.json or paths to the JSON files.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output PNG file path (e.g. experiments/diginetica_stratified_length.png).",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default=None,
        help="Optional dataset name override (e.g. Diginetica or RetailRocket).",
    )
    args = parser.parse_args()

    results_map: dict[str, dict[str, Any]] = {}
    inferred_ds = None

    for p in args.experiments:
        label, data = load_stratified_json(p)
        results_map[label] = data
        if inferred_ds is None:
            if "retailrocket" in str(p).lower():
                inferred_ds = "RetailRocket"
            elif "diginetica" in str(p).lower():
                inferred_ds = "Diginetica"

    ds_name = args.dataset or inferred_ds or "Benchmark Dataset"
    plot_stratified_results(results_map, args.output, dataset_name=ds_name)


if __name__ == "__main__":
    main()
