"""Export noise-degradation metrics from saved robustness results.

Uses the clean evaluation in each robustness_results.json as its baseline.
Inputs are read-only. Existing CSV outputs are never overwritten.
Requires only the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


COLUMNS = [
    "source_file",
    "source_sha256",
    "experiment",
    "model",
    "dataset",
    "seed",
    "metadata_file",
    "metadata_sha256",
    "condition",
    "corruption_type",
    "noise_ratio",
    "metric",
    "clean_examples",
    "condition_examples",
    "clean_score",
    "condition_score",
    "absolute_drop_percentage_points",
    "relative_degradation_pct",
]


def require_object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be a JSON object")
    return value


def read_score(
    condition: dict[str, Any],
    metric: str,
    location: str,
) -> float:
    value = condition.get(metric)

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(
            f"{location}: {metric} must be a finite score in [0, 1]"
        )

    return float(value)


def read_examples(condition: dict[str, Any], location: str) -> int:
    value = condition.get("examples")

    if type(value) is not int or value < 1:
        raise ValueError(f"{location}: examples must be a positive integer")

    return value


def read_metadata(path: Path) -> dict[str, Any]:
    """Read optional companion training metadata without guessing model names."""
    companion = path.parent / "results.json"

    metadata: dict[str, Any] = {
        "experiment": path.parent.name,
        "model": "unknown",
        "dataset": "",
        "seed": "",
        "metadata_file": "",
        "metadata_sha256": "",
    }

    if not companion.is_file():
        return metadata

    contents = companion.read_bytes()
    data = require_object(
        json.loads(contents.decode("utf-8-sig")),
        str(companion),
    )
    config = require_object(data.get("config", {}), f"{companion}: config")

    model = data.get("model_type") or config.get("model_type")
    if not model and "gnn_steps" in config:
        model = "srgnn"

    metadata.update({
        "experiment": data.get("experiment") or path.parent.name,
        "model": model or "unknown",
        "dataset": data.get("dataset") or config.get("dataset") or "",
        "seed": data.get("seed", config.get("seed", "")),
        "metadata_file": companion.resolve().as_posix(),
        "metadata_sha256": hashlib.sha256(contents).hexdigest(),
    })

    return metadata


def rows_from_result(
    path: Path,
    metrics: list[str],
) -> list[dict[str, Any]]:
    contents = path.read_bytes()
    data = require_object(
        json.loads(contents.decode("utf-8-sig")),
        str(path),
    )

    clean = require_object(data.get("clean"), f"{path}: clean")
    clean_examples = read_examples(clean, f"{path}: clean")
    clean_scores = {
        metric: read_score(clean, metric, f"{path}: clean")
        for metric in metrics
    }

    conditions = []
    for name, value in data.items():
        if name == "clean":
            continue

        match = re.fullmatch(
            r"(deletion|substitution|mixed)_([0-9]+(?:\.[0-9]+)?)",
            name,
        )
        if match is None:
            raise ValueError(f"{path}: unknown condition {name!r}")

        corruption_type, ratio_text = match.groups()
        ratio = float(ratio_text)

        if not math.isfinite(ratio) or not 0 < ratio <= 1:
            raise ValueError(f"{path}: invalid noise ratio in {name}")

        condition = require_object(value, f"{path}: {name}")
        conditions.append((corruption_type, ratio, name, condition))

    if not conditions:
        raise ValueError(f"{path}: no noisy evaluation conditions found")

    identities = [(kind, ratio) for kind, ratio, _, _ in conditions]
    if len(identities) != len(set(identities)):
        raise ValueError(
            f"{path}: duplicate corruption conditions with equivalent ratios"
        )

    conditions.sort(key=lambda entry: (entry[0], entry[1]))
    metadata = read_metadata(path)
    fingerprint = hashlib.sha256(contents).hexdigest()
    rows = []

    for corruption_type, ratio, name, condition in conditions:
        examples = read_examples(condition, f"{path}: {name}")

        if examples != clean_examples:
            raise ValueError(
                f"{path}: {name} example count differs from clean evaluation"
            )

        for metric in metrics:
            clean_score = clean_scores[metric]
            condition_score = read_score(
                condition,
                metric,
                f"{path}: {name}",
            )
            drop = clean_score - condition_score

            # Relative degradation is undefined when the baseline is zero.
            relative = (
                drop / clean_score * 100
                if clean_score > 0
                else None
            )

            rows.append({
                "source_file": path.resolve().as_posix(),
                "source_sha256": fingerprint,
                **metadata,
                "condition": name,
                "corruption_type": corruption_type,
                "noise_ratio": ratio,
                "metric": metric,
                "clean_examples": clean_examples,
                "condition_examples": examples,
                "clean_score": clean_score,
                "condition_score": condition_score,
                "absolute_drop_percentage_points": drop * 100,
                "relative_degradation_pct": relative,
            })

    return rows


def summarize_noise_degradation(
    result_paths: list[Path],
    output_path: Path,
    metrics: list[str],
) -> int:
    if output_path.suffix.lower() != ".csv":
        raise ValueError("output filename must end in .csv")

    if output_path.exists():
        raise FileExistsError(
            f"output already exists: {output_path}; "
            "choose a different filename"
        )

    if not metrics or len(metrics) != len(set(metrics)):
        raise ValueError("provide unique metric names")

    for metric in metrics:
        if re.fullmatch(r"(recall|mrr|ndcg)@[1-9][0-9]*", metric) is None:
            raise ValueError(
                f"unsupported metric: {metric}; "
                "use recall@K, mrr@K, or ndcg@K"
            )

    paths = sorted({path.resolve() for path in result_paths})
    if not paths:
        raise ValueError("provide at least one robustness result file")

    rows = []
    for path in paths:
        try:
            rows.extend(rows_from_result(path, metrics))
        except (OSError, ValueError) as error:
            raise ValueError(f"could not read {path}: {error}") from error

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("x", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        nargs="+",
        required=True,
        type=Path,
        help="Paths to robustness_results.json files.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["recall@20", "mrr@20", "ndcg@20"],
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    try:
        count = summarize_noise_degradation(
            args.results,
            args.output,
            args.metrics,
        )
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")

    print(f"Exported {count} condition-metric row(s).")
    print(f"CSV saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()