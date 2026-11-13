"""Export clean preprocessing metadata to a dataset-summary CSV.

Reads JSON metadata only. Does not load datasets or run preprocessing.
Existing output files are never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


COLUMNS = [
    "source_file",
    "summary_sha256",
    "dataset",
    "protocol_version",
    "num_items",
    "train_sessions",
    "validation_sessions",
    "test_sessions",
    "train_examples",
    "validation_examples",
    "test_examples",
    "minimum_item_frequency",
    "minimum_session_length",
    "validation_definition",
    "test_definition",
    "validation_start",
    "test_start",
    "split_type",
    "filtering_protocol",
    "item_mapping_scope",
    "prefix_target_generation",
]


def require_object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be a JSON object")
    return value


def require_integer(
    value: Any,
    location: str,
    minimum: int,
) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(
            f"{location} must be an integer >= {minimum}"
        )
    return value


def require_text(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string")
    return value


def read_summary(path: Path) -> dict[str, Any]:
    """Read one clean summary and validate the table's required fields."""
    contents = path.read_bytes()
    summary = require_object(
        json.loads(contents.decode("utf-8-sig")),
        str(path),
    )

    if "noise" in summary:
        raise ValueError(
            f"{path}: provide the clean summary, not a noisy variant"
        )

    counts = require_object(
        summary.get("counts"),
        f"{path}: counts",
    )
    split = require_object(
        summary.get("split_strategy"),
        f"{path}: split_strategy",
    )

    row: dict[str, Any] = {
        "source_file": path.resolve().as_posix(),
        "summary_sha256": hashlib.sha256(contents).hexdigest(),
        "dataset": require_text(
            summary.get("dataset"),
            f"{path}: dataset",
        ),
        "num_items": require_integer(
            summary.get("num_items"),
            f"{path}: num_items",
            minimum=1,
        ),
    }

    for name in (
        "train_sessions",
        "validation_sessions",
        "test_sessions",
        "train_examples",
        "validation_examples",
        "test_examples",
    ):
        row[name] = require_integer(
            counts.get(name),
            f"{path}: counts.{name}",
            minimum=1,
        )

    for name, minimum in (
        ("minimum_item_frequency", 1),
        ("minimum_session_length", 2),
    ):
        row[name] = require_integer(
            summary.get(name),
            f"{path}: {name}",
            minimum=minimum,
        )

    for name in (
        "protocol_version",
        "filtering_protocol",
        "item_mapping_scope",
        "prefix_target_generation",
    ):
        row[name] = require_text(
            summary.get(name),
            f"{path}: {name}",
        )

    for name in (
        "validation_definition",
        "test_definition",
        "validation_start",
        "test_start",
    ):
        row[name] = require_text(
            split.get(name),
            f"{path}: split_strategy.{name}",
        )

    row["split_type"] = require_text(
        split.get("type"),
        f"{path}: split_strategy.type",
    )

    return row


def generate_dataset_summary(
    summary_paths: list[Path],
    output_path: Path,
) -> int:
    """Write one row per distinct summary path without overwriting output."""
    if output_path.suffix.lower() != ".csv":
        raise ValueError("output filename must end in .csv")

    if output_path.exists():
        raise FileExistsError(
            f"output already exists: {output_path}; "
            "choose a different CSV filename"
        )

    paths = sorted({path.resolve() for path in summary_paths})
    if not paths:
        raise ValueError("provide at least one summary file")

    # Read and validate every input before creating the output.
    rows = []
    for path in paths:
        try:
            rows.append(read_summary(path))
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
        "--summaries",
        nargs="+",
        required=True,
        type=Path,
        help="Paths to clean preprocessing_summary.json files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/generated/dataset_summary.csv"),
        help="New CSV filename; existing files are never overwritten.",
    )
    args = parser.parse_args()

    try:
        count = generate_dataset_summary(
            args.summaries,
            args.output,
        )
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")

    print(f"Collected {count} dataset summary file(s).")
    print(f"CSV saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()