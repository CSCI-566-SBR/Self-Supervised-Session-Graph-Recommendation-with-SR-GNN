"""Load and validate the frozen model-facing processed dataset contract.

The split files use pickle for compatibility with the original SR-GNN code.
Only load artifact directories produced by this project or another trusted
source; pickle is not safe for untrusted input.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .artifacts import load_pickle

REQUIRED_FILES = {
    "train": "train.txt",
    "validation": "validation.txt",
    "test": "test.txt",
    "item_mapping": "item_mapping.pkl",
    "summary": "preprocessing_summary.json",
}

Example = tuple[tuple[int, ...], int]
Split = tuple[Example, ...]


@dataclass(frozen=True)
class ProcessedDataset:
    """Validated processed examples shared by every model."""

    train: Split
    validation: Split
    test: Split
    item_mapping: dict[Any, int]
    summary: dict[str, Any]
    checksums: dict[str, str]
    root: Path = field(default_factory=Path, compare=False)

    @property
    def num_items(self) -> int:
        return len(self.item_mapping)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 fingerprint of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Name used by the SR-GNN baseline branch.
sha256 = sha256_file


def _validate_mapping(value: Any, path: Path) -> tuple[dict[Any, int], int]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{path.name} must contain a non-empty dictionary")

    model_ids = list(value.values())
    if any(type(item_id) is not int for item_id in model_ids):
        raise ValueError(f"{path.name} values must be Python int objects")

    expected = list(range(1, len(model_ids) + 1))
    if sorted(model_ids) != expected:
        raise ValueError(
            f"{path.name} values must be contiguous and contain every integer from 1 through "
            f"{len(model_ids)} exactly once"
        )
    return value, len(model_ids)


def load_item_mapping(path: str | Path) -> dict[Any, int]:
    """Load the shared raw-item to contiguous model-ID mapping."""
    artifact_path = Path(path)
    mapping, _ = _validate_mapping(load_pickle(artifact_path), artifact_path)
    return dict(mapping)


def _validate_item_id(value: Any, *, num_items: int, location: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{location} has an invalid or out-of-range item ID; expected Python int")
    if not 1 <= value <= num_items:
        raise ValueError(
            f"{location} has an invalid or out-of-range item ID; "
            f"expected 1..{num_items}, got {value}"
        )
    return value


def load_processed_split(path: str | Path, num_items: int) -> Split:
    """Load and validate one ``(prefixes, targets)`` split artifact."""
    artifact_path = Path(path)
    value = load_pickle(artifact_path)
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(
            f"{artifact_path.name} must contain a two-element (prefixes, targets) tuple"
        )

    prefixes, targets = value
    if not isinstance(prefixes, (list, tuple)) or not isinstance(targets, (list, tuple)):
        raise ValueError(f"{artifact_path.name} prefixes and targets must be lists or tuples")
    if not prefixes or not targets:
        raise ValueError(f"{artifact_path.name} must contain at least one example")
    if len(prefixes) != len(targets):
        raise ValueError(f"{artifact_path.name} prefix and target counts differ")

    examples: list[Example] = []
    for index, (prefix, target) in enumerate(zip(prefixes, targets, strict=True)):
        if not isinstance(prefix, (list, tuple)) or not prefix:
            raise ValueError(
                f"{artifact_path.name} prefix {index} must be a non-empty list or tuple"
            )
        normalized_prefix = tuple(
            _validate_item_id(
                item,
                num_items=num_items,
                location=f"{artifact_path.name} prefix {index} item {position}",
            )
            for position, item in enumerate(prefix)
        )
        normalized_target = _validate_item_id(
            target,
            num_items=num_items,
            location=f"{artifact_path.name} target {index}",
        )
        examples.append((normalized_prefix, normalized_target))
    return tuple(examples)


def load_processed_dataset(path: str | Path) -> ProcessedDataset:
    """Load one clean or noisy dataset directory through the shared contract."""
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"missing processed artifacts: directory not found: {root}")

    paths = {name: root / filename for name, filename in REQUIRED_FILES.items()}
    missing = [path.name for path in paths.values() if not path.is_file()]
    if missing:
        raise ValueError(f"missing processed artifacts: {', '.join(sorted(missing))}")

    item_mapping = load_item_mapping(paths["item_mapping"])
    num_items = len(item_mapping)
    splits = {
        name: load_processed_split(paths[name], num_items)
        for name in ("train", "validation", "test")
    }

    with paths["summary"].open(encoding="utf-8") as handle:
        summary = json.load(handle)
    if not isinstance(summary, dict):
        raise ValueError("preprocessing_summary.json must contain a JSON object")

    reported_size = summary.get("num_items", summary.get("vocab_size"))
    if reported_size is not None and reported_size != num_items:
        raise ValueError(
            "preprocessing_summary.json vocabulary size does not match item_mapping.pkl"
        )

    checksums = {name: sha256_file(artifact_path) for name, artifact_path in paths.items()}
    return ProcessedDataset(
        train=splits["train"],
        validation=splits["validation"],
        test=splits["test"],
        item_mapping=item_mapping,
        summary=summary,
        checksums=checksums,
        root=root,
    )
