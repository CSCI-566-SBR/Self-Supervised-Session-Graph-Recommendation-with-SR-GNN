"""Load validated processed prefix/target artifacts for model training."""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SessionExample = tuple[tuple[int, ...], int]


@dataclass(frozen=True)
class ProcessedDataset:
    """Validated model-facing splits and their shared item vocabulary."""

    train: tuple[SessionExample, ...]
    validation: tuple[SessionExample, ...]
    test: tuple[SessionExample, ...]
    item_mapping: dict[Any, int]
    summary: dict[str, Any]
    checksums: dict[str, str]

    @property
    def num_items(self) -> int:
        return len(self.item_mapping)


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_pickle(path: Path) -> Any:
    # Processed pickle files are trusted project artifacts, never user uploads.
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_item_mapping(path: str | Path) -> dict[Any, int]:
    """Load a raw-item to contiguous one-based model-ID mapping."""

    path = Path(path)
    value = _load_pickle(path)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{path.name} must contain a non-empty dictionary")

    item_ids = list(value.values())
    if any(isinstance(item_id, bool) or not isinstance(item_id, int) for item_id in item_ids):
        raise ValueError(f"{path.name} values must be integer item IDs")
    expected = set(range(1, len(item_ids) + 1))
    if set(item_ids) != expected:
        raise ValueError(f"{path.name} item IDs must be contiguous from 1")
    return dict(value)


def load_processed_split(
    path: str | Path, num_items: int
) -> tuple[SessionExample, ...]:
    """Load and validate one ``(prefixes, targets)`` pickle artifact."""

    path = Path(path)
    value = _load_pickle(path)
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"{path.name} must contain (prefixes, targets)")
    prefixes, targets = value
    if not isinstance(prefixes, (list, tuple)) or not isinstance(targets, (list, tuple)):
        raise ValueError(f"{path.name} prefixes and targets must be sequences")
    if len(prefixes) != len(targets):
        raise ValueError(f"{path.name} prefix and target counts differ")
    if not prefixes:
        raise ValueError(f"{path.name} must contain at least one example")

    examples: list[SessionExample] = []
    for index, (prefix, target) in enumerate(zip(prefixes, targets)):
        if not isinstance(prefix, (list, tuple)) or not prefix:
            raise ValueError(f"{path.name} prefix {index} must be a non-empty sequence")
        if any(
            isinstance(item, bool)
            or not isinstance(item, int)
            or item < 1
            or item > num_items
            for item in prefix
        ):
            raise ValueError(
                f"{path.name} prefix {index} contains an invalid or out-of-range item ID"
            )
        if (
            isinstance(target, bool)
            or not isinstance(target, int)
            or target < 1
            or target > num_items
        ):
            raise ValueError(f"{path.name} target {index} is invalid or out of range")
        examples.append((tuple(prefix), target))
    return tuple(examples)


def load_processed_dataset(data_dir: str | Path) -> ProcessedDataset:
    """Load the shared train/validation/test artifacts from one directory."""

    data_dir = Path(data_dir)
    required = {
        "train": data_dir / "train.txt",
        "validation": data_dir / "validation.txt",
        "test": data_dir / "test.txt",
        "item_mapping": data_dir / "item_mapping.pkl",
        "summary": data_dir / "preprocessing_summary.json",
    }
    missing = [path.name for path in required.values() if not path.is_file()]
    if missing:
        raise ValueError(f"missing processed artifacts: {', '.join(sorted(missing))}")

    item_mapping = load_item_mapping(required["item_mapping"])
    num_items = len(item_mapping)
    train = load_processed_split(required["train"], num_items)
    validation = load_processed_split(required["validation"], num_items)
    test = load_processed_split(required["test"], num_items)

    with required["summary"].open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if not isinstance(summary, dict):
        raise ValueError("preprocessing_summary.json must contain a JSON object")
    reported_items = summary.get("num_items", summary.get("vocab_size"))
    if reported_items is not None and reported_items != num_items:
        raise ValueError("preprocessing summary vocabulary size does not match mapping")

    checksums = {
        name: sha256(path)
        for name, path in required.items()
    }
    return ProcessedDataset(
        train=train,
        validation=validation,
        test=test,
        item_mapping=item_mapping,
        summary=summary,
        checksums=checksums,
    )
