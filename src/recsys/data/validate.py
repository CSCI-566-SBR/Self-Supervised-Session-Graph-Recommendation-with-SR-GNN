"""Validate one processed dataset or an entire clean/noisy pipeline tree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .processed import ProcessedDataset, load_processed_dataset


def _is_subsequence(candidate: Sequence[int], source: Sequence[int]) -> bool:
    source_iterator = iter(source)
    return all(any(value == expected for value in source_iterator) for expected in candidate)


def validate_dataset(path: str | Path) -> dict[str, Any]:
    dataset = load_processed_dataset(path)
    return {
        "status": "valid",
        "path": str(dataset.root),
        "num_items": dataset.num_items,
        "examples": {
            "train": len(dataset.train),
            "validation": len(dataset.validation),
            "test": len(dataset.test),
        },
        "checksums": dataset.checksums,
    }


def _validate_noise_pair(clean: ProcessedDataset, noisy: ProcessedDataset) -> dict[str, Any]:
    if noisy.item_mapping != clean.item_mapping:
        raise ValueError(f"item mapping differs in {noisy.root}")
    if noisy.train != clean.train or noisy.validation != clean.validation:
        raise ValueError(f"train or validation data differs in {noisy.root}")
    if len(noisy.test) != len(clean.test):
        raise ValueError(f"test example count differs in {noisy.root}")

    metadata = noisy.summary.get("noise")
    if not isinstance(metadata, dict):
        raise ValueError(f"noise metadata missing from {noisy.root}")
    corruption_type = metadata.get("corruption_type")
    changed_examples = 0
    for (clean_prefix, clean_target), (noisy_prefix, noisy_target) in zip(
        clean.test, noisy.test, strict=True
    ):
        if noisy_target != clean_target:
            raise ValueError(f"target changed in {noisy.root}")
        if len(clean_prefix) == 1 and noisy_prefix != clean_prefix:
            raise ValueError(f"length-one prefix changed in {noisy.root}")
        if len(clean_prefix) >= 2 and len(noisy_prefix) < 2:
            raise ValueError(f"minimum prefix length violated in {noisy.root}")
        if noisy_prefix.count(clean_target) > clean_prefix.count(clean_target):
            raise ValueError(f"noise introduced the held-out target in {noisy.root}")
        if corruption_type == "deletion" and not _is_subsequence(noisy_prefix, clean_prefix):
            raise ValueError(f"deletion produced a non-subsequence in {noisy.root}")
        if corruption_type == "substitution" and len(noisy_prefix) != len(clean_prefix):
            raise ValueError(f"substitution changed prefix length in {noisy.root}")
        if corruption_type == "mixed" and len(noisy_prefix) > len(clean_prefix):
            raise ValueError(f"mixed noise increased prefix length in {noisy.root}")
        changed_examples += int(noisy_prefix != clean_prefix)
    return {
        "status": "valid",
        "corruption_type": corruption_type,
        "ratio": metadata.get("ratio"),
        "changed_examples": changed_examples,
        "test_examples": len(noisy.test),
        "checksums": noisy.checksums,
    }


def validate_pipeline(path: str | Path) -> dict[str, Any]:
    root = Path(path).expanduser().resolve()
    clean = load_processed_dataset(root / "clean")
    noisy_root = root / "noisy"
    variants = sorted(directory for directory in noisy_root.iterdir() if directory.is_dir())
    if not variants:
        raise ValueError(f"no noisy datasets found in {noisy_root}")
    return {
        "status": "valid",
        "clean": validate_dataset(clean.root),
        "noisy": {
            variant.name: _validate_noise_pair(clean, load_processed_dataset(variant))
            for variant in variants
        },
    }


# Compatibility with the original single-directory validator API.
def validate(data_dir: Path) -> dict[str, Any]:
    if (data_dir / "clean").is_dir() and (data_dir / "noisy").is_dir():
        return validate_pipeline(data_dir)
    return validate_dataset(data_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.data_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
