"""Validate clean and perturbed artifacts and write a JSON report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .artifacts import load_pickle

RATIOS = (0.05, 0.10, 0.20)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(data_dir: Path) -> dict:
    clean_sequences, clean_targets = load_pickle(data_dir / "test.txt")
    vocab_size = len(load_pickle(data_dir / "item_mapping.pkl"))
    if len(clean_sequences) != len(clean_targets):
        raise ValueError("clean sequence/target counts differ")
    if any(not sequence for sequence in clean_sequences):
        raise ValueError("clean data contains an empty prefix")
    if any(item < 1 or item > vocab_size for sequence in clean_sequences for item in sequence):
        raise ValueError("clean prefix contains an out-of-vocabulary item")
    if any(target < 1 or target > vocab_size for target in clean_targets):
        raise ValueError("clean target contains an out-of-vocabulary item")

    report = {
        "status": "valid",
        "examples": len(clean_targets),
        "vocab_size": vocab_size,
        "clean_input_clicks": sum(map(len, clean_sequences)),
        "clean_sha256": sha256(data_dir / "test.txt"),
        "noise": {},
    }
    for ratio in RATIOS:
        path = data_dir / f"test_noise_{ratio:.2f}.txt"
        noisy_sequences, noisy_targets = load_pickle(path)
        if noisy_targets != clean_targets:
            raise ValueError(f"targets changed in {path.name}")
        if len(noisy_sequences) != len(clean_sequences):
            raise ValueError(f"example count changed in {path.name}")
        if any(not sequence for sequence in noisy_sequences):
            raise ValueError(f"empty prefix in {path.name}")
        if any(item < 1 or item > vocab_size for sequence in noisy_sequences for item in sequence):
            raise ValueError(f"out-of-vocabulary item in {path.name}")
        for clean, noisy in zip(clean_sequences, noisy_sequences):
            if len(clean) == 1 and noisy != clean:
                raise ValueError(f"length-one prefix changed in {path.name}")
            if len(clean) >= 2 and len(noisy) < 2:
                raise ValueError(f"minimum prefix length violated in {path.name}")
        report["noise"][f"{ratio:.2f}"] = {
            "sha256": sha256(path),
            "examples": len(noisy_targets),
            "input_clicks": sum(map(len, noisy_sequences)),
            "changed_examples": sum(a != b for a, b in zip(clean_sequences, noisy_sequences)),
            "targets_identical": True,
        }
    (data_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.data_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
