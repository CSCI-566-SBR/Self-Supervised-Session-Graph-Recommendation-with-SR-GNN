"""Validate clean and perturbed dataset artifacts and write a JSON report."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .artifacts import load_pickle


RATIOS = (0.05, 0.10, 0.20)

CORRUPTION_TYPES = (
    "deletion",
    "substitution",
    "mixed",
)

REQUIRED_CLEAN_ARTIFACTS = (
    "train.txt",
    "validation.txt",
    "test.txt",
    "all_train_seq.txt",
    "item_mapping.pkl",
    "preprocessing_summary.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def validate_required_files(data_dir: Path) -> None:
    missing = [
        filename
        for filename in REQUIRED_CLEAN_ARTIFACTS
        if not (data_dir / filename).is_file()
    ]

    for corruption_type in CORRUPTION_TYPES:
        for ratio in RATIOS:
            filename = (
                f"test_noise_{corruption_type}_"
                f"{ratio:.2f}.txt"
            )

            if not (data_dir / filename).is_file():
                missing.append(filename)

    if missing:
        raise ValueError(
            "missing required artifacts: "
            + ", ".join(sorted(missing))
        )


def validate_item_mapping(
    item_mapping: Any,
) -> int:
    if not isinstance(item_mapping, dict):
        raise ValueError(
            "item_mapping.pkl must contain a dictionary"
        )

    if not item_mapping:
        raise ValueError(
            "item_mapping.pkl is empty"
        )

    item_ids = list(item_mapping.values())

    if any(
        not isinstance(item_id, int)
        for item_id in item_ids
    ):
        raise ValueError(
            "item mapping contains a non-integer ID"
        )

    if len(set(item_ids)) != len(item_ids):
        raise ValueError(
            "item mapping contains duplicate integer IDs"
        )

    expected_ids = set(
        range(1, len(item_mapping) + 1)
    )

    if set(item_ids) != expected_ids:
        raise ValueError(
            "item mapping IDs must be contiguous from 1 "
            "through the vocabulary size"
        )

    return len(item_mapping)


def validate_supervised_artifact(
    path: Path,
    vocab_size: int,
) -> tuple[list[list[int]], list[int]]:
    value = load_pickle(path)

    if (
        not isinstance(value, tuple)
        or len(value) != 2
    ):
        raise ValueError(
            f"{path.name} must contain "
            "(sequences, targets)"
        )

    sequences, targets = value

    if not isinstance(sequences, list):
        raise ValueError(
            f"{path.name} sequences must be a list"
        )

    if not isinstance(targets, list):
        raise ValueError(
            f"{path.name} targets must be a list"
        )

    if len(sequences) != len(targets):
        raise ValueError(
            f"sequence/target counts differ in {path.name}"
        )

    if not sequences:
        raise ValueError(
            f"{path.name} contains no examples"
        )

    if any(
        not isinstance(sequence, list)
        for sequence in sequences
    ):
        raise ValueError(
            f"{path.name} contains a non-list sequence"
        )

    if any(not sequence for sequence in sequences):
        raise ValueError(
            f"{path.name} contains an empty prefix"
        )

    if any(
        not isinstance(item, int)
        for sequence in sequences
        for item in sequence
    ):
        raise ValueError(
            f"{path.name} contains a non-integer item ID"
        )

    if any(
        item < 1 or item > vocab_size
        for sequence in sequences
        for item in sequence
    ):
        raise ValueError(
            f"{path.name} contains an out-of-vocabulary item"
        )

    if any(
        not isinstance(target, int)
        for target in targets
    ):
        raise ValueError(
            f"{path.name} contains a non-integer target"
        )

    if any(
        target < 1 or target > vocab_size
        for target in targets
    ):
        raise ValueError(
            f"{path.name} contains an out-of-vocabulary target"
        )

    return sequences, targets


def expand_prefixes(
    sessions: list[list[int]],
) -> tuple[list[list[int]], list[int]]:
    sequences: list[list[int]] = []
    targets: list[int] = []

    for session in sessions:
        for position in range(1, len(session)):
            sequences.append(
                session[:position]
            )
            targets.append(
                session[position]
            )

    return sequences, targets


def is_subsequence(
    candidate: list[int],
    source: list[int],
) -> bool:
    source_iterator = iter(source)

    return all(
        any(
            source_item == candidate_item
            for source_item in source_iterator
        )
        for candidate_item in candidate
    )


def validate_training_sessions(
    data_dir: Path,
    vocab_size: int,
    train_sequences: list[list[int]],
    train_targets: list[int],
) -> list[list[int]]:
    sessions = load_pickle(
        data_dir / "all_train_seq.txt"
    )

    if not isinstance(sessions, list):
        raise ValueError(
            "all_train_seq.txt must contain a list"
        )

    if not sessions:
        raise ValueError(
            "all_train_seq.txt contains no sessions"
        )

    if any(
        not isinstance(session, list)
        for session in sessions
    ):
        raise ValueError(
            "all_train_seq.txt contains a non-list session"
        )

    if any(
        len(session) < 2
        for session in sessions
    ):
        raise ValueError(
            "all_train_seq.txt contains a session "
            "shorter than two interactions"
        )

    if any(
        not isinstance(item, int)
        for session in sessions
        for item in session
    ):
        raise ValueError(
            "all_train_seq.txt contains a non-integer item"
        )

    if any(
        item < 1 or item > vocab_size
        for session in sessions
        for item in session
    ):
        raise ValueError(
            "all_train_seq.txt contains an "
            "out-of-vocabulary item"
        )

    expected_sequences, expected_targets = (
        expand_prefixes(sessions)
    )

    if expected_sequences != train_sequences:
        raise ValueError(
            "train.txt sequences do not match "
            "all_train_seq.txt prefix expansion"
        )

    if expected_targets != train_targets:
        raise ValueError(
            "train.txt targets do not match "
            "all_train_seq.txt prefix expansion"
        )

    observed_training_items = {
        item
        for session in sessions
        for item in session
    }

    expected_training_items = set(
        range(1, vocab_size + 1)
    )

    if observed_training_items != expected_training_items:
        raise ValueError(
            "item mapping contains IDs not represented "
            "in the filtered training sessions"
        )

    return sessions


def validate_preprocessing_summary(
    data_dir: Path,
    split_results: dict[str, dict[str, int]],
    vocab_size: int,
) -> dict:
    summary_path = (
        data_dir / "preprocessing_summary.json"
    )

    with summary_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        summary = json.load(handle)

    required_fields = (
        "dataset",
        "validation_start",
        "test_start",
        "train_sessions",
        "validation_sessions",
        "test_sessions",
        "train_examples",
        "validation_examples",
        "test_examples",
        "vocab_size",
        "filtering_vocabulary_protocol",
    )

    missing_fields = [
        field
        for field in required_fields
        if field not in summary
    ]

    if missing_fields:
        raise ValueError(
            "preprocessing summary is missing fields: "
            + ", ".join(missing_fields)
        )

    validation_start = datetime.fromisoformat(
        summary["validation_start"]
    )

    test_start = datetime.fromisoformat(
        summary["test_start"]
    )

    if validation_start >= test_start:
        raise ValueError(
            "validation_start must be earlier than test_start"
        )

    expected_protocol = (
        "global_frequency_filter_then_training_vocabulary"
    )

    if (
        summary["filtering_vocabulary_protocol"]
        != expected_protocol
    ):
        raise ValueError(
            "unexpected filtering/vocabulary protocol"
        )

    if summary["vocab_size"] != vocab_size:
        raise ValueError(
            "summary vocabulary size does not match "
            "item_mapping.pkl"
        )

    for split_name in (
        "train",
        "validation",
        "test",
    ):
        summary_count = summary[
            f"{split_name}_examples"
        ]

        artifact_count = split_results[
            split_name
        ]["examples"]

        if summary_count != artifact_count:
            raise ValueError(
                f"{split_name} example count does not "
                "match preprocessing summary"
            )

    return summary


def validate_noisy_artifact(
    path: Path,
    corruption_type: str,
    clean_sequences: list[list[int]],
    clean_targets: list[int],
    vocab_size: int,
) -> dict:
    noisy_sequences, noisy_targets = (
        validate_supervised_artifact(
            path,
            vocab_size,
        )
    )

    if noisy_targets != clean_targets:
        raise ValueError(
            f"targets changed in {path.name}"
        )

    if len(noisy_sequences) != len(clean_sequences):
        raise ValueError(
            f"example count changed in {path.name}"
        )

    changed_examples = 0

    for clean, noisy, target in zip(
        clean_sequences,
        noisy_sequences,
        clean_targets,
    ):
        if len(clean) == 1 and noisy != clean:
            raise ValueError(
                f"length-one prefix changed in {path.name}"
            )

        if not noisy:
            raise ValueError(
                f"empty prefix in {path.name}"
            )

        # Deletion and target-safe substitution cannot increase
        # the number of target occurrences in the input.
        if noisy.count(target) > clean.count(target):
            raise ValueError(
                f"target was introduced into a prefix "
                f"in {path.name}"
            )

        if corruption_type == "deletion":
            if len(noisy) > len(clean):
                raise ValueError(
                    f"deletion increased sequence length "
                    f"in {path.name}"
                )

            if not is_subsequence(noisy, clean):
                raise ValueError(
                    f"deletion produced items or ordering "
                    f"not present in the clean prefix "
                    f"in {path.name}"
                )

        elif corruption_type == "substitution":
            if len(noisy) != len(clean):
                raise ValueError(
                    f"substitution changed sequence length "
                    f"in {path.name}"
                )

        elif corruption_type == "mixed":
            if len(noisy) > len(clean):
                raise ValueError(
                    f"mixed corruption increased sequence "
                    f"length in {path.name}"
                )

        else:
            raise ValueError(
                f"unknown corruption type: "
                f"{corruption_type}"
            )

        changed_examples += int(clean != noisy)

    return {
        "sha256": sha256(path),
        "examples": len(noisy_targets),
        "input_clicks": sum(
            map(len, noisy_sequences)
        ),
        "changed_examples": changed_examples,
        "targets_identical": True,
        "target_injection_detected": False,
    }


def validate(data_dir: Path) -> dict:
    validate_required_files(data_dir)

    item_mapping = load_pickle(
        data_dir / "item_mapping.pkl"
    )

    vocab_size = validate_item_mapping(
        item_mapping
    )

    split_artifacts = {
        "train": data_dir / "train.txt",
        "validation": data_dir / "validation.txt",
        "test": data_dir / "test.txt",
    }

    split_data: dict[
        str,
        tuple[list[list[int]], list[int]],
    ] = {}

    split_report: dict[str, dict[str, int | str]] = {}

    for split_name, path in split_artifacts.items():
        sequences, targets = validate_supervised_artifact(
            path,
            vocab_size,
        )

        split_data[split_name] = (
            sequences,
            targets,
        )

        split_report[split_name] = {
            "examples": len(targets),
            "input_clicks": sum(
                map(len, sequences)
            ),
            "sha256": sha256(path),
        }

    train_sequences, train_targets = split_data["train"]

    training_sessions = validate_training_sessions(
        data_dir,
        vocab_size,
        train_sequences,
        train_targets,
    )

    summary = validate_preprocessing_summary(
        data_dir,
        split_report,
        vocab_size,
    )

    clean_test_sequences, clean_test_targets = (
        split_data["test"]
    )

    noise_report: dict[
        str,
        dict[str, dict]
    ] = {}

    for corruption_type in CORRUPTION_TYPES:
        noise_report[corruption_type] = {}

        for ratio in RATIOS:
            path = (
                data_dir
                / (
                    f"test_noise_{corruption_type}_"
                    f"{ratio:.2f}.txt"
                )
            )

            noise_report[corruption_type][
                f"{ratio:.2f}"
            ] = validate_noisy_artifact(
                path,
                corruption_type,
                clean_test_sequences,
                clean_test_targets,
                vocab_size,
            )

    report = {
        "status": "valid",
        "dataset": summary["dataset"],
        "vocab_size": vocab_size,
        "training_sessions": len(training_sessions),
        "splits": split_report,
        "noise": noise_report,
    }

    (
        data_dir / "validation_report.json"
    ).write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    print(
        json.dumps(
            validate(args.data_dir),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()