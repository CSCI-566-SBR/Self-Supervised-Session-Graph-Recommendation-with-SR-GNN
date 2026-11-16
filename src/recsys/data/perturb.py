"""Build self-contained noisy evaluation datasets on top of clean artifacts."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Literal, Sequence

from .artifacts import save_pickle
from .processed import ProcessedDataset, load_processed_dataset

DEFAULT_RATIOS = (0.05, 0.10, 0.20)
DEFAULT_CORRUPTION_TYPES = ("deletion", "substitution", "mixed")
DEFAULT_PERTURBATION_SEED = 2026
CorruptionType = Literal["deletion", "substitution", "mixed"]


def sample_replacement(
    vocab_size: int, forbidden_items: set[int], rng: random.Random
) -> int | None:
    """Sample uniformly without retry loops from allowed one-based item IDs."""
    forbidden = sorted(item for item in forbidden_items if 1 <= item <= vocab_size)
    allowed_count = vocab_size - len(forbidden)
    if allowed_count <= 0:
        return None
    replacement = rng.randint(1, allowed_count)
    for forbidden_item in forbidden:
        if replacement >= forbidden_item:
            replacement += 1
    return replacement


def perturb_sequence(
    sequence: Sequence[int],
    ratio: float,
    vocab_size: int,
    rng: random.Random,
    *,
    target: int | None = None,
    corruption_type: CorruptionType = "mixed",
    minimum_length: int = 2,
) -> tuple[list[int], int]:
    """Perturb one prefix while preserving its target and minimum length."""
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("ratio must lie in [0, 1]")
    if corruption_type not in DEFAULT_CORRUPTION_TYPES:
        raise ValueError(f"unsupported corruption type: {corruption_type}")
    result = list(sequence)
    if ratio <= 0 or len(result) < minimum_length:
        return result, 0

    budget = min(len(result), max(1, round(len(result) * ratio)))
    positions = list(range(len(result)))
    rng.shuffle(positions)
    changed = 0
    for position in sorted(positions[:budget], reverse=True):
        operation = corruption_type
        if operation == "mixed":
            operation = "deletion" if rng.random() < 0.5 else "substitution"
        if operation == "deletion" and len(result) > minimum_length:
            del result[position]
            changed += 1
            continue
        if operation == "deletion":
            continue

        original = result[position]
        forbidden = {original}
        if target is not None:
            forbidden.add(target)
        replacement = sample_replacement(vocab_size, forbidden, rng)
        if replacement is not None:
            result[position] = replacement
            changed += 1
    return result, changed


def _copy_shared_files(clean_dir: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for filename in ("train.txt", "validation.txt", "item_mapping.pkl"):
        source = clean_dir / filename
        target = destination / filename
        shutil.copy2(source, target)


def _select_clicks(
    prefixes: list[list[int]],
    ratio: float,
    rng: random.Random,
    corruption_type: CorruptionType,
) -> tuple[dict[int, list[int]], int]:
    eligible: list[tuple[int, int]] = []
    for example_index, prefix in enumerate(prefixes):
        if len(prefix) < 2:
            continue
        positions = list(range(len(prefix)))
        rng.shuffle(positions)
        if corruption_type == "deletion":
            positions = positions[: max(0, len(prefix) - 2)]
        eligible.extend((example_index, position) for position in positions)
    rng.shuffle(eligible)
    requested = round(sum(map(len, prefixes)) * ratio)
    budget = min(len(eligible), requested)
    selected: dict[int, list[int]] = {}
    for example_index, position in eligible[:budget]:
        selected.setdefault(example_index, []).append(position)
    return selected, requested


def _noise_one_dataset(
    clean: ProcessedDataset,
    destination: Path,
    *,
    ratio: float,
    corruption_type: CorruptionType,
    seed: int,
) -> dict[str, int | float | str]:
    rng = random.Random(seed)
    clean_prefixes = [list(prefix) for prefix, _ in clean.test]
    targets = [target for _, target in clean.test]
    selected, requested = _select_clicks(clean_prefixes, ratio, rng, corruption_type)
    noisy = [list(prefix) for prefix in clean_prefixes]
    deletions = substitutions = skipped = 0

    for example_index, positions in selected.items():
        result = noisy[example_index]
        target = targets[example_index]
        for position in sorted(positions, reverse=True):
            operation = corruption_type
            if operation == "mixed":
                operation = "deletion" if rng.random() < 0.5 else "substitution"
            if operation == "deletion" and len(result) > 2:
                del result[position]
                deletions += 1
                continue
            if operation == "deletion" and corruption_type == "deletion":
                skipped += 1
                continue

            original = result[position]
            replacement = sample_replacement(clean.num_items, {original, target}, rng)
            if replacement is None:
                skipped += 1
                continue
            result[position] = replacement
            substitutions += 1

    _copy_shared_files(clean.root, destination)
    save_pickle((noisy, targets), destination / "test.txt")
    summary = dict(clean.summary)
    summary["protocol_version"] = f"{clean.summary.get('protocol_version', 'clean')}_noise_v1"
    summary["noise"] = {
        "source_clean_checksums": clean.checksums,
        "corruption_type": corruption_type,
        "ratio": ratio,
        "seed": seed,
        "requested_clicks": requested,
        "selected_clicks": sum(map(len, selected.values())),
        "changed_clicks": deletions + substitutions,
        "deletions": deletions,
        "substitutions": substitutions,
        "skipped_operations": skipped,
        "targets_unchanged": True,
        "minimum_perturbed_prefix_length": 2,
        "length_one_clean_prefixes_unchanged": True,
    }
    with (destination / "preprocessing_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    loaded = load_processed_dataset(destination)
    if [target for _, target in loaded.test] != targets:
        raise ValueError(f"target invariance failed for {destination}")
    for clean_example, noisy_example in zip(clean.test, loaded.test, strict=True):
        clean_prefix, clean_target = clean_example
        noisy_prefix, noisy_target = noisy_example
        if clean_target != noisy_target:
            raise ValueError(f"target changed in {destination}")
        if len(clean_prefix) == 1 and noisy_prefix != clean_prefix:
            raise ValueError(f"length-one prefix changed in {destination}")
        if len(clean_prefix) >= 2 and len(noisy_prefix) < 2:
            raise ValueError(f"minimum prefix length violated in {destination}")
    return summary["noise"]


def generate_noisy_datasets(
    clean_dir: Path,
    output_dir: Path,
    *,
    ratios: tuple[float, ...] = DEFAULT_RATIOS,
    corruption_types: tuple[CorruptionType, ...] = DEFAULT_CORRUPTION_TYPES,
    seed: int = DEFAULT_PERTURBATION_SEED,
) -> dict[str, dict[str, object]]:
    """Create one contract-compliant dataset directory per noise condition."""
    clean = load_processed_dataset(clean_dir)
    manifest: dict[str, dict[str, object]] = {}
    for corruption_type in corruption_types:
        if corruption_type not in DEFAULT_CORRUPTION_TYPES:
            raise ValueError(f"unsupported corruption type: {corruption_type}")
        for ratio in ratios:
            if not 0.0 <= ratio <= 1.0:
                raise ValueError("noise ratios must lie in [0, 1]")
            variant = f"{corruption_type}_{ratio:.2f}"
            destination = output_dir / variant
            statistics = _noise_one_dataset(
                clean,
                destination,
                ratio=ratio,
                corruption_type=corruption_type,
                seed=seed,
            )
            manifest[variant] = {"path": str(destination.resolve()), **statistics}

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "noise_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


# Backward-compatible entry point with the old function name.
def generate(
    data_dir: Path,
    ratios: tuple[float, ...] = DEFAULT_RATIOS,
    corruption_types: tuple[CorruptionType, ...] = DEFAULT_CORRUPTION_TYPES,
    perturbation_seed: int = DEFAULT_PERTURBATION_SEED,
    vocab_size: int | None = None,
) -> dict[str, dict[str, object]]:
    if vocab_size is not None:
        clean = load_processed_dataset(data_dir)
        if vocab_size != clean.num_items:
            raise ValueError("vocab_size does not match item_mapping.pkl")
    return generate_noisy_datasets(
        data_dir,
        data_dir.parent / "noisy",
        ratios=ratios,
        corruption_types=corruption_types,
        seed=perturbation_seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-dir", "--data-dir", dest="clean_dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_PERTURBATION_SEED)
    parser.add_argument("--ratios", type=float, nargs="+", default=list(DEFAULT_RATIOS))
    parser.add_argument(
        "--corruption-types",
        nargs="+",
        choices=DEFAULT_CORRUPTION_TYPES,
        default=list(DEFAULT_CORRUPTION_TYPES),
    )
    args = parser.parse_args()
    generate_noisy_datasets(
        args.clean_dir,
        args.output_dir,
        ratios=tuple(args.ratios),
        corruption_types=tuple(args.corruption_types),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
