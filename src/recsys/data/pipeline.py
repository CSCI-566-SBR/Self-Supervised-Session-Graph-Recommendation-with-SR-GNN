"""Run clean preprocessing and offline noise generation as one pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .perturb import (
    DEFAULT_CORRUPTION_TYPES,
    DEFAULT_PERTURBATION_SEED,
    DEFAULT_RATIOS,
    generate_noisy_datasets,
)
from .preprocess import DEFAULT_SPLIT_DAYS, preprocess_clean
from .processed import load_processed_dataset


def run_pipeline(
    dataset: str,
    input_path: Path,
    output_root: Path,
    *,
    min_item_frequency: int = 5,
    min_session_length: int = 2,
    validation_days: int | None = None,
    test_days: int | None = None,
    ratios: tuple[float, ...] = DEFAULT_RATIOS,
    corruption_types: tuple[str, ...] = DEFAULT_CORRUPTION_TYPES,
    seed: int = DEFAULT_PERTURBATION_SEED,
) -> dict[str, Any]:
    """Generate clean/ and noisy/<type_ratio>/ model-ready directories."""
    clean_dir = output_root / "clean"
    noisy_dir = output_root / "noisy"
    clean_summary = preprocess_clean(
        dataset,
        input_path,
        clean_dir,
        min_item_frequency=min_item_frequency,
        min_session_length=min_session_length,
        validation_days=validation_days,
        test_days=test_days,
    )
    noise_manifest = generate_noisy_datasets(
        clean_dir,
        noisy_dir,
        ratios=ratios,
        corruption_types=corruption_types,
        seed=seed,
    )
    clean = load_processed_dataset(clean_dir)
    manifest = {
        "dataset": dataset,
        "clean": {
            "path": str(clean_dir.resolve()),
            "checksums": clean.checksums,
            "num_items": clean.num_items,
            "counts": clean_summary["counts"],
        },
        "noisy": noise_manifest,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "pipeline_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(DEFAULT_SPLIT_DAYS))
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--min-item-frequency", type=int, default=5)
    parser.add_argument("--min-session-length", type=int, default=2)
    parser.add_argument("--validation-days", type=int)
    parser.add_argument("--test-days", type=int)
    parser.add_argument("--ratios", type=float, nargs="+", default=list(DEFAULT_RATIOS))
    parser.add_argument(
        "--corruption-types",
        nargs="+",
        choices=DEFAULT_CORRUPTION_TYPES,
        default=list(DEFAULT_CORRUPTION_TYPES),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_PERTURBATION_SEED)
    args = parser.parse_args()
    if not args.input.is_file():
        raise SystemExit(f"raw file not found: {args.input}")
    manifest = run_pipeline(
        args.dataset,
        args.input,
        args.output_root,
        min_item_frequency=args.min_item_frequency,
        min_session_length=args.min_session_length,
        validation_days=args.validation_days,
        test_days=args.test_days,
        ratios=tuple(args.ratios),
        corruption_types=tuple(args.corruption_types),
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
