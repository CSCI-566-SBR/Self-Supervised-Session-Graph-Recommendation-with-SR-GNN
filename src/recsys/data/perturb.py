"""Create deterministic target-preserving noisy evaluation prefixes."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pickle
import random
from pathlib import Path

from .artifacts import load_pickle, save_pickle

LOGGER = logging.getLogger("recsys.data.perturb")
DEFAULT_RATIOS = (0.05, 0.10, 0.20)


def perturb_sequence(
    sequence: list[int], ratio: float, vocab_size: int, rng: random.Random
) -> tuple[list[int], int]:
    """Perturb one sequence; retained for focused unit testing and exploration."""
    if not sequence or ratio <= 0 or len(sequence) == 1:
        return list(sequence), 0
    result = list(sequence)
    budget = min(len(result), max(1, round(len(result) * ratio)))
    changed = 0
    for _ in range(budget):
        if len(result) > 2 and rng.random() < 0.5:
            del result[rng.randrange(len(result))]
            changed += 1
        else:
            position = rng.randrange(len(result))
            original = result[position]
            replacement = rng.randint(1, vocab_size)
            if vocab_size > 1:
                while replacement == original:
                    replacement = rng.randint(1, vocab_size)
            result[position] = replacement
            changed += int(replacement != original)
    return result, changed


def generate(
    data_dir: Path,
    ratios: tuple[float, ...] = DEFAULT_RATIOS,
    seed: int = 2026,
    vocab_size: int | None = None,
) -> dict[float, Path]:
    sequences, targets = load_pickle(data_dir / "test.txt")
    if vocab_size is None:
        mapping_path = data_dir / "item_mapping.pkl"
        vocab_size = len(load_pickle(mapping_path)) if mapping_path.exists() else max(
            max(targets, default=0), max((max(sequence) for sequence in sequences if sequence), default=0)
        )
    if vocab_size < 1:
        raise ValueError("could not infer a non-empty item vocabulary")

    eligible_clicks = [
        (sequence_index, position)
        for sequence_index, sequence in enumerate(sequences)
        if len(sequence) >= 2
        for position in range(len(sequence))
    ]
    total_clicks = sum(map(len, sequences))
    outputs: dict[float, Path] = {}
    target_bytes = pickle.dumps(list(targets), protocol=pickle.HIGHEST_PROTOCOL)
    report = {
        "seed": seed,
        "clean_examples": len(sequences),
        "clean_input_clicks": total_clicks,
        "length_one_prefixes_preserved": sum(len(sequence) == 1 for sequence in sequences),
        "vocab_size": vocab_size,
        "target_sha256": hashlib.sha256(target_bytes).hexdigest(),
        "ratios": {},
    }

    for ratio in ratios:
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("noise ratios must lie in [0, 1]")
        rng = random.Random(seed)
        ranked = list(eligible_clicks)
        rng.shuffle(ranked)
        budget = min(len(ranked), round(total_clicks * ratio))
        selected: dict[int, list[int]] = {}
        for sequence_index, position in ranked[:budget]:
            selected.setdefault(sequence_index, []).append(position)

        noisy = [list(sequence) for sequence in sequences]
        changed = deletions = substitutions = 0
        for sequence_index, positions in selected.items():
            result = noisy[sequence_index]
            for position in sorted(positions, reverse=True):
                if len(result) > 2 and rng.random() < 0.5:
                    del result[position]
                    changed += 1
                    deletions += 1
                else:
                    original = result[position]
                    replacement = rng.randint(1, vocab_size)
                    if vocab_size > 1:
                        while replacement == original:
                            replacement = rng.randint(1, vocab_size)
                    result[position] = replacement
                    changed += int(replacement != original)
                    substitutions += int(replacement != original)

        destination = data_dir / f"test_noise_{ratio:.2f}.txt"
        save_pickle((noisy, list(targets)), destination)
        outputs[ratio] = destination
        report["ratios"][f"{ratio:.2f}"] = {
            "requested_clicks": budget,
            "changed_clicks": changed,
            "deletions": deletions,
            "substitutions": substitutions,
            "output_input_clicks": sum(map(len, noisy)),
            "file": destination.name,
            "file_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        }
        LOGGER.info(
            "wrote %s examples=%d changed_clicks=%d seed=%d targets_unchanged=true",
            destination, len(noisy), changed, seed,
        )

    (data_dir / "perturbation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--seed", default=2026, type=int)
    parser.add_argument("--vocab-size", type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    generate(args.data_dir, seed=args.seed, vocab_size=args.vocab_size)


if __name__ == "__main__":
    main()
