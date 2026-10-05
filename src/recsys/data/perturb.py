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
DEFAULT_CORRUPTION_TYPES = ("deletion","substitution","mixed")
DEFAULT_PERTURBATION_SEED = 2026    

def sample_replacement(vocab_size: int,forbidden_items: set[int],rng: random.Random,) -> int | None:
    forbidden = sorted(
        item
        for item in set(forbidden_items)
        if 1 <= item <= vocab_size
    )

    allowed_count = vocab_size - len(forbidden)

    if allowed_count <= 0:
        return None

    replacement = rng.randint(1, allowed_count)

    for forbidden_item in forbidden:
        if replacement >= forbidden_item:
            replacement += 1

    return replacement

def perturb_sequence(sequence: list[int],ratio: float,vocab_size: int,rng: random.Random,target: int | None = None,) -> tuple[list[int], int]:
    if not sequence or ratio <= 0 or len(sequence) == 1:
        return list(sequence), 0

    result = list(sequence)
    budget = min(
        len(result),
        max(1, round(len(result) * ratio)),
    )
    changed = 0

    for _ in range(budget):
        if len(result) > 1 and rng.random() < 0.5:
            del result[rng.randrange(len(result))]
            changed += 1
        else:
            position = rng.randrange(len(result))
            original = result[position]

            forbidden_items = {original}

            if target is not None:
                forbidden_items.add(target)

            replacement = sample_replacement(vocab_size,forbidden_items,rng,)

            if replacement is None:
                continue

            result[position] = replacement
            changed += 1

    return result, changed


# def generate(
#     data_dir: Path,
#     ratios: tuple[float, ...] = DEFAULT_RATIOS,
#     seed: int = 2026,
#     vocab_size: int | None = None,
# ) -> dict[float, Path]:
#     sequences, targets = load_pickle(data_dir / "test.txt")
#     if vocab_size is None:
#         mapping_path = data_dir / "item_mapping.pkl"
#         vocab_size = len(load_pickle(mapping_path)) if mapping_path.exists() else max(
#             max(targets, default=0), max((max(sequence) for sequence in sequences if sequence), default=0)
#         )
#     if vocab_size < 1:
#         raise ValueError("could not infer a non-empty item vocabulary")

#     eligible_clicks = [
#         (sequence_index, position)
#         for sequence_index, sequence in enumerate(sequences)
#         if len(sequence) >= 2
#         for position in range(len(sequence))
#     ]
#     total_clicks = sum(map(len, sequences))
#     outputs: dict[float, Path] = {}
#     target_bytes = pickle.dumps(list(targets), protocol=pickle.HIGHEST_PROTOCOL)
#     report = {
#         "seed": seed,
#         "clean_examples": len(sequences),
#         "clean_input_clicks": total_clicks,
#         "length_one_prefixes_preserved": sum(len(sequence) == 1 for sequence in sequences),
#         "vocab_size": vocab_size,
#         "target_sha256": hashlib.sha256(target_bytes).hexdigest(),
#         "ratios": {},
#     }

#     for ratio in ratios:
#         if not 0.0 <= ratio <= 1.0:
#             raise ValueError("noise ratios must lie in [0, 1]")
#         rng = random.Random(seed)
#         ranked = list(eligible_clicks)
#         rng.shuffle(ranked)
#         budget = min(len(ranked), round(total_clicks * ratio))
#         selected: dict[int, list[int]] = {}
#         for sequence_index, position in ranked[:budget]:
#             selected.setdefault(sequence_index, []).append(position)

#         noisy = [list(sequence) for sequence in sequences]
#         changed = deletions = substitutions = 0
#         skipped_substitutions = 0   
#         for sequence_index, positions in selected.items():
#             result = noisy[sequence_index]
#             target = targets[sequence_index]
#             for position in sorted(positions, reverse=True):
#                 if len(result) > 2 and rng.random() < 0.5:
#                     del result[position]
#                     changed += 1
#                     deletions += 1
#                 else:
#                     original = result[position]
#                     replacement = sample_replacement(vocab_size,forbidden_items={original, target},rng=rng,)
#                     if replacement is None:
#                         skipped_substitutions += 1
#                         continue
#                     result[position] = replacement
#                     changed += 1
#                     substitutions += 1

#         destination = data_dir / f"test_noise_{ratio:.2f}.txt"
#         save_pickle((noisy, list(targets)), destination)
#         outputs[ratio] = destination
#         report["ratios"][f"{ratio:.2f}"] = {
#             "requested_clicks": budget,
#             "changed_clicks": changed,
#             "deletions": deletions,
#             "substitutions": substitutions,
#             "skipped_substitutions_no_candidate": skipped_substitutions,
#             "output_input_clicks": sum(map(len, noisy)),
#             "file": destination.name,
#             "file_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
#         }
#         LOGGER.info(
#             "wrote %s examples=%d changed_clicks=%d seed=%d targets_unchanged=true",
#             destination, len(noisy), changed, seed,
#         )

#     (data_dir / "perturbation_report.json").write_text(
#         json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
#     )
#     return outputs
def generate(data_dir: Path,ratios: tuple[float, ...] = DEFAULT_RATIOS,corruption_types: tuple[str, ...] = DEFAULT_CORRUPTION_TYPES, perturbation_seed: int = DEFAULT_PERTURBATION_SEED, vocab_size: int | None = None,) -> dict[str, dict[float, Path]]:
    sequences, targets = load_pickle(data_dir / "test.txt")

    if len(sequences) != len(targets):
        raise ValueError(
            "test sequences and targets must have the same length"
        )

    if vocab_size is None:
        mapping_path = data_dir / "item_mapping.pkl"

        if mapping_path.exists():
            vocab_size = len(load_pickle(mapping_path))
        else:
            vocab_size = max(
                max(targets, default=0),
                max(
                    (
                        max(sequence)
                        for sequence in sequences
                        if sequence
                    ),
                    default=0,
                ),
            )

    if vocab_size < 1:
        raise ValueError(
            "could not infer a non-empty item vocabulary"
        )

    valid_corruption_types = set(DEFAULT_CORRUPTION_TYPES)

    for corruption_type in corruption_types:
        if corruption_type not in valid_corruption_types:
            raise ValueError(
                f"unsupported corruption type: {corruption_type}"
            )

    for ratio in ratios:
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(
                "noise ratios must lie in [0, 1]"
            )

    # Length-one prefixes are deliberately left unchanged.
    eligible_clicks = [
        (sequence_index, position)
        for sequence_index, sequence in enumerate(sequences)
        if len(sequence) >= 2
        for position in range(len(sequence))
    ]

    total_clicks = sum(map(len, sequences))

    outputs: dict[str, dict[float, Path]] = {
        corruption_type: {}
        for corruption_type in corruption_types
    }

    target_bytes = pickle.dumps(
        list(targets),
        protocol=pickle.HIGHEST_PROTOCOL,
    )

    report = {
        "perturbation_seed": perturbation_seed,
        "clean_examples": len(sequences),
        "clean_input_clicks": total_clicks,
        "length_one_prefixes_preserved": sum(
            len(sequence) == 1
            for sequence in sequences
        ),
        "vocab_size": vocab_size,
        "target_sha256": hashlib.sha256(
            target_bytes
        ).hexdigest(),
        "corruption_types": {},
    }

    for corruption_type in corruption_types:
        report["corruption_types"][corruption_type] = {}

        for ratio in ratios:
            # Resetting with the same seed ensures that corruption
            # types at the same ratio begin with the same click order.
            rng = random.Random(perturbation_seed)

            ranked = list(eligible_clicks)
            rng.shuffle(ranked)

            budget = min(
                len(ranked),
                round(total_clicks * ratio),
            )

            selected: dict[int, list[int]] = {}

            for sequence_index, position in ranked[:budget]:
                selected.setdefault(
                    sequence_index,
                    [],
                ).append(position)

            noisy = [
                list(sequence)
                for sequence in sequences
            ]

            changed = 0
            deletions = 0
            substitutions = 0
            skipped_deletions = 0
            skipped_substitutions = 0

            for sequence_index, positions in selected.items():
                result = noisy[sequence_index]
                target = targets[sequence_index]

                # Descending order keeps the remaining indices valid
                # when earlier operations delete sequence elements.
                for position in sorted(
                    positions,
                    reverse=True,
                ):
                    if corruption_type == "deletion":
                        operation = "deletion"
                    elif corruption_type == "substitution":
                        operation = "substitution"
                    else:
                        operation = (
                            "deletion"
                            if rng.random() < 0.5
                            else "substitution"
                        )

                    if operation == "deletion":
                        # Preserve at least one observed input item.
                        if len(result) <= 1:
                            skipped_deletions += 1
                            continue

                        del result[position]
                        changed += 1
                        deletions += 1

                    else:
                        original = result[position]

                        replacement = sample_replacement(
                            vocab_size,
                            forbidden_items={
                                original,
                                target,
                            },
                            rng=rng,
                        )

                        if replacement is None:
                            skipped_substitutions += 1
                            continue

                        result[position] = replacement
                        changed += 1
                        substitutions += 1

            destination = (
                data_dir
                / (
                    f"test_noise_{corruption_type}_"
                    f"{ratio:.2f}.txt"
                )
            )

            save_pickle(
                (noisy, list(targets)),
                destination,
            )

            outputs[corruption_type][ratio] = destination

            report["corruption_types"][corruption_type][
                f"{ratio:.2f}"
            ] = {
                "requested_clicks": budget,
                "changed_clicks": changed,
                "deletions": deletions,
                "substitutions": substitutions,
                "skipped_deletions_minimum_length": (
                    skipped_deletions
                ),
                "skipped_substitutions_no_candidate": (
                    skipped_substitutions
                ),
                "output_input_clicks": sum(
                    map(len, noisy)
                ),
                "file": destination.name,
                "file_sha256": hashlib.sha256(
                    destination.read_bytes()
                ).hexdigest(),
            }

            LOGGER.info(
                "wrote %s corruption_type=%s examples=%d "
                "changed_clicks=%d perturbation_seed=%d "
                "targets_unchanged=true",
                destination,
                corruption_type,
                len(noisy),
                changed,
                perturbation_seed,
            )

    (data_dir / "perturbation_report.json").write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return outputs

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--perturbation-seed","--seed", dest="perturbation_seed", default=DEFAULT_PERTURBATION_SEED, type=int, help="random seed used for offline test corruption",)
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--corruption-types",nargs="+",choices=DEFAULT_CORRUPTION_TYPES,default=list(DEFAULT_CORRUPTION_TYPES),help=("test corruption types to generate; default: deletion substitution mixed"),)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    generate(data_dir=args.data_dir, corruption_types=tuple(args.corruption_types),perturbation_seed=args.perturbation_seed, vocab_size=args.vocab_size)


if __name__ == "__main__":
    main()
