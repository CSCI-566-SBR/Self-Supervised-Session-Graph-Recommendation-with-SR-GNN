"""Audit frozen team data and generate target-blind, fixed training corruption.

This module does not import torch and never writes to the source dataset.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
import tempfile
import tomllib
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

from recsys.data.perturb import _select_clicks, sample_replacement
from recsys.data.processed import (
    REQUIRED_FILES, load_processed_dataset, load_processed_split, sha256_file,
)


@dataclass(frozen=True)
class NoiseConfig:
    kind: str = "deletion"
    ratio: float = 0.10
    fraction: float = 0.50
    seed: int = 2026

    def validate(self):
        if self.kind not in {"deletion", "substitution", "mixed"}:
            raise ValueError("Unsupported training corruption type")
        if not (math.isfinite(self.ratio) and 0 <= self.ratio <= 1):
            raise ValueError("ratio must be finite and in [0,1]")
        if not (math.isfinite(self.fraction) and 0 <= self.fraction <= 1):
            raise ValueError("fraction must be finite and in [0,1]")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("noise seed must be a nonnegative integer")


def normalize_checksums(checksums):
    """Manifests use filenames; the shared loader uses logical keys."""
    result = {}
    for key, filename in REQUIRED_FILES.items():
        if key in checksums:
            result[key] = checksums[key]
        elif filename in checksums:
            result[key] = checksums[filename]
        else:
            raise ValueError(f"Missing checksum for {filename}")
    return result


def find_noise_root(root):
    root = Path(root)
    choices = [root / name for name in ("noisy", "noise") if (root / name).is_dir()]
    if len(choices) != 1:
        raise ValueError("Expected exactly one noisy/ or noise/ directory")
    return choices[0]


def prepare_archive(archive, destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                path = PurePosixPath(member.filename.replace("\\", "/"))
                if (path.is_absolute() or ".." in path.parts or
                        any(":" in part for part in path.parts) or
                        ((member.external_attr >> 16) & 0o170000) == 0o120000):
                    raise ValueError(f"Unsafe archive member: {member.filename}")
            bundle.extractall(tmp)
        roots = [p.parent for p in Path(tmp).rglob("pipeline_manifest.json")
                 if (p.parent / "clean").is_dir()]
        if len(roots) != 1:
            raise ValueError("Archive must contain one pipeline_manifest.json beside clean/")
        find_noise_root(roots[0])
        shutil.copytree(roots[0], destination)
    return destination


def examples_hash(examples):
    digest = hashlib.sha256()
    for prefix, target in examples:
        digest.update(json.dumps([prefix, target], separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def make_training_examples(examples, num_items, noise):
    """Fixed N -> N mixture. Budget is global within the chosen training subset.

    Reuses the team's position-selection and replacement helpers. Target values
    are copied through but NEVER used to sample edits or substitute items.
    Single-click prefixes remain unchanged for all operators, like frozen tests.
    """
    noise.validate()
    if not examples:
        raise ValueError("Training examples must be nonempty")
    rng = random.Random(noise.seed)
    count = math.floor(len(examples) * noise.fraction + 0.5)
    indices = sorted(rng.sample(range(len(examples)), count))
    prefixes = [list(examples[index][0]) for index in indices]
    selected, requested = _select_clicks(prefixes, noise.ratio, rng, noise.kind)
    output = list(examples)
    deletions = substitutions = skipped = 0
    affected = 0
    for local_index, positions in selected.items():
        result = prefixes[local_index][:]
        changed = 0
        for position in sorted(positions, reverse=True):
            operation = noise.kind
            if operation == "mixed":
                operation = "deletion" if rng.random() < .5 else "substitution"
            if operation == "deletion" and len(result) > 2:
                del result[position]
                deletions += 1
                changed += 1
            elif operation == "deletion" and noise.kind == "deletion":
                skipped += 1
            else:
                alternative = sample_replacement(num_items, {result[position]}, rng)
                if alternative is None:
                    skipped += 1
                else:
                    result[position] = alternative
                    substitutions += 1
                    changed += 1
        original_index = indices[local_index]
        output[original_index] = (tuple(result), examples[original_index][1])
        affected += int(changed > 0)
    total = sum(len(prefix) for prefix, _ in examples)
    subset_total = sum(map(len, prefixes))
    return tuple(output), {
        **asdict(noise), "examples": len(examples), "selected_examples": count,
        "actually_changed_examples": affected,
        "requested_clicks": requested,
        "selected_clicks": sum(map(len, selected.values())),
        "deletions": deletions, "substitutions": substitutions,
        "skipped_operations": skipped,
        "changed_clicks": deletions + substitutions,
        "total_original_clicks": total,
        "selected_original_clicks": subset_total,
        "realized_fraction_all_clicks": (deletions + substitutions) / max(1, total),
        "realized_fraction_selected_clicks": (deletions + substitutions) / max(1, subset_total),
        "targets_used_for_sampling": False,
        "budget_rule": "team global round(ratio * selected-prefix clicks), capped by eligible positions",
        "minimum_retained_length": 2, "singletons_unchanged": True,
        "last_item_protected": False,
        "examples_sha256": examples_hash(output),
    }


def load_frozen_conditions(root, clean):
    """Validate each shared noisy TEST. Do not regenerate any evaluation noise."""
    conditions, provenance = {}, {}
    noisy_root = find_noise_root(root)
    for variant in sorted(p for p in noisy_root.iterdir() if p.is_dir()):
        summary = json.loads((variant / "preprocessing_summary.json").read_text())
        noise = summary.get("noise", {})
        if normalize_checksums(noise["source_clean_checksums"]) != clean.checksums:
            raise ValueError(f"{variant.name}: clean source checksums mismatch")
        for key in ("train", "validation", "item_mapping"):
            if sha256_file(variant / REQUIRED_FILES[key]) != clean.checksums[key]:
                raise ValueError(f"{variant.name}: {key} differs from clean artifact")
        rows = load_processed_split(variant / "test.txt", clean.num_items)
        if len(rows) != len(clean.test):
            raise ValueError(f"{variant.name}: test example count mismatch")
        for (prefix, target), (noisy_prefix, noisy_target) in zip(clean.test, rows, strict=True):
            if target != noisy_target:
                raise ValueError(f"{variant.name}: held-out target changed")
            if len(prefix) == 1 and prefix != noisy_prefix:
                raise ValueError(f"{variant.name}: singleton changed")
            if len(prefix) >= 2 and len(noisy_prefix) < 2:
                raise ValueError(f"{variant.name}: short corrupted prefix")
            if noise["corruption_type"] == "substitution" and len(prefix) != len(noisy_prefix):
                raise ValueError(f"{variant.name}: substitution changed length")
            if noise["corruption_type"] == "deletion":
                iterator = iter(prefix)
                if not all(any(value == item for value in iterator) for item in noisy_prefix):
                    raise ValueError(f"{variant.name}: deletion is not a subsequence")
        conditions[variant.name] = rows
        provenance[variant.name] = {
            "test_sha256": sha256_file(variant / "test.txt"),
            "summary_sha256": sha256_file(variant / "preprocessing_summary.json"),
            "examples": len(rows), "noise": noise,
            "changed_examples": sum(p != q for (p, _), (q, _) in zip(clean.test, rows)),
        }
    expected = {f"{kind}_{ratio:.2f}" for kind in ("deletion", "substitution", "mixed")
                for ratio in (.05, .10, .20)}
    if set(conditions) != expected:
        raise ValueError(f"Expected 9 frozen conditions; got {sorted(conditions)}")
    return conditions, provenance


def audit(root, baseline_config):
    root = Path(root)
    clean = load_processed_dataset(root / "clean")
    manifest = json.loads((root / "pipeline_manifest.json").read_text())
    if normalize_checksums(manifest["clean"]["checksums"]) != clean.checksums:
        raise ValueError("Clean file bytes do not match pipeline manifest")
    with Path(baseline_config).open("rb") as stream:
        expected = tomllib.load(stream)
    if expected["data"]["expected_checksums"] != clean.checksums:
        raise ValueError("Data differs from baseline's pinned artifact")
    if clean.summary.get("protocol_version") != expected["data"]["expected_protocol_version"]:
        raise ValueError("Protocol differs from baseline")
    conditions, provenance = load_frozen_conditions(root, clean)
    return {
        "status": "passed", "dataset": clean.summary.get("dataset"),
        "num_items": clean.num_items,
        "examples": {key: len(getattr(clean, key)) for key in ("train", "validation", "test")},
        "sessions_reported_by_preprocessor": clean.summary.get("counts"),
        "clean_checksums": clean.checksums,
        "matches_baseline_pinned_artifact": True,
        "artifact_repository": expected["data"]["artifact_repository"],
        "artifact_revision_from_matching_baseline": expected["data"]["artifact_revision"],
        "noise_folder_name": find_noise_root(root).name,
        "noise_variants": provenance,
        "filtering_protocol": clean.summary.get("filtering_protocol"),
        "scope_note": "Byte hashes and model-facing invariants verified. Raw session timestamps/IDs are absent, so chronological session independence cannot be re-proven from these artifacts.",
        "evaluation_protocol_note": "Use supplied tests unchanged. Their replacement helper excludes the held-out target ID. Model2 training substitutions are target-blind; targets do not select replacements.",
    }
