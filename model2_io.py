"""Standard-library preparation and reporting; source datasets are never edited."""
from __future__ import annotations

import csv
import hashlib
import json
import pickle
import shutil
import tempfile
import tomllib
import zipfile
from pathlib import Path, PurePosixPath

FILES = {"train": "train.txt", "validation": "validation.txt", "test": "test.txt",
         "item_mapping": "item_mapping.pkl", "summary": "preprocessing_summary.json"}
CONDITIONS = [f"{kind}_{ratio:.2f}" for kind in ("deletion", "mixed", "substitution")
              for ratio in (.05, .10, .20)]
METRICS = [f"{metric}@{k}" for metric in ("recall", "mrr", "ndcg") for k in (10, 20)]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_extract(archive, target):
    """Extract trusted project ZIPs while rejecting traversal and symlinks."""
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            p = PurePosixPath(member.filename.replace("\\", "/"))
            if (p.is_absolute() or ".." in p.parts or any(":" in x for x in p.parts)
                    or ((member.external_attr >> 16) & 0o170000) == 0o120000):
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
        z.extractall(target)


def checksums(root):
    return {key: sha(Path(root) / filename) for key, filename in FILES.items()}


def normalized(hashes):
    return {key: hashes[key] if key in hashes else hashes[filename]
            for key, filename in FILES.items()}


def noise_root(root):
    choices = [Path(root) / x for x in ("noise", "noisy") if (Path(root) / x).is_dir()]
    if len(choices) != 1:
        raise ValueError("Expected one noise/ or noisy/ directory")
    return choices[0]


def prepare_data(archive, destination):
    """Repair only the staged copy, with the original manifest as authority."""
    destination = Path(destination)
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temp:
        safe_extract(archive, temp)
        roots = [p.parent for p in Path(temp).rglob("pipeline_manifest.json")
                 if (p.parent / "clean").is_dir()]
        if len(roots) != 1:
            raise ValueError("Dataset ZIP needs one pipeline manifest beside clean/")
        root = roots[0]
        expected = normalized(read_json(root / "pipeline_manifest.json")["clean"]["checksums"])
        summary = root / "clean/preprocessing_summary.json"
        if not summary.exists():
            misplaced = root / "preprocessing_summary.json"
            if not misplaced.is_file() or sha(misplaced) != expected["summary"]:
                raise ValueError("Missing clean summary; no hash-matching root summary exists")
            shutil.copy2(misplaced, summary)
        if checksums(root / "clean") != expected:
            raise ValueError("Clean dataset differs from its manifest")
        source_noise = noise_root(root)
        if source_noise.name == "noise":
            source_noise.rename(root / "noisy")
        shutil.copytree(root, destination)
    return destination


def audit_data(root, config_file):
    root = Path(root)
    with Path(config_file).open("rb") as f:
        config = tomllib.load(f)
    actual = checksums(root / "clean")
    expected = config["data"]["expected_checksums"]
    if actual != expected or actual != normalized(read_json(root / "pipeline_manifest.json")["clean"]["checksums"]):
        raise ValueError("Data hashes differ from the pinned baseline or pipeline manifest")
    summary = read_json(root / "clean/preprocessing_summary.json")
    if summary["protocol_version"] != config["data"]["expected_protocol_version"]:
        raise ValueError("Dataset protocol mismatch")
    # These pickles are supplied by the user's trusted team, not downloaded implicitly.
    with (root / "clean/item_mapping.pkl").open("rb") as f:
        mapping = pickle.load(f)
    n = len(mapping)
    if sorted(mapping.values()) != list(range(1, n + 1)) or n != summary["num_items"]:
        raise ValueError("Invalid item mapping")
    counts = {}
    for split in ("train", "validation", "test"):
        with (root / "clean" / FILES[split]).open("rb") as f:
            prefixes, targets = pickle.load(f)
        if not prefixes or len(prefixes) != len(targets):
            raise ValueError("Invalid split lengths")
        if any(not p or any(type(i) is not int or not 1 <= i <= n for i in p)
               for p in prefixes) or any(type(t) is not int or not 1 <= t <= n for t in targets):
            raise ValueError("Invalid item IDs")
        counts[split] = len(targets)
        if split == "test":
            clean_prefixes, clean_targets = prefixes, targets
    variants = noise_root(root)
    names = sorted(p.name for p in variants.iterdir() if p.is_dir())
    if names != sorted(CONDITIONS):
        raise ValueError("Exactly nine frozen test conditions are required")
    noisy_checksums = {}
    for name in CONDITIONS:
        d = variants / name
        noise = read_json(d / FILES["summary"])["noise"]
        if normalized(noise["source_clean_checksums"]) != actual:
            raise ValueError(f"{name}: wrong source dataset")
        for key in ("train", "validation", "item_mapping"):
            if sha(d / FILES[key]) != actual[key]:
                raise ValueError(f"{name}: shared {key} differs from clean")
        with (d / "test.txt").open("rb") as f:
            ps, ts = pickle.load(f)
        if ts != clean_targets or len(ps) != len(clean_prefixes):
            raise ValueError(f"{name}: target or example count changed")
        for p, q in zip(clean_prefixes, ps, strict=True):
            if (len(p) == 1 and p != q) or (len(p) >= 2 and len(q) < 2):
                raise ValueError(f"{name}: invalid short prefix")
            if any(type(i) is not int or not 1 <= i <= n for i in q):
                raise ValueError(f"{name}: invalid noisy item ID")
        noisy_checksums[name] = sha(d / "test.txt")
    return {"status": "passed", "dataset": summary["dataset"], "num_items": n,
            "examples": counts, "clean_checksums": actual, "noisy_test_checksums": noisy_checksums,
            "artifact_repository": config["data"]["artifact_repository"],
            "artifact_revision": config["data"]["artifact_revision"],
            "note": "Model-facing identities verified; raw session chronology cannot be re-proven here."}


def import_runs(archive, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination) as temp:
        safe_extract(archive, temp)
        for name in ("clean_seed2028", "model2_deletion_seed2028"):
            sources = [p.parent for p in Path(temp).rglob("run.json") if p.parent.name == name]
            if len(sources) != 1:
                raise ValueError(f"Saved outputs ZIP must contain one {name}")
            source, target = sources[0], destination / name
            if target.exists():
                if sha(source / "best_model.pt") != sha(target / "best_model.pt"):
                    raise ValueError(f"Refusing to replace different checkpoint: {target}")
                continue
            target.mkdir()
            for f in ("best_model.pt", "run.json", "history.json", "evaluation.json"):
                shutil.copy2(source / f, target / f)
    return destination


def report(root, destination):
    root, destination = Path(root), Path(destination)
    rows = []
    for f in sorted(root.rglob("robustness_results.json")):
        meta_file = f.parent / "results.json"
        if not meta_file.is_file():
            continue
        meta, metrics = read_json(meta_file), read_json(f)
        for condition, values in metrics.items():
            rows.append({"dataset": meta.get("dataset"), "run": str(f.parent.relative_to(root)),
                         "model_type": meta.get("model_type"), "seed": meta.get("seed"),
                         "training_kind": meta.get("training_kind", "clean"),
                         "condition": condition, **{m: values.get(m) for m in METRICS},
                         "loss": values.get("loss"), "examples": values.get("examples"),
                         "runtime_seconds": meta.get("runtime_seconds"),
                         "peak_gpu_memory_bytes": meta.get("peak_gpu_memory_bytes"),
                         "parameter_count": meta.get("parameter_count"),
                         "hardware": meta.get("hardware", "See original run provenance"),
                         "checkpoint_sha256": values.get("checkpoint_sha256", metrics.get("clean", {}).get("checkpoint_sha256")),
                         "test_sha256": values.get("test_checksum_sha256")})
    if not rows:
        raise ValueError("No completed robustness results found")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(destination.with_suffix(".json"), {"rows": rows,
        "notes": ["Metrics are fractions, not percentages.",
                  "Single seed; no statistical significance is claimed.",
                  "Hardware and runtime definitions differ; do not rank speed across machines."]})
    return destination
