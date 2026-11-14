"""Create comparable result tables without importing torch."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


def build_report(runs, output):
    output = Path(output)
    summary_path = output.with_suffix(".summary.json")
    if output.exists() or summary_path.exists():
        raise FileExistsError("Choose a new report output; previous results are never overwritten")
    rows, common_checksums, common_conditions, common_design = [], None, None, None
    for run in runs:
        result = json.loads((Path(run) / "evaluation.json").read_text())
        identity = result["training_identity"]
        design = {key: identity.get(key) for key in (
            "model", "batch_size", "optimizer", "scheduler", "patience", "k",
            "deterministic", "debug_limit")}
        fingerprints = {name: info["test_sha256"] for name, info in result["frozen_conditions"].items()}
        if common_checksums is None:
            common_checksums, common_conditions = identity["clean_checksums"], fingerprints
            common_design = design
        if identity["clean_checksums"] != common_checksums or fingerprints != common_conditions:
            raise ValueError("Cannot compare runs with different clean data or frozen test conditions")
        if design != common_design:
            raise ValueError("Cannot compare runs with different backbone/training settings")
        clean_recall = result["metrics"]["clean"]["recall@20"]
        noise = identity["training_noise"]
        for condition, metrics in result["metrics"].items():
            rows.append({"run": str(run), "training_seed": identity["training_seed"],
                         "training_kind": noise["kind"], "training_ratio": noise["ratio"],
                         "selected_training_fraction": noise["fraction"],
                         "actual_training_click_fraction": noise["realized_fraction_all_clicks"],
                         "condition": condition, "recall@20": metrics["recall@20"],
                         "mrr@20": metrics["mrr@20"], "examples": metrics["examples"],
                         "relative_recall_drop": (clean_recall - metrics["recall@20"]) / clean_recall
                         if clean_recall else None})
    if not rows:
        raise ValueError("At least one evaluated run is required")
    grouped = {}
    for row in rows:
        key = f"{row['training_kind']}:r{row['training_ratio']}:f{row['selected_training_fraction']}:{row['condition']}"
        grouped.setdefault(key, []).append(row)
    summary = {}
    for key, values in grouped.items():
        seeds = [row["training_seed"] for row in values]
        if len(seeds) != len(set(seeds)):
            raise ValueError("Duplicate training seed in one experiment group")
        summary[key] = {"n_seeds": len(values), "seeds": seeds}
        for metric in ("recall@20", "mrr@20"):
            scores = [row[metric] for row in values]
            summary[key][metric] = {"mean": statistics.mean(scores),
                                    "sample_std": statistics.stdev(scores) if len(scores) > 1 else None}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return output
