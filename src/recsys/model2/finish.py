"""Full-metric evaluation of old checkpoints and latest-branch Model2 training."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from recsys.data.processed import load_processed_dataset
from recsys.data.session_graph import collate_session_graphs
from recsys.models.baselines import SRGNN
from recsys.training.baseline import evaluate_ranking
from recsys.utils import format_results_dict, make_data_loader, resolve_device
from recsys.model2.data import load_frozen_conditions
from model2_io import CONDITIONS, read_json, sha, write_json


@torch.no_grad()
def evaluate_full(run, data, dataset, config_file, device_arg="cuda", workers=0):
    run, data = Path(run), Path(data)
    import tomllib
    with Path(config_file).open("rb") as f:
        baseline = tomllib.load(f)
    destination = run / "robustness_results.json"
    meta = read_json(run / "run.json")
    checkpoint = run / "best_model.pt"
    checkpoint_sha = sha(checkpoint)
    if destination.exists():
        saved = read_json(destination)
        if saved["clean"]["checkpoint_sha256"] != checkpoint_sha:
            raise ValueError("Completed results refer to another checkpoint")
        if not (run / "results.json").is_file():
            raise ValueError("Incomplete result pair; rename robustness_results.json and rerun evaluation")
        print(f"Already evaluated; keeping {destination}", flush=True)
        return saved
    clean = load_processed_dataset(data / "clean")
    if clean.checksums != meta["identity"]["clean_checksums"]:
        raise ValueError("Checkpoint metadata does not match the clean data")
    device = resolve_device(device_arg)
    model = SRGNN(**meta["identity"]["model"]).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
    conditions, provenance = load_frozen_conditions(data, clean)
    rows_by_name = {"clean": clean.test, **conditions}
    batch_size = meta["identity"]["batch_size"]
    seed = meta["identity"]["training_seed"]
    params = {k: v for k, v in meta["identity"]["model"].items() if k != "num_items"}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    tick = time.perf_counter()
    validation = format_results_dict(evaluate_ranking(model, make_data_loader(
        clean.validation, batch_size, False, seed, collate_session_graphs,
        num_workers=workers, pin_memory=device.type == "cuda"), device, k=20, cutoffs=(10, 20)), 20)
    metrics = {}
    original = read_json(run / "evaluation.json") if (run / "evaluation.json").exists() else None
    for name in ["clean", *CONDITIONS]:
        loader = make_data_loader(rows_by_name[name], batch_size, False, seed,
                                  collate_session_graphs, num_workers=workers,
                                  pin_memory=device.type == "cuda")
        start = time.perf_counter()
        result = evaluate_ranking(model, loader, device, k=20, cutoffs=(10, 20))
        values = format_results_dict(result, 20)
        values.update({"eval_time_seconds": time.perf_counter() - start,
                       "corruption_type": name.split("_")[0],
                       "corruption_rate": 0.0 if name == "clean" else float(name.split("_")[1]),
                       "test_checksum_sha256": clean.checksums["test"] if name == "clean" else provenance[name]["test_sha256"]})
        if original:
            values["original_metric_delta"] = {m: values[m] - original["metrics"][name][m]
                                               for m in ("recall@20", "mrr@20")}
            if max(map(abs, values["original_metric_delta"].values())) > 0.0001:
                raise ValueError(f"{name}: old checkpoint scores changed by more than 0.01 percentage points")
        metrics[name] = values
        print(f"{dataset} | {run.name} | {name}: Recall@20={values['recall@20']:.6f}, "
              f"MRR@20={values['mrr@20']:.6f}, NDCG@20={values['ndcg@20']:.6f}", flush=True)
        # Persist progress without presenting partial runs as completed results.
        write_json(run / "evaluation_progress.json", metrics)
    seconds = time.perf_counter() - tick
    metrics["clean"].update({"checkpoint_sha256": checkpoint_sha, "model_type": "srgnn",
        "model_params": params, "artifact_repository": meta["artifact_repository"],
        "artifact_revision": meta["artifact_revision_from_matching_baseline"],
        "device": str(device), "source_branch_commit": "c25d6624d54edfda1c255e888227277a9fd2251b"})
    old_training = meta["identity"]["training_noise"]
    training_kind = "clean" if old_training["fraction"] == 0 else "fixed_deletion"
    history = read_json(run / "history.json")
    config = {"model_type": "srgnn", "model_params": params, "seed": seed,
              "expected_protocol_version": clean.summary["protocol_version"],
              "expected_checksums": clean.checksums,
              "processed_data_dir": str(data / "clean"), "output_dir": str(run),
              "batch_size": batch_size, "training_noise": old_training}
    results = {"experiment": run.name, "dataset": dataset, "model_type": "srgnn",
        "model_params": params, "training_kind": training_kind, "seed": seed,
        "best_epoch": meta["best_epoch"], "checkpoint_metric": "validation_mrr@20",
        "best_validation": validation, "test": metrics["clean"], "history": history,
        "parameter_count": meta["parameter_count"],
        "runtime_seconds": meta["train_validation_seconds"] + seconds,
        "training_validation_seconds": meta["train_validation_seconds"],
        "full_evaluation_seconds": seconds,
        "runtime_definition": "Original train+validation wall time plus this full-metric validation and ten-condition evaluation; differs from upstream baseline runtime.",
        "peak_gpu_memory_bytes": meta.get("peak_gpu_allocated_bytes"),
        "evaluation_peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "peak_memory_definition": "Training/validation peak PyTorch allocated bytes; not reserved memory or nvidia-smi device usage.",
        "hardware": meta.get("hardware", "NVIDIA GeForce RTX 5060 Laptop GPU (user-reported original run)"),
        "software": meta["software"], "evaluation_software": {"torch": str(torch.__version__)},
        "dataset_artifact": {"repository": meta["artifact_repository"],
            "revision": meta["artifact_revision_from_matching_baseline"],
            "protocol_version": clean.summary["protocol_version"], "checksums": clean.checksums},
        "config": config, "checkpoint_sha256": checkpoint_sha,
        "source_branch_commit": "c25d6624d54edfda1c255e888227277a9fd2251b",
        "notes": ["Legacy imported weights retain original initialization and optimizer history.",
                  "New RetailRocket weights use current-branch initialization and CUDA fused Adam.",
                  "Historical validation metrics absent from original logs are not fabricated.",
                  "No contrastive SSL objective; this is a supervised fixed-noise ablation."]}
    write_json(run / "results.json", results)
    write_json(destination, metrics)
    return metrics
