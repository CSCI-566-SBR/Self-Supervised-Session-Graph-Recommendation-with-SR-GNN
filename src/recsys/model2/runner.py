"""Training around the unchanged baseline; full-state resume and frozen tests."""
from __future__ import annotations

import json
import math
import platform
import random
import time
import tomllib
from pathlib import Path

import numpy as np
import torch

from recsys.data.processed import load_processed_dataset
from recsys.data.session_graph import collate_session_graphs
from recsys.models.baselines import SRGNN
from recsys.reproducibility import set_random_seed
from recsys.training.baseline import evaluate_ranking, train_epoch
from recsys.training.runner import _make_loader, _results_dict, resolve_device
from .data import NoiseConfig, audit, load_frozen_conditions, make_training_examples

ROOT = Path(__file__).resolve().parents[3]
BASELINE_CONFIG = ROOT / "configs/srgnn_diginetica_global_recursive.toml"


def _save_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _save_torch(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, tmp)
    tmp.replace(path)


def _rng_snapshot(generator):
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(), "torch": torch.get_rng_state(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), int(numpy_state[2]),
                  int(numpy_state[3]), float(numpy_state[4])],
        "loader": generator.get_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(state, generator):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
                         numpy_state[2], numpy_state[3], numpy_state[4]))
    generator.set_state(state["loader"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def doctor():
    result = {"python": platform.python_version(), "torch": str(torch.__version__),
              "numpy": str(np.__version__), "cuda_build": torch.version.cuda,
              "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        result["gpu"] = torch.cuda.get_device_name(0)
        result["gpu_capability"] = list(torch.cuda.get_device_capability(0))
        result["wheel_architectures"] = torch.cuda.get_arch_list()
        result["gpu_matrix_test"] = float((torch.ones(8, 8, device="cuda") @
                                               torch.ones(8, 8, device="cuda")).sum().item())
    else:
        result["gpu_training_verified"] = False
    print(json.dumps(result, indent=2))
    return result


def train_run(args, *, baseline_config=BASELINE_CONFIG):
    if min(args.epochs, args.patience, args.batch_size, args.hidden_size, args.cpu_threads) < 1:
        raise ValueError("Epochs, patience, batch size, hidden size and threads must be positive")
    if args.debug_limit < 0 or args.seed < 0:
        raise ValueError("debug limit and seed must be nonnegative")
    torch.set_num_threads(args.cpu_threads)
    data_root = Path(args.data).resolve()
    output = Path(args.output).resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {output}; use a new name or --resume")
    if args.resume and (output / "evaluation.json").exists():
        raise ValueError("An evaluated run is frozen; use a new experiment directory")
    noise_config = NoiseConfig(args.kind, args.ratio, args.fraction, args.noise_seed)
    noise_config.validate()
    clean = load_processed_dataset(data_root / "clean")
    with Path(baseline_config).open("rb") as stream:
        base = tomllib.load(stream)
    if clean.checksums != base["data"]["expected_checksums"]:
        raise ValueError("Dataset differs from the team's baseline-pinned artifact")
    if clean.summary.get("protocol_version") != base["data"]["expected_protocol_version"]:
        raise ValueError("Dataset protocol differs from baseline")
    original_train = clean.train[:args.debug_limit] if args.debug_limit else clean.train
    validation = clean.validation[:100] if args.debug_limit else clean.validation
    train_rows, noise = make_training_examples(original_train, clean.num_items, noise_config)
    set_random_seed(args.seed, deterministic=args.deterministic)
    device = resolve_device(args.device)
    model_config = {"num_items": clean.num_items, "hidden_size": args.hidden_size,
                    "gnn_steps": base["model"]["gnn_steps"], "hybrid": base["model"]["hybrid"]}
    model = SRGNN(**model_config).to(device)
    training = base["training"]
    optimizer = torch.optim.Adam(model.parameters(), lr=training["learning_rate"],
                                 weight_decay=training["weight_decay"],
                                 fused=(device.type == "cuda" and args.fused))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=training["lr_decay_step"],
                                                gamma=training["lr_decay_gamma"])
    train_loader = _make_loader(train_rows, args.batch_size, True, args.seed,
                               collate_session_graphs, num_workers=args.num_workers,
                               pin_memory=(device.type == "cuda"))
    validation_loader = _make_loader(validation, args.batch_size, False, args.seed,
                                    collate_session_graphs, num_workers=args.num_workers,
                                    pin_memory=(device.type == "cuda"))
    identity = {
        "clean_checksums": clean.checksums, "training_noise": noise,
        "model": model_config, "training_seed": args.seed,
        "batch_size": args.batch_size, "patience": args.patience,
        "optimizer": {"name": "Adam", "lr": training["learning_rate"],
                      "weight_decay": training["weight_decay"]},
        "scheduler": {"name": "StepLR", "step_size": training["lr_decay_step"],
                      "gamma": training["lr_decay_gamma"]},
        "deterministic": args.deterministic, "debug_limit": args.debug_limit,
        "fused_adam": device.type == "cuda" and args.fused,
        "num_workers": args.num_workers,
        "selection": "clean validation MRR@20", "k": 20,
    }
    metadata = {
        "status": "training", "identity": identity, "device": str(device),
        "artifact_repository": base["data"]["artifact_repository"],
        "artifact_revision_from_matching_baseline": base["data"]["artifact_revision"],
        "filtering_protocol": clean.summary.get("filtering_protocol"),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "software": {"python": platform.python_version(), "torch": str(torch.__version__),
                     "numpy": str(np.__version__)},
        "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "test_used_for_training_or_selection": False,
        "note": "Same team model/collator/loss/ranking functions; only training input mixture changes.",
    }
    history, best_mrr, best_epoch, stale, updates = [], -1.0, 0, 0, 0
    elapsed = 0.0
    start_epoch = 1
    prior_peak = 0
    if args.resume:
        saved = torch.load(output / "last_training.pt", map_location="cpu", weights_only=True)
        if saved["identity"] != identity:
            raise ValueError("Resume rejected: data/noise/model/training configuration differs")
        terminal_resume = saved["epoch"] >= args.epochs or saved["stale_epochs"] >= args.patience
        if not (output / "best_model.pt").is_file():
            raise FileNotFoundError("Resume requires this run's best_model.pt")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        _restore_rng(saved["rng"], train_loader.generator)
        history, best_mrr, best_epoch = saved["history"], saved["best_mrr"], saved["best_epoch"]
        stale, updates = saved["stale_epochs"], saved["optimizer_updates"]
        elapsed = saved["train_validation_seconds"]
        prior_peak = saved["peak_gpu_allocated_bytes"] or 0
        start_epoch = args.epochs + 1 if terminal_resume else saved["epoch"] + 1
    else:
        output.mkdir(parents=True)
    _save_json(output / "run.json", metadata)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    print(json.dumps({"device": str(device), "training_examples": len(train_rows),
                      "parameters": metadata["parameter_count"], "noise": noise}), flush=True)
    started = time.perf_counter()
    peak = prior_peak if device.type == "cuda" else None
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.perf_counter()
        lr = optimizer.param_groups[0]["lr"]
        loss = train_epoch(model, train_loader, optimizer, device)
        if not math.isfinite(loss):
            raise RuntimeError("Nonfinite loss; training stopped")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - epoch_start
        scores = evaluate_ranking(model, validation_loader, device, k=20)
        if scores.mrr_at_k > best_mrr:
            best_mrr, best_epoch, stale = scores.mrr_at_k, epoch, 0
            # Keep the team's checkpoint format: a plain state_dict.
            _save_torch(output / "best_model.pt", model.state_dict())
        else:
            stale += 1
        updates += len(train_loader)
        scheduler.step()
        peak = max(prior_peak, torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        record = {"epoch": epoch, "train_loss": loss, "learning_rate": lr,
                  "validation": _results_dict(scores, 20), "training_seconds": train_seconds,
                  "epoch_seconds": time.perf_counter() - epoch_start,
                  "optimizer_updates": updates}
        history.append(record)
        _save_json(output / "history.json", history)
        _save_torch(output / "last_training.pt", {
            "identity": identity, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "rng": _rng_snapshot(train_loader.generator),
            "epoch": epoch, "history": history, "best_mrr": best_mrr, "best_epoch": best_epoch,
            "stale_epochs": stale, "optimizer_updates": updates,
            "train_validation_seconds": elapsed + time.perf_counter() - started,
            "peak_gpu_allocated_bytes": peak,
        })
        print(json.dumps(record), flush=True)
        if stale >= args.patience:
            break
    metadata.update({"status": "trained_not_tested", "best_epoch": best_epoch,
                     "best_validation_mrr@20": best_mrr, "epochs_completed": len(history),
                     "optimizer_updates": updates,
                     "train_validation_seconds": elapsed + time.perf_counter() - started,
                     "peak_gpu_allocated_bytes": peak})
    _save_json(output / "run.json", metadata)
    return metadata


