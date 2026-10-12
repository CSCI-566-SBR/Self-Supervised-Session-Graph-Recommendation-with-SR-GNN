"""Run a complete SR-GNN experiment from frozen processed artifacts."""

from __future__ import annotations

import json
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from recsys.data.processed import ProcessedDataset, load_processed_dataset
from recsys.data.session_graph import collate_session_graphs
from recsys.models.baselines import SRGNN
from recsys.reproducibility import set_random_seed
from recsys.training.baseline import RankingResults, evaluate_ranking, train_epoch
from recsys.training.config import ExperimentConfig


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but is unavailable")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _make_loader(
    examples,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        examples,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_session_graphs,
        generator=generator,
        num_workers=0,
    )


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _results_dict(results: RankingResults, k: int) -> dict[str, Any]:
    return {
        f"recall@{k}": results.recall_at_k,
        f"mrr@{k}": results.mrr_at_k,
        "examples": results.examples,
    }


def _prepare_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(
            f"output directory already exists; refusing to overwrite: {output_dir}"
        )
    output_dir.mkdir(parents=True)


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    """Train with validation MRR@K selection, then evaluate test exactly once."""

    set_random_seed(config.seed, deterministic=config.deterministic)
    device = resolve_device(config.device)
    dataset: ProcessedDataset = load_processed_dataset(config.processed_data_dir)
    _prepare_output_dir(config.output_dir)

    train_loader = _make_loader(dataset.train, config.batch_size, True, config.seed)
    validation_loader = _make_loader(
        dataset.validation, config.batch_size, False, config.seed
    )
    test_loader = _make_loader(dataset.test, config.batch_size, False, config.seed)

    model = SRGNN(
        num_items=dataset.num_items,
        hidden_size=config.hidden_size,
        gnn_steps=config.gnn_steps,
        hybrid=config.hybrid,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.lr_decay_step,
        gamma=config.lr_decay_gamma,
    )

    checkpoint_path = config.output_dir / "best_model.pt"
    history: list[dict[str, Any]] = []
    best_mrr = float("-inf")
    best_epoch = 0
    stale_epochs = 0
    start_time = time.perf_counter()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, config.max_epochs + 1):
        mean_loss = train_epoch(model, train_loader, optimizer, device)
        validation = evaluate_ranking(
            model, validation_loader, device, k=config.metric_k
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": mean_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "validation": _results_dict(validation, config.metric_k),
            }
        )

        if validation.mrr_at_k > best_mrr:
            best_mrr = validation.mrr_at_k
            best_epoch = epoch
            stale_epochs = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            stale_epochs += 1

        scheduler.step()
        if stale_epochs >= config.patience:
            break

    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    best_validation = evaluate_ranking(
        model, validation_loader, device, k=config.metric_k
    )
    test_results = evaluate_ranking(model, test_loader, device, k=config.metric_k)
    runtime_seconds = time.perf_counter() - start_time

    peak_gpu_memory = None
    if device.type == "cuda":
        peak_gpu_memory = torch.cuda.max_memory_allocated(device)

    results = {
        "experiment": config.name,
        "dataset": config.dataset,
        "filtering_protocol": dataset.summary.get(
            "filtering_protocol",
            dataset.summary.get("filtering_vocabulary_protocol"),
        ),
        "seed": config.seed,
        "device": str(device),
        "best_epoch": best_epoch,
        "checkpoint_metric": f"validation_mrr@{config.metric_k}",
        "best_validation": _results_dict(best_validation, config.metric_k),
        "test": _results_dict(test_results, config.metric_k),
        "runtime_seconds": runtime_seconds,
        "peak_gpu_memory_bytes": peak_gpu_memory,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "git_commit": _git_commit(),
        "dataset_checksums": dataset.checksums,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "config": config.as_serializable_dict(),
        "history": history,
    }
    with (config.output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return results
