"""Run a complete baseline experiment (SR-GNN, SASRec, etc.) from frozen processed artifacts."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from recsys.data.processed import ProcessedDataset, load_processed_dataset
from recsys.models.registry import get_model_builder
from recsys.training.baseline import evaluate_ranking, train_epoch
from recsys.training.config import ExperimentConfig
from recsys.utils import (
    format_results_dict,
    get_git_commit,
    make_data_loader,
    prepare_output_dir,
    resolve_device,
    resolve_num_workers,
    set_random_seed,
    validate_artifact_identity,
)

# Re-exports for backward compatibility
_make_loader = make_data_loader
_prepare_output_dir = prepare_output_dir
_validate_artifact_identity = validate_artifact_identity
_results_dict = format_results_dict
_git_commit = get_git_commit
_resolve_num_workers = resolve_num_workers


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    """Train with validation MRR@K selection, then evaluate test exactly once."""

    set_random_seed(config.seed, deterministic=config.deterministic)
    device = resolve_device(config.device)

    if not config.processed_data_dir.exists() and config.artifact_repository:
        print(
            f"Processed data directory not found at {config.processed_data_dir}.\n"
            f"Auto-downloading pinned artifacts from {config.artifact_repository} (revision: {config.artifact_revision})...",
            flush=True,
        )
        from recsys.data.hub import download_pipeline

        download_pipeline(
            repo_id=config.artifact_repository,
            destination=config.processed_data_dir.parent,
            revision=config.artifact_revision,
        )

    dataset: ProcessedDataset = load_processed_dataset(config.processed_data_dir)
    validate_artifact_identity(config, dataset)
    prepare_output_dir(config.output_dir)

    builder = get_model_builder(config.model_type)
    model, collate_fn = builder(config.model_params, dataset)
    model = model.to(device)

    workers = resolve_num_workers(device, requested=config.num_workers)
    pin_memory = device.type == "cuda"
    train_loader = make_data_loader(
        dataset.train,
        config.batch_size,
        True,
        config.seed,
        collate_fn,
        num_workers=workers,
        pin_memory=pin_memory,
    )
    validation_loader = make_data_loader(
        dataset.validation,
        config.batch_size,
        False,
        config.seed,
        collate_fn,
        num_workers=workers,
        pin_memory=pin_memory,
    )
    test_loader = make_data_loader(
        dataset.test,
        config.batch_size,
        False,
        config.seed,
        collate_fn,
        num_workers=workers,
        pin_memory=pin_memory,
    )

    use_fused = (device.type == "cuda")
    try:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            fused=use_fused,
        )
    except (TypeError, RuntimeError):
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

    print(
        f"Starting training on {device} ({config.model_type.upper()}) | "
        f"Train sessions: {len(dataset.train):,} | "
        f"Val sessions: {len(dataset.validation):,} | "
        f"Workers: {workers} | "
        f"Max epochs: {config.max_epochs}",
        flush=True,
    )

    for epoch in range(1, config.max_epochs + 1):
        epoch_start = time.perf_counter()
        mean_loss = train_epoch(model, train_loader, optimizer, device)
        validation = evaluate_ranking(
            model,
            validation_loader,
            device,
            k=config.metric_k,
            cutoffs=config.metric_cutoffs,
        )
        epoch_seconds = time.perf_counter() - epoch_start
        history.append(
            {
                "epoch": epoch,
                "epoch_seconds": epoch_seconds,
                "train_loss": mean_loss,
                "val_loss": validation.val_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "validation": format_results_dict(validation, config.metric_k),
            }
        )

        rec20 = validation.metrics.get("recall@20", validation.recall_at_k)
        mrr20 = validation.metrics.get("mrr@20", validation.mrr_at_k)
        ndcg20 = validation.metrics.get("ndcg@20", validation.ndcg_at_k)
        is_best = validation.mrr_at_k > best_mrr
        status_tag = " (*best*)" if is_best else ""

        print(
            f"Epoch {epoch:02d}/{config.max_epochs:02d} ({epoch_seconds:.1f}s) | "
            f"Train Loss: {mean_loss:.4f} | "
            f"Val Loss: {validation.val_loss:.4f} | "
            f"Val Recall@20: {rec20:.4f} | "
            f"Val MRR@20: {mrr20:.4f} | "
            f"Val NDCG@20: {ndcg20:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}{status_tag}",
            flush=True,
        )

        if is_best:
            best_mrr = validation.mrr_at_k
            best_epoch = epoch
            stale_epochs = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            stale_epochs += 1

        scheduler.step()
        if stale_epochs >= config.patience:
            print(f"Early stopping triggered after {epoch} epochs.", flush=True)
            break

    if best_epoch == 0 or not checkpoint_path.exists():
        raise RuntimeError(
            f"No checkpoint was saved during training across {config.max_epochs} epochs; "
            f"validation MRR@{config.metric_k} never exceeded -inf."
        )

    print(f"Loading best checkpoint from epoch {best_epoch:02d} for test evaluation...", flush=True)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    best_validation = evaluate_ranking(
        model,
        validation_loader,
        device,
        k=config.metric_k,
        cutoffs=config.metric_cutoffs,
    )
    test_results = evaluate_ranking(
        model,
        test_loader,
        device,
        k=config.metric_k,
        cutoffs=config.metric_cutoffs,
    )
    runtime_seconds = time.perf_counter() - start_time

    test_rec20 = test_results.metrics.get("recall@20", test_results.recall_at_k)
    test_mrr20 = test_results.metrics.get("mrr@20", test_results.mrr_at_k)
    test_ndcg20 = test_results.metrics.get("ndcg@20", test_results.ndcg_at_k)
    print(
        f"Test Results | Loss: {test_results.val_loss:.4f} | "
        f"Recall@20: {test_rec20:.4f} | MRR@20: {test_mrr20:.4f} | "
        f"NDCG@20: {test_ndcg20:.4f} | Total Runtime: {runtime_seconds:.1f}s",
        flush=True,
    )

    peak_gpu_memory = None
    if device.type == "cuda":
        peak_gpu_memory = torch.cuda.max_memory_allocated(device)

    results = {
        "experiment": config.name,
        "dataset": config.dataset,
        "model_type": config.model_type,
        "model_params": config.model_params,
        "dataset_artifact": {
            "repository": config.artifact_repository,
            "revision": config.artifact_revision,
            "protocol_version": dataset.summary.get("protocol_version"),
            "checksums": dataset.checksums,
            "training_noise": dataset.summary.get("training_noise"),
        },
        "filtering_protocol": dataset.summary.get(
            "filtering_protocol",
            dataset.summary.get("filtering_vocabulary_protocol"),
        ),
        "seed": config.seed,
        "device": str(device),
        "best_epoch": best_epoch,
        "checkpoint_metric": f"validation_mrr@{config.metric_k}",
        "best_validation": format_results_dict(best_validation, config.metric_k),
        "test": format_results_dict(test_results, config.metric_k),
        "runtime_seconds": runtime_seconds,
        "peak_gpu_memory_bytes": peak_gpu_memory,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "git_commit": get_git_commit(),
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
