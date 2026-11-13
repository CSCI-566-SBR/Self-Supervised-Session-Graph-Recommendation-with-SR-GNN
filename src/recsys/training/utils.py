"""System, DataLoader, and artifact validation utilities for model training."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader

from recsys.data.processed import ProcessedDataset
from recsys.training.baseline import RankingResults
from recsys.training.config import ExperimentConfig


def resolve_device(requested: str) -> torch.device:
    """Resolve device string to torch.device with CUDA/MPS availability checks."""

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


def resolve_num_workers(device: torch.device, requested: int | None = None) -> int:
    """Resolve DataLoader worker count with conservative defaults for shared machines.

    If explicitly requested, returns max(0, requested).
    If environment variable RECSYS_NUM_WORKERS is set, uses that value.
    Otherwise on CUDA, defaults to min(4, available_cores). On CPU/MPS, defaults to 0.
    """
    if requested is not None:
        return max(0, requested)

    env_val = os.environ.get("RECSYS_NUM_WORKERS")
    if env_val is not None:
        try:
            return max(0, int(env_val))
        except ValueError:
            pass

    if device.type != "cuda":
        return 0

    try:
        available = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        available = os.cpu_count() or 1
    return max(1, min(4, available))


def make_data_loader(
    examples: Any,
    batch_size: int,
    shuffle: bool,
    seed: int,
    collate_fn: Callable,
    num_workers: int = 0,
    pin_memory: bool = False,
    prefetch_factor: int | None = 4,
) -> DataLoader:
    """Construct a DataLoader with reproducible seeding, multiprocessing, and prefetching."""

    generator = torch.Generator()
    generator.manual_seed(seed)
    loader_kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor or 2

    return DataLoader(
        examples,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        generator=generator,
        **loader_kwargs,
    )


def get_git_commit() -> str | None:
    """Retrieve the current HEAD git commit hash if running in a git repository."""

    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def format_results_dict(results: RankingResults, k: int) -> dict[str, Any]:
    """Format RankingResults into a serializable metric dictionary."""

    output = dict(results.metrics)
    output["loss"] = results.val_loss
    output["examples"] = results.examples
    return output


def prepare_output_dir(output_dir: Path) -> None:
    """Create the output directory defensively, raising FileExistsError if non-empty."""

    if output_dir.exists():
        raise FileExistsError(
            f"output directory already exists; refusing to overwrite: {output_dir}"
        )
    output_dir.mkdir(parents=True)


def validate_artifact_identity(
    config: ExperimentConfig, dataset: ProcessedDataset
) -> None:
    """Validate dataset protocol and checksums against experiment configuration."""

    actual_protocol = dataset.summary.get("protocol_version")
    if (
        config.expected_protocol_version is not None
        and actual_protocol != config.expected_protocol_version
    ):
        raise ValueError(
            "processed dataset protocol does not match the experiment configuration: "
            f"expected {config.expected_protocol_version!r}, got {actual_protocol!r}"
        )
    if (
        config.expected_checksums is not None
        and dataset.checksums != config.expected_checksums
    ):
        raise ValueError("processed dataset checksums do not match the frozen artifact")
