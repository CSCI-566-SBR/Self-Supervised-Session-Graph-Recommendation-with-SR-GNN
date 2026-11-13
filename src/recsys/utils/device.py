"""Hardware device and multiprocessing allocation utilities."""

from __future__ import annotations

import os
import torch


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
