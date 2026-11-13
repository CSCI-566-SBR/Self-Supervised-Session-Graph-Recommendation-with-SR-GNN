"""PyTorch DataLoader factory and data batching utilities."""

from __future__ import annotations

from typing import Any, Callable
import torch
from torch.utils.data import DataLoader


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
