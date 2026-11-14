"""Session sequence dataclasses and collator for sequential recommendation models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class SessionSequenceBatch:
    """Left-padded sequence tensor and target items for a batch of session examples."""

    input_ids: Tensor
    targets: Tensor

    def to(self, device: torch.device | str, non_blocking: bool = False) -> SessionSequenceBatch:
        """Return the batch with tensors moved to ``device``."""

        return SessionSequenceBatch(
            input_ids=self.input_ids.to(device, non_blocking=non_blocking),
            targets=self.targets.to(device, non_blocking=non_blocking),
        )


def collate_session_sequences(
    examples: Sequence[tuple[Sequence[int], int]],
    maxlen: int = 50,
) -> SessionSequenceBatch:
    """Collate (prefix, target) examples with left-padding to a fixed capacity ``maxlen``."""

    if not examples:
        raise ValueError("cannot collate an empty batch")
    if maxlen <= 0:
        raise ValueError("maxlen must be a positive integer")

    batch_size = len(examples)
    input_ids = np.zeros((batch_size, maxlen), dtype=np.int64)
    targets = np.empty(batch_size, dtype=np.int64)

    for row, (prefix, target) in enumerate(examples):
        if not prefix:
            raise ValueError("a session prefix must contain at least one item")
        if isinstance(target, bool) or not isinstance(target, int) or target <= 0:
            raise ValueError("target item IDs must be positive integers")

        truncated = prefix[-maxlen:]
        if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in truncated):
            raise ValueError("session item IDs must be positive integers; zero is reserved for padding")

        start_col = maxlen - len(truncated)
        input_ids[row, start_col:] = truncated
        targets[row] = target

    return SessionSequenceBatch(
        input_ids=torch.from_numpy(input_ids),
        targets=torch.from_numpy(targets),
    )
