"""PyTorch target-rank extraction matching the shared metric semantics."""

from __future__ import annotations

import torch
from torch import Tensor


_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def get_target_ranks_tensor(
    scores: Tensor,
    targets: Tensor,
    item_ids: Tensor,
) -> list[int]:
    """Return exact 1-based target ranks without moving all scores to Python.

    This is the tensor equivalent of ``get_target_ranks``. Higher scores rank
    first, and equal scores favor the smaller item ID. Only the resulting rank
    vector is transferred to the CPU.
    """

    if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] == 0:
        raise ValueError("scores must have shape [examples, candidates] and be non-empty")
    if targets.ndim != 1 or targets.shape[0] != scores.shape[0]:
        raise ValueError("targets must have shape [examples]")
    if item_ids.ndim != 1 or item_ids.shape[0] != scores.shape[1]:
        raise ValueError("item_ids must have shape [candidates]")
    if targets.dtype not in _INTEGER_DTYPES or item_ids.dtype not in _INTEGER_DTYPES:
        raise ValueError("targets and item_ids must contain integer IDs")
    if scores.is_complex():
        raise ValueError("scores must contain real numbers")
    if scores.device != targets.device or scores.device != item_ids.device:
        raise ValueError("scores, targets, and item_ids must be on the same device")
    if torch.unique(item_ids).numel() != item_ids.numel():
        raise ValueError("item_ids must be unique")
    if not torch.isfinite(scores).all():
        raise ValueError("scores must contain only finite numbers")

    target_matches = targets.unsqueeze(1).eq(item_ids.unsqueeze(0))
    if not torch.all(target_matches.sum(dim=1) == 1):
        raise ValueError("every target must be present exactly once in item_ids")

    target_columns = target_matches.to(torch.int64).argmax(dim=1, keepdim=True)
    target_scores = scores.gather(1, target_columns)
    candidates_ahead = scores > target_scores
    tied_smaller_items = scores.eq(target_scores) & (
        item_ids.unsqueeze(0) < targets.unsqueeze(1)
    )
    ranks = 1 + (candidates_ahead | tied_smaller_items).sum(dim=1)
    return ranks.detach().cpu().tolist()
