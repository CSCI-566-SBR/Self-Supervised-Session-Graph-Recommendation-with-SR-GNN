"""Small training and evaluation functions for the SR-GNN baseline."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Integral

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from recsys.data.session_graph import SessionGraphBatch
from recsys.evaluation import get_target_ranks_tensor, mrr_at_k, recall_at_k


@dataclass(frozen=True)
class RankingResults:
    recall_at_k: float
    mrr_at_k: float
    examples: int


def recommendation_loss(logits: Tensor, target_item_ids: Tensor) -> Tensor:
    """Return cross-entropy for one-based item IDs and zero-based score columns."""

    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch_size, num_items]")
    if target_item_ids.ndim != 1 or target_item_ids.shape[0] != logits.shape[0]:
        raise ValueError("targets must have shape [batch_size]")
    if torch.any(target_item_ids < 1) or torch.any(target_item_ids > logits.shape[1]):
        raise ValueError("target item IDs must lie in 1..num_items")
    return F.cross_entropy(logits, target_item_ids - 1)


def train_epoch(
    model: nn.Module,
    batches: Iterable[SessionGraphBatch],
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
) -> float:
    """Train for one epoch and return example-weighted mean loss."""

    model.train()
    total_loss = 0.0
    total_examples = 0

    for batch in batches:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = recommendation_loss(logits, batch.targets)
        loss.backward()
        optimizer.step()

        examples = batch.targets.shape[0]
        total_loss += loss.item() * examples
        total_examples += examples

    if total_examples == 0:
        raise ValueError("training requires at least one example")
    return total_loss / total_examples


@torch.no_grad()
def evaluate_ranking(
    model: nn.Module,
    batches: Iterable[SessionGraphBatch],
    device: torch.device | str,
    k: int = 20,
) -> RankingResults:
    """Evaluate with the shared exact-rank Recall@K and MRR@K definitions."""

    if isinstance(k, bool) or not isinstance(k, Integral) or k <= 0:
        raise ValueError("k must be a positive integer")

    model.eval()
    ranks: list[int] = []

    for batch in batches:
        batch = batch.to(device)
        logits = model(batch)
        item_ids = torch.arange(
            1,
            logits.shape[1] + 1,
            dtype=batch.targets.dtype,
            device=logits.device,
        )
        ranks.extend(get_target_ranks_tensor(logits, batch.targets, item_ids))

    if not ranks:
        raise ValueError("evaluation requires at least one example")
    return RankingResults(
        recall_at_k=recall_at_k(ranks, k=k),
        mrr_at_k=mrr_at_k(ranks, k=k),
        examples=len(ranks),
    )
