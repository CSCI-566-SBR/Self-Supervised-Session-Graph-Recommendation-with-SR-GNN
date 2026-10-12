"""Small training and evaluation functions for the SR-GNN baseline."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from recsys.data.session_graph import SessionGraphBatch


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
    """Evaluate Recall@K and MRR@K without materializing all candidate ranks."""

    if k <= 0:
        raise ValueError("k must be positive")

    model.eval()
    recall_sum = 0.0
    reciprocal_rank_sum = 0.0
    total_examples = 0

    for batch in batches:
        batch = batch.to(device)
        logits = model(batch)
        if torch.any(batch.targets < 1) or torch.any(batch.targets > logits.shape[1]):
            raise ValueError("target item IDs must lie in 1..num_items")
        cutoff = min(k, logits.shape[1])
        top_items = torch.topk(logits, cutoff, dim=1).indices
        target_classes = (batch.targets - 1).unsqueeze(1)
        matches = top_items.eq(target_classes)

        hits = matches.any(dim=1)
        recall_sum += hits.sum().item()
        positions = torch.arange(1, cutoff + 1, device=logits.device)
        reciprocal_ranks = torch.where(
            matches,
            positions.reciprocal(),
            torch.zeros((), device=logits.device),
        ).sum(dim=1)
        reciprocal_rank_sum += reciprocal_ranks.sum().item()
        total_examples += batch.targets.shape[0]

    if total_examples == 0:
        raise ValueError("evaluation requires at least one example")
    return RankingResults(
        recall_at_k=recall_sum / total_examples,
        mrr_at_k=reciprocal_rank_sum / total_examples,
        examples=total_examples,
    )
