"""Training and evaluation routines shared by baseline models."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from recsys.evaluation import get_target_ranks_tensor, mrr_at_k, ndcg_at_k, recall_at_k


@dataclass(frozen=True)
class RankingResults:
    recall_at_k: float
    mrr_at_k: float
    examples: int
    ndcg_at_k: float = 0.0
    val_loss: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)


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
    batches: Iterable[Any],
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
) -> float:
    """Train for one epoch and return example-weighted mean loss."""

    model.train()
    total_loss = 0.0
    total_examples = 0

    for batch in batches:
        batch = batch.to(device, non_blocking=True)
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
    batches: Iterable[Any],
    device: torch.device | str,
    k: int = 20,
    cutoffs: Sequence[int] = (10, 20),
) -> RankingResults:
    """Evaluate with shared exact-rank Recall@K, MRR@K, and NDCG@K across cutoffs."""

    if isinstance(k, bool) or not isinstance(k, Integral) or k <= 0:
        raise ValueError("k must be a positive integer")

    model.eval()
    ranks: list[int] = []
    item_ids: Tensor | None = None
    total_loss: float = 0.0

    for batch in batches:
        batch = batch.to(device, non_blocking=True)
        logits = model(batch)
        loss = recommendation_loss(logits, batch.targets)
        total_loss += loss.item() * batch.targets.shape[0]

        if (
            item_ids is None
            or item_ids.shape[0] != logits.shape[1]
            or item_ids.device != logits.device
            or item_ids.dtype != batch.targets.dtype
        ):
            item_ids = torch.arange(
                1,
                logits.shape[1] + 1,
                dtype=batch.targets.dtype,
                device=logits.device,
            )
        ranks.extend(get_target_ranks_tensor(logits, batch.targets, item_ids))

    if not ranks:
        raise ValueError("evaluation requires at least one example")

    val_loss = total_loss / len(ranks)

    all_cutoffs = sorted(set(list(cutoffs) + [k]))
    computed_metrics: dict[str, float] = {}
    for cutoff in all_cutoffs:
        computed_metrics[f"recall@{cutoff}"] = recall_at_k(ranks, k=cutoff)
        computed_metrics[f"mrr@{cutoff}"] = mrr_at_k(ranks, k=cutoff)
        computed_metrics[f"ndcg@{cutoff}"] = ndcg_at_k(ranks, k=cutoff)

    return RankingResults(
        recall_at_k=computed_metrics[f"recall@{k}"],
        mrr_at_k=computed_metrics[f"mrr@{k}"],
        ndcg_at_k=computed_metrics[f"ndcg@{k}"],
        examples=len(ranks),
        val_loss=val_loss,
        metrics=computed_metrics,
    )
