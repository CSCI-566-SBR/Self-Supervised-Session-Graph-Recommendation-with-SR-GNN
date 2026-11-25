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

@dataclass(frozen=True)
class TrainingMetric:
    """One batch's contribution to an epoch metric.

    count=None: sum totals.
    count supplied: divide summed totals by summed counts.
    """

    total: Tensor | float | int
    count: float | int | None = None

def train_epoch(
    model: nn.Module,
    batches: Iterable[Any],
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    *,
    metrics: dict[str, float | int | None] | None = None,
) -> float:
    """Train one epoch, optionally collecting model-defined metrics."""
    model.train()
    total_loss = 0.0
    total_examples = 0

    metric_totals: dict[str, Tensor | float | int] = {}
    metric_counts: dict[str, float | int | None] = {}

    if metrics is not None:
        metrics.clear()

    for batch in batches:
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        training_loss_fn = getattr(model, "training_loss", None)
        if training_loss_fn is None:
            logits = model(batch)
            loss = recommendation_loss(logits, batch.targets)
            step_metrics = {}
        elif callable(training_loss_fn):
            loss, step_metrics = training_loss_fn(batch)
        else:
            raise TypeError("model.training_loss must be callable")

        for name, contribution in step_metrics.items():
            if not isinstance(contribution, TrainingMetric):
                raise TypeError(
                    "training metrics must contain TrainingMetric values"
                )

            value = contribution.total
            if isinstance(value, Tensor):
                if value.ndim != 0:
                    raise ValueError("training metrics must be scalars")
                value = value.detach()

            count = contribution.count
            if count is not None and count < 0:
                raise ValueError("metric counts must be non-negative")

            if name in metric_counts:
                if (metric_counts[name] is None) != (count is None):
                    raise ValueError(
                        f"inconsistent reduction for metric {name!r}"
                    )
            else:
                metric_counts[name] = None if count is None else 0

            metric_totals[name] = metric_totals.get(name, 0) + value
            if count is not None:
                metric_counts[name] = (
                    metric_counts[name] or 0
                ) + count

        loss.backward()
        optimizer.step()

        examples = batch.targets.shape[0]
        total_loss += loss.item() * examples
        total_examples += examples

    if total_examples == 0:
        raise ValueError("training requires at least one example")

    if metrics is not None:
        for name, total in metric_totals.items():
            value = total.item() if isinstance(total, Tensor) else total
            count = metric_counts[name]
            metrics[name] = (
                value
                if count is None
                else value / count if count else None
            )

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
