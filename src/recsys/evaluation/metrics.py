"""Ranking metrics for one correct next item per evaluation example."""

from collections.abc import Sequence
from math import isfinite, log2
from numbers import Integral, Real


def get_target_ranks(
    scores: Sequence[Sequence[float]],
    targets: Sequence[int],
    item_ids: Sequence[int],
) -> list[int]:
    """Return each target's 1-based rank, independently of any cutoff.

    Each score row corresponds to one target. Columns correspond to the unique
    integer IDs in item_ids, which must exclude padding. Higher scores rank
    first; equal scores favor the smaller item ID. Previously clicked items
    remain eligible if supplied as candidates.

    Counts candidates ahead of each target without sorting. Reuse the returned
    list for any metric or cutoff. Raises ValueError for empty or inconsistent
    inputs, invalid IDs, absent targets, or non-finite/non-numeric scores.
    """
    if len(scores) == 0:
        raise ValueError("scores must contain at least one example")
    if len(scores) != len(targets):
        raise ValueError("scores and targets must have the same number of examples")
    if len(item_ids) == 0:
        raise ValueError("item_ids must contain at least one candidate")
    if any(isinstance(item, bool) or not isinstance(item, Integral) for item in item_ids):
        raise ValueError("item_ids must contain integer IDs")

    columns = {item: column for column, item in enumerate(item_ids)}
    if len(columns) != len(item_ids):
        raise ValueError("item_ids must be unique")

    ranks = []
    for row_index, (row, target) in enumerate(zip(scores, targets)):
        if len(row) != len(item_ids):
            raise ValueError(f"score row {row_index} must have one score per item ID")
        if isinstance(target, bool) or not isinstance(target, Integral):
            raise ValueError(f"target at row {row_index} must be an integer ID")
        if target not in columns:
            raise ValueError(f"target {target} at row {row_index} is absent from item_ids")

        target_score = row[columns[target]]
        if not isinstance(target_score, Real) or not isfinite(target_score):
            raise ValueError(f"target score at row {row_index} must be a finite number")

        rank = 1
        for item, score in zip(item_ids, row):
            if not isinstance(score, Real) or not isfinite(score):
                raise ValueError(f"scores in row {row_index} must be finite numbers")
            if score > target_score or (score == target_score and item < target):
                rank += 1
        ranks.append(rank)

    return ranks


def _validate_ranks(ranks: Sequence[int], k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, Integral) or k <= 0:
        raise ValueError("k must be a positive integer")
    if len(ranks) == 0:
        raise ValueError("ranks must contain at least one example")
    if any(
        isinstance(rank, bool) or not isinstance(rank, Integral) or rank <= 0
        for rank in ranks
    ):
        raise ValueError("ranks must contain positive integers (ranks start at 1)")


def recall_at_k(ranks: Sequence[int], k: int = 20) -> float:
    """Return the fraction of targets within the top k (single-target hit rate).

    Pass 1-based ranks from get_target_ranks. All examples, including misses,
    contribute to the average. Raises ValueError for empty/invalid ranks or k.
    """
    _validate_ranks(ranks, k)
    return sum(rank <= k for rank in ranks) / len(ranks)


def mrr_at_k(ranks: Sequence[int], k: int = 20) -> float:
    """Return mean reciprocal rank, assigning zero to targets below the top k.

    Pass 1-based ranks from get_target_ranks. All examples, including misses,
    contribute to the average. Raises ValueError for empty/invalid ranks or k.
    """
    _validate_ranks(ranks, k)
    return sum(1 / rank for rank in ranks if rank <= k) / len(ranks)


def ndcg_at_k(ranks: Sequence[int], k: int = 20) -> float:
    """Return mean normalized discounted gain for one relevant item per example.

    A hit contributes 1 / log2(rank + 1); a miss contributes zero. The ideal gain
    is 1 for this single-target task. Raises ValueError for empty/invalid ranks
    or k. Pass 1-based ranks from get_target_ranks.
    """
    _validate_ranks(ranks, k)
    return sum(1 / log2(rank + 1) for rank in ranks if rank <= k) / len(ranks)
