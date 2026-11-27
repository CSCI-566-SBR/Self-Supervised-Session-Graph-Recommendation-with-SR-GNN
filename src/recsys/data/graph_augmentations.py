"""Model-independent session-graph augmentations for SSL training.

These functions create stochastic graph views in memory. They do not write
static augmented training files: callers should generate fresh views while
training and build their normalized adjacency matrices from ``GraphView``.
"""

from __future__ import annotations

import random
import math
from dataclasses import dataclass
from typing import Literal, Sequence
from recsys.data.session_graph import _validate_sequence

AugmentationMode = Literal["uniform_edge", "uniform_node", "recency", "recency_substitution"]


@dataclass(frozen=True)
class Transition:
    """A retained chronological transition from position ``index`` to ``index + 1``."""

    source: int
    target: int
    index: int


@dataclass(frozen=True)
class GraphView:
    """One augmented view of a session prefix.

    ``sequence`` is used for session readout, ``nodes`` lists retained unique
    item IDs in first-appearance order, and ``edges`` preserves chronological
    transition positions. Repeated edges remain repeated so a downstream graph
    builder can aggregate their weights exactly as SR-GNN requires.
    """

    sequence: tuple[int, ...]
    nodes: tuple[int, ...]
    edges: tuple[Transition, ...]


def _validate(sequence: Sequence[int], probability: float, gamma: float) -> tuple[int, ...]:
    values = tuple(int(item) for item in sequence)
    if not values:
        raise ValueError("a session prefix cannot be empty")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0, 1]")
    if gamma < 1.0:
        raise ValueError("gamma must be at least one")
    return values


def _view(sequence: tuple[int, ...], kept_edge_indices: set[int]) -> GraphView:
    edges = tuple(
        Transition(sequence[index], sequence[index + 1], index)
        for index in range(len(sequence) - 1)
        if index in kept_edge_indices
    )
    return GraphView(sequence, tuple(dict.fromkeys(sequence)), edges)


def augment_session(
    sequence: Sequence[int],
    *,
    mode: AugmentationMode,
    probability: float,
    rng: random.Random,
    gamma: float = 1.0,
) -> GraphView:
    """Create one uniform-edge, uniform-node, or recency-aware graph view."""
    values = _validate(sequence, probability, gamma)
    all_edges = set(range(max(0, len(values) - 1)))
    if len(values) <= 2:
        return _view(values, all_edges)

    if mode == "uniform_edge":
        kept = {index for index in all_edges if rng.random() >= probability}
        if all_edges and not kept:
            restored_edge = rng.choice(sorted(all_edges))
            kept.add(restored_edge)
        return _view(values, kept)

    if mode == "recency":
        kept: set[int] = set()
        length = len(values)
        for edge_index in range(length - 1):
            chronological_step = edge_index + 1
            drop_probability = probability * (1.0 - chronological_step / length) ** gamma
            if rng.random() >= drop_probability:
                kept.add(edge_index)
        # Strictly protect the edge entering the final observed item.
        kept.add(length - 2)
        return _view(values, kept)

    # if mode == "uniform_node":
    #     final_item = values[-1]
    #     candidates = tuple(dict.fromkeys(item for item in values[:-1] if item != final_item))
    #     dropped = {item for item in candidates if rng.random() < probability}
    #     filtered = tuple(item for item in values if item not in dropped)
    #     if len(filtered) < 2:
    #         # Restore the most recent dropped node identity and all occurrences.
    #         restored = next(item for item in reversed(values[:-1]) if item != final_item)
    #         dropped.discard(restored)
    #         filtered = tuple(item for item in values if item not in dropped)
    #     nodes = tuple(dict.fromkeys(filtered))
    #     edges = tuple(
    #         Transition(values[index], values[index + 1], index)
    #         for index in range(len(values) - 1)
    #         if values[index] not in dropped and values[index + 1] not in dropped
    #     )
    #     return GraphView(filtered, nodes, edges)

    if mode == "uniform_node":
        final_item = values[-1]
        candidates = tuple(dict.fromkeys(item for item in values[:-1] if item != final_item))

        dropped = {
            item
            for item in candidates
            if rng.random() < probability
        }

        def construct_identity_dropout_view(dropped_items: set[int],) -> GraphView:
            filtered_sequence = tuple(
                item
                for item in values
                if item not in dropped_items
            )

            retained_nodes = tuple(dict.fromkeys(filtered_sequence))

            retained_edges = tuple(
                Transition(
                    values[index],
                    values[index + 1],
                    index,
                )
                for index in range(len(values) - 1)
                if (
                    values[index] not in dropped_items
                    and values[index + 1] not in dropped_items
                )
            )

            return GraphView(filtered_sequence,retained_nodes,retained_edges,)

        view = construct_identity_dropout_view(dropped)

        while (len(view.sequence) < 2 or not view.edges):
            item_to_restore = next(
                (
                    item
                    for item in reversed(values[:-1])
                    if item in dropped
                ),
                None,
            )

            if item_to_restore is None:
                break

            dropped.remove(item_to_restore)
            view = construct_identity_dropout_view(dropped)
        return view

    raise ValueError(f"unknown augmentation mode: {mode}")

def count_edge_deletions(
    transition_count: int,
    budget: float,
    *,
    budget_rounding: str = "nearest",
    rng: random.Random | None = None,
) -> int:
    """Calculate a protected per-prefix transition-deletion count."""
    if (
        isinstance(transition_count, bool)
        or not isinstance(transition_count, int)
        or transition_count < 0
    ):
        raise ValueError("transition_count must be a non-negative integer")

    if not 0.0 <= budget <= 1.0:
        raise ValueError("budget must lie in [0, 1]")

    if budget_rounding not in {"nearest", "stochastic"}:
        raise ValueError(
            "budget_rounding must be nearest or stochastic"
        )

    if budget_rounding == "stochastic" and rng is None:
        raise ValueError("stochastic rounding requires a budget RNG")

    # Prefixes with at most two items remain unchanged.
    if transition_count <= 1:
        return 0

    requested = budget * transition_count

    if budget_rounding == "nearest":
        count = round(requested)
    else:
        lower = math.floor(requested)
        count = lower + int(rng.random() < requested - lower)

    # Protect the final observed transition.
    return min(count, transition_count - 1)

def make_two_views(
    sequence: Sequence[int],
    *,
    mode: AugmentationMode,
    probability: float | None = None,
    budget: float | None = None,
    budget_rounding: str = "nearest",
    budget_rng: random.Random | None = None,
    gamma: float = 1.0,
    seed: int | None = None,
    rng: random.Random | None = None,
    num_items: int | None = None,
    target: int | None = None,
) -> tuple[GraphView, GraphView]:
    """Generate paired views using probability or fixed-budget augmentation.

    Supply exactly one of probability or budget.
    Existing probability-based callers retain their original behavior.
    """
    if rng is not None and seed is not None:
        raise ValueError("pass either rng or seed, not both")

    if (probability is None) == (budget is None):
        raise ValueError("provide exactly one of probability or budget")

    source = rng if rng is not None else random.Random(seed)
    if budget_rounding not in {"nearest", "stochastic"}:
        raise ValueError(
            "budget_rounding must be nearest or stochastic"
        )

    if budget is None and (
        budget_rounding != "nearest" or budget_rng is not None
    ):
        raise ValueError(
            "rounding options require budget-based augmentation"
        )

    if budget is not None:
        if mode not in {"uniform_edge", "recency", "recency_substitution"}:
            raise ValueError(
                "budgeted augmentation supports uniform_edge, recency, or recency_substitution"
            )
        if budget_rounding == "stochastic" and budget_rng is source:
            raise ValueError(
                "budget_rng must be separate from the edge-selection RNG"
            )

        deletion_count = count_edge_deletions(
            len(sequence) - 1,
            budget,
            budget_rounding=budget_rounding,
            rng=budget_rng,
        )
        if mode == "recency_substitution":
            return tuple(
                augment_recency_substitution(
                    sequence, num_items=num_items, count=deletion_count,
                    gamma=gamma, rng=source, target=target,
                )
                for _ in range(2)
            )
        return (
            augment_budgeted_edge_view(
                sequence,
                mode=mode,
                budget=budget,
                gamma=gamma,
                rng=source,
                deletion_count=deletion_count,
            ),
            augment_budgeted_edge_view(
                sequence,
                mode=mode,
                budget=budget,
                gamma=gamma,
                rng=source,
                deletion_count=deletion_count,
            ),
        )

    if mode == "recency_substitution":
        raise ValueError("recency_substitution requires a budget")
    return (
        augment_session(
            sequence,
            mode=mode,
            probability=probability,
            gamma=gamma,
            rng=source,
        ),
        augment_session(
            sequence,
            mode=mode,
            probability=probability,
            gamma=gamma,
            rng=source,
        ),
    )

def augment_budgeted_edge_view(
    sequence: Sequence[int],
    *,
    mode: Literal["uniform_edge", "recency"],
    budget: float,
    rng: random.Random,
    gamma: float = 2.0,
    deletion_count: int | None = None,
) -> GraphView:
    """Drop a fixed number of transition occurrences from one prefix.

    Both policies use the same deletion count and protect the final
    observed transition. Only their sampling weights differ.

    Budget denominator: all original transition occurrences.
    Rounding: Python round(), including ties-to-even.
    Prefixes containing at most two items remain unchanged.

    Counts refer to transition occurrences, not unique graph edges.
    """
    if mode not in {"uniform_edge", "recency"}:
        raise ValueError("mode must be uniform_edge or recency")
    if not 0.0 <= budget <= 1.0:
        raise ValueError("budget must lie in [0, 1]")
    if not math.isfinite(gamma) or gamma < 1.0:
        raise ValueError("gamma must be finite and at least one")

    values = tuple(sequence)
    _validate_sequence(values)

    transition_count = len(values) - 1
    all_edges = set(range(transition_count))
    eligible = list(range(max(0, transition_count - 1)))

    if deletion_count is None:
        deletion_count = count_edge_deletions(
            transition_count,
            budget,
        )
    elif (
        isinstance(deletion_count, bool)
        or not isinstance(deletion_count, int)
        or not 0 <= deletion_count <= len(eligible)
    ):
        raise ValueError(
            "deletion_count must be an integer within the eligible count"
        )

    if deletion_count == 0:
        return _view(values, all_edges)

    if mode == "uniform_edge":
        dropped = set(rng.sample(eligible, deletion_count))
    else:
        # For one-based transition position i:
        # weight = ((M - i) / (M - 1)) ** gamma.
        candidates = eligible.copy()
        weights = [
            (
                (transition_count - (index + 1))
                / (transition_count - 1)
            ) ** gamma
            for index in candidates
        ]

        dropped = set()
        for _ in range(deletion_count):
            selected_position = rng.choices(
                range(len(candidates)),
                weights=weights,
                k=1,
            )[0]
            dropped.add(candidates.pop(selected_position))
            weights.pop(selected_position)

    return _view(values, all_edges - dropped)


def augment_recency_substitution(
    sequence: Sequence[int],
    *,
    num_items: int,
    count: int,
    rng: random.Random,
    gamma: float = 2.0,
    target: int | None = None,
) -> GraphView:
    """Replace older clicks using the training vocabulary; protect the final click.

    Paired callers use the existing protected/rounded edit count so that the
    requested number of substitutions matches the deletion experiment. Rebuild
    all transitions from the resulting sequence, including repeated items.
    """
    from recsys.data.perturb import sample_replacement

    values = list(sequence)
    _validate_sequence(values)
    if isinstance(num_items, bool) or not isinstance(num_items, int) or num_items <= 0:
        raise ValueError("num_items must be a positive training vocabulary size")
    if any(item > num_items for item in values):
        raise ValueError("session item exceeds the training vocabulary")
    if not math.isfinite(gamma) or gamma < 1:
        raise ValueError("gamma must be finite and at least one")
    candidates = list(range(len(values) - 1))
    if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= len(candidates):
        raise ValueError("invalid substitution count")
    weights = [((len(values) - 1 - pos) / (len(values) - 1)) ** gamma for pos in candidates]
    for _ in range(count):
        selected = rng.choices(range(len(candidates)), weights=weights, k=1)[0]
        position = candidates.pop(selected)
        weights.pop(selected)
        forbidden = {values[position]}
        if target is not None:
            forbidden.add(target)
        replacement = sample_replacement(num_items, forbidden, rng)
        if replacement is not None:
            values[position] = replacement
    return _view(tuple(values), set(range(len(values) - 1)))
