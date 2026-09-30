"""Model-independent session-graph augmentations for SSL training.

These functions create stochastic graph views in memory. They do not write
static augmented training files: callers should generate fresh views while
training and build their normalized adjacency matrices from ``GraphView``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal, Sequence

AugmentationMode = Literal["uniform_edge", "uniform_node", "recency"]


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

    if mode == "uniform_node":
        final_item = values[-1]
        candidates = tuple(dict.fromkeys(item for item in values[:-1] if item != final_item))
        dropped = {item for item in candidates if rng.random() < probability}
        filtered = tuple(item for item in values if item not in dropped)
        if len(filtered) < 2:
            # Restore the most recent dropped node identity and all occurrences.
            restored = next(item for item in reversed(values[:-1]) if item != final_item)
            dropped.discard(restored)
            filtered = tuple(item for item in values if item not in dropped)
        nodes = tuple(dict.fromkeys(filtered))
        edges = tuple(
            Transition(values[index], values[index + 1], index)
            for index in range(len(values) - 1)
            if values[index] not in dropped and values[index + 1] not in dropped
        )
        return GraphView(filtered, nodes, edges)

    raise ValueError(f"unknown augmentation mode: {mode}")


def make_two_views(
    sequence: Sequence[int],
    *,
    mode: AugmentationMode,
    probability: float,
    gamma: float = 1.0,
    seed: int | None = None,
    rng: random.Random | None = None,
) -> tuple[GraphView, GraphView]:
    """Generate two independent views from a caller-owned or seeded RNG.

    Training code should create one persistent ``random.Random(seed)`` and pass
    it on every call. Supplying ``seed`` directly is convenient for isolated,
    exactly reproducible calls and tests.
    """
    if rng is not None and seed is not None:
        raise ValueError("pass either rng or seed, not both")
    source = rng if rng is not None else random.Random(seed)
    return (
        augment_session(sequence, mode=mode, probability=probability, gamma=gamma, rng=source),
        augment_session(sequence, mode=mode, probability=probability, gamma=gamma, rng=source),
    )
