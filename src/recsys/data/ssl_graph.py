"""CPU-side graph-view collation for SR-GNN SSL."""

from __future__ import annotations

from typing import Sequence

import torch

from recsys.data.graph_augmentations import GraphView
from recsys.data.session_graph import (
    SessionGraphBatch,
    _validate_sequence,
    build_session_adjacency,
    collate_session_graphs,
)
import random
from dataclasses import dataclass, fields, replace
from typing import Callable

from torch.utils.data import get_worker_info

from recsys.data.graph_augmentations import make_two_views


def collate_graph_views(
    examples: Sequence[tuple[GraphView, int]],
    *,
    template: SessionGraphBatch | None = None,
) -> SessionGraphBatch:
    """Collate retained edges, optionally reusing clean-batch metadata."""
    if not examples:
        raise ValueError("cannot collate an empty batch")

    for view, _ in examples:
        _validate_sequence(view.sequence)
        if set(view.nodes) != set(view.sequence):
            raise ValueError("view nodes must match its sequence items")

    if template is None:
        return collate_session_graphs(
            [(view.sequence, target) for view, target in examples],
            edge_lists=[
                [(edge.source, edge.target) for edge in view.edges]
                for view, _ in examples
            ],
        )

    tensors = (
        template.node_ids,
        template.adjacency,
        template.alias_indices,
        template.sequence_mask,
        template.targets,
    )
    if any(tensor.device.type != "cpu" for tensor in tensors):
        raise ValueError("graph-view collation requires a CPU template")

    if template.targets.shape[0] != len(examples):
        raise ValueError("template batch size does not match examples")

    adjacency = torch.zeros_like(template.adjacency)
    max_nodes = template.node_ids.shape[1]

    for row, (view, target) in enumerate(examples):
        original_sequence = template.node_ids[row].gather(
            0, template.alias_indices[row]
        )[template.sequence_mask[row]].tolist()

        if tuple(original_sequence) != view.sequence:
            raise ValueError(
                "template reuse requires an unchanged item sequence"
            )
        if (
            isinstance(target, bool)
            or not isinstance(target, int)
            or target <= 0
            or target != template.targets[row].item()
        ):
            raise ValueError("view target does not match the template")

        nodes = sorted(set(view.sequence))
        node_index = {
            item: index for index, item in enumerate(nodes)
        }
        num_nodes = len(nodes)

        view_adjacency = build_session_adjacency(
            node_index,
            ((edge.source, edge.target) for edge in view.edges),
        )
        adjacency[row, :num_nodes, :num_nodes] = (
            view_adjacency[:, :num_nodes]
        )
        adjacency[
            row, :num_nodes, max_nodes:max_nodes + num_nodes
        ] = view_adjacency[:, num_nodes:]

    return SessionGraphBatch(
        node_ids=template.node_ids,
        adjacency=adjacency,
        alias_indices=template.alias_indices,
        sequence_mask=template.sequence_mask,
        targets=template.targets,
    )

@dataclass(frozen=True)
class SSLGraphBatch:
    """Clean graph batch, two views, and CPU-computed diagnostics."""

    clean: SessionGraphBatch
    first_view: SessionGraphBatch
    second_view: SessionGraphBatch
    original_transitions: int
    deleted_transitions: int
    prefixes: int
    zero_deletion_prefixes: int

    @property
    def targets(self) -> torch.Tensor:
        return self.clean.targets

    def _map_tensors(
        self,
        transform: Callable[[torch.Tensor], torch.Tensor],
    ) -> SSLGraphBatch:
        clean = replace(
            self.clean,
            **{
                item.name: transform(getattr(self.clean, item.name))
                for item in fields(self.clean)
            },
        )

        # Edge-only views share the unchanged node/sequence metadata.
        return replace(
            self,
            clean=clean,
            first_view=replace(
                clean,
                adjacency=transform(self.first_view.adjacency),
            ),
            second_view=replace(
                clean,
                adjacency=transform(self.second_view.adjacency),
            ),
        )

    def to(
        self,
        device: torch.device | str,
        non_blocking: bool = False,
    ) -> SSLGraphBatch:
        return self._map_tensors(
            lambda tensor: tensor.to(
                device, non_blocking=non_blocking
            )
        )

    def pin_memory(self) -> SSLGraphBatch:
        return self._map_tensors(lambda tensor: tensor.pin_memory())


class SSLGraphCollator:
    """Generate budgeted graph views in the CPU data-loading pipeline."""

    def __init__(
        self,
        *,
        mode: str,
        budget: float,
        gamma: float,
        seed: int,
        budget_rounding: str = "nearest",
    ) -> None:
        if budget_rounding not in {"nearest", "stochastic"}:
            raise ValueError(
                "budget_rounding must be nearest or stochastic"
            )
        self.budget_rounding = budget_rounding
        self.mode = mode
        self.budget = budget
        self.gamma = gamma
        self.seed = seed
        self._rng: random.Random | None = None
        self._budget_rng: random.Random | None = None
        self._worker_seed: int | None = None

    def _get_rng(self) -> random.Random:
        worker = get_worker_info()
        worker_seed = None if worker is None else worker.seed

        if self._rng is None or self._worker_seed != worker_seed:
            # Separate worker streams; reproduce them for the same loader seed.
            combined_seed = (
                self.seed
                if worker_seed is None
                else (self.seed << 64) + worker_seed
            )
            self._rng = random.Random(combined_seed)
            self._budget_rng = random.Random(f"{combined_seed}:budget")
            self._worker_seed = worker_seed

        return self._rng

    def __call__(
        self,
        examples: Sequence[tuple[Sequence[int], int]],
    ) -> SSLGraphBatch:
        clean = collate_session_graphs(examples)
        rng = self._get_rng()

        first_examples = []
        second_examples = []
        original_transitions = 0
        deleted_transitions = 0
        zero_deletion_prefixes = 0

        for prefix, target in examples:
            first, second = make_two_views(
                prefix,
                mode=self.mode,
                budget=self.budget,
                budget_rounding=self.budget_rounding,
                budget_rng=self._budget_rng,
                gamma=self.gamma,
                rng=rng,
            )
            first_examples.append((first, target))
            second_examples.append((second, target))

            transitions = len(prefix) - 1
            original_transitions += 2 * transitions
            deleted_transitions += (
                2 * transitions
                - len(first.edges)
                - len(second.edges)
            )
            zero_deletion_prefixes += int(
                len(first.edges) == transitions
            )

        return SSLGraphBatch(
            clean=clean,
            first_view=collate_graph_views(
                first_examples, template=clean
            ),
            second_view=collate_graph_views(
                second_examples, template=clean
            ),
            original_transitions=original_transitions,
            deleted_transitions=deleted_transitions,
            prefixes=len(examples),
            zero_deletion_prefixes=zero_deletion_prefixes,
        )