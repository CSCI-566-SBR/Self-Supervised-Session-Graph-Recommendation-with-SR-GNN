"""Build and batch the per-prefix session graphs consumed by SR-GNN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class SessionGraph:
    """Graph tensors for one observed session prefix."""

    node_ids: Tensor
    alias_indices: Tensor
    adjacency: Tensor


@dataclass(frozen=True)
class SessionGraphBatch:
    """Padded graph tensors for a batch of prefix/target examples."""

    node_ids: Tensor
    adjacency: Tensor
    alias_indices: Tensor
    sequence_mask: Tensor
    targets: Tensor

    def to(self, device: torch.device | str) -> SessionGraphBatch:
        """Return the batch with every tensor moved to ``device``."""

        return SessionGraphBatch(
            node_ids=self.node_ids.to(device),
            adjacency=self.adjacency.to(device),
            alias_indices=self.alias_indices.to(device),
            sequence_mask=self.sequence_mask.to(device),
            targets=self.targets.to(device),
        )


def _validate_sequence(sequence: Sequence[int]) -> None:
    if not sequence:
        raise ValueError("a session prefix must contain at least one item")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in sequence):
        raise ValueError("session item IDs must be integers")
    if any(item <= 0 for item in sequence):
        raise ValueError("session item IDs must be positive; zero is reserved for padding")


def build_session_graph(sequence: Sequence[int]) -> SessionGraph:
    """Convert one ordered prefix to the binary graph used by official SR-GNN.

    Repeated items share one node. Repeated directed transitions remain binary,
    matching the official PyTorch implementation. The returned adjacency has
    shape ``[num_nodes, 2 * num_nodes]`` and concatenates incoming then outgoing
    normalized connections.
    """

    _validate_sequence(sequence)

    # The official implementation uses np.unique, which sorts item IDs.
    nodes = sorted(set(sequence))
    node_index = {item: index for index, item in enumerate(nodes)}
    aliases = [node_index[item] for item in sequence]
    num_nodes = len(nodes)

    transitions = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    for source, destination in zip(sequence, sequence[1:]):
        transitions[node_index[source], node_index[destination]] = 1.0

    incoming_degree = transitions.sum(dim=0)
    incoming_degree = incoming_degree.masked_fill(incoming_degree == 0, 1.0)
    incoming = (transitions / incoming_degree).transpose(0, 1)

    outgoing_degree = transitions.sum(dim=1)
    outgoing_degree = outgoing_degree.masked_fill(outgoing_degree == 0, 1.0)
    outgoing = transitions / outgoing_degree.unsqueeze(1)

    return SessionGraph(
        node_ids=torch.tensor(nodes, dtype=torch.long),
        alias_indices=torch.tensor(aliases, dtype=torch.long),
        adjacency=torch.cat((incoming, outgoing), dim=1),
    )


def collate_session_graphs(
    examples: Sequence[tuple[Sequence[int], int]],
) -> SessionGraphBatch:
    """Build graphs and pad a batch of ``(prefix, target_item_id)`` examples."""

    if not examples:
        raise ValueError("cannot collate an empty batch")

    graphs: list[SessionGraph] = []
    targets: list[int] = []
    for prefix, target in examples:
        if isinstance(target, bool) or not isinstance(target, int) or target <= 0:
            raise ValueError("target item IDs must be positive integers")
        graphs.append(build_session_graph(prefix))
        targets.append(target)

    batch_size = len(graphs)
    max_nodes = max(graph.node_ids.numel() for graph in graphs)
    max_sequence_length = max(graph.alias_indices.numel() for graph in graphs)

    node_ids = torch.zeros((batch_size, max_nodes), dtype=torch.long)
    adjacency = torch.zeros(
        (batch_size, max_nodes, 2 * max_nodes), dtype=torch.float32
    )
    alias_indices = torch.zeros(
        (batch_size, max_sequence_length), dtype=torch.long
    )
    sequence_mask = torch.zeros(
        (batch_size, max_sequence_length), dtype=torch.bool
    )

    for row, graph in enumerate(graphs):
        num_nodes = graph.node_ids.numel()
        sequence_length = graph.alias_indices.numel()
        node_ids[row, :num_nodes] = graph.node_ids
        alias_indices[row, :sequence_length] = graph.alias_indices
        sequence_mask[row, :sequence_length] = True

        adjacency[row, :num_nodes, :num_nodes] = graph.adjacency[:, :num_nodes]
        adjacency[row, :num_nodes, max_nodes : max_nodes + num_nodes] = (
            graph.adjacency[:, num_nodes:]
        )

    return SessionGraphBatch(
        node_ids=node_ids,
        adjacency=adjacency,
        alias_indices=alias_indices,
        sequence_mask=sequence_mask,
        targets=torch.tensor(targets, dtype=torch.long),
    )
