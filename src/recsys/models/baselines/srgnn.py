"""Faithful, modern PyTorch implementation of the SR-GNN baseline."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from recsys.data.session_graph import SessionGraphBatch


class GatedGraphNeuralNetwork(nn.Module):
    """Propagate incoming and outgoing messages with the SR-GNN gated update."""

    def __init__(self, hidden_size: int, steps: int = 1) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if steps <= 0:
            raise ValueError("steps must be positive")

        self.hidden_size = hidden_size
        self.steps = steps
        gate_size = 3 * hidden_size

        self.input_weights = nn.Parameter(torch.empty(gate_size, 2 * hidden_size))
        self.hidden_weights = nn.Parameter(torch.empty(gate_size, hidden_size))
        self.input_bias = nn.Parameter(torch.empty(gate_size))
        self.hidden_bias = nn.Parameter(torch.empty(gate_size))
        self.incoming_bias = nn.Parameter(torch.empty(hidden_size))
        self.outgoing_bias = nn.Parameter(torch.empty(hidden_size))
        self.incoming_projection = nn.Linear(hidden_size, hidden_size)
        self.outgoing_projection = nn.Linear(hidden_size, hidden_size)

    def _cell(self, adjacency: Tensor, hidden: Tensor) -> Tensor:
        num_nodes = adjacency.shape[1]
        incoming = (
            torch.matmul(
                adjacency[:, :, :num_nodes], self.incoming_projection(hidden)
            )
            + self.incoming_bias
        )
        outgoing = (
            torch.matmul(
                adjacency[:, :, num_nodes:], self.outgoing_projection(hidden)
            )
            + self.outgoing_bias
        )
        messages = torch.cat((incoming, outgoing), dim=2)

        input_gates = F.linear(messages, self.input_weights, self.input_bias)
        hidden_gates = F.linear(hidden, self.hidden_weights, self.hidden_bias)
        input_reset, input_update, input_candidate = input_gates.chunk(3, dim=2)
        hidden_reset, hidden_update, hidden_candidate = hidden_gates.chunk(3, dim=2)

        reset_gate = torch.sigmoid(input_reset + hidden_reset)
        update_gate = torch.sigmoid(input_update + hidden_update)
        candidate = torch.tanh(input_candidate + reset_gate * hidden_candidate)
        return candidate + update_gate * (hidden - candidate)

    def forward(self, adjacency: Tensor, hidden: Tensor) -> Tensor:
        for _ in range(self.steps):
            hidden = self._cell(adjacency, hidden)
        return hidden


class SRGNN(nn.Module):
    """Session-based recommendation with a gated session graph encoder."""

    def __init__(
        self,
        num_items: int,
        hidden_size: int = 100,
        gnn_steps: int = 1,
        hybrid: bool = True,
    ) -> None:
        super().__init__()
        if num_items <= 0:
            raise ValueError("num_items must be positive")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")

        self.num_items = num_items
        self.hidden_size = hidden_size
        self.hybrid = hybrid

        # ID zero is padding. Candidate items occupy rows 1..num_items.
        self.item_embedding = nn.Embedding(
            num_items + 1, hidden_size, padding_idx=0
        )
        self.gnn = GatedGraphNeuralNetwork(hidden_size, steps=gnn_steps)
        self.local_projection = nn.Linear(hidden_size, hidden_size)
        self.sequence_projection = nn.Linear(hidden_size, hidden_size)
        self.attention_projection = nn.Linear(hidden_size, 1, bias=False)
        self.hybrid_projection = nn.Linear(2 * hidden_size, hidden_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.hidden_size)
        for parameter in self.parameters():
            nn.init.uniform_(parameter, -bound, bound)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

    def encode(self, batch: SessionGraphBatch) -> Tensor:
        """Return one hybrid session representation per batch example."""

        node_hidden = self.item_embedding(batch.node_ids)
        node_hidden = self.gnn(batch.adjacency, node_hidden)

        gather_index = batch.alias_indices.unsqueeze(-1).expand(
            -1, -1, self.hidden_size
        )
        sequence_hidden = node_hidden.gather(1, gather_index)

        sequence_lengths = batch.sequence_mask.sum(dim=1)
        if torch.any(sequence_lengths == 0):
            raise ValueError("every session must contain at least one observed item")
        last_positions = sequence_lengths - 1
        local = sequence_hidden[
            torch.arange(sequence_hidden.shape[0], device=sequence_hidden.device),
            last_positions,
        ]

        query = self.local_projection(local).unsqueeze(1)
        keys = self.sequence_projection(sequence_hidden)
        attention = self.attention_projection(torch.sigmoid(query + keys))
        mask = batch.sequence_mask.unsqueeze(-1).to(sequence_hidden.dtype)
        global_preference = torch.sum(attention * sequence_hidden * mask, dim=1)

        if not self.hybrid:
            return global_preference
        return self.hybrid_projection(torch.cat((global_preference, local), dim=1))

    def forward(self, batch: SessionGraphBatch) -> Tensor:
        session_embeddings = self.encode(batch)
        candidate_embeddings = self.item_embedding.weight[1:]
        return torch.matmul(session_embeddings, candidate_embeddings.transpose(0, 1))
