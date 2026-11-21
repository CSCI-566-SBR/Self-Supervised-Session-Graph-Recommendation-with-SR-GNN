"""SR-GNN SSL extension with shared augmented-view batching and loss."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Sequence

import torch
import math
import random
from torch import Tensor, nn
from torch.nn import functional as F
from recsys.data.graph_augmentations import (
    GraphView,
    augment_budgeted_edge_view,
    augment_session,
)
from recsys.data.session_graph import (
    SessionGraphBatch,
    collate_session_graphs,
)
from recsys.models.baselines.srgnn import SRGNN, build_srgnn
from recsys.models.registry import register_model
from recsys.training.baseline import recommendation_loss

if TYPE_CHECKING:
    from recsys.data.processed import ProcessedDataset


def collate_graph_views(
    examples: Sequence[tuple[GraphView, int]],
) -> SessionGraphBatch:
    """Create SR-GNN batches using only each view's retained edges.

    Reuse baseline node ordering, aliases, padding, and target validation,
    but replace its sequence-derived adjacency with view-derived adjacency.
    Repeated directed edges remain binary, matching the baseline.
    """
    if not examples:
        raise ValueError("cannot collate an empty batch")

    batch = collate_session_graphs(
        [(view.sequence, target) for view, target in examples]
    )
    adjacency = torch.zeros_like(batch.adjacency)
    max_nodes = batch.node_ids.shape[1]

    for row, (view, _) in enumerate(examples):
        nodes = sorted(set(view.sequence))
        if set(view.nodes) != set(nodes):
            raise ValueError("view nodes must match its sequence items")

        node_index = {
            item: index for index, item in enumerate(nodes)
        }
        num_nodes = len(nodes)
        transitions = torch.zeros(
            (num_nodes, num_nodes),
            dtype=torch.float32,
        )

        # Use retained edges, not consecutive items in view.sequence.
        for edge in view.edges:
            if (
                edge.source not in node_index
                or edge.target not in node_index
            ):
                raise ValueError("view edge references a missing node")

            transitions[
                node_index[edge.source],
                node_index[edge.target],
            ] = 1.0

        incoming_degree = transitions.sum(dim=0).clamp_min(1.0)
        incoming = (
            transitions / incoming_degree
        ).transpose(0, 1)

        outgoing_degree = transitions.sum(dim=1).clamp_min(1.0)
        outgoing = transitions / outgoing_degree.unsqueeze(1)

        adjacency[
            row, :num_nodes, :num_nodes
        ] = incoming
        adjacency[
            row, :num_nodes, max_nodes:max_nodes + num_nodes
        ] = outgoing

    return SessionGraphBatch(
        node_ids=batch.node_ids,
        adjacency=adjacency,
        alias_indices=batch.alias_indices,
        sequence_mask=batch.sequence_mask,
        targets=batch.targets,
    )

def session_contrastive_loss(
    first_view: Tensor,
    second_view: Tensor,
    *,
    temperature: float = 0.2,
) -> Tensor:
    """Compute one-way, batch-mean InfoNCE over paired session views.

    Matching rows are positives; other rows are in-batch negatives.
    No same-session negative masking is applied.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")

    if first_view.ndim != 2 or second_view.ndim != 2:
        raise ValueError(
            "view embeddings must have shape [batch_size, hidden_size]"
        )

    if first_view.shape != second_view.shape:
        raise ValueError("both views must have the same shape")

    batch_size, hidden_size = first_view.shape
    if batch_size == 0 or hidden_size == 0:
        raise ValueError("view embeddings must not be empty")

    if first_view.device != second_view.device:
        raise ValueError("both views must be on the same device")

    if first_view.dtype != second_view.dtype:
        raise ValueError("both views must have the same dtype")

    if not (
        first_view.is_floating_point()
        and second_view.is_floating_point()
    ):
        raise ValueError("view embeddings must be floating-point tensors")

    first_normalized = F.normalize(first_view, p=2, dim=1)
    second_normalized = F.normalize(second_view, p=2, dim=1)

    logits = (
        first_normalized @ second_normalized.transpose(0, 1)
    ) / temperature

    positive_indices = torch.arange(
        batch_size,
        device=logits.device,
    )

    return F.cross_entropy(
        logits,
        positive_indices,
        reduction="mean",
    )

class SRGNNSSL(nn.Module):
    """Reuse SR-GNN with an auxiliary session-contrastive training loss."""

    def __init__(
        self,
        encoder: SRGNN,
        *,
        augmentation_mode: str = "uniform_edge",
        deletion_budget: float = 0.1,
        ssl_weight: float = 0.01,
        temperature: float = 0.2,
        recency_gamma: float = 2.0,
        augmentation_seed: int = 2026,
    ) -> None:
        super().__init__()

        if augmentation_mode not in {"uniform_edge", "recency", "uniform_node"}:
            raise ValueError(
                "augmentation_mode must be uniform_edge, recency, or uniform_node"
            )
        if not 0.0 <= deletion_budget <= 1.0:
            raise ValueError("deletion_budget must lie in [0, 1]")
        if not math.isfinite(ssl_weight) or ssl_weight < 0:
            raise ValueError("ssl_weight must be finite and non-negative")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not math.isfinite(recency_gamma) or recency_gamma < 1:
            raise ValueError("recency_gamma must be finite and at least one")
        if (
            isinstance(augmentation_seed, bool)
            or not isinstance(augmentation_seed, int)
            or augmentation_seed < 0
        ):
            raise ValueError(
                "augmentation_seed must be a non-negative integer"
            )

        self.encoder = encoder
        self.augmentation_mode = augmentation_mode
        self.deletion_budget = deletion_budget
        self.ssl_weight = ssl_weight
        self.temperature = temperature
        self.recency_gamma = recency_gamma
        self._augmentation_rng = random.Random(augmentation_seed)

        self.last_training_metrics: dict[str, float | int] = {}

    def encode(self, batch: SessionGraphBatch) -> Tensor:
        """Return session embeddings using the existing SR-GNN encoder."""
        return self.encoder.encode(batch)

    def forward(self, batch: SessionGraphBatch) -> Tensor:
        """Predict next items without generating augmented views."""
        return self.encoder(batch)

    def training_loss(self, batch: SessionGraphBatch) -> Tensor:
        """Combine clean recommendation loss with dual-view InfoNCE."""
        rec_loss = recommendation_loss(
            self(batch),
            batch.targets,
        )

        self.last_training_metrics = {
            "recommendation_loss": rec_loss.detach().item(),
            "ssl_loss": 0.0,
            "augmentation_original_transitions": 0,
            "augmentation_deleted_transitions": 0,
        }

        # Preserve ordinary recommendation training when SSL is disabled.
        if self.ssl_weight == 0:
            return rec_loss

        # Recover only observed prefix items; targets are never added.
        sequence_ids = batch.node_ids.gather(
            1, batch.alias_indices
        ).detach().cpu()
        masks = batch.sequence_mask.detach().cpu()
        targets = batch.targets.detach().cpu().tolist()

        first_examples = []
        second_examples = []
        original_transitions = 0
        deleted_transitions = 0

        for row, mask, target in zip(sequence_ids, masks, targets):
            prefix = row[mask].tolist()

            if self.augmentation_mode == "uniform_node":
                first_view = augment_session(
                    prefix,
                    mode="uniform_node",
                    probability=self.deletion_budget,
                    rng=self._augmentation_rng,
                )
                second_view = augment_session(
                    prefix,
                    mode="uniform_node",
                    probability=self.deletion_budget,
                    rng=self._augmentation_rng,
                )
            else:
                first_view = augment_budgeted_edge_view(
                    prefix,
                    mode=self.augmentation_mode,
                    budget=self.deletion_budget,
                    rng=self._augmentation_rng,
                    gamma=self.recency_gamma,
                )
                second_view = augment_budgeted_edge_view(
                    prefix,
                    mode=self.augmentation_mode,
                    budget=self.deletion_budget,
                    rng=self._augmentation_rng,
                    gamma=self.recency_gamma,
                )

            first_examples.append((first_view, target))
            second_examples.append((second_view, target))

            transitions_per_view = len(prefix) - 1
            original_transitions += 2 * transitions_per_view
            deleted_transitions += (
                2 * transitions_per_view
                - len(first_view.edges)
                - len(second_view.edges)
            )

        device = batch.node_ids.device
        first_batch = collate_graph_views(first_examples).to(device)
        second_batch = collate_graph_views(second_examples).to(device)

        ssl_loss = session_contrastive_loss(
            self.encode(first_batch),
            self.encode(second_batch),
            temperature=self.temperature,
        )

        self.last_training_metrics.update({
            "ssl_loss": ssl_loss.detach().item(),
            "augmentation_original_transitions": original_transitions,
            "augmentation_deleted_transitions": deleted_transitions,
        })

        return rec_loss + self.ssl_weight * ssl_loss

@register_model("srgnn_ssl")
def build_srgnn_ssl(
    params: dict[str, Any],
    dataset: ProcessedDataset,
) -> tuple[nn.Module, Callable[[list[Any]], Any]]:
    """Build SSL using the existing validated SR-GNN builder."""
    encoder_keys = {"hidden_size", "gnn_steps", "hybrid"}
    ssl_keys = {
        "augmentation_mode",
        "deletion_budget",
        "ssl_weight",
        "temperature",
        "recency_gamma",
        "augmentation_seed",
    }

    unknown = set(params) - encoder_keys - ssl_keys
    if unknown:
        raise ValueError(
            "unknown SR-GNN SSL settings: " + ", ".join(sorted(unknown))
        )

    encoder_params = {
        key: params[key]
        for key in encoder_keys
        if key in params
    }
    encoder, collate_fn = build_srgnn(encoder_params, dataset)

    model = SRGNNSSL(
        encoder,
        augmentation_mode=params.get(
            "augmentation_mode", "uniform_edge"
        ),
        deletion_budget=params.get("deletion_budget", 0.1),
        ssl_weight=params.get("ssl_weight", 0.01),
        temperature=params.get("temperature", 0.2),
        recency_gamma=params.get("recency_gamma", 2.0),
        augmentation_seed=params.get("augmentation_seed", 2026),
    )

    return model, collate_fn