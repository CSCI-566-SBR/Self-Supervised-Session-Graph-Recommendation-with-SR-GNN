"""SR-GNN SSL extension with shared augmented-view batching and loss."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import torch
import math
from torch import Tensor, nn
from torch.nn import functional as F
from recsys.data.session_graph import (
    SessionGraphBatch,
    collate_session_graphs,
)
from recsys.data.ssl_graph import SSLGraphBatch, SSLGraphCollator
from recsys.models.baselines.srgnn import SRGNN, build_srgnn
from recsys.models.registry import register_model
from recsys.training.baseline import TrainingMetric, recommendation_loss

if TYPE_CHECKING:
    from recsys.data.processed import ProcessedDataset



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
        budget_rounding: str = "nearest",
    ) -> None:
        super().__init__()

        if augmentation_mode not in {"uniform_edge", "recency"}:
            raise ValueError(
                "augmentation_mode must be uniform_edge or recency"
            )
        if budget_rounding not in {"nearest", "stochastic"}:
            raise ValueError(
                "budget_rounding must be nearest or stochastic"
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
        self.budget_rounding = budget_rounding
        self.deletion_budget = deletion_budget
        self.ssl_weight = ssl_weight
        self.temperature = temperature
        self.recency_gamma = recency_gamma
        self.augmentation_seed = augmentation_seed


    def encode(self, batch: SessionGraphBatch) -> Tensor:
        """Return session embeddings using the existing SR-GNN encoder."""
        return self.encoder.encode(batch)

    def forward(
        self,
        batch: SessionGraphBatch | SSLGraphBatch,
    ) -> Tensor:
        """Predict next items without generating augmented views."""
        clean = batch.clean if isinstance(batch, SSLGraphBatch) else batch
        return self.encoder(clean)

    def make_training_collator(self) -> Callable:
        """Provide CPU-side training collation without changing evaluation."""
        if self.ssl_weight == 0:
            return collate_session_graphs

        return SSLGraphCollator(
            mode=self.augmentation_mode,
            budget=self.deletion_budget,
            gamma=self.recency_gamma,
            seed=self.augmentation_seed,
            budget_rounding=self.budget_rounding,
        )

    def training_loss(
        self,
        batch: SessionGraphBatch | SSLGraphBatch,
    ) -> tuple[Tensor, dict[str, TrainingMetric]]:
        """Return the joint loss and generic metric contributions."""
        if self.ssl_weight > 0 and not isinstance(batch, SSLGraphBatch):
            raise TypeError(
                "SSL training requires batches from make_training_collator()"
            )

        clean = batch.clean if isinstance(batch, SSLGraphBatch) else batch
        examples = clean.targets.shape[0]

        rec_loss = recommendation_loss(
            self.encoder(clean),
            clean.targets,
        )
        metrics = {
            "recommendation_loss": TrainingMetric(
                rec_loss.detach() * examples,
                examples,
            ),
            "ssl_loss": TrainingMetric(0.0, examples),
        }

        if self.ssl_weight == 0:
            return rec_loss, metrics

        ssl_loss = session_contrastive_loss(
            self.encode(batch.first_view),
            self.encode(batch.second_view),
            temperature=self.temperature,
        )

        metrics.update({
            "ssl_loss": TrainingMetric(
                ssl_loss.detach() * examples,
                examples,
            ),
            "augmentation_original_transitions": TrainingMetric(
                batch.original_transitions
            ),
            "augmentation_deleted_transitions": TrainingMetric(
                batch.deleted_transitions
            ),
            "augmentation_prefixes": TrainingMetric(
                batch.prefixes
            ),
            "augmentation_zero_deletion_prefixes": TrainingMetric(
                batch.zero_deletion_prefixes
            ),
            "realized_deletion_rate": TrainingMetric(
                batch.deleted_transitions,
                batch.original_transitions,
            ),
            "zero_deletion_prefix_percentage": TrainingMetric(
                100.0 * batch.zero_deletion_prefixes,
                batch.prefixes,
            ),
        })

        return rec_loss + self.ssl_weight * ssl_loss, metrics
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
        "budget_rounding",
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
        budget_rounding=params.get("budget_rounding", "nearest"),
    )

    return model, collate_fn