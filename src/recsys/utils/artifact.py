"""Artifact validation and results serialization helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from recsys.data.processed import ProcessedDataset
    from recsys.training.baseline import RankingResults
    from recsys.training.config import ExperimentConfig


def format_results_dict(results: RankingResults, k: int) -> dict[str, Any]:
    """Format RankingResults into a serializable metric dictionary."""
    output = dict(results.metrics)
    output["loss"] = results.val_loss
    output["examples"] = results.examples
    return output


def validate_artifact_identity(
    config: ExperimentConfig, dataset: ProcessedDataset
) -> None:
    """Validate dataset protocol and checksums against experiment configuration."""
    actual_protocol = dataset.summary.get("protocol_version")
    if (
        config.expected_protocol_version is not None
        and actual_protocol != config.expected_protocol_version
    ):
        raise ValueError(
            "processed dataset protocol does not match the experiment configuration: "
            f"expected {config.expected_protocol_version!r}, got {actual_protocol!r}"
        )
    if (
        config.expected_checksums is not None
        and dataset.checksums != config.expected_checksums
    ):
        raise ValueError("processed dataset checksums do not match the frozen artifact")


def validate_robustness_artifacts(
    dataset: ProcessedDataset,
    meta: dict[str, Any],
    checkpoint_state: dict[str, Any] | None = None,
) -> None:
    """Validate dataset protocol, checksums, and checkpoint vocabulary compatibility.

    Raises:
        ValueError: If protocol version, checksums, or vocabulary dimensions mismatch.
    """
    cfg = meta.get("config", {})

    expected_protocol = cfg.get("expected_protocol_version") or meta.get("dataset_artifact", {}).get("protocol_version")
    actual_protocol = dataset.summary.get("protocol_version") if hasattr(dataset, "summary") and dataset.summary else None
    if expected_protocol is not None and actual_protocol != expected_protocol:
        raise ValueError(
            f"Dataset protocol mismatch: expected {expected_protocol!r}, got {actual_protocol!r}"
        )

    expected_checksums = cfg.get("expected_checksums") or meta.get("dataset_artifact", {}).get("checksums")
    if expected_checksums is not None and hasattr(dataset, "checksums") and dataset.checksums:
        for file_key, expected_hash in expected_checksums.items():
            actual_hash = dataset.checksums.get(file_key)
            if actual_hash is not None and actual_hash != expected_hash:
                raise ValueError(
                    f"Dataset artifact checksum mismatch for {file_key}: expected {expected_hash}, got {actual_hash}"
                )

    if checkpoint_state is not None:
        embedding_weight = checkpoint_state.get("embedding.weight")
        if embedding_weight is None:
            embedding_weight = checkpoint_state.get("item_embedding.weight")

        if embedding_weight is not None:
            checkpoint_vocab_size = embedding_weight.shape[0]
            expected_vocab_size = dataset.num_items + 1
            if checkpoint_vocab_size != expected_vocab_size:
                raise ValueError(
                    f"Vocabulary size mismatch: clean dataset has {dataset.num_items} items "
                    f"({expected_vocab_size} embedding rows), but checkpoint expects {checkpoint_vocab_size} rows."
                )
