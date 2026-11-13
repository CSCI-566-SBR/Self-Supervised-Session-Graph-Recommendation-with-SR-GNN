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
