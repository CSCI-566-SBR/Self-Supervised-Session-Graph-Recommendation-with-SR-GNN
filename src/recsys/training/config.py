"""Minimal TOML configuration for reproducible baseline experiments."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    dataset: str
    processed_data_dir: Path
    output_dir: Path
    model_type: str = "srgnn"
    model_params: dict[str, Any] = field(default_factory=dict)
    artifact_repository: str | None = None
    artifact_revision: str | None = None
    expected_protocol_version: str | None = None
    expected_checksums: dict[str, str] | None = None
    # Training settings
    batch_size: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    max_epochs: int = 30
    patience: int = 10
    lr_decay_step: int = 3
    lr_decay_gamma: float = 0.1
    seed: int = 2028
    device: str = "auto"
    deterministic: bool = False
    num_workers: int | None = None
    metric_k: int = 20
    metric_cutoffs: list[int] = field(default_factory=lambda: [10, 20])
    # Kept for backward compatibility with existing tests and scripts
    hidden_size: int = 100

    def __post_init__(self) -> None:
        if "hidden_size" not in self.model_params and self.hidden_size is not None:
            params = dict(self.model_params)
            params["hidden_size"] = self.hidden_size
            object.__setattr__(self, "model_params", params)

    def as_serializable_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["processed_data_dir"] = str(self.processed_data_dir)
        values["output_dir"] = str(self.output_dir)
        return values


def _require_table(document: dict[str, Any], name: str) -> dict[str, Any]:
    value = document.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"configuration requires a [{name}] table")
    return value


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Read and validate experiment configuration from a TOML document."""

    path = Path(path)
    with path.open("rb") as handle:
        document = tomllib.load(handle)

    experiment = _require_table(document, "experiment")
    data = _require_table(document, "data")
    model = _require_table(document, "model")
    training = _require_table(document, "training")

    model_type = str(model.get("type", "srgnn")).lower()
    model_params = {k: v for k, v in model.items() if k != "type"}
    hidden_size = model_params.get("hidden_size", 100)

    try:
        config = ExperimentConfig(
            name=experiment["name"],
            dataset=experiment["dataset"],
            processed_data_dir=Path(data["processed_data_dir"]),
            output_dir=Path(experiment["output_dir"]),
            model_type=model_type,
            model_params=model_params,
            hidden_size=hidden_size,
            artifact_repository=data.get("artifact_repository"),
            artifact_revision=data.get("artifact_revision"),
            expected_protocol_version=data.get("expected_protocol_version"),
            expected_checksums=data.get("expected_checksums"),
            batch_size=training.get("batch_size", 100),
            learning_rate=training.get("learning_rate", 0.001),
            weight_decay=training.get("weight_decay", 1e-5),
            max_epochs=training.get("max_epochs", 30),
            patience=training.get("patience", 10),
            lr_decay_step=training.get("lr_decay_step", 3),
            lr_decay_gamma=training.get("lr_decay_gamma", 0.1),
            seed=training.get("seed", 2028),
            device=training.get("device", "auto"),
            deterministic=training.get("deterministic", False),
            num_workers=training.get("num_workers"),
            metric_k=training.get("metric_k", 20),
            metric_cutoffs=training.get("metric_cutoffs", [10, 20]),
        )
    except KeyError as error:
        raise ValueError(f"missing required configuration value: {error.args[0]}") from error

    _validate_config(config)
    return config


def _validate_config(config: ExperimentConfig) -> None:
    if not config.name or not config.dataset:
        raise ValueError("experiment name and dataset must be non-empty")
    if not config.model_type:
        raise ValueError("model type must be non-empty")

    positive_integers = {
        "batch_size": config.batch_size,
        "max_epochs": config.max_epochs,
        "patience": config.patience,
        "lr_decay_step": config.lr_decay_step,
        "metric_k": config.metric_k,
    }
    for name, value in positive_integers.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    if not config.metric_cutoffs or any(
        isinstance(c, bool) or not isinstance(c, int) or c <= 0 for c in config.metric_cutoffs
    ):
        raise ValueError("metric_cutoffs must be a non-empty list of positive integers")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if config.weight_decay < 0:
        raise ValueError("weight_decay cannot be negative")
    if not 0 < config.lr_decay_gamma <= 1:
        raise ValueError("lr_decay_gamma must lie in (0, 1]")
    if isinstance(config.seed, bool) or not isinstance(config.seed, int) or config.seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if config.device not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("device must be auto, cpu, cuda, or mps")
    if bool(config.artifact_repository) != bool(config.artifact_revision):
        raise ValueError(
            "artifact_repository and artifact_revision must be provided together"
        )
    if config.expected_checksums is not None:
        expected_names = {"train", "validation", "test", "item_mapping", "summary"}
        if set(config.expected_checksums) != expected_names:
            raise ValueError(
                "expected_checksums must contain train, validation, test, "
                "item_mapping, and summary"
            )
        for digest in config.expected_checksums.values():
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError("expected checksums must be 64-character SHA-256 values")
            try:
                int(digest, 16)
            except ValueError as error:
                raise ValueError("expected checksums must be hexadecimal") from error
