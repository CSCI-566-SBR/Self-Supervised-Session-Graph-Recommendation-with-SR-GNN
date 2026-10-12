"""Minimal TOML configuration for reproducible SR-GNN experiments."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    dataset: str
    processed_data_dir: Path
    output_dir: Path
    hidden_size: int = 100
    gnn_steps: int = 1
    hybrid: bool = True
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
    metric_k: int = 20

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
    """Read and validate the small set of settings needed by the baseline."""

    path = Path(path)
    with path.open("rb") as handle:
        document = tomllib.load(handle)

    experiment = _require_table(document, "experiment")
    data = _require_table(document, "data")
    model = _require_table(document, "model")
    training = _require_table(document, "training")

    try:
        config = ExperimentConfig(
            name=experiment["name"],
            dataset=experiment["dataset"],
            processed_data_dir=Path(data["processed_data_dir"]),
            output_dir=Path(experiment["output_dir"]),
            hidden_size=model.get("hidden_size", 100),
            gnn_steps=model.get("gnn_steps", 1),
            hybrid=model.get("hybrid", True),
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
            metric_k=training.get("metric_k", 20),
        )
    except KeyError as error:
        raise ValueError(f"missing required configuration value: {error.args[0]}") from error

    _validate_config(config)
    return config


def _validate_config(config: ExperimentConfig) -> None:
    if not config.name or not config.dataset:
        raise ValueError("experiment name and dataset must be non-empty")
    positive_integers = {
        "hidden_size": config.hidden_size,
        "gnn_steps": config.gnn_steps,
        "batch_size": config.batch_size,
        "max_epochs": config.max_epochs,
        "patience": config.patience,
        "lr_decay_step": config.lr_decay_step,
        "metric_k": config.metric_k,
    }
    for name, value in positive_integers.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
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
