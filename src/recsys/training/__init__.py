"""Model training harnesses, configuration, and execution utilities."""

from .baseline import (
    RankingResults,
    evaluate_ranking,
    recommendation_loss,
    train_epoch,
)
from .config import ExperimentConfig, load_experiment_config
from .runner import run_experiment
from .utils import (
    format_results_dict,
    get_git_commit,
    make_data_loader,
    prepare_output_dir,
    resolve_device,
    resolve_num_workers,
    validate_artifact_identity,
)

__all__ = [
    "ExperimentConfig",
    "RankingResults",
    "evaluate_ranking",
    "format_results_dict",
    "get_git_commit",
    "load_experiment_config",
    "make_data_loader",
    "prepare_output_dir",
    "recommendation_loss",
    "resolve_device",
    "resolve_num_workers",
    "run_experiment",
    "train_epoch",
    "validate_artifact_identity",
]
