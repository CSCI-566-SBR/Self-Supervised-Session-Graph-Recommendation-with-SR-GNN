"""Extract target ranks once, then calculate, report, and plot selected metrics."""

from .metrics import get_target_ranks, mrr_at_k, ndcg_at_k, recall_at_k
from .plotting import (
    load_experiment_history,
    load_robustness_results,
    plot_experiment_curves,
    plot_robustness_bars,
    plot_robustness_curves,
)
from .reporting import format_results
from .torch_ranking import get_target_ranks_tensor

__all__ = [
    "format_results",
    "get_target_ranks",
    "get_target_ranks_tensor",
    "load_experiment_history",
    "load_robustness_results",
    "mrr_at_k",
    "ndcg_at_k",
    "plot_experiment_curves",
    "plot_robustness_bars",
    "plot_robustness_curves",
    "recall_at_k",
]
