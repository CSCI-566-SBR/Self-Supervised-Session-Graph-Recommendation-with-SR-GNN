"""Extract target ranks once, then calculate and report selected metrics."""

from .metrics import get_target_ranks, mrr_at_k, ndcg_at_k, recall_at_k
from .reporting import format_results
from .torch_ranking import get_target_ranks_tensor

__all__ = [
    "get_target_ranks",
    "get_target_ranks_tensor",
    "recall_at_k",
    "mrr_at_k",
    "ndcg_at_k",
    "format_results",
]
