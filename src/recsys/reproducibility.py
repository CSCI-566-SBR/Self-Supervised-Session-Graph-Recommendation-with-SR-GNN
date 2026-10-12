"""Random-seed handling shared by baseline experiments."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_random_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch for a repeatable experiment run."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(deterministic)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic
