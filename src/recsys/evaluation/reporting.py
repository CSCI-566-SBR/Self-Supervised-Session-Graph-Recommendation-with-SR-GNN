"""Consistent text formatting for whichever ranking metrics a caller selects."""

from collections.abc import Mapping
from math import isfinite
from numbers import Real


def format_results(results: Mapping[str, float]) -> str:
    """Format supplied metric fractions as percentages, preserving their order.

    Returns one 'name: 12.34%' line per metric, without printing or adding a
    trailing newline. An empty mapping returns an empty string. Values must
    be finite numbers between 0 and 1; otherwise raises ValueError.
    """
    lines = []
    for name, value in results.items():
        if not isinstance(value, Real) or not isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be a finite metric value between 0 and 1")
        lines.append(f"{name}: {value:.2%}")
    return "\n".join(lines)
