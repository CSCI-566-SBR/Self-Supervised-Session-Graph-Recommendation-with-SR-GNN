"""Model and data collator registry for recommendation architectures."""

from __future__ import annotations

from typing import Any, Callable
import torch.nn as nn

from recsys.data.processed import ProcessedDataset

ModelBuilder = Callable[
    [dict[str, Any], ProcessedDataset],
    tuple[nn.Module, Callable[[list[Any]], Any]],
]

_MODEL_REGISTRY: dict[str, ModelBuilder] = {}


def register_model(name: str) -> Callable[[ModelBuilder], ModelBuilder]:
    """Decorator to register a model builder factory."""

    def decorator(builder: ModelBuilder) -> ModelBuilder:
        _MODEL_REGISTRY[name.lower()] = builder
        return builder

    return decorator


def _ensure_defaults_registered() -> None:
    if "srgnn" not in _MODEL_REGISTRY or "sasrec" not in _MODEL_REGISTRY:
        import recsys.models.baselines.sasrec  # noqa: F401
        import recsys.models.baselines.srgnn  # noqa: F401


def get_model_builder(name: str) -> ModelBuilder:
    """Retrieve the registered builder for the given model architecture."""

    _ensure_defaults_registered()
    key = name.lower()
    if key not in _MODEL_REGISTRY:
        available = ", ".join(repr(k) for k in sorted(_MODEL_REGISTRY.keys()))
        raise ValueError(
            f"unknown model architecture {name!r}; available models: [{available}]"
        )
    return _MODEL_REGISTRY[key]


def list_registered_models() -> list[str]:
    """Return all currently registered model architecture names."""

    _ensure_defaults_registered()
    return sorted(_MODEL_REGISTRY.keys())
