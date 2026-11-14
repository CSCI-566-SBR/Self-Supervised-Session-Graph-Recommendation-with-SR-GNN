"""Recommendation model implementations and architectural registry."""

from __future__ import annotations

from .registry import get_model_builder, list_registered_models, register_model

__all__ = [
    "get_model_builder",
    "list_registered_models",
    "register_model",
]
