"""Baseline recommendation models."""

from .srgnn import GatedGraphNeuralNetwork, SRGNN

__all__ = ["GatedGraphNeuralNetwork", "SRGNN"]
