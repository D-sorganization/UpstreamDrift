"""Pinocchio perturbation analysis package.

Provides the ``PinocchioPerturbationAnalyzer`` for injecting polynomial torque
perturbations into a Pinocchio simulation and collecting metrics.
"""

from src.shared.python.perturbation.analyzer_base import ComparisonReport

from .analyzer import (
    PinocchioPerturbationAnalyzer,
    PinocchioSimResult,
)

__all__ = [
    "ComparisonReport",
    "PinocchioPerturbationAnalyzer",
    "PinocchioSimResult",
]
