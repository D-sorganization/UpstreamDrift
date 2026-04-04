"""Pinocchio perturbation analysis package.

Provides the ``PinocchioPerturbationAnalyzer`` for injecting polynomial torque
perturbations into a Pinocchio simulation and collecting metrics.
"""

from .analyzer import (
    ComparisonReport,
    PinocchioPerturbationAnalyzer,
    PinocchioSimResult,
)

__all__ = [
    "ComparisonReport",
    "PinocchioPerturbationAnalyzer",
    "PinocchioSimResult",
]
