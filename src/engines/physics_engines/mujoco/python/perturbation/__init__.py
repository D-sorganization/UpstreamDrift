"""MuJoCo perturbation analysis package.

Provides the ``MuJoCoPerturbationAnalyzer`` for injecting polynomial torque
perturbations into a MuJoCo simulation and collecting metrics.
"""

from src.shared.python.perturbation.analyzer_base import ComparisonReport

from .analyzer import MuJoCoPerturbationAnalyzer, MuJoCoSimResult

__all__ = [
    "ComparisonReport",
    "MuJoCoPerturbationAnalyzer",
    "MuJoCoSimResult",
]
