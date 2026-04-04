"""MuJoCo perturbation analysis package.

Provides the ``MuJoCoPerturbationAnalyzer`` for injecting polynomial torque
perturbations into a MuJoCo simulation and collecting metrics.
"""

from .analyzer import ComparisonReport, MuJoCoPerturbationAnalyzer, MuJoCoSimResult

__all__ = [
    "ComparisonReport",
    "MuJoCoPerturbationAnalyzer",
    "MuJoCoSimResult",
]
