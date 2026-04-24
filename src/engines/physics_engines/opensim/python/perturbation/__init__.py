"""OpenSim perturbation analysis package.

Provides the ``OpenSimPerturbationAnalyzer`` for injecting polynomial torque
perturbations into an OpenSim simulation and collecting metrics.
"""

from src.shared.python.perturbation.analyzer_base import ComparisonReport

from .analyzer import OpenSimPerturbationAnalyzer, OpenSimSimResult

__all__ = [
    "ComparisonReport",
    "OpenSimPerturbationAnalyzer",
    "OpenSimSimResult",
]
