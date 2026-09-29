"""OpenSim perturbation analysis package.

Provides the ``OpenSimPerturbationAnalyzer`` for injecting polynomial torque
perturbations into an OpenSim simulation and collecting metrics.
"""

from .analyzer import ComparisonReport, OpenSimPerturbationAnalyzer, OpenSimSimResult

__all__ = [
    "ComparisonReport",
    "OpenSimPerturbationAnalyzer",
    "OpenSimSimResult",
]
