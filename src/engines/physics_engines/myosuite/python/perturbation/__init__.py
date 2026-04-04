"""MyoSuite perturbation analysis package.

Provides the ``MyoSuitePerturbationAnalyzer`` for injecting polynomial torque
perturbations into a MyoSuite simulation and collecting metrics.
"""

from src.shared.python.perturbation.analyzer_base import ComparisonReport

from .analyzer import MyoSuitePerturbationAnalyzer, MyoSuiteSimResult

__all__ = [
    "ComparisonReport",
    "MyoSuitePerturbationAnalyzer",
    "MyoSuiteSimResult",
]
