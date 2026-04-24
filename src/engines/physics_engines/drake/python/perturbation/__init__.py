"""Drake perturbation analysis package.

Provides the ``DrakePerturbationAnalyzer`` for injecting polynomial torque
perturbations into a Drake ``MultibodyPlant`` and collecting simulation metrics.
"""

from src.shared.python.perturbation.analyzer_base import ComparisonReport

from .analyzer import DrakePerturbationAnalyzer, DrakeSimResult

__all__ = [
    "ComparisonReport",
    "DrakePerturbationAnalyzer",
    "DrakeSimResult",
]
