"""Drake perturbation analysis package.

Provides the ``DrakePerturbationAnalyzer`` for injecting polynomial torque
perturbations into a Drake ``MultibodyPlant`` and collecting simulation metrics.
"""

from .analyzer import ComparisonReport, DrakePerturbationAnalyzer, DrakeSimResult

__all__ = [
    "ComparisonReport",
    "DrakePerturbationAnalyzer",
    "DrakeSimResult",
]
