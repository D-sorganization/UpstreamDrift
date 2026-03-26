"""Perturbation analysis package for upstream drift.

Unified platform utilities for conducting Monte Carlo sensitivity
analysis on engine dynamics.
"""

from .config import PerturbationAnalyzer, PerturbationConfig, PerturbationSummary
from .noise import generate_noise
from .robustness_score import compute_robustness_score
from .statistics import MetricStatistics, compute_metric_statistics

__all__ = [
    "PerturbationAnalyzer",
    "PerturbationConfig",
    "PerturbationSummary",
    "generate_noise",
    "compute_robustness_score",
    "MetricStatistics",
    "compute_metric_statistics",
]
