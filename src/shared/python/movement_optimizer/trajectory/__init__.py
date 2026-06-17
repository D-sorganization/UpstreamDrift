# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Trajectory module extracted from monolith."""

from __future__ import annotations

from ..observability import MetricSample as MetricSample
from ..observability import metrics as metrics
from .cache import SolutionCache as SolutionCache
from .optimizer import TrajectoryOptimizer as TrajectoryOptimizer
from .optimizer_diagnostics import run_minimize as run_minimize
from .optimizer_diagnostics import run_single_start as run_single_start
from .optimizer_guess import build_bounds as build_bounds
from .optimizer_guess import build_initial_guess as build_initial_guess
from .optimizer_guess import build_perturbed_guess as build_perturbed_guess
from .optimizer_progress import ProgressTracker as ProgressTracker
from .optimizer_progress import detect_stall as detect_stall
from .optimizer_spline import build_splines as build_splines
from .optimizer_spline import eval_trajectory as eval_trajectory
from .result import CancelledError as CancelledError
from .result import OptimizationResult as OptimizationResult
from .result import ProgressReport as ProgressReport

__all__ = [
    "CancelledError",
    "MetricSample",
    "OptimizationResult",
    "ProgressReport",
    "ProgressTracker",
    "SolutionCache",
    "TrajectoryOptimizer",
    "build_bounds",
    "build_initial_guess",
    "build_perturbed_guess",
    "build_splines",
    "detect_stall",
    "eval_trajectory",
    "metrics",
    "run_minimize",
    "run_single_start",
]
