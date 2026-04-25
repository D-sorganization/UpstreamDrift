"""Data fitting and parameter estimation for golf biomechanics (Guideline A3).

This module implements the A3 pipeline per project design guidelines:
- Fit kinematics to observed trajectories
- Estimate body segment parameters (lengths, mass, inertia)
- Report sensitivity analysis and fit quality metrics

Issue #754: Implements complete A3 model fitting and parameter identification.
Issue #3060: Split into sub-modules; this file re-exports the public API for
             backward compatibility.

  Sub-modules:
    _data_fitting_models   — data structures / dataclasses
    _data_fitting_solvers  — InverseKinematicsSolver, ParameterEstimator
    fitting_pipeline       — SensitivityAnalyzer, A3FittingPipeline
    ui/data_fitting_widget — convert_poses_to_markers (GUI-facing helper)

Reference: docs/assessments/project_design_guidelines.qmd Section A3
"""

from __future__ import annotations

# Re-export data structures
from ._data_fitting_models import (
    BodySegmentParams,
    FitResult,
    KinematicState,
    ParameterEstimationReport,
    SensitivityResult,
)

# Re-export numerical solvers
from ._data_fitting_solvers import (
    InverseKinematicsSolver,
    ParameterEstimator,
)

# Re-export pipeline and sensitivity analysis
from .fitting_pipeline import (
    A3FittingPipeline,
    SensitivityAnalyzer,
)

# Re-export GUI-facing helper (kept here for backward compatibility)
from .ui.data_fitting_widget import convert_poses_to_markers

__all__ = [
    # Data structures
    "BodySegmentParams",
    "FitResult",
    "KinematicState",
    "ParameterEstimationReport",
    "SensitivityResult",
    # Numerical solvers
    "InverseKinematicsSolver",
    "ParameterEstimator",
    # Pipeline
    "A3FittingPipeline",
    "SensitivityAnalyzer",
    # GUI-facing helper
    "convert_poses_to_markers",
]
