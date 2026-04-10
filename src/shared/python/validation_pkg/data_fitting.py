"""Public facade for the A3 data-fitting pipeline.

The implementation now lives in focused helper modules for:
- shared datamodels
- inverse kinematics
- parameter estimation
- sensitivity analysis
- pipeline orchestration
"""

from __future__ import annotations

from .data_fitting_inverse_kinematics import InverseKinematicsSolver
from .data_fitting_models import (
    BodySegmentParams,
    FitResult,
    KinematicState,
    ParameterEstimationReport,
    SensitivityResult,
)
from .data_fitting_parameter_estimator import ParameterEstimator
from .data_fitting_pipeline import A3FittingPipeline, convert_poses_to_markers
from .data_fitting_sensitivity import SensitivityAnalyzer

__all__ = [
    "A3FittingPipeline",
    "BodySegmentParams",
    "FitResult",
    "InverseKinematicsSolver",
    "KinematicState",
    "ParameterEstimationReport",
    "ParameterEstimator",
    "SensitivityAnalyzer",
    "SensitivityResult",
    "convert_poses_to_markers",
]
