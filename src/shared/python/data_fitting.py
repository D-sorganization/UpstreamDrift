"""Backward compatibility shim - module moved to validation_pkg.data_fitting."""

import sys as _sys

from .validation_pkg import data_fitting as _real_module  # noqa: E402
from .validation_pkg.data_fitting import (  # noqa: F401
    A3FittingPipeline,
    BodySegmentParams,
    FitResult,
    InverseKinematicsSolver,
    KinematicState,
    ParameterEstimationReport,
    ParameterEstimator,
    SensitivityAnalyzer,
    SensitivityResult,
    convert_poses_to_markers,
    logger,
)

_sys.modules[__name__] = _real_module
