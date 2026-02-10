"""Backward compatibility shim - module moved to validation_pkg.statistical_analysis."""

import sys as _sys

from .validation_pkg import statistical_analysis as _real_module  # noqa: E402
from .validation_pkg.statistical_analysis import (  # noqa: F401
    AngularMomentumMetrics,
    CoordinationMetrics,
    GRFMetrics,
    ImpulseMetrics,
    JerkMetrics,
    JointPowerMetrics,
    JointStiffnessMetrics,
    KinematicSequenceInfo,
    PCAResult,
    PeakInfo,
    RQAMetrics,
    StabilityMetrics,
    StatisticalAnalyzer,
    SummaryStatistics,
    SwingPhase,
    SwingProfileMetrics,
)

_sys.modules[__name__] = _real_module
