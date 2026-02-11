"""Statistical analysis package for golf swing biomechanics.

This package provides modular statistical analysis components:

- dataclasses: Data structures for analysis results
- basic_stats: Summary statistics and peak detection
- swing_metrics: Tempo, X-factor
- energy_metrics: Kinetic energy transfer analysis
- phase_detection: Swing phase detection
- grf_metrics: Ground reaction force metrics
- stability_metrics: Balance and stability analysis
- angular_momentum: Angular momentum analysis
- coordination_metrics: Coupling angles, CRP, coordination
- nonlinear_dynamics: Lyapunov, RQA, entropy, fractal analysis
- power_work_metrics: Work, power, impulse, stiffness
- pca_analysis: PCA, principal movements, kinematic sequence
- reporting: Reports, CSV export, frequency, jerk, swing profile

For backward compatibility, the main StatisticalAnalyzer class
is still available via:
    from shared.python.validation_pkg.statistical_analysis import StatisticalAnalyzer

For new code, prefer importing specific modules:
    from shared.python.analysis.dataclasses import PeakInfo, SummaryStatistics
    from shared.python.analysis.basic_stats import BasicStatsMixin
    from shared.python.analysis.coordination_metrics import CoordinationMetricsMixin
"""

from src.shared.python.analysis.angular_momentum import AngularMomentumMetricsMixin
from src.shared.python.analysis.basic_stats import BasicStatsMixin
from src.shared.python.analysis.coordination_metrics import CoordinationMetricsMixin
from src.shared.python.analysis.dataclasses import (
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
    SegmentTimingInfo,
    StabilityMetrics,
    SummaryStatistics,
    SwingPhase,
    SwingProfileMetrics,
)
from src.shared.python.analysis.energy_metrics import EnergyMetricsMixin
from src.shared.python.analysis.grf_metrics import GRFMetricsMixin
from src.shared.python.analysis.nonlinear_dynamics import NonlinearDynamicsMixin
from src.shared.python.analysis.pca_analysis import PCAAnalysisMixin
from src.shared.python.analysis.phase_detection import PhaseDetectionMixin
from src.shared.python.analysis.power_work_metrics import PowerWorkMetricsMixin
from src.shared.python.analysis.reporting import ReportingMixin
from src.shared.python.analysis.stability_metrics import StabilityMetricsMixin
from src.shared.python.analysis.swing_metrics import SwingMetricsMixin

__all__ = [
    # Dataclasses
    "AngularMomentumMetrics",
    "CoordinationMetrics",
    "GRFMetrics",
    "ImpulseMetrics",
    "JerkMetrics",
    "JointPowerMetrics",
    "JointStiffnessMetrics",
    "KinematicSequenceInfo",
    "SegmentTimingInfo",
    "PCAResult",
    "PeakInfo",
    "RQAMetrics",
    "StabilityMetrics",
    "SummaryStatistics",
    "SwingPhase",
    "SwingProfileMetrics",
    # Mixins
    "BasicStatsMixin",
    "SwingMetricsMixin",
    "EnergyMetricsMixin",
    "PhaseDetectionMixin",
    "GRFMetricsMixin",
    "StabilityMetricsMixin",
    "AngularMomentumMetricsMixin",
    "CoordinationMetricsMixin",
    "NonlinearDynamicsMixin",
    "PCAAnalysisMixin",
    "PowerWorkMetricsMixin",
    "ReportingMixin",
]
