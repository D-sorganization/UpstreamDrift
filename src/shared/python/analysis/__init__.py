"""Statistical analysis package for golf swing biomechanics.

This package provides modular statistical analysis components:

- dataclasses: Data structures for analysis results
- (future) basic_stats: Summary statistics and peak detection
- (future) swing_metrics: Tempo, X-factor, phase detection
- (future) coordination: Coordination and stability analysis
- (future) advanced: RQA, entropy, frequency analysis

For backward compatibility, the main StatisticalAnalyzer class
is still available via:
    from shared.python.statistical_analysis import StatisticalAnalyzer

For new code, prefer importing specific dataclasses:
    from shared.python.analysis.dataclasses import PeakInfo, SummaryStatistics
    from shared.python.analysis.basic_stats import BasicStatsMixin
    from shared.python.analysis.swing_metrics import SwingMetricsMixin
    from shared.python.analysis.energy_metrics import EnergyMetricsMixin
    from shared.python.analysis.phase_detection import PhaseDetectionMixin
    from shared.python.analysis.grf_metrics import GRFMetricsMixin
    from shared.python.analysis.stability_metrics import StabilityMetricsMixin
    from shared.python.analysis.angular_momentum import AngularMomentumMetricsMixin
"""

from shared.python.analysis.angular_momentum import AngularMomentumMetricsMixin
from shared.python.analysis.basic_stats import BasicStatsMixin
from shared.python.analysis.dataclasses import (
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
    SummaryStatistics,
    SwingPhase,
    SwingProfileMetrics,
)
from shared.python.analysis.energy_metrics import EnergyMetricsMixin
from shared.python.analysis.grf_metrics import GRFMetricsMixin
from shared.python.analysis.phase_detection import PhaseDetectionMixin
from shared.python.analysis.stability_metrics import StabilityMetricsMixin
from shared.python.analysis.swing_metrics import SwingMetricsMixin

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
]
