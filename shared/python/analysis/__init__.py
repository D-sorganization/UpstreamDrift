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
"""

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
]
