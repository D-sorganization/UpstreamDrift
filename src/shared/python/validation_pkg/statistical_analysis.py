"""Statistical analysis module for golf swing biomechanics.

Provides comprehensive statistical analysis including:
- Peak detection
- Summary statistics
- Swing quality metrics
- Phase-specific analysis
- Advanced stability and coordination metrics

Note: Most methods have been extracted into focused mixin modules
in the analysis package. This module provides the composed
StatisticalAnalyzer class for backward compatibility.
"""

from __future__ import annotations

import numpy as np

from src.shared.python.analysis.angular_momentum import AngularMomentumMetricsMixin
from src.shared.python.analysis.basic_stats import BasicStatsMixin
from src.shared.python.analysis.coordination_metrics import CoordinationMetricsMixin

# Import dataclasses from modular package
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

# Re-export for backward compatibility
__all__ = [
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
    "StatisticalAnalyzer",
    "SummaryStatistics",
    "SwingPhase",
    "SwingProfileMetrics",
]


class StatisticalAnalyzer(
    ReportingMixin,
    PCAAnalysisMixin,
    PowerWorkMetricsMixin,
    NonlinearDynamicsMixin,
    CoordinationMetricsMixin,
    EnergyMetricsMixin,
    PhaseDetectionMixin,
    GRFMetricsMixin,
    StabilityMetricsMixin,
    AngularMomentumMetricsMixin,
    SwingMetricsMixin,
    BasicStatsMixin,
):
    """Comprehensive statistical analysis for golf swing data.

    This class composes all analysis mixins to provide a unified
    interface for statistical analysis of biomechanical data.

    Methods are organized into focused mixin modules:
    - BasicStatsMixin: Summary statistics, peak detection
    - SwingMetricsMixin: Tempo, X-factor
    - EnergyMetricsMixin: Kinetic energy transfer
    - PhaseDetectionMixin: Swing phase detection
    - GRFMetricsMixin: Ground reaction force metrics
    - StabilityMetricsMixin: Balance and stability
    - AngularMomentumMetricsMixin: Angular momentum analysis
    - CoordinationMetricsMixin: Coupling angles, CRP, coordination
    - NonlinearDynamicsMixin: Lyapunov, RQA, entropy, fractal
    - PowerWorkMetricsMixin: Work, power, impulse, stiffness
    - PCAAnalysisMixin: PCA, principal movements, kinematic sequence
    - ReportingMixin: Reports, CSV export, frequency, jerk, swing profile
    """

    def __init__(
        self,
        times: np.ndarray,
        joint_positions: np.ndarray,
        joint_velocities: np.ndarray,
        joint_torques: np.ndarray,
        club_head_speed: np.ndarray | None = None,
        club_head_position: np.ndarray | None = None,
        cop_position: np.ndarray | None = None,
        ground_forces: np.ndarray | None = None,
        com_position: np.ndarray | None = None,
        angular_momentum: np.ndarray | None = None,
        joint_accelerations: np.ndarray | None = None,
    ) -> None:
        """Initialize analyzer with recorded data.

        Args:
            times: Time array (N,)
            joint_positions: Joint positions (N, nq)
            joint_velocities: Joint velocities (N, nv)
            joint_torques: Joint torques (N, nu)
            club_head_speed: Club head speed (N,) [optional]
            club_head_position: Club head 3D position (N, 3) [optional]
            cop_position: Center of Pressure position (N, 2) or (N, 3) [optional]
            ground_forces: Ground reaction forces (N, 3) or (N, 6) [optional]
            com_position: Center of Mass position (N, 3) [optional]
            angular_momentum: System angular momentum (N, 3) [optional]
            joint_accelerations: Joint accelerations (N, nv) [optional]
        """
        self.times = times
        self.joint_positions = joint_positions
        self.joint_velocities = joint_velocities
        self.joint_torques = joint_torques
        self.club_head_speed = club_head_speed
        self.club_head_position = club_head_position
        self.cop_position = cop_position
        self.ground_forces = ground_forces
        self.com_position = com_position
        self.angular_momentum = angular_momentum
        self.joint_accelerations = joint_accelerations

        self.dt = float(np.mean(np.diff(times))) if len(times) > 1 else 0.0
        self.duration = times[-1] - times[0] if len(times) > 1 else 0.0

        # Performance optimization: Cache for expensive computations
        self._work_metrics_cache: dict[int, dict[str, float]] = {}
