"""Dataclass definitions for statistical analysis.

This module contains all dataclass definitions used by the statistical
analysis system. Extracted from statistical_analysis.py for modularity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PeakInfo:
    """Information about a detected peak."""

    value: float
    time: float
    index: int
    prominence: float | None = None
    width: float | None = None


@dataclass
class SummaryStatistics:
    """Summary statistics for a time series."""

    mean: float
    median: float
    std: float
    min: float
    max: float
    range: float
    min_time: float
    max_time: float
    rms: float  # Root mean square


@dataclass
class SwingPhase:
    """Information about a swing phase."""

    name: str
    start_time: float
    end_time: float
    start_index: int
    end_index: int
    duration: float


@dataclass
class KinematicSequenceInfo:
    """Information about the kinematic sequence."""

    segment_name: str
    peak_velocity: float
    peak_time: float
    peak_index: int
    order_index: int


@dataclass
class GRFMetrics:
    """Ground Reaction Force and Center of Pressure metrics."""

    cop_path_length: float
    cop_max_velocity: float
    cop_x_range: float
    cop_y_range: float
    peak_vertical_force: float | None = None
    peak_shear_force: float | None = None


@dataclass
class AngularMomentumMetrics:
    """Metrics related to system angular momentum."""

    peak_magnitude: float
    peak_time: float
    mean_magnitude: float
    # Component peaks
    peak_lx: float
    peak_ly: float
    peak_lz: float
    # Conservation error (std dev / mean) if no external torques were present
    variability: float


@dataclass
class StabilityMetrics:
    """Metrics related to postural stability."""

    # Dynamic Stability Margin proxies
    min_com_cop_distance: float  # Minimum horizontal distance between CoM and CoP
    max_com_cop_distance: float
    mean_com_cop_distance: float

    # Inclination Angles (Angle between vertical and CoP-CoM vector)
    peak_inclination_angle: float  # Maximum lean
    mean_inclination_angle: float


@dataclass
class CoordinationMetrics:
    """Metrics quantifying inter-segment coordination patterns."""

    # Percentage of swing duration in each coordination state
    in_phase_pct: float  # Both segments rotating same direction
    anti_phase_pct: float  # Segments rotating opposite directions
    proximal_leading_pct: float  # Proximal segment dominant
    distal_leading_pct: float  # Distal segment dominant

    # Mean coupling angle (if meaningful)
    mean_coupling_angle: float
    coordination_variability: float  # Std dev of coupling angle


@dataclass
class JointPowerMetrics:
    """Metrics related to joint power generation and absorption."""

    peak_generation: float  # Max positive power (W)
    peak_absorption: float  # Max negative power (W)
    avg_generation: float  # Mean positive power (W)
    avg_absorption: float  # Mean negative power (W)
    net_work: float  # Total work done (J)
    generation_duration: float  # Time spent generating power (s)
    absorption_duration: float  # Time spent absorbing power (s)


@dataclass
class ImpulseMetrics:
    """Metrics related to force/torque impulse."""

    net_impulse: float  # Integrated force/torque over time
    positive_impulse: float  # Integrated positive force/torque
    negative_impulse: float  # Integrated negative force/torque


@dataclass
class RQAMetrics:
    """Recurrence Quantification Analysis metrics.

    Quantifies the structure of the recurrence plot.
    """

    recurrence_rate: float  # Density of recurrence points (RR)
    determinism: float  # Percentage of recurrence points forming diagonal lines (DET)
    laminarity: float  # Percentage of recurrence points forming vertical lines (LAM)
    longest_diagonal_line: int  # L_max
    trapping_time: float  # Average length of vertical lines (TT)


@dataclass
class SwingProfileMetrics:
    """Normalized metrics (0-100) for Swing Profile visualization."""

    speed_score: float
    sequence_score: float
    stability_score: float
    efficiency_score: float
    power_score: float


@dataclass
class PCAResult:
    """Result of Principal Component Analysis."""

    components: np.ndarray  # (n_components, n_features) - Eigenvectors
    explained_variance: np.ndarray  # (n_components,) - Variance of each PC
    explained_variance_ratio: np.ndarray  # (n_components,) - Ratio of variance
    projected_data: np.ndarray  # (n_samples, n_components) - Scores
    mean: np.ndarray  # (n_features,) - Mean of original data


@dataclass
class JointStiffnessMetrics:
    """Metrics related to joint stiffness (Moment-Angle relationship)."""

    stiffness: float  # Slope of the regression line (Nm/rad)
    r_squared: float  # Goodness of fit
    hysteresis_area: float  # Area inside the loop (Energy dissipated/generated)
    intercept: float  # Y-intercept of the regression line


@dataclass
class JerkMetrics:
    """Metrics related to movement smoothness (Jerk)."""

    peak_jerk: float  # Max absolute jerk
    rms_jerk: float  # Root mean square jerk
    dimensionless_jerk: float  # Normalized for duration and amplitude


# Re-export all for convenience
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
    "SummaryStatistics",
    "SwingPhase",
    "SwingProfileMetrics",
]
