"""Dataclass definitions for statistical analysis.

This module contains all dataclass definitions used by the statistical
analysis system. Extracted from statistical_analysis.py for modularity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MethodCitation:
    """Methodology reference for a biomechanical metric.

    Provides structured, machine-readable citation metadata so that
    downstream consumers (reports, cross-engine validators, audit logs)
    can trace every metric back to its methodological source.

    Attributes:
        name: Short identifier (e.g. "X-Factor Stretch").
        authors: Primary author list (e.g. "Cheetham et al.").
        year: Publication year.
        title: Paper or textbook title.
        doi: Digital Object Identifier (optional).
        notes: Implementation-specific notes (optional).
    """

    name: str
    authors: str
    year: int
    title: str
    doi: str | None = None
    notes: str | None = None


# ---------------------------------------------------------------------------
# Pre-defined citations for reuse across the codebase (Section U1-U3)
# ---------------------------------------------------------------------------

CITATION_KINEMATIC_SEQUENCE = MethodCitation(
    name="Proximal-to-Distal Sequencing",
    authors="Putnam",
    year=1993,
    title="Sequential motions of body segments in striking and throwing skills",
    doi="10.1016/0021-9290(93)90084-R",
    notes="User-supplied expected order; no proprietary methodology.",
)

CITATION_X_FACTOR = MethodCitation(
    name="X-Factor",
    authors="Cheetham et al.",
    year=2001,
    title="The importance of stretching the X-Factor in the downswing of golf",
    notes="Pelvis-thorax separation angle and stretch-shortening cycle.",
)

CITATION_CRUNCH_FACTOR = MethodCitation(
    name="Crunch Factor",
    authors="McHardy & Pollard",
    year=2005,
    title="Muscle activity during the golf swing",
    doi="10.1136/bjsm.2004.014514",
    notes="Lateral bend + axial rotation coupling metric.",
)

CITATION_SPINAL_LOAD = MethodCitation(
    name="Spinal Load Analysis",
    authors="Hosea et al.",
    year=1990,
    title="Biomechanical analysis of the golfer's back",
    notes="Up to 8x body weight compression; validated by Lindsay et al. (2002).",
)


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


# ---------------------------------------------------------------------------
# Cross-engine validation utilities (Section U)
# ---------------------------------------------------------------------------

# Default tolerances for cross-engine metric comparison
TIMING_TOLERANCE_S = 0.005  # 5 ms for peak timing
ANGLE_TOLERANCE_DEG = 2.0  # 2 degrees for angle metrics


def validate_timing_cross_engine(
    times_a: np.ndarray,
    times_b: np.ndarray,
    tolerance_s: float = TIMING_TOLERANCE_S,
) -> dict[str, bool | float]:
    """Validate peak-timing consistency between two engines.

    Args:
        times_a: Peak times from engine A [s] (N,)
        times_b: Peak times from engine B [s] (N,)
        tolerance_s: Maximum acceptable difference per peak [s]

    Returns:
        Dictionary with ``passed`` bool and ``max_diff_s`` float.
    """
    if len(times_a) != len(times_b):
        return {"passed": False, "max_diff_s": float("inf")}

    diffs = np.abs(np.asarray(times_a) - np.asarray(times_b))
    max_diff = float(np.max(diffs)) if len(diffs) > 0 else 0.0
    return {"passed": bool(max_diff <= tolerance_s), "max_diff_s": max_diff}


def validate_angle_cross_engine(
    angles_a: np.ndarray,
    angles_b: np.ndarray,
    tolerance_deg: float = ANGLE_TOLERANCE_DEG,
) -> dict[str, bool | float]:
    """Validate angle-metric consistency between two engines.

    Args:
        angles_a: Angle time series from engine A [deg] (N,)
        angles_b: Angle time series from engine B [deg] (N,)
        tolerance_deg: Maximum acceptable peak difference [deg]

    Returns:
        Dictionary with ``passed`` bool and ``max_diff_deg`` float.
    """
    a = np.asarray(angles_a)
    b = np.asarray(angles_b)
    if a.shape != b.shape:
        return {"passed": False, "max_diff_deg": float("inf")}

    diffs = np.abs(a - b)
    max_diff = float(np.max(diffs)) if diffs.size > 0 else 0.0
    return {"passed": bool(max_diff <= tolerance_deg), "max_diff_deg": max_diff}


# Re-export all for convenience
__all__ = [
    "AngularMomentumMetrics",
    "CITATION_CRUNCH_FACTOR",
    "CITATION_KINEMATIC_SEQUENCE",
    "CITATION_SPINAL_LOAD",
    "CITATION_X_FACTOR",
    "CoordinationMetrics",
    "GRFMetrics",
    "ImpulseMetrics",
    "JerkMetrics",
    "JointPowerMetrics",
    "JointStiffnessMetrics",
    "KinematicSequenceInfo",
    "MethodCitation",
    "PCAResult",
    "PeakInfo",
    "RQAMetrics",
    "StabilityMetrics",
    "SummaryStatistics",
    "SwingPhase",
    "SwingProfileMetrics",
    "validate_angle_cross_engine",
    "validate_timing_cross_engine",
]
