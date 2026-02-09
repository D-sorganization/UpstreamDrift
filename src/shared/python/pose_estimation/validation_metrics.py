"""Validation metrics for pose estimation pipelines per S3 tolerances.

Evaluates motion matching quality against project design guidelines
(Section S3) which define acceptable error bounds for:
- Joint angle RMSE
- Marker position RMSE
- Temporal consistency (jitter)
- Fit quality (R-squared)

Issue #759: Complete motion matching pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.shared.python.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------
# S3 tolerance thresholds
# ---------------------------------------------------------------

S3_TOLERANCES: dict[str, float] = {
    # Joint angle RMSE thresholds (radians)
    "joint_angle_rmse_excellent": np.radians(2.0),  # < 2 deg
    "joint_angle_rmse_acceptable": np.radians(5.0),  # < 5 deg
    "joint_angle_rmse_poor": np.radians(10.0),  # < 10 deg
    # Marker position RMSE thresholds (meters)
    "marker_rmse_excellent": 0.010,  # < 10 mm
    "marker_rmse_acceptable": 0.025,  # < 25 mm
    "marker_rmse_poor": 0.050,  # < 50 mm
    # Temporal jitter (frame-to-frame angular velocity std, rad/s)
    "jitter_excellent": np.radians(5.0),
    "jitter_acceptable": np.radians(15.0),
    # Fit quality (R-squared)
    "r_squared_excellent": 0.99,
    "r_squared_acceptable": 0.95,
}


def _grade(value: float, excellent: float, acceptable: float) -> str:
    """Return quality grade for a metric (lower is better)."""
    if value <= excellent:
        return "excellent"
    if value <= acceptable:
        return "acceptable"
    return "poor"


def _grade_higher_better(value: float, excellent: float, acceptable: float) -> str:
    """Return quality grade for a metric where higher is better."""
    if value >= excellent:
        return "excellent"
    if value >= acceptable:
        return "acceptable"
    return "poor"


@dataclass
class ValidationReport:
    """Complete validation report for a pose estimation run.

    Attributes:
        joint_angle_metrics: Per-joint RMSE and grade.
        marker_metrics: Per-marker RMSE and grade (if available).
        temporal_metrics: Jitter and consistency.
        fit_quality: R-squared and related metrics.
        overall_grade: Aggregate quality assessment.
        details: Extra detail dictionary.
    """

    joint_angle_metrics: dict[str, Any] = field(default_factory=dict)
    marker_metrics: dict[str, Any] = field(default_factory=dict)
    temporal_metrics: dict[str, Any] = field(default_factory=dict)
    fit_quality: dict[str, Any] = field(default_factory=dict)
    overall_grade: str = "unknown"
    details: dict[str, Any] = field(default_factory=dict)


def compute_joint_angle_rmse(
    predicted: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Compute per-joint RMSE between predicted and reference angles.

    Args:
        predicted: Joint name -> array of angles (radians) per frame.
        reference: Joint name -> array of reference angles (radians).

    Returns:
        Dictionary with per-joint RMSE, grade, and aggregate RMSE.
    """
    per_joint: dict[str, dict[str, Any]] = {}
    all_errors: list[float] = []

    for joint_name in predicted:
        if joint_name not in reference:
            continue
        pred = np.asarray(predicted[joint_name])
        ref = np.asarray(reference[joint_name])
        n = min(len(pred), len(ref))
        if n == 0:
            continue
        errors = pred[:n] - ref[:n]
        rmse = float(np.sqrt(np.mean(errors**2)))
        all_errors.append(rmse)
        per_joint[joint_name] = {
            "rmse_rad": rmse,
            "rmse_deg": float(np.degrees(rmse)),
            "grade": _grade(
                rmse,
                S3_TOLERANCES["joint_angle_rmse_excellent"],
                S3_TOLERANCES["joint_angle_rmse_acceptable"],
            ),
        }

    aggregate_rmse = float(np.mean(all_errors)) if all_errors else float("inf")
    aggregate_grade = _grade(
        aggregate_rmse,
        S3_TOLERANCES["joint_angle_rmse_excellent"],
        S3_TOLERANCES["joint_angle_rmse_acceptable"],
    )

    return {
        "per_joint": per_joint,
        "aggregate_rmse_rad": aggregate_rmse,
        "aggregate_rmse_deg": float(np.degrees(aggregate_rmse)),
        "aggregate_grade": aggregate_grade,
    }


def compute_marker_rmse(
    predicted: np.ndarray,
    reference: np.ndarray,
) -> dict[str, Any]:
    """Compute marker position RMSE.

    Args:
        predicted: Predicted marker positions [frames x markers x 3].
        reference: Reference marker positions [frames x markers x 3].

    Returns:
        Dictionary with per-marker and aggregate RMSE in meters.
    """
    n = min(predicted.shape[0], reference.shape[0])
    if n == 0:
        return {"aggregate_rmse_m": float("inf"), "aggregate_grade": "poor"}

    pred = predicted[:n]
    ref = reference[:n]

    # Per-marker RMSE
    errors = np.linalg.norm(pred - ref, axis=2)  # [frames x markers]
    per_marker_rmse = np.sqrt(np.mean(errors**2, axis=0))  # [markers]
    aggregate_rmse = float(np.sqrt(np.mean(errors**2)))

    return {
        "per_marker_rmse_m": per_marker_rmse.tolist(),
        "aggregate_rmse_m": aggregate_rmse,
        "aggregate_grade": _grade(
            aggregate_rmse,
            S3_TOLERANCES["marker_rmse_excellent"],
            S3_TOLERANCES["marker_rmse_acceptable"],
        ),
    }


def compute_temporal_jitter(
    joint_angles_series: dict[str, np.ndarray],
    dt: float,
) -> dict[str, Any]:
    """Compute temporal jitter (angular velocity standard deviation).

    High jitter indicates noisy or unstable tracking.

    Args:
        joint_angles_series: Joint name -> array of angles per frame.
        dt: Time step between frames (seconds).

    Returns:
        Dictionary with per-joint jitter and aggregate jitter.
    """
    per_joint: dict[str, dict[str, Any]] = {}
    all_jitter: list[float] = []

    for joint_name, angles in joint_angles_series.items():
        angles = np.asarray(angles)
        if len(angles) < 3:
            continue
        # Angular velocity via finite differences
        omega = np.diff(angles) / dt
        jitter = float(np.std(omega))
        all_jitter.append(jitter)
        per_joint[joint_name] = {
            "jitter_rad_s": jitter,
            "grade": _grade(
                jitter,
                S3_TOLERANCES["jitter_excellent"],
                S3_TOLERANCES["jitter_acceptable"],
            ),
        }

    aggregate_jitter = float(np.mean(all_jitter)) if all_jitter else 0.0

    return {
        "per_joint": per_joint,
        "aggregate_jitter_rad_s": aggregate_jitter,
        "aggregate_grade": _grade(
            aggregate_jitter,
            S3_TOLERANCES["jitter_excellent"],
            S3_TOLERANCES["jitter_acceptable"],
        ),
    }


def compute_fit_quality(
    r_squared: float,
    condition_number: float,
    rms_error: float,
) -> dict[str, Any]:
    """Evaluate fit quality metrics.

    Args:
        r_squared: Coefficient of determination from fitting.
        condition_number: Condition number of the Jacobian.
        rms_error: RMS fitting error.

    Returns:
        Dictionary with quality grades.
    """
    return {
        "r_squared": r_squared,
        "r_squared_grade": _grade_higher_better(
            r_squared,
            S3_TOLERANCES["r_squared_excellent"],
            S3_TOLERANCES["r_squared_acceptable"],
        ),
        "condition_number": condition_number,
        "condition_well_posed": condition_number < 1e6,
        "rms_error": rms_error,
    }


def validate_pipeline_output(
    joint_angles_series: dict[str, np.ndarray] | None = None,
    reference_angles: dict[str, np.ndarray] | None = None,
    predicted_markers: np.ndarray | None = None,
    reference_markers: np.ndarray | None = None,
    dt: float = 1.0 / 30.0,
    r_squared: float = 0.0,
    condition_number: float = float("inf"),
    rms_error: float = float("inf"),
) -> ValidationReport:
    """Run full validation of a pipeline run against S3 tolerances.

    Args:
        joint_angles_series: Joint name -> angles array (predicted).
        reference_angles: Joint name -> angles array (ground truth).
        predicted_markers: Predicted marker positions [F x M x 3].
        reference_markers: Reference marker positions [F x M x 3].
        dt: Frame time step in seconds.
        r_squared: R-squared from fitting.
        condition_number: Condition number from fitting.
        rms_error: RMS error from fitting.

    Returns:
        Comprehensive ValidationReport.
    """
    report = ValidationReport()

    # Joint angle RMSE
    if joint_angles_series and reference_angles:
        report.joint_angle_metrics = compute_joint_angle_rmse(
            joint_angles_series, reference_angles
        )

    # Marker RMSE
    if predicted_markers is not None and reference_markers is not None:
        report.marker_metrics = compute_marker_rmse(
            predicted_markers, reference_markers
        )

    # Temporal jitter
    if joint_angles_series:
        report.temporal_metrics = compute_temporal_jitter(joint_angles_series, dt)

    # Fit quality
    report.fit_quality = compute_fit_quality(r_squared, condition_number, rms_error)

    # Overall grade: worst of all sub-grades
    grades = []
    if report.joint_angle_metrics.get("aggregate_grade"):
        grades.append(report.joint_angle_metrics["aggregate_grade"])
    if report.marker_metrics.get("aggregate_grade"):
        grades.append(report.marker_metrics["aggregate_grade"])
    if report.temporal_metrics.get("aggregate_grade"):
        grades.append(report.temporal_metrics["aggregate_grade"])
    if report.fit_quality.get("r_squared_grade"):
        grades.append(report.fit_quality["r_squared_grade"])

    grade_order = {"excellent": 0, "acceptable": 1, "poor": 2}
    if grades:
        report.overall_grade = max(grades, key=lambda g: grade_order.get(g, 3))
    else:
        report.overall_grade = "unknown"

    return report
