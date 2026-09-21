"""Physical 3D Euclidean marker tracking fit metrics and unconflated statuses (TB-02 #10587).

Defines:
- 5 unconflated statuses: solver convergence, kinematic accuracy, dynamic feasibility,
  scientific qualification, and product promotion.
- Physical 3D Euclidean marker RMSE formula sqrt(sum(valid ||pred - obs||^2) / N_valid),
  p95, max, per-marker, per-phase, and impact errors.
- Separate reporting of optimizer weighted loss from physical marker RMSE.
- Landmark set hashing preventing ranking different landmark sets as equivalent.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import postcondition, precondition


class SolverStatus(str, Enum):
    """Solver numerical termination state."""

    CONVERGED = "converged"
    REACHED_MAX_ITER = "reached_max_iter"
    NUMERICAL_FAILURE = "numerical_failure"
    TIMEOUT = "timeout"
    FAILED = "failed"


class KinematicAccuracyStatus(str, Enum):
    """Kinematic marker trajectory fidelity state."""

    ACCURATE = "accurate"
    INACCURATE = "inaccurate"
    UNTESTED = "untested"


class DynamicFeasibilityStatus(str, Enum):
    """Dynamical consistency and constraint feasibility state."""

    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    NOT_APPLICABLE = "not_applicable"


class ScientificQualificationStatus(str, Enum):
    """Rigor of scientific evidence and native replay reproduction."""

    QUALIFIED = "qualified"
    UNQUALIFIED = "unqualified"
    REJECTED = "rejected"
    HISTORICAL = "historical"


class ProductPromotionStatus(str, Enum):
    """Deployment readiness for end-user tour baseline consumption."""

    PROMOTED = "promoted"
    UNPROMOTED = "unpromoted"
    CANDIDATE = "candidate"
    REJECTED = "rejected"


@dataclass(frozen=True)
class TourFitMetrics:
    """Comprehensive, un-conflated fit metrics for tour motion matching."""

    observed_valid_denominator: int
    excluded_sample_count: int
    coverage_fraction: float
    whole_marker_rmse_m: float
    p95_marker_error_m: float
    max_marker_error_m: float
    per_marker_rmse_m: dict[str, float]
    per_phase_rmse_m: dict[str, float]
    impact_marker_error_m: float
    endpoint_clubhead_rmse_m: float
    optimizer_weighted_loss: float
    original_frame_rmse_m: float
    in_plane_rmse_m: float
    out_of_plane_residual_m: float
    landmarks_hash: str
    clubhead_speed_error_m_s: float | None = None

    def as_dict(self) -> dict[str, Any]:
        """Convert metrics to a serialized JSON-compatible dictionary."""
        return {
            "observed_valid_denominator": self.observed_valid_denominator,
            "excluded_sample_count": self.excluded_sample_count,
            "coverage_fraction": self.coverage_fraction,
            "whole_marker_rmse_m": self.whole_marker_rmse_m,
            "p95_marker_error_m": self.p95_marker_error_m,
            "max_marker_error_m": self.max_marker_error_m,
            "per_marker_rmse_m": dict(self.per_marker_rmse_m),
            "per_phase_rmse_m": dict(self.per_phase_rmse_m),
            "impact_marker_error_m": self.impact_marker_error_m,
            "endpoint_clubhead_rmse_m": self.endpoint_clubhead_rmse_m,
            "optimizer_weighted_loss": self.optimizer_weighted_loss,
            "original_frame_rmse_m": self.original_frame_rmse_m,
            "in_plane_rmse_m": self.in_plane_rmse_m,
            "out_of_plane_residual_m": self.out_of_plane_residual_m,
            "landmarks_hash": self.landmarks_hash,
            "clubhead_speed_error_m_s": self.clubhead_speed_error_m_s,
        }


def hash_landmark_set(labels: Iterable[str]) -> str:
    """Compute deterministic SHA-256 hash for an observation landmark set."""
    canonical_labels = sorted({lbl.strip() for lbl in labels if lbl.strip()})
    blob = "\n".join(canonical_labels).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _compute_per_marker_errors(
    pred: NDArray[np.float64],
    obs: NDArray[np.float64],
    labels: Sequence[str],
) -> dict[str, float]:
    """Calculate per-marker RMSE over valid frames."""
    per_marker: dict[str, float] = {}
    for idx, lbl in enumerate(labels):
        p_m = pred[:, idx, :]
        o_m = obs[:, idx, :]
        valid = ~np.isnan(o_m).any(axis=-1) & ~np.isnan(p_m).any(axis=-1)
        if not np.any(valid):
            per_marker[lbl] = float("nan")
            continue
        diff = p_m[valid] - o_m[valid]
        rmse = float(np.sqrt(np.mean(np.sum(diff**2, axis=-1))))
        per_marker[lbl] = rmse
    return per_marker


def _compute_per_phase_errors(
    pred: NDArray[np.float64],
    obs: NDArray[np.float64],
    phase_intervals: dict[str, tuple[int, int]] | None,
) -> dict[str, float]:
    """Calculate per-phase RMSE across all markers."""
    if not phase_intervals:
        return {}
    per_phase: dict[str, float] = {}
    for phase_name, (start, end) in phase_intervals.items():
        if start >= end or start >= pred.shape[0]:
            continue
        p_slice = pred[start:end]
        o_slice = obs[start:end]
        valid = ~np.isnan(o_slice).any(axis=-1) & ~np.isnan(p_slice).any(axis=-1)
        if not np.any(valid):
            per_phase[phase_name] = float("nan")
            continue
        diff = p_slice[valid] - o_slice[valid]
        rmse = float(np.sqrt(np.mean(np.sum(diff**2, axis=-1))))
        per_phase[phase_name] = rmse
    return per_phase


DEFAULT_CLUBHEAD_LABELS: tuple[str, ...] = (
    "Marker_3:3:1",
    "Marker_3:3:2",
    "Marker_3:3:3",
)


def _compute_impact_error(
    predicted_points_m: NDArray[np.float64],
    observed_points_m: NDArray[np.float64],
    valid_mask: NDArray[np.bool_],
    impact_frame: int | None,
    whole_rmse: float,
) -> float:
    """Compute marker RMSE at the designated impact frame."""
    n_frames = predicted_points_m.shape[0]
    imp_idx = impact_frame if impact_frame is not None else int(0.73 * n_frames)
    imp_idx = min(max(0, imp_idx), n_frames - 1)
    imp_valid = valid_mask[imp_idx]
    if np.any(imp_valid):
        imp_diff = (
            predicted_points_m[imp_idx, imp_valid]
            - observed_points_m[imp_idx, imp_valid]
        )
        return float(np.sqrt(np.mean(np.sum(imp_diff**2, axis=-1))))
    return whole_rmse


def _compute_clubhead_error(
    predicted_points_m: NDArray[np.float64],
    observed_points_m: NDArray[np.float64],
    valid_mask: NDArray[np.bool_],
    marker_labels: Sequence[str],
    whole_rmse: float,
) -> float:
    """Compute marker RMSE specifically for clubhead markers."""
    head_indices = [
        i for i, lbl in enumerate(marker_labels) if lbl in DEFAULT_CLUBHEAD_LABELS
    ]
    if not head_indices:
        return whole_rmse
    h_mask = valid_mask[:, head_indices]
    if not np.any(h_mask):
        return whole_rmse
    h_diffs = (
        predicted_points_m[:, head_indices][h_mask]
        - observed_points_m[:, head_indices][h_mask]
    )
    return float(np.sqrt(np.mean(np.sum(h_diffs**2, axis=-1))))


def _compute_planar_residuals(
    diffs: NDArray[np.float64],
    whole_rmse: float,
    swing_plane_normal: NDArray[np.float64] | None,
) -> tuple[float, float]:
    """Compute in-plane RMSE and out-of-plane residual."""
    if swing_plane_normal is None:
        return whole_rmse, 0.0
    norm_val = float(np.linalg.norm(swing_plane_normal))
    if norm_val < 1e-12:
        return whole_rmse, 0.0
    normal = swing_plane_normal / norm_val
    out_of_plane = float(np.sqrt(np.mean(np.dot(diffs, normal) ** 2)))
    in_plane = float(np.sqrt(max(0.0, whole_rmse**2 - out_of_plane**2)))
    return in_plane, out_of_plane


def compute_tour_fit_metrics(
    predicted_points_m: NDArray[np.float64],
    observed_points_m: NDArray[np.float64],
    marker_labels: Sequence[str],
    time_s: Sequence[float],
    *,
    optimizer_loss: float = 0.0,
    impact_frame: int | None = None,
    phase_intervals: dict[str, tuple[int, int]] | None = None,
    swing_plane_normal: NDArray[np.float64] | None = None,
) -> TourFitMetrics:
    """Compute physical 3D marker tracking errors over valid observations."""
    if predicted_points_m.shape != observed_points_m.shape:
        raise ValueError(
            f"Shape mismatch: predicted {predicted_points_m.shape} != observed {observed_points_m.shape}"
        )
    if predicted_points_m.ndim != 3 or predicted_points_m.shape[-1] != 3:
        raise ValueError("Arrays must have 3D coordinates (..., 3)")
    if len(marker_labels) != predicted_points_m.shape[1]:
        raise ValueError("marker_labels count must match array marker dimension")
    if len(time_s) != predicted_points_m.shape[0]:
        raise ValueError("time_s length must match array frame dimension")

    n_frames, n_markers, _ = predicted_points_m.shape
    total_samples = n_frames * n_markers

    valid_mask: NDArray[np.bool_] = np.logical_and(
        ~np.isnan(observed_points_m).any(axis=-1),
        ~np.isnan(predicted_points_m).any(axis=-1),
    )
    n_valid = int(np.sum(valid_mask))
    if n_valid == 0:
        raise ValueError("No valid observations present in observed points")

    diffs = predicted_points_m[valid_mask] - observed_points_m[valid_mask]
    squared_euclidean = np.sum(diffs**2, axis=-1)
    whole_rmse = float(np.sqrt(np.sum(squared_euclidean) / n_valid))

    euclidean_errors = np.sqrt(squared_euclidean)
    max_err = float(np.max(euclidean_errors))
    p95_err = float(np.percentile(euclidean_errors, 95.0))

    per_marker = _compute_per_marker_errors(
        predicted_points_m, observed_points_m, marker_labels
    )
    per_phase = _compute_per_phase_errors(
        predicted_points_m, observed_points_m, phase_intervals
    )
    impact_err = _compute_impact_error(
        predicted_points_m, observed_points_m, valid_mask, impact_frame, whole_rmse
    )
    head_rmse = _compute_clubhead_error(
        predicted_points_m, observed_points_m, valid_mask, marker_labels, whole_rmse
    )
    in_plane, out_of_plane = _compute_planar_residuals(
        diffs, whole_rmse, swing_plane_normal
    )

    return TourFitMetrics(
        observed_valid_denominator=n_valid,
        excluded_sample_count=total_samples - n_valid,
        coverage_fraction=float(n_valid / total_samples),
        whole_marker_rmse_m=whole_rmse,
        p95_marker_error_m=p95_err,
        max_marker_error_m=max_err,
        per_marker_rmse_m=per_marker,
        per_phase_rmse_m=per_phase,
        impact_marker_error_m=impact_err,
        endpoint_clubhead_rmse_m=head_rmse,
        optimizer_weighted_loss=float(optimizer_loss),
        original_frame_rmse_m=whole_rmse,
        in_plane_rmse_m=in_plane,
        out_of_plane_residual_m=out_of_plane,
        landmarks_hash=hash_landmark_set(marker_labels),
    )
