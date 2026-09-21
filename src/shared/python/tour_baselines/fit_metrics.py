"""Physical 3D Euclidean marker tracking metrics and error formulas (TB-02 #10587).

Specifies unambiguous, typed error formulas for tour motion matching:
1. Physical 3D Euclidean marker RMSE: sqrt(sum(valid ||pred - obs||^2) / N_valid).
2. p95 and max error across all valid observations.
3. Per-marker and per-phase breakdown (address, backswing, downswing, impact, follow-through).
4. Endpoint and impact errors.
5. In-plane error and out-of-plane residual relative to swing or frontal plane.
6. Observation accounting: observed denominator N_valid, excluded counts, whole-capture coverage.
7. Optimizer weighted loss reported distinctly and separately from physical RMSE.
8. Landmark set cryptographic signatures to forbid ranking disparate marker sets as equivalent.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MarkerMetricSummary:
    """Error summary for a single tracked marker landmark."""

    rmse_m: float
    max_m: float
    p95_m: float
    valid_count: int
    total_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "rmse_m": self.rmse_m,
            "max_m": self.max_m,
            "p95_m": self.p95_m,
            "valid_count": self.valid_count,
            "total_count": self.total_count,
        }


@dataclass(frozen=True)
class PhaseMetricSummary:
    """Error summary for a biomechanical swing phase."""

    rmse_m: float
    max_m: float
    p95_m: float
    valid_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "rmse_m": self.rmse_m,
            "max_m": self.max_m,
            "p95_m": self.p95_m,
            "valid_count": self.valid_count,
        }


@dataclass(frozen=True)
class PhysicalFitMetrics:
    """Standardized, comprehensive physical tracking metrics."""

    whole_marker_rmse_m: float
    p95_marker_error_m: float
    max_marker_error_m: float
    per_marker: dict[str, MarkerMetricSummary]
    per_phase: dict[str, PhaseMetricSummary]
    endpoint_error_m: float | None
    impact_error_m: float | None
    in_plane_rmse_m: float | None
    out_of_plane_residual_m: float | None
    pelvis_yaw_rmse_rad: float | None
    optimizer_weighted_loss: float | None
    n_valid: int
    n_excluded: int
    total_observations: int
    coverage_fraction: float
    landmark_set_signature: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize metrics deterministically to JSON-compatible dictionary."""
        return {
            "whole_marker_rmse_m": self.whole_marker_rmse_m,
            "p95_marker_error_m": self.p95_marker_error_m,
            "max_marker_error_m": self.max_marker_error_m,
            "per_marker": {k: v.to_dict() for k, v in sorted(self.per_marker.items())},
            "per_phase": {k: v.to_dict() for k, v in sorted(self.per_phase.items())},
            "endpoint_error_m": self.endpoint_error_m,
            "impact_error_m": self.impact_error_m,
            "in_plane_rmse_m": self.in_plane_rmse_m,
            "out_of_plane_residual_m": self.out_of_plane_residual_m,
            "pelvis_yaw_rmse_rad": self.pelvis_yaw_rmse_rad,
            "optimizer_weighted_loss": self.optimizer_weighted_loss,
            "n_valid": self.n_valid,
            "n_excluded": self.n_excluded,
            "total_observations": self.total_observations,
            "coverage_fraction": self.coverage_fraction,
            "landmark_set_signature": self.landmark_set_signature,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PhysicalFitMetrics:
        """Deserialize metrics from dictionary."""
        d = dict(data)
        per_marker = {
            k: MarkerMetricSummary(**v) for k, v in d.get("per_marker", {}).items()
        }
        per_phase = {
            k: PhaseMetricSummary(**v) for k, v in d.get("per_phase", {}).items()
        }
        return cls(
            whole_marker_rmse_m=float(d["whole_marker_rmse_m"]),
            p95_marker_error_m=float(d["p95_marker_error_m"]),
            max_marker_error_m=float(d["max_marker_error_m"]),
            per_marker=per_marker,
            per_phase=per_phase,
            endpoint_error_m=d.get("endpoint_error_m"),
            impact_error_m=d.get("impact_error_m"),
            in_plane_rmse_m=d.get("in_plane_rmse_m"),
            out_of_plane_residual_m=d.get("out_of_plane_residual_m"),
            pelvis_yaw_rmse_rad=d.get("pelvis_yaw_rmse_rad"),
            optimizer_weighted_loss=d.get("optimizer_weighted_loss"),
            n_valid=int(d["n_valid"]),
            n_excluded=int(d["n_excluded"]),
            total_observations=int(d["total_observations"]),
            coverage_fraction=float(d["coverage_fraction"]),
            landmark_set_signature=str(d["landmark_set_signature"]),
        )


def compute_landmark_signature(labels: Sequence[str]) -> str:
    """Compute deterministic SHA-256 fingerprint for a sequence of landmark labels.

    Ensures that distinct landmark configurations cannot be ranked as equivalent.
    """
    sorted_unique = sorted(set(labels))
    canonical_json = json.dumps(sorted_unique, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def _compute_distribution_metrics(errors_sq: np.ndarray) -> tuple[float, float, float]:
    """Compute RMSE, max, and p95 given an array of squared Euclidean errors."""
    if len(errors_sq) == 0:
        return 0.0, 0.0, 0.0
    rmse = float(np.sqrt(np.mean(errors_sq)))
    linear_errors = np.sqrt(errors_sq)
    max_err = float(np.max(linear_errors))
    p95_err = float(np.percentile(linear_errors, 95.0))
    return rmse, max_err, p95_err


def _compute_per_marker(
    dist_sq: np.ndarray,
    valid: np.ndarray,
    labels: Sequence[str],
) -> dict[str, MarkerMetricSummary]:
    """Calculate per-marker error summaries."""
    summaries: dict[str, MarkerMetricSummary] = {}
    n_frames = dist_sq.shape[0]
    for idx, name in enumerate(labels):
        m_mask = valid[:, idx]
        m_errs_sq = dist_sq[m_mask, idx]
        m_valid_count = int(np.sum(m_mask))
        rmse, max_val, p95_val = _compute_distribution_metrics(m_errs_sq)
        summaries[name] = MarkerMetricSummary(
            rmse_m=rmse,
            max_m=max_val,
            p95_m=p95_val,
            valid_count=m_valid_count,
            total_count=n_frames,
        )
    return summaries


def _compute_per_phase(
    dist_sq: np.ndarray,
    valid: np.ndarray,
    phase_indices: Mapping[str, tuple[int, int]] | None,
) -> dict[str, PhaseMetricSummary]:
    """Calculate per-phase error summaries."""
    if not phase_indices:
        return {}
    summaries: dict[str, PhaseMetricSummary] = {}
    for phase_name, (f_start, f_end) in phase_indices.items():
        phase_mask = np.zeros_like(valid)
        phase_mask[f_start:f_end, :] = valid[f_start:f_end, :]
        p_errs_sq = dist_sq[phase_mask]
        p_valid_count = int(np.sum(phase_mask))
        rmse, max_val, p95_val = _compute_distribution_metrics(p_errs_sq)
        summaries[phase_name] = PhaseMetricSummary(
            rmse_m=rmse,
            max_m=max_val,
            p95_m=p95_val,
            valid_count=p_valid_count,
        )
    return summaries


def _compute_planar_residuals(
    diff: np.ndarray,
    valid: np.ndarray,
    plane_normal: np.ndarray | None,
) -> tuple[float | None, float | None]:
    """Project error vectors onto in-plane and out-of-plane components."""
    if plane_normal is None:
        return None, None
    norm_val = np.linalg.norm(plane_normal)
    if norm_val < 1e-9:
        return None, None
    unit_norm = plane_normal / norm_val
    # Out-of-plane projection vector: (diff . n) * n
    dot_prod = np.sum(diff * unit_norm[None, None, :], axis=-1, keepdims=True)
    out_of_plane_vec = dot_prod * unit_norm[None, None, :]
    in_plane_vec = diff - out_of_plane_vec

    out_sq = np.sum(out_of_plane_vec**2, axis=-1)[valid]
    in_sq = np.sum(in_plane_vec**2, axis=-1)[valid]

    out_rmse = float(np.sqrt(np.mean(out_sq))) if len(out_sq) > 0 else 0.0
    in_rmse = float(np.sqrt(np.mean(in_sq))) if len(in_sq) > 0 else 0.0
    return in_rmse, out_rmse


def _validate_fit_inputs(
    pred_arr: np.ndarray,
    obs_arr: np.ndarray,
    valid_arr: np.ndarray,
    labels: Sequence[str],
) -> int:
    """Validate shapes, finite values, and non-empty valid observations."""
    if pred_arr.shape != obs_arr.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred_arr.shape} vs obs {obs_arr.shape}"
        )
    if pred_arr.ndim != 3 or pred_arr.shape[2] != 3:
        raise ValueError(
            f"Expected 3D coordinates (frames, markers, 3), got {pred_arr.shape}"
        )

    n_frames, n_markers, _ = pred_arr.shape
    if valid_arr.shape != (n_frames, n_markers):
        raise ValueError(
            f"Valid mask shape {valid_arr.shape} != {(n_frames, n_markers)}"
        )
    if len(labels) != n_markers:
        raise ValueError(f"Labels length {len(labels)} != n_markers {n_markers}")

    n_valid = int(np.sum(valid_arr))
    if n_valid == 0:
        raise ValueError(
            "Cannot compute fit metrics for empty valid set (N_valid == 0)"
        )

    valid_pred = pred_arr[valid_arr]
    if not np.all(np.isfinite(valid_pred)):
        raise ValueError(
            "Predicted coordinates contain non-finite (NaN/Inf) values at valid positions"
        )
    return n_valid


def compute_fit_metrics(
    predicted: np.ndarray,
    observed: np.ndarray,
    valid: np.ndarray,
    labels: Sequence[str],
    *,
    time_s: np.ndarray | None = None,
    phase_indices: Mapping[str, tuple[int, int]] | None = None,
    impact_frame: int | None = None,
    **kwargs: Any,
) -> PhysicalFitMetrics:
    """Compute comprehensive physical 3D marker fit metrics."""
    plane_normal: np.ndarray | None = kwargs.get("plane_normal")
    pelvis_yaw_rmse_rad: float | None = kwargs.get("pelvis_yaw_rmse_rad")
    optimizer_loss: float | None = kwargs.get("optimizer_loss")

    pred_arr = np.asarray(predicted, dtype=np.float64)
    obs_arr = np.asarray(observed, dtype=np.float64)
    valid_arr = np.asarray(valid, dtype=bool)

    n_valid = _validate_fit_inputs(pred_arr, obs_arr, valid_arr, labels)
    n_frames, n_markers, _ = pred_arr.shape

    diff = pred_arr - obs_arr
    dist_sq = np.sum(diff**2, axis=-1)

    valid_errs_sq = dist_sq[valid_arr]
    whole_rmse, max_err, p95_err = _compute_distribution_metrics(valid_errs_sq)

    total_obs = n_frames * n_markers
    n_excluded = total_obs - n_valid
    coverage = float(n_valid / total_obs)

    per_marker = _compute_per_marker(dist_sq, valid_arr, labels)
    per_phase = _compute_per_phase(dist_sq, valid_arr, phase_indices)
    in_plane_rmse, out_of_plane = _compute_planar_residuals(
        diff, valid_arr, plane_normal
    )

    # Endpoint error: error across valid markers at the final frame with valid observations
    valid_frames = np.where(np.any(valid_arr, axis=1))[0]
    last_frame = int(valid_frames[-1])
    end_mask = valid_arr[last_frame, :]
    end_err_sq = dist_sq[last_frame, end_mask]
    endpoint_error = (
        float(np.sqrt(np.mean(end_err_sq))) if len(end_err_sq) > 0 else None
    )

    # Impact error
    impact_error: float | None = None
    if impact_frame is not None and 0 <= impact_frame < n_frames:
        imp_mask = valid_arr[impact_frame, :]
        imp_err_sq = dist_sq[impact_frame, imp_mask]
        if len(imp_err_sq) > 0:
            impact_error = float(np.sqrt(np.mean(imp_err_sq)))

    sig = compute_landmark_signature(labels)

    return PhysicalFitMetrics(
        whole_marker_rmse_m=whole_rmse,
        p95_marker_error_m=p95_err,
        max_marker_error_m=max_err,
        per_marker=per_marker,
        per_phase=per_phase,
        endpoint_error_m=endpoint_error,
        impact_error_m=impact_error,
        in_plane_rmse_m=in_plane_rmse,
        out_of_plane_residual_m=out_of_plane,
        pelvis_yaw_rmse_rad=pelvis_yaw_rmse_rad,
        optimizer_weighted_loss=optimizer_loss,
        n_valid=n_valid,
        n_excluded=n_excluded,
        total_observations=total_obs,
        coverage_fraction=coverage,
        landmark_set_signature=sig,
    )
