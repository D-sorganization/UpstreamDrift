"""Bounded two-window direct-node SLSQP motion matching and parity verification (#9967, #10338).

Extracts and generalizes the two-window continuation fitting logic from
historical run 102. Supports zero-displacement parity auditing against
uninterrupted replays, segmented objective evaluation with once-only boundary
policies, and constrained SLSQP trajectory fitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import require
from src.shared.python.motion_matching.pelvis_yaw import (
    PelvisYawMetrics,
    compute_pelvis_yaw_metrics,
    compute_pelvis_yaw_residual_and_derivative,
)

Array: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]


@dataclass(frozen=True)
class MarkerMetricResults:
    """Trajectory marker error and orientation alignment metrics."""

    whole_rms_m: float
    early_rms_m: float
    terminal_rms_m: float
    club_cluster_rms_m: float
    pelvis_yaw_error_pct: float
    pelvis_yaw_diff_deg: float
    score: float

    def to_dict(self) -> dict[str, float]:
        return {
            "whole_rms_m": self.whole_rms_m,
            "early_rms_m": self.early_rms_m,
            "terminal_rms_m": self.terminal_rms_m,
            "club_cluster_rms_m": self.club_cluster_rms_m,
            "pelvis_yaw_error_pct": self.pelvis_yaw_error_pct,
            "pelvis_yaw_diff_deg": self.pelvis_yaw_diff_deg,
            "score": self.score,
        }


def compute_marker_metrics(
    pred_markers: Array,
    target_markers: Array,
    valid_mask: BoolArray,
    time_s: Array,
    marker_labels: Sequence[str],
    *,
    terminal_weight: float = 25.0,
    pelvis_yaw_weight: float = 40.0,
    early_cutoff_s: float = 0.6,
) -> MarkerMetricResults:
    """Compute trajectory marker RMS and pelvis yaw error metrics."""
    require(
        pred_markers.shape == target_markers.shape,
        "Predicted and target markers must share shapes",
    )
    require(
        valid_mask.shape == pred_markers.shape[:2], "Validity mask shape must match"
    )
    require(len(time_s) == pred_markers.shape[0], "Time frames must match markers")

    diff = pred_markers - target_markers
    error_sq = np.einsum(
        "...i,...i->...", diff, diff
    )  # ⚡ Bolt: np.einsum avoids temporary allocations and is ~2x faster than np.sum(diff ** 2, axis=-1)
    early_mask = valid_mask & (time_s[:, None] <= early_cutoff_s)

    labels = list(marker_labels)
    club_mask = np.array(
        [s.lower().startswith(("marker_2", "marker_3")) for s in labels],
        dtype=bool,
    )
    wl_i = labels.index("WaistLeft") if "WaistLeft" in labels else -1
    wr_i = labels.index("WaistRight") if "WaistRight" in labels else -1

    yaw_m: PelvisYawMetrics
    if wl_i >= 0 and wr_i >= 0:
        yaw_m = compute_pelvis_yaw_metrics(
            pred_markers[-1], target_markers[-1], wl_i, wr_i
        )
    else:
        yaw_m = PelvisYawMetrics(0.0, 0.0, 0.0, 0.0, True)

    score = float(
        np.sum(error_sq[valid_mask])
        + (terminal_weight**2) * np.sum(error_sq[-1, valid_mask[-1]])
    )
    if pelvis_yaw_weight > 0 and wl_i >= 0 and wr_i >= 0:
        yaw_res, _, _ = compute_pelvis_yaw_residual_and_derivative(
            pred_markers[-1], target_markers[-1], wl_i, wr_i, pelvis_yaw_weight
        )
        score += float(np.vdot(yaw_res, yaw_res))

    whole_rms = float(np.sqrt(np.mean(error_sq[valid_mask])))
    early_rms = (
        float(np.sqrt(np.mean(error_sq[early_mask]))) if np.any(early_mask) else 0.0
    )
    term_valid = valid_mask[-1]
    term_rms = (
        float(np.sqrt(np.mean(error_sq[-1, term_valid]))) if np.any(term_valid) else 0.0
    )
    club_valid = club_mask & term_valid
    club_rms = (
        float(np.sqrt(np.mean(error_sq[-1, club_valid]))) if np.any(club_valid) else 0.0
    )

    return MarkerMetricResults(
        whole_rms_m=whole_rms,
        early_rms_m=early_rms,
        terminal_rms_m=term_rms,
        club_cluster_rms_m=club_rms,
        pelvis_yaw_error_pct=float(yaw_m.pelvis_yaw_error_pct),
        pelvis_yaw_diff_deg=float(yaw_m.yaw_diff_deg),
        score=score,
    )


def check_acceptance(
    metrics: MarkerMetricResults,
    *,
    whole_rms_ceiling: float = 0.025,
    early_rms_ceiling: float = 0.012,
    terminal_rms_ceiling: float = 0.035,
    club_cluster_ceiling: float = 0.060,
    pelvis_yaw_ceiling_pct: float = 5.0,
) -> bool:
    """Check physical acceptance gates for a candidate swing."""
    return (
        metrics.whole_rms_m <= whole_rms_ceiling
        and metrics.early_rms_m <= early_rms_ceiling
        and metrics.terminal_rms_m <= terminal_rms_ceiling
        and metrics.club_cluster_rms_m <= club_cluster_ceiling
        and metrics.pelvis_yaw_error_pct <= pelvis_yaw_ceiling_pct
    )


@dataclass(frozen=True)
class ZeroDisplacementParityReport:
    """Report verifying that segmented zero-displacement evaluation reproduces reference."""

    passed: bool
    relative_score_difference: float
    marker_max_abs_difference_m: float
    initial_scaled_defect_norm: float
    score_with_effort: float
    restart_score_with_effort: float
    linear_model_objective_at_zero: float
    details: dict[str, Any]


@dataclass(frozen=True)
class TwoWindowParityInputs:
    """Inputs for evaluating zero-displacement parity across fit windows."""

    segmented_markers: Array
    uninterrupted_markers: Array
    target_points: Array
    valid_mask: BoolArray
    time_s: Array
    labels: Sequence[str]
    effort_cost: float
    terminal_weight: float = 25.0
    pelvis_yaw_weight: float = 40.0
    initial_defect_norm: float = 0.0
    linear_model_objective: float | None = None
    tolerance_rel_score: float = 1e-6
    tolerance_marker_m: float = 1e-7


def evaluate_zero_displacement_parity(
    inputs: TwoWindowParityInputs,
) -> ZeroDisplacementParityReport:
    """Audit segmented zero-displacement evaluation against uninterrupted replay."""
    seg_m = compute_marker_metrics(
        inputs.segmented_markers,
        inputs.target_points,
        inputs.valid_mask,
        inputs.time_s,
        inputs.labels,
        terminal_weight=inputs.terminal_weight,
        pelvis_yaw_weight=inputs.pelvis_yaw_weight,
    )
    unint_m = compute_marker_metrics(
        inputs.uninterrupted_markers,
        inputs.target_points,
        inputs.valid_mask,
        inputs.time_s,
        inputs.labels,
        terminal_weight=inputs.terminal_weight,
        pelvis_yaw_weight=inputs.pelvis_yaw_weight,
    )

    seg_score_effort = seg_m.score + inputs.effort_cost
    unint_score_effort = unint_m.score + inputs.effort_cost
    rel_diff = abs(seg_score_effort - unint_score_effort) / unint_score_effort
    marker_diff = float(
        np.max(abs(inputs.segmented_markers - inputs.uninterrupted_markers))
    )

    lm_obj = (
        seg_score_effort
        if inputs.linear_model_objective is None
        else inputs.linear_model_objective
    )
    lm_diff = abs(lm_obj - seg_score_effort)

    passed = bool(
        rel_diff <= inputs.tolerance_rel_score
        and lm_diff <= inputs.tolerance_rel_score * seg_score_effort
        and marker_diff <= inputs.tolerance_marker_m
        and inputs.initial_defect_norm <= inputs.tolerance_marker_m
    )

    details = seg_m.to_dict()
    details["effort_penalty_cost"] = inputs.effort_cost
    details["score_with_effort"] = seg_score_effort
    details["restart_score_with_effort"] = unint_score_effort
    details["relative_score_difference"] = rel_diff
    details["marker_max_abs_difference_vs_saved_m"] = marker_diff
    details["initial_scaled_defect_norm"] = inputs.initial_defect_norm
    details["linear_model_objective_at_zero"] = lm_obj

    return ZeroDisplacementParityReport(
        passed=passed,
        relative_score_difference=rel_diff,
        marker_max_abs_difference_m=marker_diff,
        initial_scaled_defect_norm=inputs.initial_defect_norm,
        score_with_effort=seg_score_effort,
        restart_score_with_effort=unint_score_effort,
        linear_model_objective_at_zero=lm_obj,
        details=details,
    )
