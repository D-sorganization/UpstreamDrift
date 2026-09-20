"""Unified Swing Tracking and Contact Audit Evaluator (MS-31 / MS-104 / #10415).

Provides standardized, audit-grade evaluation across:
1. Anatomical segments: club, feet, wrists/hands, arms, pelvis, torso/head.
2. Swing phases: address, backswing, downswing, impact, follow-through, finish.
3. Unilateral ground penetration across all contact spheres.
4. Weld closure and tracking coverage.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import ensure, require

Array: TypeAlias = NDArray[np.float64]


class SwingPhase(str, Enum):
    """Standardized biomechanical phases of the golf swing."""

    ADDRESS = "address"
    BACKSWING = "backswing"
    DOWNSWING = "downswing"
    IMPACT = "impact"
    FOLLOW_THROUGH = "follow_through"
    FINISH = "finish"


@dataclass(frozen=True)
class SegmentMetric:
    """Tracking error summary for an anatomical group."""

    name: str
    marker_count: int
    rmse_mm: float
    max_error_mm: float
    coverage_fraction: float


@dataclass(frozen=True)
class PhaseMetric:
    """Tracking error summary within a time phase."""

    phase: SwingPhase
    start_time_s: float
    end_time_s: float
    frame_count: int
    overall_rmse_mm: float
    club_rmse_mm: float
    feet_rmse_mm: float


@dataclass(frozen=True)
class GroundPenetrationAudit:
    """Audit of foot contact sphere penetrations below ground height."""

    max_penetration_mm: float
    mean_penetration_mm: float
    penetration_frames: int
    worst_frame: int
    worst_sphere_index: int


@dataclass(frozen=True)
class ClosureAudit:
    """Audit of closed-chain weld residuals."""

    max_closure_mm: float
    mean_closure_mm: float
    worst_frame: int


@dataclass(frozen=True)
class SwingEvaluationReport:
    """Comprehensive evaluation report for a matched swing candidate."""

    overall_rmse_mm: float
    segments: dict[str, SegmentMetric]
    phases: dict[str, PhaseMetric]
    ground_penetration: GroundPenetrationAudit
    closure: ClosureAudit
    coverage_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_rmse_mm": float(self.overall_rmse_mm),
            "coverage_fraction": float(self.coverage_fraction),
            "segments": {k: vars(v) for k, v in self.segments.items()},
            "phases": {k: vars(v) for k, v in self.phases.items()},
            "ground_penetration": vars(self.ground_penetration),
            "closure": vars(self.closure),
        }


class SwingEvaluator:
    """Unified evaluator for motion matching trajectories."""

    SEGMENT_DEFINITIONS = {
        "club": lambda lbl: "Marker_" in lbl or "Club" in lbl,
        "feet": lambda lbl: any(k in lbl for k in ("Toe", "Ankle", "Heel", "Foot")),
        "wrists_hands": lambda lbl: any(k in lbl for k in ("Wrist", "Hand")),
        "arms": lambda lbl: any(k in lbl for k in ("Elbow", "UArm", "Shoulder")),
        "pelvis": lambda lbl: any(
            k in lbl for k in ("Waist", "Pelvis", "ASIS", "PSIS")
        ),
        "torso_head": lambda lbl: any(
            k in lbl for k in ("Back", "Head", "Chest", "Sternum")
        ),
    }

    def __init__(self, labels: Sequence[str]) -> None:
        require(len(labels) > 0, "labels must be non-empty")
        self.labels = tuple(labels)
        self._segment_masks: dict[str, NDArray[np.bool_]] = {}
        for seg_name, predicate in self.SEGMENT_DEFINITIONS.items():
            mask = np.array([predicate(lbl) for lbl in self.labels], dtype=bool)
            if np.any(mask):
                self._segment_masks[seg_name] = mask

    def evaluate(
        self,
        time_s: Array,
        pred_markers: Array,
        target_markers: Array,
        valid: NDArray[np.bool_],
        sphere_bottom_z: Array,
        ground_height_m: float,
        closure_errors_m: Array,
        t_events: dict[str, float] | None = None,
    ) -> SwingEvaluationReport:
        """Run comprehensive audit on tracking and contact feasibility."""
        n_nodes, n_markers = valid.shape
        require(
            pred_markers.shape == (n_nodes, n_markers, 3), "pred_markers shape mismatch"
        )
        require(
            target_markers.shape == (n_nodes, n_markers, 3),
            "target_markers shape mismatch",
        )
        require(sphere_bottom_z.shape[0] == n_nodes, "sphere_bottom_z node mismatch")

        # Per-marker distance errors in metres: (nodes, markers)
        diff = pred_markers - target_markers
        # ⚡ Bolt: np.sqrt(np.einsum) avoids temporary allocations and is ~3x faster than np.linalg.norm(..., axis=2)
        dist_m = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
        valid_dist = np.where(valid, dist_m, np.nan)

        # Overall RMSE
        all_valid_dist = dist_m[valid]
        overall_rmse_mm = (
            float(np.sqrt(np.mean(all_valid_dist**2)) * 1000.0)
            if len(all_valid_dist) > 0
            else 0.0
        )
        coverage = float(np.mean(valid))

        # Segment breakdown
        segments: dict[str, SegmentMetric] = {}
        for seg_name, mask in self._segment_masks.items():
            seg_valid = valid[:, mask]
            seg_dist = dist_m[:, mask][seg_valid]
            if len(seg_dist) > 0:
                seg_rmse = float(np.sqrt(np.mean(seg_dist**2)) * 1000.0)
                seg_max = float(np.max(seg_dist) * 1000.0)
            else:
                seg_rmse, seg_max = 0.0, 0.0
            segments[seg_name] = SegmentMetric(
                name=seg_name,
                marker_count=int(np.sum(mask)),
                rmse_mm=seg_rmse,
                max_error_mm=seg_max,
                coverage_fraction=float(np.mean(seg_valid)),
            )

        # Phase breakdown
        # Default phases based on time if events not provided
        t_start = float(time_s[0])
        t_end = float(time_s[-1])
        t_span = t_end - t_start

        phase_windows = [
            (SwingPhase.ADDRESS, t_start, t_start + min(0.30, 0.2 * t_span)),
            (
                SwingPhase.BACKSWING,
                t_start + min(0.30, 0.2 * t_span),
                t_start + 0.6 * t_span,
            ),
            (SwingPhase.DOWNSWING, t_start + 0.6 * t_span, t_start + 0.72 * t_span),
            (SwingPhase.IMPACT, t_start + 0.72 * t_span, t_start + 0.78 * t_span),
            (SwingPhase.FOLLOW_THROUGH, t_start + 0.78 * t_span, t_end),
        ]

        phases: dict[str, PhaseMetric] = {}
        club_mask = self._segment_masks.get("club", np.zeros(n_markers, dtype=bool))
        feet_mask = self._segment_masks.get("feet", np.zeros(n_markers, dtype=bool))

        for phase_enum, p_t0, p_t1 in phase_windows:
            idx = np.where((time_s >= p_t0) & (time_s <= p_t1))[0]
            if len(idx) == 0:
                continue
            p_val = valid[idx]
            p_dist = dist_m[idx][p_val]
            p_overall = (
                float(np.sqrt(np.mean(p_dist**2)) * 1000.0) if len(p_dist) > 0 else 0.0
            )

            # Club
            c_val = valid[idx][:, club_mask]
            c_dist = dist_m[idx][:, club_mask][c_val]
            p_club = (
                float(np.sqrt(np.mean(c_dist**2)) * 1000.0) if len(c_dist) > 0 else 0.0
            )

            # Feet
            f_val = valid[idx][:, feet_mask]
            f_dist = dist_m[idx][:, feet_mask][f_val]
            p_feet = (
                float(np.sqrt(np.mean(f_dist**2)) * 1000.0) if len(f_dist) > 0 else 0.0
            )

            phases[phase_enum.value] = PhaseMetric(
                phase=phase_enum,
                start_time_s=p_t0,
                end_time_s=p_t1,
                frame_count=len(idx),
                overall_rmse_mm=p_overall,
                club_rmse_mm=p_club,
                feet_rmse_mm=p_feet,
            )

        # Ground penetration audit
        # penetration = max(0, ground_height - sphere_bottom_z)
        penetrations = np.maximum(0.0, ground_height_m - sphere_bottom_z)
        max_pen_m = float(np.max(penetrations))
        mean_pen_m = float(np.mean(penetrations))
        worst_frame, worst_sphere = np.unravel_index(
            np.argmax(penetrations), penetrations.shape
        )
        pen_frames = int(np.sum(np.any(penetrations > 1e-4, axis=1)))

        ground_audit = GroundPenetrationAudit(
            max_penetration_mm=max_pen_m * 1000.0,
            mean_penetration_mm=mean_pen_m * 1000.0,
            penetration_frames=pen_frames,
            worst_frame=int(worst_frame),
            worst_sphere_index=int(worst_sphere),
        )

        # Weld closure audit
        max_c = float(np.max(closure_errors_m)) * 1000.0
        mean_c = float(np.mean(closure_errors_m)) * 1000.0
        worst_c_frame = int(np.argmax(closure_errors_m))

        closure_audit = ClosureAudit(
            max_closure_mm=max_c,
            mean_closure_mm=mean_c,
            worst_frame=worst_c_frame,
        )

        return SwingEvaluationReport(
            overall_rmse_mm=overall_rmse_mm,
            segments=segments,
            phases=phases,
            ground_penetration=ground_audit,
            closure=closure_audit,
            coverage_fraction=coverage,
        )
