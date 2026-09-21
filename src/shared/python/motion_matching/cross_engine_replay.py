"""Cross-engine full-body forward dynamics replay, parity, and visual review (FB-6, #10070).

Provides:
1. CrossEngineReplayConfig for multi-engine replay specification.
2. Step-size convergence verification for stiff contact simulation.
3. CrossEngineComparisonReport for metric and kinematic comparison.
4. Visual review frame generation for marker overlay animations.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
import json
import logging
import math
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.shared.python.motion_matching.acceptance import Horizon, evaluate
from src.shared.python.motion_matching.full_body_forward_dynamics import (
    ContactAuditResult,
)
from src.shared.python.motion_matching.tour_metrics import SharedMetrics

logger = logging.getLogger(__name__)

Array: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]

VALID_ENGINES = frozenset({"mujoco", "pinocchio", "drake"})


@dataclass(frozen=True)
class CrossEngineReplayConfig:
    """Immutable specification for multi-engine forward simulation replay."""

    candidate_path: Path
    duration_s: float
    engines: tuple[str, ...]
    substeps_nominal: int = 2
    substeps_refined: int = 4
    tolerance_rad: float = 0.05

    def __post_init__(self) -> None:
        p = Path(self.candidate_path)
        if not p.is_file():
            raise FileNotFoundError(f"Candidate file not found: {p}")
        if self.duration_s <= 0.0:
            raise ValueError("duration_s must be strictly positive")
        if not self.engines:
            raise ValueError("At least one engine must be specified")
        invalid = set(self.engines) - VALID_ENGINES
        if invalid:
            raise ValueError(f"Unsupported engine(s): {sorted(invalid)}")
        if self.substeps_nominal < 1 or self.substeps_refined <= self.substeps_nominal:
            raise ValueError(
                "substeps_refined must be strictly greater than substeps_nominal >= 1"
            )
        if self.tolerance_rad <= 0.0:
            raise ValueError("tolerance_rad must be strictly positive")


@dataclass(frozen=True)
class StepSizeConvergenceResult:
    """Results of step-size refinement convergence analysis."""

    h_nominal: float
    h_refined: float
    max_q_difference: float
    max_qd_difference: float
    is_converged: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "h_nominal": self.h_nominal,
            "h_refined": self.h_refined,
            "max_q_difference": self.max_q_difference,
            "max_qd_difference": self.max_qd_difference,
            "is_converged": self.is_converged,
        }


def compute_step_size_convergence(
    time_s: Array,
    q_nominal: Array,
    qd_nominal: Array,
    q_refined: Array,
    qd_refined: Array,
    h_nominal: float,
    h_refined: float,
    tolerance_rad: float = 0.05,
) -> StepSizeConvergenceResult:
    """Measure trajectory difference across integration step refinement."""
    if q_nominal.shape != q_refined.shape or qd_nominal.shape != qd_refined.shape:
        raise ValueError("Nominal and refined trajectories must have identical shape")
    if len(time_s) != q_nominal.shape[0]:
        raise ValueError("time_s length must match trajectory frame count")

    q_diff = np.abs(q_nominal - q_refined)
    qd_diff = np.abs(qd_nominal - qd_refined)

    max_q_diff = float(np.max(q_diff))
    max_qd_diff = float(np.max(qd_diff))
    is_converged = bool(max_q_diff <= tolerance_rad)

    return StepSizeConvergenceResult(
        h_nominal=h_nominal,
        h_refined=h_refined,
        max_q_difference=max_q_diff,
        max_qd_difference=max_qd_diff,
        is_converged=is_converged,
    )


@dataclass(frozen=True)
class EngineReplayOutcome:
    """Outcome of forward simulation replay on a single physical engine."""

    engine: str
    status: str
    shared_metrics: SharedMetrics
    contact_audit: ContactAuditResult
    convergence: StepSizeConvergenceResult
    max_closure_residual_m: float
    max_closure_translation_m: float | None = None
    max_closure_rotation_rad: float | None = None

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.max_closure_residual_m)
            or self.max_closure_residual_m < 0.0
        ):
            raise ValueError(
                f"max_closure_residual_m must be finite and non-negative, got {self.max_closure_residual_m}"
            )
        if self.max_closure_translation_m is not None:
            if (
                not math.isfinite(self.max_closure_translation_m)
                or self.max_closure_translation_m < 0.0
            ):
                raise ValueError(
                    f"max_closure_translation_m must be finite and non-negative, got {self.max_closure_translation_m}"
                )
        if self.max_closure_rotation_rad is not None:
            if (
                not math.isfinite(self.max_closure_rotation_rad)
                or self.max_closure_rotation_rad < 0.0
            ):
                raise ValueError(
                    f"max_closure_rotation_rad must be finite and non-negative, got {self.max_closure_rotation_rad}"
                )

    def as_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "status": self.status,
            "shared_metrics": self.shared_metrics.as_dict(),
            "contact_audit": self.contact_audit.as_dict(),
            "convergence": self.convergence.as_dict(),
            "max_closure_residual_m": self.max_closure_residual_m,
            "max_closure_translation_m": self.max_closure_translation_m,
            "max_closure_rotation_rad": self.max_closure_rotation_rad,
            "legacy_mixed_closure_residual": self.max_closure_residual_m,
        }


@dataclass(frozen=True)
class CrossEngineComparisonReport:
    """Comparative evaluation report across multiple physics engine rollouts."""

    engines: tuple[str, ...]
    outcomes: dict[str, EngineReplayOutcome]
    pairwise_metric_diffs: dict[str, dict[str, float]]
    max_marker_rmse_diff_m: float

    @property
    def is_physically_accepted(self) -> bool:
        """True only if every engine outcome passes the physical acceptance contract."""
        return all(
            evaluate(outcome.as_dict(), horizon=Horizon.G3).is_physically_accepted
            for outcome in self.outcomes.values()
        )

    @property
    def status(self) -> str:
        """Authoritative status string ('PASSED' or 'REJECTED') derived from physical gates."""
        return "PASSED" if self.is_physically_accepted else "REJECTED"

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "is_physically_accepted": self.is_physically_accepted,
            "engines": list(self.engines),
            "outcomes": {k: v.as_dict() for k, v in self.outcomes.items()},
            "pairwise_metric_diffs": self.pairwise_metric_diffs,
            "max_marker_rmse_diff_m": self.max_marker_rmse_diff_m,
        }


def _compute_pairwise_diffs(
    outcomes: Sequence[EngineReplayOutcome],
) -> tuple[dict[str, dict[str, float]], float]:
    """Compute pairwise metric differentials and find maximum marker RMSE diff."""
    pairwise: dict[str, dict[str, float]] = {}
    max_diff = 0.0
    metric_keys = (
        "whole_marker_rmse_m",
        "early_marker_rmse_m",
        "terminal_marker_rmse_m",
        "club_marker_rmse_m",
        "pelvis_yaw_rmse_rad",
    )

    for i in range(len(outcomes)):
        for j in range(i + 1, len(outcomes)):
            a = outcomes[i]
            b = outcomes[j]
            pair_key = f"{a.engine}_vs_{b.engine}"
            dict_a = a.shared_metrics.as_dict()
            dict_b = b.shared_metrics.as_dict()
            pair_diffs: dict[str, float] = {}
            for k in metric_keys:
                diff_val = abs(dict_a[k] - dict_b[k])
                pair_diffs[f"{k}_diff"] = diff_val
                if "marker" in k:
                    max_diff = max(max_diff, diff_val)
            pairwise[pair_key] = pair_diffs

    return pairwise, max_diff


def compare_engine_replays(
    outcomes: Sequence[EngineReplayOutcome],
) -> CrossEngineComparisonReport:
    """Assemble cross-engine comparative report from individual engine outcomes."""
    if not outcomes:
        raise ValueError("At least one engine replay outcome required")

    engines = tuple(o.engine for o in outcomes)
    outcome_map = {o.engine: o for o in outcomes}
    pairwise, max_marker_diff = _compute_pairwise_diffs(outcomes)

    return CrossEngineComparisonReport(
        engines=engines,
        outcomes=outcome_map,
        pairwise_metric_diffs=pairwise,
        max_marker_rmse_diff_m=max_marker_diff,
    )


def generate_overlay_frame(
    frame_idx: int,
    time_s: float,
    target_markers_m: Array,
    model_markers_m: Array,
    valid: BoolArray | None = None,
) -> dict[str, Any]:
    """Generate 3D marker overlay frame data for visual review animation."""
    t_pts = np.asarray(target_markers_m, dtype=np.float64)
    m_pts = np.asarray(model_markers_m, dtype=np.float64)
    if t_pts.shape != m_pts.shape or t_pts.ndim != 2:
        raise ValueError(
            "target and model markers must have matching 2D (markers, 3) shape"
        )

    if valid is not None:
        mask = np.asarray(valid, dtype=bool)
        v_diff = m_pts[mask] - t_pts[mask]
    else:
        v_diff = m_pts - t_pts

    rmse = (
        float(np.sqrt(np.mean(np.sum(v_diff**2, axis=-1)))) if len(v_diff) > 0 else 0.0
    )

    return {
        "frame_idx": frame_idx,
        "time_s": float(time_s),
        "target_markers": t_pts,
        "model_markers": m_pts,
        "valid_mask": valid,
        "rmse_m": rmse,
    }


def render_marker_overlay_animation(
    time_s: Array,
    target_markers_m: Array,
    model_markers_m: Array,
    output_gif_path: Path | str,
    *,
    engine_name: str = "engine",
    stride: int = 5,
    valid_mask: BoolArray | None = None,
) -> Path:
    """Render a lightweight 3D marker overlay animation comparing target and model markers."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio

    out_p = Path(output_gif_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    n_frames = len(time_s)
    frames: list[Any] = []
    color_map = {
        "mujoco": "#d62728",
        "pinocchio": "#1f77b4",
        "drake": "#2ca02c",
    }
    m_color = color_map.get(engine_name.lower(), "#ff7f0e")

    fig = plt.figure(figsize=(6, 6), dpi=80)
    ax: Any = fig.add_subplot(111, projection="3d")
    t0 = target_markers_m[0]
    center = np.nanmean(t0, axis=0) if np.isnan(t0).any() else np.mean(t0, axis=0)
    box_half = 1.0

    for k in range(0, n_frames, stride):
        ax.clear()
        t_k = target_markers_m[k]
        m_k = model_markers_m[k]
        mask_k = (
            valid_mask[k] if valid_mask is not None else np.ones(len(t_k), dtype=bool)
        )

        t_valid = t_k[mask_k]
        ax.scatter(
            t_valid[:, 0],
            t_valid[:, 2],
            zs=t_valid[:, 1],
            c="black",
            s=20,
            alpha=0.7,
            label="Capture (C3D)",
        )
        m_valid = m_k[mask_k]
        ax.scatter(
            m_valid[:, 0],
            m_valid[:, 2],
            zs=m_valid[:, 1],
            c=m_color,
            s=25,
            alpha=0.9,
            label=f"Model ({engine_name})",
        )

        err = (
            float(np.sqrt(np.mean(np.sum((m_valid - t_valid) ** 2, axis=-1))))
            if len(t_valid) > 0
            else 0.0
        )

        ax.set_xlim(center[0] - box_half, center[0] + box_half)
        ax.set_ylim(center[2] - box_half, center[2] + box_half)
        ax.set_zlim(center[1] - box_half, center[1] + box_half)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.set_zlabel("Y (m)")
        ax.set_title(
            f"{engine_name.upper()} | t={time_s[k]:.3f}s | RMSE={err * 1000.0:.1f} mm"
        )
        ax.legend(loc="upper right", fontsize=8)

        canvas: Any = fig.canvas
        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba())
        frames.append(rgba[:, :, :3].copy())

    plt.close(fig)
    duration_ms = (
        float(1000.0 * stride * (time_s[1] - time_s[0])) if len(time_s) > 1 else 50.0
    )
    imageio.mimsave(str(out_p), frames, duration=duration_ms, loop=0)
    logger.info("Saved overlay animation to %s (%d frames)", out_p, len(frames))
    return out_p
