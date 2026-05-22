"""
Reusable cost-function building blocks for motion matching.

Part of issue #4568. Pure functions over CIR types — no engine imports.
Each function validates inputs (DbC) and returns finite scalars.
"""

from __future__ import annotations

import logging
from typing import Any
from collections.abc import Mapping

import numpy as np

from ..contracts import (
    JointTrajectory,
    MarkerTrajectory,
    TorqueTrajectory,
)
from .base import CostWeights

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Lightweight typing aliases for non-CIR shapes.
# -----------------------------------------------------------------------------

# ``TorqueTrajectory`` is now a distinct Pydantic model imported from
# ``contracts``; it is no longer aliased to ``JointTrajectory``. See #4667.
ResidualReport = Mapping[str, Any]


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------


def _q_matrix(traj: JointTrajectory) -> np.ndarray:
    """Stack q values from every frame into a (T, n_dof) array."""
    if traj is None or not traj.frames:
        raise ValueError("Trajectory must contain at least one frame")
    return np.asarray([list(f.q) for f in traj.frames], dtype=float)


def _qddot_matrix(traj: JointTrajectory) -> np.ndarray:
    """
    Return a (T, n_dof) matrix of joint accelerations.

    Falls back to second-order finite differences if `qddot` is not stored.
    """
    if traj is None or not traj.frames:
        raise ValueError("Trajectory must contain at least one frame")

    times = np.asarray([f.timestamp for f in traj.frames], dtype=float)
    if traj.frames[0].qddot is not None and all(
        f.qddot is not None for f in traj.frames
    ):
        # The guard above proves ``f.qddot`` is non-None; mypy can't see it
        # through the generator. The cast keeps the runtime semantics.
        return np.asarray(
            [list(f.qddot) for f in traj.frames],  # type: ignore[arg-type]
            dtype=float,
        )

    q = _q_matrix(traj)
    if q.shape[0] < 3:
        # Not enough samples for a second derivative; treat as zero.
        return np.zeros_like(q)

    # Non-uniform finite differences
    t = times
    qddot = np.zeros_like(q)
    for i in range(1, len(t) - 1):
        dt_back = t[i] - t[i - 1]
        dt_fwd = t[i + 1] - t[i]
        if dt_back <= 0 or dt_fwd <= 0:
            continue
        qddot[i] = (
            2.0
            * (q[i + 1] * dt_back - q[i] * (dt_back + dt_fwd) + q[i - 1] * dt_fwd)
            / (dt_back * dt_fwd * (dt_back + dt_fwd))
        )
    qddot[0] = qddot[1]
    qddot[-1] = qddot[-2]
    return qddot


def _ensure_finite(value: float, name: str) -> float:
    if not np.isfinite(value):
        raise ValueError(f"Cost component '{name}' is not finite: {value}")
    return float(value)


# -----------------------------------------------------------------------------
# Public cost components
# -----------------------------------------------------------------------------


def joint_tracking_cost(
    reference: JointTrajectory,
    actual: JointTrajectory,
    weights: Mapping[str, float] | None = None,
) -> float:
    """
    Root-mean-square error between two joint trajectories.

    Args:
        reference: Reference (ground-truth) joint trajectory.
        actual: Tracked joint trajectory.
        weights: Optional mapping ``{joint_name: weight}`` applied per-DOF.
            If `None`, every DOF is weighted equally.

    Returns:
        Non-negative finite RMSE in radians.

    Raises:
        ValueError: On empty trajectories or frame-count mismatch.
    """
    if reference is None or actual is None:
        raise ValueError("Both trajectories must be provided")
    if not reference.frames or not actual.frames:
        raise ValueError("Trajectories must have at least one frame")
    if len(reference.frames) != len(actual.frames):
        raise ValueError(
            f"Frame count mismatch: reference={len(reference.frames)} "
            f"actual={len(actual.frames)}"
        )

    ref = _q_matrix(reference)
    act = _q_matrix(actual)
    if ref.shape != act.shape:
        raise ValueError(f"q shape mismatch: {ref.shape} vs {act.shape}")

    diff_sq = (ref - act) ** 2

    if weights is not None and weights:
        joint_names = list(reference.skeleton.joints.keys())
        # Build per-DOF weight vector aligned with q
        w_vec = np.ones(ref.shape[1], dtype=float)
        idx = 0
        for jname in joint_names:
            jdef = reference.skeleton.joints[jname]
            jw = float(weights.get(jname, 1.0))
            for _ in jdef.axes:
                if idx < len(w_vec):
                    w_vec[idx] = jw
                    idx += 1
        diff_sq = diff_sq * w_vec[None, :]

    return _ensure_finite(float(np.sqrt(np.mean(diff_sq))), "joint_tracking")


def marker_tracking_cost(
    reference_markers: MarkerTrajectory,
    actual_markers: MarkerTrajectory,
) -> float:
    """
    RMSE between two marker trajectories, in meters.

    Markers present in both trajectories are paired by name. Occluded markers
    in the actual frame are skipped for that frame.

    Args:
        reference_markers: Ground-truth marker trajectory.
        actual_markers: Reconstructed marker trajectory.

    Returns:
        Non-negative finite RMSE.

    Raises:
        ValueError: On empty trajectories or frame mismatch.
    """
    if reference_markers is None or actual_markers is None:
        raise ValueError("Both marker trajectories must be provided")
    if not reference_markers.frames or not actual_markers.frames:
        raise ValueError("Marker trajectories must have at least one frame")
    if len(reference_markers.frames) != len(actual_markers.frames):
        raise ValueError("Marker trajectories must have the same number of frames")

    sq_errs: list[float] = []
    for ref_frame, act_frame in zip(
        reference_markers.frames, actual_markers.frames, strict=False
    ):
        common = set(ref_frame.marker_names) & set(act_frame.marker_names)
        for name in common:
            ref_m = ref_frame.markers[name]
            act_m = act_frame.markers[name]
            if act_m.occluded or ref_m.occluded:
                continue
            sq_errs.append(
                (ref_m.x - act_m.x) ** 2
                + (ref_m.y - act_m.y) ** 2
                + (ref_m.z - act_m.z) ** 2
            )

    if not sq_errs:
        return 0.0
    return _ensure_finite(float(np.sqrt(np.mean(sq_errs))), "marker_tracking")


def smoothness_cost(trajectory: JointTrajectory) -> float:
    """
    Sum-of-squares of joint accelerations.

    Uses ``qddot`` if present in every frame; otherwise falls back to a
    second-order finite difference of ``q``.

    Args:
        trajectory: Trajectory to evaluate.

    Returns:
        Non-negative finite scalar.
    """
    if trajectory is None:
        raise ValueError("Trajectory must be provided")
    qddot = _qddot_matrix(trajectory)
    return _ensure_finite(float(np.vdot(qddot, qddot)), "smoothness")


def effort_cost(torques: TorqueTrajectory) -> float:
    """
    Integral of squared torque (rectangle rule).

    Args:
        torques: Torque trajectory whose frames carry per-DOF ``tau``.

    Returns:
        Non-negative finite scalar.
    """
    if torques is None or not torques.frames:
        raise ValueError("Torque trajectory must have at least one frame")
    tau = np.asarray([list(f.tau) for f in torques.frames], dtype=float)
    times = np.asarray([f.timestamp for f in torques.frames], dtype=float)

    if len(times) == 1:
        return _ensure_finite(float(np.vdot(tau, tau)), "effort")

    dts = np.diff(times)
    # Pair each interval with the average of its two endpoint sq-norms
    sq = np.einsum("ij,ij->i", tau, tau)
    integral = float(np.sum(0.5 * (sq[:-1] + sq[1:]) * dts))
    return _ensure_finite(integral, "effort")


def residual_cost(residuals: ResidualReport) -> float:
    """
    Aggregate scalar derived from a residual report.

    Recognises the standard keys produced by
    ``BaseMotionMatchingSolver._compute_residual_report``
    (``mean_residual``, ``max_residual``, ``std_residual``).

    Args:
        residuals: Residual report (dict-like).

    Returns:
        Non-negative finite scalar.
    """
    if residuals is None:
        raise ValueError("Residual report must be provided")
    mean_r = float(residuals.get("mean_residual", 0.0))
    max_r = float(residuals.get("max_residual", 0.0))
    std_r = float(residuals.get("std_residual", 0.0))
    total = abs(mean_r) + 0.5 * abs(max_r) + 0.25 * abs(std_r)
    return _ensure_finite(total, "residual")


def composite_cost(
    reference: JointTrajectory,
    actual: JointTrajectory,
    weights: CostWeights,
    *,
    reference_markers: MarkerTrajectory | None = None,
    actual_markers: MarkerTrajectory | None = None,
    torques: TorqueTrajectory | None = None,
    residuals: ResidualReport | None = None,
    joint_weights: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """
    Compute every cost component plus a weighted total.

    Args:
        reference: Reference joint trajectory.
        actual: Tracked joint trajectory.
        weights: Component weights (see :class:`CostWeights`).
        reference_markers: Optional reference marker trajectory.
        actual_markers: Optional reconstructed marker trajectory.
        torques: Optional torque trajectory.
        residuals: Optional residual report.
        joint_weights: Optional per-joint weights for joint tracking.

    Returns:
        Dict with each component plus ``"total"``.

    Postcondition (DbC):
        Every value is finite; ``total >= 0``.
    """
    if weights is None:
        raise ValueError("CostWeights must be provided")

    out: dict[str, float] = {}

    j = joint_tracking_cost(reference, actual, joint_weights)
    out["joint_tracking"] = j

    if reference_markers is not None and actual_markers is not None:
        out["marker_tracking"] = marker_tracking_cost(reference_markers, actual_markers)
    else:
        out["marker_tracking"] = 0.0

    out["smoothness"] = smoothness_cost(actual)
    out["effort"] = effort_cost(torques) if torques is not None else 0.0
    out["residual"] = residual_cost(residuals) if residuals is not None else 0.0

    total = (
        weights.joint_tracking * out["joint_tracking"]
        + weights.marker_tracking * out["marker_tracking"]
        + weights.smoothness * out["smoothness"]
        + weights.effort * out["effort"]
        + weights.residual * out["residual"]
    )
    total = _ensure_finite(total, "total")
    if total < 0:
        raise ValueError(f"Composite total must be non-negative, got {total}")

    out["total"] = total

    # Postcondition
    for k, v in out.items():
        if not np.isfinite(v):
            raise ValueError(f"composite_cost: '{k}' is not finite ({v})")
    return out


__all__ = [
    "joint_tracking_cost",
    "marker_tracking_cost",
    "smoothness_cost",
    "effort_cost",
    "residual_cost",
    "composite_cost",
    "TorqueTrajectory",
    "ResidualReport",
]
