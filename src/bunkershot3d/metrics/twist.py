"""Twist metrics: moment about shaft axis and CG (issue #8614).

The reason bounce and relief exist — they control how the club twists.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.shared.python.core.contracts import require

__all__ = [
    "TwistMetrics",
    "compute_twist_metrics",
]


@dataclass(frozen=True, slots=True)
class TwistMetrics:
    """Twist analysis metrics.

    Attributes:
        peak_shaft_moment_nm: Peak moment about shaft axis [N.m].
        peak_shaft_moment_time_s: Time of peak shaft moment [s].
        mean_shaft_moment_nm: Mean moment about shaft axis [N.m].
        shaft_impulse_nm_s: Integrated moment about shaft [N.m.s].
        peak_cg_moment_nm: Peak moment about CG [N.m].
        net_twist_direction: "opening", "closing", or "neutral".
    """

    peak_shaft_moment_nm: float
    peak_shaft_moment_time_s: float
    mean_shaft_moment_nm: float
    shaft_impulse_nm_s: float
    peak_cg_moment_nm: float
    net_twist_direction: str


def compute_twist_metrics(
    t: np.ndarray,
    torques: np.ndarray,
    shaft_axis: np.ndarray,
    *,
    forces: np.ndarray | None = None,
    contact_points: np.ndarray | None = None,
    cg_position: np.ndarray | None = None,
) -> TwistMetrics:
    """Compute twist metrics from torque trace.

    Args:
        t: Time array (T,) [s].
        torques: Contact torques (T, 3) [N.m].
        shaft_axis: Unit vector along shaft axis (3,).
        forces: Contact forces (T, 3) [N], optional for CG moment.
        contact_points: Points where forces are applied (T, 3) [m], optional.
        cg_position: Clubhead CG position (3,) [m], optional.

    Returns:
        TwistMetrics with shaft and CG moment analysis.
    """
    require(len(t) == len(torques), "t and torques must have same length")

    torques = np.asarray(torques)
    t = np.asarray(t)
    shaft_axis = np.asarray(shaft_axis)
    shaft_axis = shaft_axis / np.linalg.norm(shaft_axis)

    if len(t) < 2:
        return TwistMetrics(
            peak_shaft_moment_nm=0.0,
            peak_shaft_moment_time_s=0.0,
            mean_shaft_moment_nm=0.0,
            shaft_impulse_nm_s=0.0,
            peak_cg_moment_nm=0.0,
            net_twist_direction="neutral",
        )

    shaft_moments = torques @ shaft_axis

    abs_shaft_moments = np.abs(shaft_moments)
    peak_idx = int(np.argmax(abs_shaft_moments))
    peak_shaft = float(shaft_moments[peak_idx])
    peak_shaft_time = float(t[peak_idx])

    dt = np.diff(t)
    dt = np.where(dt > 0, dt, 1e-9)
    moment_avg = (shaft_moments[:-1] + shaft_moments[1:]) / 2.0
    total_time = float(np.sum(dt))

    mean_shaft = float(np.sum(moment_avg * dt) / total_time) if total_time > 0 else 0.0
    impulse = float(np.sum(moment_avg * dt))

    if forces is not None and contact_points is not None and cg_position is not None:
        cg_moments = _compute_cg_moments(forces, contact_points, cg_position, torques)
        cg_moment_mag = np.linalg.norm(cg_moments, axis=1)
        peak_cg = float(np.max(cg_moment_mag))
    else:
        cg_moments = torques
        cg_moment_mag = np.linalg.norm(cg_moments, axis=1)
        peak_cg = float(np.max(cg_moment_mag))

    net_impulse = impulse
    if net_impulse > 0.001:
        direction = "opening"
    elif net_impulse < -0.001:
        direction = "closing"
    else:
        direction = "neutral"

    return TwistMetrics(
        peak_shaft_moment_nm=peak_shaft,
        peak_shaft_moment_time_s=peak_shaft_time,
        mean_shaft_moment_nm=mean_shaft,
        shaft_impulse_nm_s=impulse,
        peak_cg_moment_nm=peak_cg,
        net_twist_direction=direction,
    )


def _compute_cg_moments(
    forces: np.ndarray,
    contact_points: np.ndarray,
    cg_position: np.ndarray,
    torques: np.ndarray,
) -> np.ndarray:
    """Compute moment about CG: M_CG = (r_cg - r_contact) x F + M_contact."""
    forces = np.asarray(forces)
    contact_points = np.asarray(contact_points)
    cg_position = np.asarray(cg_position)
    torques = np.asarray(torques)

    r = cg_position - contact_points
    cross_moment = np.cross(r, forces)
    return cross_moment + torques
