"""Force metrics: peak force, deceleration, contact duration (issue #8614).

Characterize the forces and accelerations experienced by the clubhead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.shared.python.core.contracts import require

__all__ = [
    "ForceMetrics",
    "compute_force_metrics",
]


@dataclass(frozen=True, slots=True)
class ForceMetrics:
    """Force and acceleration metrics.

    Attributes:
        peak_force_n: Maximum resultant force magnitude [N].
        peak_force_time_s: Time of peak force [s].
        peak_force_components: Force components at peak (x, y, z) [N].
        mean_force_n: Time-weighted mean force magnitude [N].
        peak_moment_nm: Maximum resultant moment magnitude [N.m].
        peak_moment_time_s: Time of peak moment [s].
        peak_deceleration_m_s2: Maximum deceleration magnitude [m/s^2].
        mean_deceleration_m_s2: Mean deceleration over contact [m/s^2].
        contact_duration_s: Duration of sand contact [s].
    """

    peak_force_n: float
    peak_force_time_s: float
    peak_force_components: tuple[float, float, float]
    mean_force_n: float
    peak_moment_nm: float
    peak_moment_time_s: float
    peak_deceleration_m_s2: float
    mean_deceleration_m_s2: float
    contact_duration_s: float


def compute_force_metrics(
    t: np.ndarray,
    forces: np.ndarray,
    torques: np.ndarray,
    head_mass_kg: float,
    *,
    velocities: np.ndarray | None = None,
    force_threshold_n: float = 10.0,
) -> ForceMetrics:
    """Compute force and acceleration metrics.

    Args:
        t: Time array (T,) [s].
        forces: Contact forces (T, 3) [N].
        torques: Contact torques (T, 3) [N.m].
        head_mass_kg: Clubhead mass [kg].
        velocities: Clubhead velocities (T, 3) [m/s], optional.
        force_threshold_n: Threshold for defining contact [N].

    Returns:
        ForceMetrics with peak/mean values and contact duration.
    """
    require(len(t) == len(forces), "t and forces must have same length")
    require(len(t) == len(torques), "t and torques must have same length")
    require(head_mass_kg > 0, "head mass must be positive")

    forces = np.asarray(forces)
    torques = np.asarray(torques)
    t = np.asarray(t)

    if len(t) < 2:
        return ForceMetrics(
            peak_force_n=0.0,
            peak_force_time_s=0.0,
            peak_force_components=(0.0, 0.0, 0.0),
            mean_force_n=0.0,
            peak_moment_nm=0.0,
            peak_moment_time_s=0.0,
            peak_deceleration_m_s2=0.0,
            mean_deceleration_m_s2=0.0,
            contact_duration_s=0.0,
        )

    force_mag = np.linalg.norm(forces, axis=1)
    torque_mag = np.linalg.norm(torques, axis=1)

    peak_force_idx = int(np.argmax(force_mag))
    peak_force = float(force_mag[peak_force_idx])
    peak_force_time = float(t[peak_force_idx])
    peak_force_components = tuple(float(f) for f in forces[peak_force_idx])

    peak_moment_idx = int(np.argmax(torque_mag))
    peak_moment = float(torque_mag[peak_moment_idx])
    peak_moment_time = float(t[peak_moment_idx])

    dt = np.diff(t)
    dt = np.where(dt > 0, dt, 1e-9)
    force_avg = (force_mag[:-1] + force_mag[1:]) / 2.0
    total_time = float(np.sum(dt))
    mean_force = float(np.sum(force_avg * dt) / total_time) if total_time > 0 else 0.0

    in_contact = force_mag > force_threshold_n
    contact_duration = _compute_contact_duration(t, in_contact)

    peak_decel, mean_decel = _compute_deceleration(t, forces, head_mass_kg, velocities)

    return ForceMetrics(
        peak_force_n=peak_force,
        peak_force_time_s=peak_force_time,
        peak_force_components=peak_force_components,
        mean_force_n=mean_force,
        peak_moment_nm=peak_moment,
        peak_moment_time_s=peak_moment_time,
        peak_deceleration_m_s2=peak_decel,
        mean_deceleration_m_s2=mean_decel,
        contact_duration_s=contact_duration,
    )


def _compute_contact_duration(t: np.ndarray, in_contact: np.ndarray) -> float:
    """Compute total time in contact."""
    if len(t) < 2:
        return 0.0

    dt = np.diff(t)
    contact_mask = in_contact[:-1] | in_contact[1:]
    return float(np.sum(dt[contact_mask]))


def _compute_deceleration(
    t: np.ndarray,
    forces: np.ndarray,
    head_mass_kg: float,
    velocities: np.ndarray | None,
) -> tuple[float, float]:
    """Compute peak and mean deceleration.

    Uses velocities if available (finite diff), else F/m.
    """
    if velocities is not None and len(velocities) >= 2:
        velocities = np.asarray(velocities)
        dt = np.diff(t)
        dt = np.where(dt > 0, dt, 1e-9)
        dv = np.diff(velocities, axis=0)
        accel = dv / dt[:, np.newaxis]
        accel_mag = np.linalg.norm(accel, axis=1)

        peak_decel = float(np.max(accel_mag))

        total_time = float(np.sum(dt))
        if total_time > 0:
            v_in = np.linalg.norm(velocities[0])
            v_out = np.linalg.norm(velocities[-1])
            mean_decel = abs(v_in - v_out) / total_time
        else:
            mean_decel = 0.0
    else:
        accel_from_force = forces / head_mass_kg
        accel_mag = np.linalg.norm(accel_from_force, axis=1)
        peak_decel = float(np.max(accel_mag))

        dt = np.diff(t)
        total_time = float(np.sum(dt)) if len(dt) > 0 else 0.0
        mean_decel = float(np.mean(accel_mag)) if len(accel_mag) > 0 else 0.0

    return peak_decel, mean_decel
