"""Reference-explicit geometry gates for momentum-transfer observables."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def force_velocity_projection(
    force_magnitude_n: float,
    speed_m_s: float,
    force_velocity_angle_rad: float | npt.ArrayLike,
) -> float | FloatArray:
    """Return force power from magnitude, speed, and their included angle."""

    values = np.asarray([force_magnitude_n, speed_m_s], dtype=np.float64)
    angle = np.asarray(force_velocity_angle_rad, dtype=np.float64)
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(angle)):
        raise ValueError("force, speed, and angle must be finite")
    if force_magnitude_n < 0.0 or speed_m_s < 0.0:
        raise ValueError("force magnitude and speed must be nonnegative")
    result = force_magnitude_n * speed_m_s * np.cos(angle)
    return float(result) if result.ndim == 0 else result


def relative_link_gates(
    relative_angle_rad: npt.ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    """Return distal tangential and centripetal projection coefficients."""

    angle = np.asarray(relative_angle_rad, dtype=np.float64)
    if not np.all(np.isfinite(angle)):
        raise ValueError("relative angle must be finite")
    return np.cos(angle), -np.sin(angle)


def bilateral_force_couple(
    signed_separation_m: float,
    separation_axis: npt.ArrayLike,
    differential_force_n: npt.ArrayLike,
) -> FloatArray:
    """Return the midpoint couple from an opposed bilateral force mode.

    ``differential_force_n`` is the force at the positive half-contact; the
    negative half-contact carries its opposite.  Common-mode force is omitted
    because its midpoint couple is identically zero.
    """

    axis = np.asarray(separation_axis, dtype=np.float64)
    force = np.asarray(differential_force_n, dtype=np.float64)
    if axis.shape != (3,) or force.shape != (3,):
        raise ValueError("axis and differential force must have shape (3,)")
    if (
        not np.isfinite(signed_separation_m)
        or not np.all(np.isfinite(axis))
        or not np.all(np.isfinite(force))
    ):
        raise ValueError("separation, axis, and force must be finite")
    norm = float(np.linalg.norm(axis))
    if norm <= 0.0:
        raise ValueError("separation axis must have nonzero length")
    return signed_separation_m * np.cross(axis / norm, force)


__all__ = [
    "bilateral_force_couple",
    "force_velocity_projection",
    "relative_link_gates",
]
