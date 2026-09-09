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


def distributed_contact_couple(
    station_offsets_m: npt.ArrayLike,
    separation_axis: npt.ArrayLike,
    station_forces_n: npt.ArrayLike,
) -> FloatArray:
    """Return the midpoint couple of a distributed contact-station set.

    ``station_offsets_m`` are signed station positions along ``separation_axis``
    measured from the declared reference midpoint, and ``station_forces_n`` is
    one force per station.  The result is the cross product of the axis with the
    first moment of the station forces, so a distributed grip is gated by the
    same signed-separation geometry as an ideal opposed pair: two half-contacts
    at plus and minus half of a signed separation, carrying opposite forces,
    reproduce :func:`bilateral_force_couple` exactly, and any station spread
    that leaves the first moment unchanged leaves the couple unchanged.
    """

    offsets = np.asarray(station_offsets_m, dtype=np.float64)
    axis = np.asarray(separation_axis, dtype=np.float64)
    forces = np.asarray(station_forces_n, dtype=np.float64)
    if offsets.ndim != 1 or forces.shape != (offsets.size, 3):
        raise ValueError("offsets must be (n,) and station forces must be (n, 3)")
    if axis.shape != (3,):
        raise ValueError("separation axis must have shape (3,)")
    if (
        not np.all(np.isfinite(offsets))
        or not np.all(np.isfinite(axis))
        or not np.all(np.isfinite(forces))
    ):
        raise ValueError("offsets, axis, and station forces must be finite")
    norm = float(np.linalg.norm(axis))
    if norm <= 0.0:
        raise ValueError("separation axis must have nonzero length")
    first_moment = offsets @ forces
    return np.cross(axis / norm, first_moment)


__all__ = [
    "bilateral_force_couple",
    "distributed_contact_couple",
    "force_velocity_projection",
    "relative_link_gates",
]
