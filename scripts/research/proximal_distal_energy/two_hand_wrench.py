"""Frame-explicit planar wrench mechanics for two hand contacts.

The functions in this module are deliberately independent of a simulation
backend.  They operate on SI-valued arrays and make the distinction between a
physical force system and its reference-dependent moment explicit.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

CrossingDirection = Literal["positive_to_negative", "negative_to_positive"]


def _finite_array(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _cross_z(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return the z component of the row-wise planar cross product."""
    return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]


def two_contact_wrench(
    positions: np.ndarray,
    forces: np.ndarray,
    reference: np.ndarray,
    applied_couples: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Reduce two planar contact actions to a resultant and moment.

    ``positions`` and ``forces`` are ordered pairs of Cartesian vectors.  The
    returned scalar is the out-of-plane moment about ``reference``.  A positive
    value is counterclockwise.  Free couples, if supplied, use the same sign.
    """
    checked_positions = _finite_array(positions, (2, 2), "positions")
    checked_forces = _finite_array(forces, (2, 2), "forces")
    checked_reference = _finite_array(reference, (2,), "reference")
    couples = (
        np.zeros(2)
        if applied_couples is None
        else _finite_array(applied_couples, (2,), "applied_couples")
    )
    resultant = np.sum(checked_forces, axis=0)
    moment = float(
        np.sum(_cross_z(checked_positions - checked_reference, checked_forces))
        + np.sum(couples)
    )
    return resultant, moment


def transport_planar_moment(
    moment_at_a: float,
    resultant: np.ndarray,
    point_a: np.ndarray,
    point_b: np.ndarray,
) -> float:
    """Transport a planar wrench moment from reference point A to point B."""
    checked_resultant = _finite_array(resultant, (2,), "resultant")
    checked_a = _finite_array(point_a, (2,), "point_a")
    checked_b = _finite_array(point_b, (2,), "point_b")
    if not np.isfinite(moment_at_a):
        raise ValueError("moment_at_a must be finite")
    return float(moment_at_a - _cross_z(checked_b - checked_a, checked_resultant))


def resolve_grip_components(
    forces: np.ndarray, grip_axis: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resolve two forces along and normal to a declared grip axis."""
    checked_forces = _finite_array(forces, (2, 2), "forces")
    checked_axis = _finite_array(grip_axis, (2,), "grip_axis")
    magnitude = float(np.linalg.norm(checked_axis))
    if magnitude <= np.finfo(float).eps:
        raise ValueError("grip_axis must have nonzero length")
    axial_axis = checked_axis / magnitude
    normal_axis = np.array([-axial_axis[1], axial_axis[0]])
    axial = checked_forces @ axial_axis
    normal = checked_forces @ normal_axis
    return axial, normal, axial_axis, normal_axis


def two_contact_power(
    forces: np.ndarray,
    velocities: np.ndarray,
    applied_couples: np.ndarray,
    angular_velocity: float,
) -> tuple[float, float, float]:
    """Return translational, rotational, and total contact power."""
    checked_forces = _finite_array(forces, (2, 2), "forces")
    checked_velocities = _finite_array(velocities, (2, 2), "velocities")
    checked_couples = _finite_array(applied_couples, (2,), "applied_couples")
    if not np.isfinite(angular_velocity):
        raise ValueError("angular_velocity must be finite")
    force_power = float(np.sum(checked_forces * checked_velocities))
    torque_power = float(np.sum(checked_couples) * angular_velocity)
    return force_power, torque_power, force_power + torque_power


def find_zero_crossings(
    time: np.ndarray, values: np.ndarray
) -> list[tuple[float, CrossingDirection]]:
    """Locate sign changes by linear interpolation without extrapolation."""
    checked_time = np.asarray(time, dtype=float)
    checked_values = np.asarray(values, dtype=float)
    if checked_time.ndim != 1 or checked_values.shape != checked_time.shape:
        raise ValueError("time and values must be equal-length one-dimensional arrays")
    if checked_time.size < 2 or not np.all(np.diff(checked_time) > 0.0):
        raise ValueError("time must contain at least two strictly increasing values")
    if not np.all(np.isfinite(checked_time)) or not np.all(np.isfinite(checked_values)):
        raise ValueError("time and values must contain only finite values")

    crossings: list[tuple[float, CrossingDirection]] = []
    for index in range(checked_time.size - 1):
        left = checked_values[index]
        right = checked_values[index + 1]
        if left == 0.0 or left * right >= 0.0:
            continue
        fraction = -left / (right - left)
        crossing_time = checked_time[index] + fraction * (
            checked_time[index + 1] - checked_time[index]
        )
        direction: CrossingDirection = (
            "positive_to_negative" if left > 0.0 else "negative_to_positive"
        )
        crossings.append((float(crossing_time), direction))
    return crossings
