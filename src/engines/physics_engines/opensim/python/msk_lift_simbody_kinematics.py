"""Simbody generalized speeds for Rajagopal lift inverse dynamics."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.engine_core.opensim_runtime import import_opensim

Array = NDArray[np.float64]


def spline_coordinate_derivatives(
    splines: Any, n: int, t: float
) -> tuple[Array, Array, Array]:
    """Quintic GCV spline position and first/second derivatives at time *t*."""
    osim = import_opensim()
    x = osim.Vector(1, float(t))
    first, second = osim.StdVectorInt(), osim.StdVectorInt()
    first.append(0)
    second.append(0)
    second.append(0)
    values = np.empty((3, n))
    for j in range(n):
        f = splines.get(j)
        values[0, j] = f.calcValue(x)
        values[1, j] = f.calcDerivative(first, x)
        values[2, j] = f.calcDerivative(second, x)
    return values[0], values[1], values[2]


_PELVIS_ROTATION_COORDINATES = ("pelvis_tilt", "pelvis_list", "pelvis_rotation")
_PELVIS_EULER_SEQUENCE = "ZXY"


def generalized_speeds_from_spline_derivatives(
    coordinates: tuple[str, ...],
    q: Array,
    qd: Array,
    qdd: Array,
) -> tuple[Array, Array]:
    """Map spline ``(q, qd, qdd)`` to Simbody ``(u, udot)`` for inverse dynamics.

    Rajagopal pelvis orientation coordinates are intrinsic Euler angles in
    :data:`_PELVIS_EULER_SEQUENCE` order; their generalized speeds and
    accelerations are not the raw spline derivatives.  Every other coordinate
    uses the spline first and second derivatives directly.
    """
    if q.shape != qd.shape or q.shape != qdd.shape:
        raise ValueError("q, qd and qdd must share the same shape")
    if len(coordinates) != q.size:
        raise ValueError("coordinates length must match derivative vectors")
    u = np.asarray(qd, dtype=float).copy()
    udot = np.asarray(qdd, dtype=float).copy()
    names = list(coordinates)
    if all(name in names for name in _PELVIS_ROTATION_COORDINATES):
        from src.engines.physics_engines.opensim.python.spine.inverse_dynamics import (
            euler_rate_columns,
        )

        indices = [names.index(name) for name in _PELVIS_ROTATION_COORDINATES]
        q_p = q[indices]
        qd_p = qd[indices]
        qdd_p = qdd[indices]
        omega, alpha = euler_rate_columns(q_p, qd_p, qdd_p, _PELVIS_EULER_SEQUENCE)
        u[indices] = omega
        udot[indices] = alpha
    return u, udot


def _mapped_speeds_at_time(
    splines: Any,
    coordinates: tuple[str, ...],
    n: int,
    t: float,
) -> Array:
    q, qd, qdd = spline_coordinate_derivatives(splines, n, t)
    u, _ = generalized_speeds_from_spline_derivatives(coordinates, q, qd, qdd)
    return u


def mapped_acceleration_from_speeds(
    t: float,
    t_min: float,
    t_max: float,
    sample_dt: float,
    speeds_at: Callable[[float], Array],
    analytic_udot: Array,
) -> Array:
    """Finite-difference acceleration on mapped speeds when the window is interior.

    Uses a centred difference only when ``t ± sample_dt/2`` lies inside
    ``[t_min, t_max]``.  At motion endpoints (or when the window is clipped),
    returns the spline-derived *analytic_udot* so boundary times do not use a
    one-sided secant (which is first-order and biased — e.g. ``u=t²`` at ``t=0``).
    """
    half = 0.5 * sample_dt
    t_lo = t - half
    t_hi = t + half
    if t_lo >= t_min and t_hi <= t_max:
        u_lo = speeds_at(t_lo)
        u_hi = speeds_at(t_hi)
        return (u_hi - u_lo) / sample_dt
    return np.asarray(analytic_udot, dtype=float).copy()


def simbody_kinematics_at_spline_time(
    splines: Any,
    coordinates: tuple[str, ...],
    n: int,
    t: float,
    *,
    t_min: float,
    t_max: float,
    sample_dt: float,
) -> tuple[Array, Array, Array]:
    """Return ``(q, u, udot)`` for inverse dynamics at spline time *t*.

    Interior times use centred differences on mapped generalized speeds so
    ``udot`` stays consistent with Simbody's realized ``State.getUDot()``
    after ``InverseDynamicsSolver``.  Endpoints use the analytic spline
    accelerations (including pelvis Euler mapping) because a clipped window
    would be a one-sided secant.
    """
    if len(coordinates) != n:
        raise ValueError("coordinates length must match n")
    if t_max < t_min:
        raise ValueError("t_max must be >= t_min")
    if sample_dt <= 0.0 or not math.isfinite(sample_dt):
        raise ValueError("sample_dt must be positive and finite")

    q, qd, qdd = spline_coordinate_derivatives(splines, n, t)
    u, analytic_udot = generalized_speeds_from_spline_derivatives(
        coordinates, q, qd, qdd
    )
    def speeds_at(tt: float) -> Array:
        return _mapped_speeds_at_time(splines, coordinates, n, tt)

    udot = mapped_acceleration_from_speeds(
        t, t_min, t_max, sample_dt, speeds_at, analytic_udot
    )
    return q, u, udot
