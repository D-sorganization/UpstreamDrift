"""Simbody generalized speeds for Rajagopal lift inverse dynamics."""

from __future__ import annotations

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
