"""Passive-coordinate response under a prescribed motion (#15150 Track Z, Z6b).

Inverse dynamics and allocation solve ``M a + b = S^T tau + J^T lambda +
S_root^T delta``. A passive coordinate (a flexible shaft's bend joints) is
neither actuated nor on the root, and the hand closure does not reach it, so
its rows carry no unknown: the reference must satisfy them exactly. A shaft
trajectory copied from a different simulation never does, and the allocator
then reports the whole swing infeasible.

:func:`passive_response` makes the reference consistent. With every other
coordinate prescribed, each passive row is an ordinary differential equation

    M_pp a_p = -(b_p(q, v) + M_po a_o),

which this module integrates with semi-implicit Euler substeps, the active
state interpolated linearly between frames. It starts from quasi-static
equilibrium with the frame-0 load. At every frame the passive acceleration is
then evaluated from the same equation, so the passive rows balance to
round-off and the allocator sees only loads the golfer and the ground can
supply.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

__all__ = ["ROOT_DOF", "passive_indices", "passive_response"]

ROOT_DOF = 6
_EQUILIBRIUM_ITERATIONS = 20
_EQUILIBRIUM_TOLERANCE = 1e-12
_FD_STEP = 1e-7


def passive_indices(adapter: Any) -> np.ndarray:
    """Native indices that are neither on the root nor actuated."""
    actuated = {int(i) for i in adapter.actuated_indices}
    return np.array(
        [i for i in range(ROOT_DOF, int(adapter.nv)) if i not in actuated], dtype=int
    )


def _passive_acceleration(adapter, idx, other, q, v, a) -> np.ndarray:
    mass, bias = adapter.compute_mass_and_bias(q, v)
    load = bias[idx] + mass[np.ix_(idx, other)] @ a[other]
    return np.linalg.solve(mass[np.ix_(idx, idx)], -load)


def _static_load(adapter, idx, other, q, a) -> np.ndarray:
    mass, bias = adapter.compute_mass_and_bias(q, np.zeros_like(q))
    return bias[idx] + mass[np.ix_(idx, other)] @ a[other]


def _equilibrium(adapter, idx, other, q, a) -> np.ndarray:
    """Passive pose where the static load vanishes (Newton, FD Jacobian)."""
    q = q.copy()
    for _ in range(_EQUILIBRIUM_ITERATIONS):
        load = _static_load(adapter, idx, other, q, a)
        if np.abs(load).max() <= _EQUILIBRIUM_TOLERANCE:
            break
        jac = np.empty((idx.size, idx.size))
        for col, i in enumerate(idx):
            nudged = q.copy()
            nudged[i] += _FD_STEP
            jac[:, col] = (
                _static_load(adapter, idx, other, nudged, a) - load
            ) / _FD_STEP
        q[idx] -= np.linalg.solve(jac, load)
    return q[idx]


def passive_response(
    adapter: Any,
    times: np.ndarray,
    q: np.ndarray,
    v: np.ndarray,
    a: np.ndarray,
    substep_s: float,
) -> int:
    """Replace the passive columns of ``q, v, a`` by the passive response.

    Preconditions: ``q, v, a`` are ``(frames, adapter.nv)`` arrays in the
    adapter's native order; ``times`` is increasing with one entry per frame;
    ``substep_s`` is finite and positive. ``ValueError`` otherwise.
    Postconditions: the active columns are unchanged; the passive rows of
    ``M a + b`` are zero to round-off at every frame. Returns the number of
    passive coordinates (0 leaves every array untouched).
    """
    if not math.isfinite(substep_s) or substep_s <= 0.0:
        raise ValueError("substep_s must be finite and positive")
    if not (q.shape == v.shape == a.shape) or q.shape != (times.size, adapter.nv):
        raise ValueError("q, v and a must share the shape (frames, nv)")
    idx = passive_indices(adapter)
    if idx.size == 0:
        return 0
    other = np.setdiff1d(np.arange(adapter.nv), idx)
    q[0, idx] = _equilibrium(adapter, idx, other, q[0], a[0])
    v[0, idx] = 0.0
    qp, vp = q[0, idx].copy(), v[0, idx].copy()
    for k in range(times.size - 1):
        steps = max(1, math.ceil((times[k + 1] - times[k]) / substep_s))
        h = (times[k + 1] - times[k]) / steps
        for j in range(steps):
            s = j / steps
            qs, vs, acc = ((1.0 - s) * x[k] + s * x[k + 1] for x in (q, v, a))
            qs[idx], vs[idx] = qp, vp
            vp = vp + h * _passive_acceleration(adapter, idx, other, qs, vs, acc)
            qp = qp + h * vp
        q[k + 1, idx], v[k + 1, idx] = qp, vp
    for k in range(times.size):
        a[k, idx] = _passive_acceleration(adapter, idx, other, q[k], v[k], a[k])
    return int(idx.size)
