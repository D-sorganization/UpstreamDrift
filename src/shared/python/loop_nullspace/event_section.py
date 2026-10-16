"""Event section projection and saltation for hybrid flows (NSX-83, V2).

Outcomes are evaluated at the physical event ``g(x, t) = 0`` rather than at a
fixed time.  With ``f^-`` the flow just before the event, a state perturbation
``dx`` at the nominal event time moves the event by

    dt* = -(g_x dx) / (g_x f^- + g_t)

and the perturbation *on the event section* is ``Pi dx`` with

    Pi = I - f^- g_x / (g_x f^- + g_t)      (so Pi f^- = 0, g_x Pi = 0 when g_t = 0).

The perturbation after a reset ``x+ = R(x-)`` is propagated by the saltation
matrix ``S = R_x + (f^+ - R_x f^-) g_x / (g_x f^- + g_t)``, which satisfies
``S f^- = f^+``.  Specification: ``docs/research/loop_nullspace`` section
``sec:var-salt``; goldens: ``variability_sensitivity.json``.  Pure numpy.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]

#: Below this ``|g_x f| / (|g_x||f|)`` the contact is near-grazing and a linear
#: estimate is unreliable (Monte Carlo is required).
GRAZING_TRANSVERSALITY = 0.1
_DEGENERATE = 1e-12


def _vector(value: ArrayLike, name: str) -> Array:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _pair(grad_g: ArrayLike, flow: ArrayLike) -> tuple[Array, Array]:
    g_x = _vector(grad_g, "grad_g")
    f = _vector(flow, "flow")
    if g_x.shape != f.shape:
        raise ValueError(
            f"grad_g and flow must have the same shape, got {g_x.shape} and {f.shape}"
        )
    return g_x, f


def transversality(grad_g: ArrayLike, flow: ArrayLike) -> float:
    """Return ``|g_x f| / (|g_x||f|)`` in ``[0, 1]`` (1 = head-on crossing).

    Raises:
        ValueError: On shape mismatch, non-finite input or a zero vector.
    """
    g_x, f = _pair(grad_g, flow)
    denom = float(np.linalg.norm(g_x) * np.linalg.norm(f))
    if denom <= 0.0:
        raise ValueError("grad_g and flow must be nonzero")
    return abs(float(g_x @ f)) / denom


def _denominator(g_x: Array, f: Array, g_t: float) -> float:
    denom = float(g_x @ f) + float(g_t)
    scale = float(np.linalg.norm(g_x) * np.linalg.norm(f)) + abs(float(g_t))
    if abs(denom) <= _DEGENERATE * max(scale, 1.0):
        raise ValueError(
            "event is not transversal to the flow (g_x f + g_t = 0); "
            "the event time is undefined"
        )
    return denom


def event_time_shift_row(grad_g: ArrayLike, flow: ArrayLike, g_t: float = 0.0) -> Array:
    """Row ``r`` with ``dt* = r @ dx`` (the event-time shift per unit ``dx``).

    Postcondition: ``dt* = -(g_x dx) / (g_x f + g_t)``.

    Raises:
        ValueError: If shapes differ or the event is not transversal.
    """
    g_x, f = _pair(grad_g, flow)
    return -g_x / _denominator(g_x, f, g_t)


def event_projection(grad_g: ArrayLike, flow: ArrayLike, g_t: float = 0.0) -> Array:
    """Projection ``Pi = I + f r`` of a state perturbation onto the event section.

    Postcondition: ``Pi @ f = 0`` and ``g_x @ Pi = 0`` when ``g_t = 0``.
    """
    g_x, f = _pair(grad_g, flow)
    return np.eye(f.size) + np.outer(f, event_time_shift_row(g_x, f, g_t))


def saltation_matrix(
    reset_jacobian: ArrayLike,
    flow_minus: ArrayLike,
    flow_plus: ArrayLike,
    grad_g: ArrayLike,
    g_t: float = 0.0,
) -> Array:
    """Saltation matrix ``S`` mapping pre-event to post-reset perturbations.

    Postcondition: ``S @ flow_minus == flow_plus``.

    Raises:
        ValueError: On shape mismatch or a non-transversal event.
    """
    g_x, f_minus = _pair(grad_g, flow_minus)
    f_plus = _vector(flow_plus, "flow_plus")
    r_x = np.asarray(reset_jacobian, dtype=np.float64)
    n = f_minus.size
    if f_plus.shape != (n,) or r_x.shape != (n, n):
        raise ValueError(
            f"flow_plus must have shape ({n},) and reset_jacobian ({n}, {n})"
        )
    denom = _denominator(g_x, f_minus, g_t)
    return r_x + np.outer(f_plus - r_x @ f_minus, g_x) / denom
