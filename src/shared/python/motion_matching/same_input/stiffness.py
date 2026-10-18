"""Stiff-mode RK4 substep rule for same-input bundles (OSL-16, #13104).

Classical RK4 on a real decaying mode ``x' = lambda x`` multiplies the state
by ``R(z) = 1 + z + z^2/2 + z^3/6 + z^4/24`` per substep, with
``z = lambda * h``.  ``|R(z)| <= 1`` only for ``-2.785 <= z <= 0`` on the
real axis.  The penalty foot contact under a 100 kg bar has real
open-loop foot modes of ``-1.2e4`` to ``-1.3e5 1/s`` over the squat window
(the Coulomb regularisation slope ``mu f_n / v_t`` on the light foot
segments, growing with the ground force), so eight 0.125 ms substeps give
``|z|`` up to 16 and the foot joints run away.  The rule here measures the
stiffest rate with a finite-difference Jacobian and picks the smallest
power-of-two substep count that keeps ``|z|`` at or below ``target``, by
default the RK4 real stability limit itself; a run at the selected count
must still show trajectory convergence against twice the substeps (the
loaded squat's 64-substep run is within 1.7e-8 rad of 128, and its 32-substep
run, at ``|z| = 2.80``, is 2e-3 rad off).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from src.shared.python.motion_matching.same_input.integrator import (
    DEFAULT_SUBSTEPS,
)

Array = NDArray[np.float64]

#: Real-axis extent of the classical RK4 stability region (|R(-x)| = 1).
RK4_REAL_STABILITY_LIMIT = 2.785293563405282
#: Default ``|lambda| h`` target of :func:`substeps_for`: stability.
DEFAULT_TARGET = RK4_REAL_STABILITY_LIMIT
MAX_SUBSTEPS = 1024


class AccelerationPlant(Protocol):
    """Anything that maps ``(q, v, tau)`` to generalised accelerations."""

    def acceleration(self, q: Array, v: Array, tau: Array) -> Array:
        """Constrained accelerations at ``(q, v)`` under efforts ``tau``."""
        ...


def rk4_amplification(z: complex) -> float:
    """``|R(z)|`` for classical RK4 at ``z = lambda * h``."""
    return float(abs(1.0 + z + z**2 / 2.0 + z**3 / 6.0 + z**4 / 24.0))


def _state_jacobian(
    plant: AccelerationPlant, q: Array, v: Array, tau: Array, eps: float
) -> Array:
    """Central-difference Jacobian of ``(q', v') = (v, a(q, v, tau))``."""
    nv = q.shape[0]
    jac = np.zeros((2 * nv, 2 * nv))
    jac[:nv, nv:] = np.eye(nv)
    for j in range(nv):
        dq = np.zeros(nv)
        dq[j] = eps
        jac[nv:, j] = (
            plant.acceleration(q + dq, v, tau) - plant.acceleration(q - dq, v, tau)
        ) / (2.0 * eps)
        jac[nv:, nv + j] = (
            plant.acceleration(q, v + dq, tau) - plant.acceleration(q, v - dq, tau)
        ) / (2.0 * eps)
    return jac


def stiffest_rate(
    plant: AccelerationPlant, q: Array, v: Array, tau: Array, *, eps: float = 1e-7
) -> float:
    """Largest ``|lambda|`` (1/s) of the open-loop state Jacobian at a state.

    Preconditions: ``q``, ``v`` and ``tau`` are finite vectors of equal length
    and ``eps`` is positive.  Postcondition: the result is finite and >= 0.
    """
    q, v, tau = (np.asarray(x, dtype=float) for x in (q, v, tau))
    if not (q.ndim == 1 and q.shape == v.shape == tau.shape):
        raise ValueError("q, v and tau must be vectors of equal length")
    if not all(np.isfinite(x).all() for x in (q, v, tau)):
        raise ValueError("q, v and tau must be finite")
    if not (np.isfinite(eps) and eps > 0.0):
        raise ValueError("eps must be positive")
    rate = float(
        np.max(np.abs(np.linalg.eigvals(_state_jacobian(plant, q, v, tau, eps))))
    )
    if not np.isfinite(rate):
        raise ValueError("plant Jacobian has nonfinite eigenvalues")
    return rate


def substeps_for(rate: float, dt_s: float, *, target: float = DEFAULT_TARGET) -> int:
    """Smallest power-of-two substeps (>= the default eight) with
    ``rate * dt_s / substeps <= target``.

    Preconditions: ``rate`` >= 0, ``dt_s`` > 0 and ``0 < target`` <= the RK4
    real stability limit.  Raises ``ValueError`` if more than
    ``MAX_SUBSTEPS`` would be needed.
    """
    if not (np.isfinite(rate) and rate >= 0.0):
        raise ValueError("rate must be non-negative and finite")
    if not (np.isfinite(dt_s) and dt_s > 0.0):
        raise ValueError("dt_s must be positive")
    if not 0.0 < target <= RK4_REAL_STABILITY_LIMIT:
        raise ValueError("target must lie in (0, RK4 real stability limit]")
    substeps = DEFAULT_SUBSTEPS
    while rate * dt_s / substeps > target:
        substeps *= 2
        if substeps > MAX_SUBSTEPS:
            raise ValueError(f"rate {rate} needs more than {MAX_SUBSTEPS} substeps")
    return substeps


def stable_substeps(record: Mapping[str, Any], dt_s: float, substeps: int) -> int:
    """Substeps keeping a run's recorded stiffest rate inside the RK4 region.

    ``record`` is an :func:`rk4_stability_record` of a run integrated at
    ``substeps`` per ``dt_s`` step.  Returns ``substeps`` when its rate is
    already inside the real stability limit, otherwise the count
    :func:`substeps_for` picks for that rate (#13041: the eight-substep
    deadlift sat at ``|z| = 4.24``).  Never fewer than ``substeps``, so an
    explicit policy (the loaded 64) is kept.  Preconditions: ``substeps`` is
    a positive int; the rate and ``dt_s`` as for :func:`substeps_for`.
    """
    if isinstance(substeps, bool) or not isinstance(substeps, int) or substeps < 1:
        raise ValueError("substeps must be a positive int")
    return max(substeps, substeps_for(float(record["max_rate_per_s"]), dt_s))


def rk4_stability_record(
    plant: AccelerationPlant,
    q_rows: Array,
    v_rows: Array,
    tau_rows: Array,
    substep_s: float,
) -> dict[str, float | int]:
    """Stiffest open-loop rate over sampled states and its RK4 margin.

    ``q_rows``, ``v_rows`` and ``tau_rows`` are (samples, nv) states with the
    effort held at each.  Postcondition: ``max_abs_z = max_rate_per_s *
    substep_s`` and ``max_amplification`` is ``|R(-max_abs_z)|``; the run is
    inside the RK4 real stability region iff ``max_abs_z`` is below
    :data:`RK4_REAL_STABILITY_LIMIT`.
    """
    q_rows, v_rows, tau_rows = (np.atleast_2d(x) for x in (q_rows, v_rows, tau_rows))
    if not (q_rows.shape == v_rows.shape == tau_rows.shape and q_rows.shape[0] > 0):
        raise ValueError("q_rows, v_rows and tau_rows must be equal non-empty arrays")
    if not (np.isfinite(substep_s) and substep_s > 0.0):
        raise ValueError("substep_s must be positive")
    rate = max(
        stiffest_rate(plant, q, v, tau)
        for q, v, tau in zip(q_rows, v_rows, tau_rows, strict=True)
    )
    z = rate * substep_s
    return {
        "samples": int(q_rows.shape[0]),
        "max_rate_per_s": float(rate),
        "substep_s": float(substep_s),
        "max_abs_z": float(z),
        "max_amplification": rk4_amplification(-z),
        "real_stability_limit": RK4_REAL_STABILITY_LIMIT,
    }
