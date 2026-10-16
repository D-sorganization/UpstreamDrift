"""Constrained forward dynamics through the KKT projector (NSX-30, #12570).

One closed-loop instant obeys ``M a + h = tau + J_c^T lambda`` with the closure
held at acceleration level, ``J_c a + Jdot_v = 0``.  Eliminating ``lambda``::

    Lambda_c = (J_c M^-1 J_c^T)^+
    P_c      = M^-1 (I - J_c^T Lambda_c J_c M^-1)
    a        = P_c (tau - h) - M^-1 J_c^T Lambda_c Jdot_v
    lambda   = -Lambda_c (J_c M^-1 (tau - h) + Jdot_v)

(``loop_nullspace.tex``, Section "Constrained Counterfactuals").  ``tau`` is the
*generalised* applied load (``S^T tau_a`` for actuated torques ``tau_a``).  The
acceleration is unique even when ``J_c`` is rank deficient; ``lambda`` is then
the minimum-norm member of its gauge family.

Baumgarte stabilisation is declared, never implicit: pass a :class:`Baumgarte`
to replace ``Jdot_v`` by ``Jdot_v + 2 alpha e_dot + beta^2 e`` where ``e`` is
the position-level closure error and ``e_dot`` its rate.  This is the single
implementation of the projector; the pendulum simulator calls it.  Pure numpy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from src.shared.python.loop_nullspace.evidence import Array

logger = logging.getLogger(__name__)

__all__ = [
    "Baumgarte",
    "closure_acceleration_residual",
    "constrained_acceleration",
    "kkt_projector",
    "solve_kkt_block",
]

_PINV_RTOL = 1e-12


@dataclass(frozen=True)
class Baumgarte:
    """Declared Baumgarte terms ``Jdot_v -> Jdot_v + 2 alpha e_dot + beta^2 e``.

    Attributes:
        alpha: Velocity-error gain [1/s], ``>= 0``.
        beta: Position-error gain [1/s], ``>= 0``.
        position_error: Closure error ``e = Phi(q)``, shape ``(m,)``.
        velocity_error: Its rate ``e_dot = J_c v``, shape ``(m,)``.
    """

    alpha: float
    beta: float
    position_error: Array
    velocity_error: Array

    def __post_init__(self) -> None:
        if not (np.isfinite(self.alpha) and self.alpha >= 0.0):
            raise ValueError(f"alpha must be finite and >= 0, got {self.alpha}")
        if not (np.isfinite(self.beta) and self.beta >= 0.0):
            raise ValueError(f"beta must be finite and >= 0, got {self.beta}")
        pos = np.array(self.position_error, dtype=np.float64).reshape(-1)
        vel = np.array(self.velocity_error, dtype=np.float64).reshape(-1)
        if pos.shape != vel.shape:
            raise ValueError(
                "position_error and velocity_error must have equal length, got "
                f"{pos.shape} and {vel.shape}"
            )
        if not (np.isfinite(pos).all() and np.isfinite(vel).all()):
            raise ValueError("Baumgarte errors must be finite")
        object.__setattr__(self, "position_error", pos)
        object.__setattr__(self, "velocity_error", vel)

    def drift(self, jdot_v: Array) -> Array:
        """``Jdot_v`` plus the declared stabilisation terms."""
        if jdot_v.shape != self.position_error.shape:
            raise ValueError(
                f"Baumgarte errors have shape {self.position_error.shape} but "
                f"Jdot_v has shape {jdot_v.shape}"
            )
        return (
            jdot_v
            + 2.0 * self.alpha * self.velocity_error
            + (self.beta**2 * self.position_error)
        )


def _vector(value: ArrayLike, n: int, name: str) -> Array:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},), got {arr.shape}")
    return arr


def _checked(
    mass: ArrayLike, jac: ArrayLike, jdot_v: ArrayLike
) -> tuple[Array, Array, Array]:
    m_arr = np.asarray(mass, dtype=np.float64)
    if m_arr.ndim != 2 or m_arr.shape[0] != m_arr.shape[1]:
        raise ValueError(f"M must be a square matrix (shape), got {m_arr.shape}")
    n = m_arr.shape[0]
    j_arr = np.asarray(jac, dtype=np.float64)
    if j_arr.ndim != 2 or j_arr.shape[1] != n:
        raise ValueError(f"J_c must have shape (m, {n}), got {j_arr.shape}")
    d_arr = _vector(jdot_v, j_arr.shape[0], "Jdot_v")
    for name, arr in (("M", m_arr), ("J_c", j_arr), ("Jdot_v", d_arr)):
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} must be finite")
    return m_arr, j_arr, d_arr


def _inverse_mass_products(mass: Array, jac: Array, rhs: Array) -> tuple[Array, Array]:
    """``(M^-1 rhs, M^-1 J^T)`` by Cholesky; raises if ``M`` is not SPD."""
    try:
        chol = np.linalg.cholesky(0.5 * (mass + mass.T))
    except np.linalg.LinAlgError as exc:
        raise ValueError("M must be symmetric positive definite") from exc

    def solve(b: Array) -> Array:
        return np.linalg.solve(chol.T, np.linalg.solve(chol, b))

    return solve(rhs), solve(jac.T)


def _is_spd(mass: Array) -> bool:
    try:
        np.linalg.cholesky(0.5 * (mass + mass.T))
    except np.linalg.LinAlgError:
        return False
    return True


def _block_solve(
    mass: Array, jac: Array, free_load: Array, drift: Array
) -> tuple[Array, Array]:
    """Augmented-system solve for a semi-definite ``M`` (see the caller)."""
    n, m = mass.shape[0], jac.shape[0]
    kkt = np.block([[mass, -jac.T], [jac, np.zeros((m, m))]])
    rhs = np.concatenate([free_load, -drift])
    # Minimum-norm least squares: exact for the consistent singular system a
    # massless degree of freedom leaves (zero columns in M and J), and equal
    # to the direct solve whenever the block matrix is nonsingular.
    sol = np.linalg.lstsq(kkt, rhs, rcond=None)[0]
    if not np.isfinite(sol).all():
        raise ValueError("constrained dynamics produced non-finite values")
    return np.asarray(sol[:n]), np.asarray(sol[n:])


def kkt_projector(mass: ArrayLike, jac: ArrayLike) -> Array:
    """``P_c = M^-1 (I - J_c^T Lambda_c J_c M^-1)`` with ``Lambda_c`` a pinv.

    Preconditions: ``M`` symmetric positive definite ``(n, n)``, ``J_c``
    ``(m, n)``, both finite (``ValueError``).  Postconditions: ``J_c P_c = 0``,
    ``P_c`` symmetric, and ``P_c`` annihilates nothing the closure permits.
    """
    m_arr, j_arr, _ = _checked(mass, jac, np.zeros(np.shape(jac)[0]))
    n = m_arr.shape[0]
    minv, minv_jt = _inverse_mass_products(m_arr, j_arr, np.eye(n))
    if j_arr.shape[0] == 0:
        return minv
    lam_c = np.linalg.pinv(j_arr @ minv_jt, rcond=_PINV_RTOL, hermitian=True)
    return np.asarray(minv - minv_jt @ lam_c @ minv_jt.T)


def constrained_acceleration(
    M: ArrayLike,  # noqa: N803 - spec name
    h: ArrayLike,
    tau: ArrayLike,
    J_c: ArrayLike,  # noqa: N803 - spec name
    Jdot_v: ArrayLike,  # noqa: N803 - spec name
    *,
    baumgarte: Baumgarte | None = None,
    singular_mass_fallback: bool = False,
) -> tuple[Array, Array]:
    """Closure-consistent acceleration and reaction ``(a, lambda)``.

    Solves ``M a + h = tau + J_c^T lambda`` with ``J_c a + Jdot_v = 0`` by the
    KKT projector.  ``tau`` is the generalised load (``S^T tau_a``), not the
    actuated vector.  ``lambda`` is re-solved on every call; it is never held.

    Preconditions (``ValueError``): ``M`` symmetric positive definite
    ``(n, n)``; ``h`` and ``tau`` ``(n,)``; ``J_c`` ``(m, n)``; ``Jdot_v``
    ``(m,)``; everything finite; a ``baumgarte`` whose errors have length ``m``.

    Postconditions: ``J_c a + Jdot_v_eff = 0`` and
    ``M a + h - tau - J_c^T lambda = 0`` to ``1e-9`` scaled for full-rank or
    consistent rank-deficient ``J_c``; ``lambda`` is the minimum-norm
    multiplier (unique only up to ``null(J_c^T)``).

    ``singular_mass_fallback`` is for degenerate legacy models whose ``M`` is
    only positive semi-definite (the pendulum golfer's massless club degree of
    freedom): the projector does not exist there, so the equivalent augmented
    block system ``[[M, -J^T], [J, 0]]`` is solved (least squares if singular),
    which agrees with the projector whenever ``M`` is positive definite.
    Without the flag a non-positive-definite ``M`` raises ``ValueError``.
    """
    m_arr, j_arr, drift = _checked(M, J_c, Jdot_v)
    n = m_arr.shape[0]
    bias = _vector(h, n, "h")
    load = _vector(tau, n, "tau")
    if not (np.isfinite(bias).all() and np.isfinite(load).all()):
        raise ValueError("h and tau must be finite")
    if baumgarte is not None:
        drift = baumgarte.drift(drift)
    if singular_mass_fallback and not _is_spd(m_arr):
        return _block_solve(m_arr, j_arr, load - bias, drift)
    free, minv_jt = _inverse_mass_products(m_arr, j_arr, load - bias)
    if j_arr.shape[0] == 0:
        return free, np.zeros(0)
    lam_c = np.linalg.pinv(j_arr @ minv_jt, rcond=_PINV_RTOL, hermitian=True)
    lam = -lam_c @ (j_arr @ free + drift)
    return np.asarray(free + minv_jt @ lam), np.asarray(lam)


def closure_acceleration_residual(
    J_c: ArrayLike,
    a: ArrayLike,
    Jdot_v: ArrayLike,  # noqa: N803 - spec name
) -> Array:
    """``J_c a + Jdot_v``: zero when the acceleration respects the closure."""
    j_arr = np.asarray(J_c, dtype=np.float64)
    acc = _vector(a, j_arr.shape[1], "a")
    return np.asarray(j_arr @ acc + _vector(Jdot_v, j_arr.shape[0], "Jdot_v"))


def solve_kkt_block(
    mass: ArrayLike, jac: ArrayLike, free_load: ArrayLike, drift: ArrayLike
) -> tuple[Array, Array]:
    """Solve ``[[M, -J^T], [J, 0]] [x; y] = [free_load; -drift]`` (NSX-84).

    The differentiated KKT system of the tangent linearization has the same
    block matrix as the forward solve, so it reuses the same minimum-norm
    least-squares solve.  Preconditions (``ValueError``): ``M`` square
    ``(n, n)``, ``J`` ``(m, n)``, ``free_load`` ``(n,)``, ``drift`` ``(m,)``,
    everything finite.  Returns ``(x, y)``.
    """
    m_arr, j_arr, d_arr = _checked(mass, jac, drift)
    load = _vector(free_load, m_arr.shape[0], "free_load")
    if not np.isfinite(load).all():
        raise ValueError("free_load must be finite")
    return _block_solve(m_arr, j_arr, load, d_arr)
