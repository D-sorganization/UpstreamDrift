"""Operational-space (dynamic) mobility matrices and ellipsoids.

The mobility matrix at a task point is ``J M^-1 J^T`` (the inverse of the
operational-space inertia).  Unlike the kinematic manipulability in
``manipulability.py`` it includes the joint-space mass matrix ``M``.  A
loop-closure Jacobian ``Jc`` (e.g. the other hand holding the club) reduces
the mobility to that of the constrained system.

Units: for k = 3 the mobility is [kg^-1]: acceleration per unit force
(equivalently velocity change per unit impulse).  For k = 6 the rows
are ordered [linear; angular] and the blocks are not commensurate; pass
``scales`` (a ``CharacteristicScales``) to rescale the angular rows by L so that all
rows are [m/s]-equivalent (angular velocity times L), yielding uniform [kg^-1] blocks.

Reference: Khatib (1987), "A unified approach for motion and force control of
robot manipulators: the operational space formulation".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve

from src.shared.python.core.numerical_constants import EPSILON_SINGULARITY_DETECTION
from src.shared.python.loop_nullspace.constrained_dynamics import kkt_projector
from src.shared.python.loop_nullspace.decomposition import CharacteristicScales
from src.shared.python.spatial_algebra.spatial_vectors import skew

SYMMETRY_TOLERANCE = 1e-9  # [relative] allowed asymmetry of M

__all__ = [
    "CapacityEllipsoid",
    "Ellipsoid",
    "acceleration_capacity",
    "drift_acceleration",
    "force_ellipsoid",
    "mobility_matrix",
    "transport_spatial_mobility",
    "velocity_ellipsoid",
]


@dataclass(frozen=True)
class Ellipsoid:
    """Ellipsoid in an orthonormal eigenbasis.

    Attributes:
        axes: (k, k) matrix whose columns are orthonormal principal axes.
        radii: (k,) semi-axis lengths, ascending with the eigenvalues.
        singular: (k,) bool mask of axes whose radius was clamped (force
            ellipsoids only; all False for velocity ellipsoids).
    """

    axes: np.ndarray
    radii: np.ndarray
    singular: np.ndarray


def _finite_matrix(name: str, a: np.ndarray) -> np.ndarray:
    arr = np.asarray(a, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _validated_task(
    J: np.ndarray, M: np.ndarray, Jc: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, tuple]:
    """Validate ``(J, M, Jc)``; return them plus the Cholesky factor of ``M``."""
    J = _finite_matrix("J", J)
    M = _finite_matrix("M", M)
    k, nv = J.shape
    if k not in (3, 6):
        raise ValueError(f"J must have 3 or 6 rows, got {k}")
    if M.shape != (nv, nv):
        raise ValueError(f"M must be ({nv}, {nv}) to match J, got {M.shape}")
    if not np.allclose(M, M.T, rtol=SYMMETRY_TOLERANCE, atol=SYMMETRY_TOLERANCE):
        raise ValueError("M must be symmetric")
    if Jc is not None:
        Jc = _finite_matrix("Jc", Jc)
        if Jc.shape[1] != nv:
            raise ValueError(f"Jc must have {nv} columns, got {Jc.shape[1]}")
    try:  # raises on non-SPD M before any projector work
        factor = cho_factor(0.5 * (M + M.T), lower=True)
    except LinAlgError as exc:
        raise ValueError("M must be positive definite (Cholesky failed)") from exc
    return J, M, Jc, factor


def mobility_matrix(
    J: np.ndarray,
    M: np.ndarray,
    Jc: np.ndarray | None = None,
    *,
    scales: CharacteristicScales | None = None,
) -> np.ndarray:
    """Return the (k, k) operational-space mobility ``J A_c J^T``.

    Without ``Jc`` this is ``J M^-1 J^T``.  With a loop-closure Jacobian it is
    ``J P_c J^T`` where ``P_c`` is the closure projector of
    ``loop_nullspace.constrained_dynamics.kkt_projector`` (not re-derived here).

    Args:
        J: (k, nv) task Jacobian, k in {3, 6} (rows [linear; angular]).
        M: (nv, nv) symmetric positive-definite joint-space mass matrix.
        Jc: optional (c, nv) closure Jacobian; may be rank deficient.
        scales: if given (k = 6 only), angular rows and columns are multiplied
            by the characteristic length ``scales.moment / scales.force`` [m]
            (0.05 m for the defaults) so all blocks share the uniform unit [kg^-1].

    Returns:
        Symmetric positive-semidefinite (k, k) array; Loewner-below the
        unconstrained mobility when ``Jc`` is given.

    Raises:
        ValueError: on non-finite or mis-shaped inputs, non-symmetric or
            non-positive-definite ``M``, or ``scales`` with a 3-row ``J``.
    """
    J, M, Jc, factor = _validated_task(J, M, Jc)
    k = J.shape[0]
    if scales is not None and k != 6:
        raise ValueError("scales applies to a 6-row J only")

    if Jc is not None and Jc.shape[0] > 0:
        result = J @ kkt_projector(M, Jc) @ J.T
    else:
        result = J @ cho_solve(factor, J.T)

    if scales is not None:
        length = scales.moment / scales.force
        scale = np.concatenate([np.ones(3), np.full(3, length)])
        result = result * scale[:, None] * scale[None, :]
    return 0.5 * (result + result.T)


@dataclass(frozen=True)
class CapacityEllipsoid(Ellipsoid):
    """:class:`Ellipsoid` of task accelerations [m/s^2] plus its ``centre``.

    ``radii`` are acceleration semi-axes; ``singular`` flags axes with no
    actuator authority; ``centre`` (k,) is the shift ``J A B mid`` from
    asymmetric torque limits (zero for symmetric limits).
    """

    centre: np.ndarray = field(default_factory=lambda: np.zeros(0))


def _limits(name: str, v: object, nu: int) -> np.ndarray:
    arr = np.asarray(v, dtype=float)
    if arr.shape != (nu,):
        raise ValueError(f"{name} must have shape ({nu},), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def acceleration_capacity(
    J: np.ndarray,
    M: np.ndarray,
    B: np.ndarray,
    tau_lo: np.ndarray,
    tau_hi: np.ndarray,
    Jc: np.ndarray | None = None,
    *,
    tol: float = EPSILON_SINGULARITY_DETECTION,
) -> CapacityEllipsoid:
    """Task-acceleration capacity ellipsoid of the actuator torque box.

    With ``M a + h = B u (+ Jc^T lambda)`` and ``A = M^-1`` (or the closure
    projector ``P_c`` of ``kkt_projector`` when ``Jc`` is given), torques
    ``u in [tau_lo, tau_hi]`` reach the task accelerations of the zonotope
    ``J A B [tau_lo, tau_hi]``.  Writing ``u = mid + T s`` with
    ``T = diag((tau_hi - tau_lo)/2)``, ``|s_i| <= 1``, and ``G = J A B T``:

    - ``Q_a = G G^T`` (symmetrised); radii are ``sqrt(eig(Q_a))`` [m/s^2];
    - the centre is ``J A B (tau_hi + tau_lo)/2``.

    The result is an ellipsoidal **inner approximation** of the zonotope
    (the unit ball of ``s`` lies inside the box; the box corners reach up to
    ``sqrt(nu)`` further).  Gravity/Coriolis drift is excluded: see
    :func:`drift_acceleration`.  Axes with ``eig <= tol * max(eig_max, 1)``
    are flagged in ``singular`` (radii are kept, not clamped, since the
    capacity is a bound, not an inverse).

    Args:
        J: (k, nv) task Jacobian, k in {3, 6}.
        M: (nv, nv) symmetric positive-definite mass matrix.
        B: (nv, nu) map from actuator torques to generalised forces.
        tau_lo, tau_hi: (nu,) finite limits with ``tau_lo < 0 < tau_hi``.
        Jc: optional (c, nv) closure Jacobian (may be rank deficient).

    Raises:
        ValueError: on bad shapes, non-finite values, non-SPD ``M`` or limits
            that do not bracket zero strictly.
    """
    if not tol > 0:
        raise ValueError("tol must be positive")
    J, M, Jc, factor = _validated_task(J, M, Jc)
    B = _finite_matrix("B", B)
    nv = M.shape[0]
    if B.shape[0] != nv:
        raise ValueError(f"B must have {nv} rows, got {B.shape[0]}")
    nu = B.shape[1]
    lo = _limits("tau_lo", tau_lo, nu)
    hi = _limits("tau_hi", tau_hi, nu)
    if not (np.all(lo < 0) and np.all(hi > 0)):
        raise ValueError("torque limits must satisfy tau_lo < 0 < tau_hi")

    if Jc is not None and Jc.shape[0] > 0:
        JA = J @ kkt_projector(M, Jc)
    else:
        JA = cho_solve(factor, J.T).T
    JAB = JA @ B
    G = JAB * ((hi - lo) / 2)[None, :]
    lam, vec = np.linalg.eigh(0.5 * (G @ G.T + (G @ G.T).T))
    lam = np.maximum(lam, 0.0)
    singular = lam <= tol * max(float(lam.max()), 1.0)
    return CapacityEllipsoid(vec, np.sqrt(lam), singular, JAB @ ((hi + lo) / 2))


def drift_acceleration(
    J: np.ndarray,
    M: np.ndarray,
    h: np.ndarray,
    Jdot_v: np.ndarray,
    Jc: np.ndarray | None = None,
    Jcdot_v: np.ndarray | None = None,
) -> np.ndarray:
    """Task acceleration with zero actuator torque: ``c = J a0 + Jdot v``.

    ``a0 = -A h - M^-1 Jc^T Lambda_c Jcdot_v`` with ``Lambda_c`` the
    pseudo-inverse of ``Jc M^-1 Jc^T`` (the ``tau = 0`` case of
    ``constrained_dynamics.constrained_acceleration``, same sign convention
    ``M a + h = tau + Jc^T lambda``); without ``Jc``, ``a0 = -M^-1 h``.
    The feasible task accelerations are ``c + capacity``.

    Raises:
        ValueError: on bad shapes, non-finite values or non-SPD ``M``.
    """
    J, M, Jc, factor = _validated_task(J, M, Jc)
    nv = M.shape[0]
    h = _limits("h", h, nv)
    jdv = _limits("Jdot_v", Jdot_v, J.shape[0])
    if Jc is None or Jc.shape[0] == 0:
        return np.asarray(J @ -cho_solve(factor, h) + jdv)
    jcdv = (
        np.zeros(Jc.shape[0])
        if Jcdot_v is None
        else _limits("Jcdot_v", Jcdot_v, Jc.shape[0])
    )
    minv_jct = cho_solve(factor, Jc.T)
    lam_c = np.linalg.pinv(Jc @ minv_jct, hermitian=True)
    a0 = -kkt_projector(M, Jc) @ h - minv_jct @ (lam_c @ jcdv)
    return np.asarray(J @ a0 + jdv)


def _eigh_psd(mobility: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = _finite_matrix("mobility", mobility)
    if a.shape[0] != a.shape[1]:
        raise ValueError(f"mobility must be square, got {a.shape}")
    if not np.allclose(a, a.T, rtol=SYMMETRY_TOLERANCE, atol=SYMMETRY_TOLERANCE):
        raise ValueError("mobility must be symmetric")
    lam, vec = np.linalg.eigh(0.5 * (a + a.T))
    return lam, vec


def velocity_ellipsoid(mobility: np.ndarray) -> Ellipsoid:
    """Mobility (velocity) ellipsoid: radii ``sqrt(max(lambda, 0))`` [kg^-1/2]."""
    lam, vec = _eigh_psd(mobility)
    return Ellipsoid(vec, np.sqrt(np.maximum(lam, 0.0)), np.zeros(lam.shape, bool))


def force_ellipsoid(
    mobility: np.ndarray, *, tol: float = EPSILON_SINGULARITY_DETECTION
) -> Ellipsoid:
    """Force ellipsoid: radii ``1/sqrt(lambda)`` [kg^1/2], the inertial dual.

    ``1/lambda`` along each axis is the effective mass [kg] there.  This is an
    inertial property of the (constrained) system, not a force capacity.

    Eigenvalues at or below ``tol * max(lambda_max, 1)`` (the unit floor keeps
    round-off noise on a locked system singular) are flagged in ``singular``
    and clamped to that floor, so radii stay finite (cap = ``1/sqrt(floor)``) and renderers can special-case them.
    """
    if not tol > 0:
        raise ValueError("tol must be positive")
    lam, vec = _eigh_psd(mobility)
    lam_max = float(lam.max()) if lam.size else 0.0
    floor = tol * max(lam_max, 1.0)
    singular = lam <= floor
    return Ellipsoid(vec, 1.0 / np.sqrt(np.maximum(lam, floor)), singular)


def transport_spatial_mobility(mobility6: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Move a 6x6 [linear; angular] mobility from point a to b = a + r.

    ``r`` [m] is the vector a -> b in the frame of the mobility.  With
    ``X = [[I, -skew(r)], [0, I]]`` (twist_b = X twist_a, power invariance
    gives wrench_a = X^T wrench_b), ``Lambda_b^-1 = X Lambda_a^-1 X^T``.
    Valid only for points on one rigid link (e.g. across a rigid weld).
    """
    mob = _finite_matrix("mobility6", mobility6)
    if mob.shape != (6, 6):
        raise ValueError(f"mobility6 must be (6, 6), got {mob.shape}")
    r_arr = np.asarray(r, dtype=float)
    if r_arr.shape != (3,) or not np.all(np.isfinite(r_arr)):
        raise ValueError("r must be a finite 3-vector")
    X = np.eye(6)
    X[:3, 3:] = -skew(r_arr)
    out = X @ mob @ X.T
    return 0.5 * (out + out.T)
