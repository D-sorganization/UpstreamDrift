"""Operational-space (dynamic) mobility matrices and ellipsoids.

The mobility matrix at a task point is ``J M^-1 J^T`` (the inverse of the
operational-space inertia).  Unlike the kinematic manipulability in
``manipulability.py`` it includes the joint-space mass matrix ``M``.  A
loop-closure Jacobian ``Jc`` (e.g. the other hand holding the club) reduces
the mobility to that of the constrained system.

Units: for k = 3 the mobility is [kg^-1]: acceleration per unit force
(equivalently velocity change per unit impulse).  For k = 6 the rows
are ordered [linear; angular] and the blocks are not commensurate; pass
``scales`` (a ``CharacteristicScales``) to rescale the angular rows by 1/L so that all
rows are [m/s]-equivalent (angular velocity times L).

Reference: Khatib (1987), "A unified approach for motion and force control of
robot manipulators: the operational space formulation".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve

from src.shared.python.core.numerical_constants import EPSILON_SINGULARITY_DETECTION
from src.shared.python.loop_nullspace.constrained_dynamics import kkt_projector
from src.shared.python.loop_nullspace.decomposition import CharacteristicScales
from src.shared.python.spatial_algebra.spatial_vectors import skew

SYMMETRY_TOLERANCE = 1e-9  # [relative] allowed asymmetry of M

__all__ = [
    "Ellipsoid",
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
        scales: if given (k = 6 only), angular rows and columns are divided
            by the characteristic length ``scales.moment / scales.force`` [m]
            (0.05 m for the defaults) so the blocks share units.

    Returns:
        Symmetric positive-semidefinite (k, k) array; Loewner-below the
        unconstrained mobility when ``Jc`` is given.

    Raises:
        ValueError: on non-finite or mis-shaped inputs, non-symmetric or
            non-positive-definite ``M``, or ``scales`` with a 3-row ``J``.
    """
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
    if scales is not None and k != 6:
        raise ValueError("scales applies to a 6-row J only")

    try:  # raises on non-SPD M before any projector work
        factor = cho_factor(0.5 * (M + M.T), lower=True)
    except LinAlgError as exc:
        raise ValueError("M must be positive definite (Cholesky failed)") from exc

    if Jc is not None and Jc.shape[0] > 0:
        result = J @ kkt_projector(M, Jc) @ J.T
    else:
        result = J @ cho_solve(factor, J.T)

    if scales is not None:
        length = scales.moment / scales.force
        scale = np.concatenate([np.ones(3), np.full(3, 1.0 / length)])
        result = result * scale[:, None] * scale[None, :]
    return 0.5 * (result + result.T)


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
