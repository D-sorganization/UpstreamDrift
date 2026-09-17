"""Local scaled shooting-node retraction with an implicit derivative."""

from collections.abc import Callable
from typing import NamedTuple
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root

Array = NDArray[np.float64]


class NodeRetraction(NamedTuple):
    """Closure-valid node and physical derivative versus chart coordinates."""

    state: Array
    state_jacobian: Array
    closure_max_abs: float
    scaled_displacement: float


def scaled_tangent_basis(jacobian: Array, state_scales: Array) -> Array:
    """Return an orthonormal chart basis tangent in scaled state coordinates.

    If physical perturbations are `dx = scales * dy`, this returns N with
    J*diag(scales)*N=0. The weld must have full row rank; otherwise choosing a
    chart would silently hide a singular closure.
    """
    matrix = np.asarray(jacobian, dtype=float)
    scales = np.asarray(state_scales, dtype=float)
    if (
        matrix.ndim != 2
        or not matrix.shape[0]
        or matrix.shape[0] >= matrix.shape[1]
        or scales.shape != (matrix.shape[1],)
        or not np.isfinite(matrix).all()
        or not np.isfinite(scales).all()
        or np.any(scales <= 0)
    ):
        raise ValueError("Invalid closure Jacobian or positive state scales")
    _, singular, right = np.linalg.svd(matrix * scales[None, :], full_matrices=True)
    if singular[-1] <= 1e-10 * singular[0]:
        raise ValueError("Closure Jacobian lost row rank")
    rank = matrix.shape[0]
    basis = right[rank:].T.copy()
    if basis.shape != (matrix.shape[1], matrix.shape[1] - rank):
        raise ValueError("Invalid tangent basis dimension")
    basis.setflags(write=False)
    return basis


def retract_node(
    reference: Array,
    basis: Array,
    coordinates: Array,
    closure: Callable[[Array], Array],
    jacobian: Callable[[Array], Array],
    *,
    state_scales: Array,
    residual_scales: Array,
    radius: float,
    tolerance: float = 1e-9,
) -> NodeRetraction:
    """Solve c(x)=0 and N.T ((x-reference)/scales)=coordinates locally.

    N is orthonormal in dimensionless scaled-state space and tangent at the
    reference. A finite radius limits this chart; rank loss or failed closure
    raises. Callers must qualify closure derivatives and choose physical scales.
    This correction belongs to a shooting node, never inside forward integration.
    """
    ref, n, z, d, cs = (
        np.array(v, dtype=float, copy=True)
        for v in (reference, basis, coordinates, state_scales, residual_scales)
    )
    if (
        ref.ndim != 1
        or not ref.size
        or n.ndim != 2
        or n.shape[0] != ref.size
        or not 0 < n.shape[1] < ref.size
        or z.shape != (n.shape[1],)
        or d.shape != ref.shape
        or cs.shape != (ref.size - n.shape[1],)
        or any(not np.isfinite(v).all() for v in (ref, n, z, d, cs))
        or np.any(d <= 0)
        or np.any(cs <= 0)
        or not np.isfinite(radius)
        or radius <= 0
        or not np.isfinite(tolerance)
        or tolerance <= 0
    ):
        raise ValueError("Invalid node chart dimensions, scales or tolerances")
    if np.linalg.norm(z) > radius:
        raise ValueError("Requested node exceeds chart radius")
    if not np.allclose(n.T @ n, np.eye(n.shape[1]), atol=1e-10, rtol=0):
        raise ValueError("Node tangent basis must be orthonormal")

    def c(x: Array) -> Array:
        value = np.asarray(closure(x), dtype=float)
        if value.shape != cs.shape or not np.isfinite(value).all():
            raise ValueError("Invalid closure residual")
        return value / cs

    def j(x: Array) -> Array:
        value = np.asarray(jacobian(x), dtype=float)
        if value.shape != (cs.size, ref.size) or not np.isfinite(value).all():
            raise ValueError("Invalid closure Jacobian")
        return value * d[None, :] / cs[:, None]

    if np.max(abs(c(ref))) > tolerance:
        raise ValueError("Reference node violates closure")
    if np.max(abs(j(ref) @ n)) > 1e-7:
        raise ValueError("Node basis is not tangent to closure")

    def residual(y: Array) -> Array:
        return np.concatenate((c(ref + d * y), n.T @ y - z))

    def matrix(y: Array) -> Array:
        return np.vstack((j(ref + d * y), n.T))

    result = root(residual, n @ z, jac=matrix, method="hybr", options={"xtol": 1e-10})
    y = np.asarray(result.x)
    if not np.isfinite(y).all() or np.max(abs(residual(y))) > tolerance:
        raise ValueError("Node retraction failed closure or coordinate contract")
    displacement = float(np.linalg.norm(y))
    if displacement > radius:
        raise ValueError("Retracted node exceeds chart radius")
    m = matrix(y)
    singular = np.linalg.svd(m, compute_uv=False)
    if singular[-1] <= 1e-10 * singular[0]:
        raise ValueError("Node chart lost rank")
    rhs = np.vstack((np.zeros((cs.size, z.size)), np.eye(z.size)))
    dx = d[:, None] * np.linalg.solve(m, rhs)
    state = ref + d * y
    if not np.isfinite(dx).all():
        raise ValueError("Nonfinite node derivative")
    state.setflags(write=False)
    dx.setflags(write=False)
    return NodeRetraction(state, dx, float(np.max(abs(c(state)))), displacement)
