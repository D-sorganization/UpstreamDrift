"""Force redistribution solvers on a null space (NSX-03 split from force_nullspace).

NSX-24 (#12567): both solvers are convex (weighted least squares plus a rate
penalty over bounds, linear rows and exact Coulomb cones), so they are shims
over the conic core (:func:`src.shared.python.loop_nullspace.qp_core.solve_qp`,
Clarabel) and no longer run SLSQP.  A missing ``clarabel`` raises
``ImportError`` naming the ``loop-nullspace`` extra; there is no fallback.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import scipy.sparse as sp

from src.shared.python.loop_nullspace.qp_core import friction_cone_rows, solve_qp
from src.shared.python.loop_nullspace.solvers import SOCBlock, SolveStatus

from ._common import Array, _array
from .constraints import ForceConstraints
from .space import ForceNullSpace

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RedistributionResult:
    """Audited numerical candidate; feasible never means replay accepted."""

    forces: Array
    feasible: bool
    converged: bool
    equality_error: float
    constraint_violation: float
    cost: float
    message: str


def _margin_violation(margins: Array, *, finite: bool) -> float:
    """Reduce constraint margins to a single non-negative violation scalar.

    Returns 0 when every margin is non-negative, the magnitude of the worst
    negative margin otherwise, or infinity when the solution (or any margin)
    is non-finite.
    """
    if finite and np.isfinite(margins).all():
        return max(0.0, float(-np.min(margins)))
    return float("inf")


def _finalize_redistribution(
    x: Array,
    *,
    finite: bool,
    converged: bool,
    error: float,
    violation: float,
    cost: float,
    message: str,
    tolerance: float,
) -> RedistributionResult:
    """Assemble the immutable `RedistributionResult` for a solved system."""
    feasible = _is_solution_feasible(
        finite=finite,
        cost=cost,
        error=error,
        violation=violation,
        tolerance=tolerance,
    )
    x.setflags(write=False)
    return RedistributionResult(
        x, bool(feasible), converged, error, violation, cost, message
    )


def _is_solution_feasible(
    *,
    finite: bool,
    cost: float,
    error: float,
    violation: float,
    tolerance: float,
) -> bool:
    """Evaluate solution feasibility against tolerance guards."""
    feasibility_tolerance = max(tolerance, 1e-6)
    return bool(
        finite
        and np.isfinite(cost)
        and error <= feasibility_tolerance
        and violation <= feasibility_tolerance
    )


def _stacked_problem(
    spaces: Sequence[ForceNullSpace],
    constraints: Sequence[ForceConstraints],
    w: Array,
    references: Array,
    smoothness_weight: float,
    dts: Array,
) -> tuple[sp.csr_matrix, Array, dict[str, Any]]:
    """Assemble the convex QP in the stacked null-space coordinates ``z``.

    ``x_k = particular_k + basis_k z_k``.  Cost
    ``sum ||W (x_k - ref_k)||^2 + smoothness sum ||(x_{k+1}-x_k)/dt_k||^2``
    is evaluated on physical ``x`` (basis-invariant).  Returns the Hessian,
    the gradient offset and the keyword rows for ``solve_qp``.
    """
    t_count, n = len(spaces), spaces[0].particular.size
    sizes = [s.basis.shape[1] for s in spaces]
    offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
    total = int(offsets[-1])
    basis = sp.lil_matrix((t_count * n, total))
    for k, s in enumerate(spaces):
        basis[k * n : (k + 1) * n, offsets[k] : offsets[k + 1]] = s.basis
    b_mat = basis.tocsr()
    x0 = np.concatenate([s.particular for s in spaces])
    w_sq = sp.diags(np.tile(w**2, t_count))
    quad = w_sq
    if t_count > 1 and smoothness_weight > 0.0:
        diff_lil = sp.lil_matrix(((t_count - 1) * n, t_count * n))
        for k in range(t_count - 1):
            row_sel = slice(k * n, (k + 1) * n)
            diff_lil[row_sel, k * n : (k + 1) * n] = -sp.identity(n) / dts[k]
            diff_lil[row_sel, (k + 1) * n : (k + 2) * n] = sp.identity(n) / dts[k]
        diff = diff_lil.tocsr()
        quad = w_sq + smoothness_weight * (diff.T @ diff)
    hess = (2.0 * (b_mat.T @ quad @ b_mat)).tocsr()
    grad = 2.0 * b_mat.T @ (quad @ x0 - w_sq @ references.ravel())
    rows = _constraint_rows(constraints, b_mat, x0, n)
    return hess, grad, rows


def _constraint_rows(
    constraints: Sequence[ForceConstraints],
    b_mat: sp.csr_matrix,
    x0: Array,
    n: int,
) -> dict[str, Any]:
    """Bounds, linear rows and exact friction cones as rows in ``z``."""
    t_count = len(constraints)
    lower = np.concatenate([c.lower for c in constraints])
    upper = np.concatenate([c.upper_bound for c in constraints])
    fixed = np.isclose(lower, upper, atol=1e-12)
    a_eq, b_eq = b_mat[fixed], lower[fixed] - x0[fixed]
    ub_blocks, ub_rhs = [], []
    for sel, sign, rhs in (
        (~fixed, 1.0, upper - x0),
        (~fixed, -1.0, x0 - lower),
    ):
        ub_blocks.append(sign * b_mat[sel])
        ub_rhs.append(rhs[sel])
    cones = []
    for k, c in enumerate(constraints):
        if c.matrix is not None and c.upper is not None and c.matrix.shape[0] > 0:
            g = sp.hstack(
                [sp.csr_matrix((c.matrix.shape[0], k * n)), sp.csr_matrix(c.matrix)]
                + [sp.csr_matrix((c.matrix.shape[0], (t_count - k - 1) * n))]
            ).tocsr()
            ub_blocks.append(g @ b_mat)
            ub_rhs.append(c.upper - g @ x0)
        for cone in c.cones:
            i0, i1, i2 = cone.indices
            idx = (k * n + i0, k * n + i1, k * n + i2)
            block, a_uni, b_uni = friction_cone_rows(
                t_count * n, idx, cone.frame, cone.mu
            )
            ub_blocks.append(a_uni @ b_mat)
            ub_rhs.append(b_uni - a_uni @ x0)
            cones.append(_cone_in_z(block, b_mat, x0))
    return {
        "A_eq": a_eq,
        "b_eq": b_eq,
        "A_ub": sp.vstack(ub_blocks).tocsr(),
        "b_ub": np.concatenate(ub_rhs),
        "cones": cones,
    }


def _cone_in_z(block: SOCBlock, b_mat: sp.csr_matrix, x0: Array) -> SOCBlock:
    """Express an x-space SOC block ``-A x in SOC`` on ``x = x0 + B z``."""
    return SOCBlock(A=block.A @ b_mat, b=-(block.A @ x0))


def _solve_stacked(
    spaces: Sequence[ForceNullSpace],
    constraints: Sequence[ForceConstraints],
    w: Array,
    references: Array,
    smoothness_weight: float,
    dts: Array,
    *,
    tolerance: float,
    max_iterations: int,
) -> tuple[Array, bool, str]:
    """Solve for the stacked ``z``; non-optimal statuses return ``z = 0``.

    ``z = 0`` is the particular solution: it satisfies the equality, so the
    independent audit then reports any bound violation honestly (no repair,
    no clipping) and ``converged`` is False.
    """
    hess, grad, rows = _stacked_problem(
        spaces, constraints, w, references, smoothness_weight, dts
    )
    total = hess.shape[0]
    res = solve_qp(
        hess,
        grad,
        A_eq=rows["A_eq"],
        b_eq=rows["b_eq"],
        A_ub=rows["A_ub"],
        b_ub=rows["b_ub"],
        cones=rows["cones"],
        tol=max(tolerance, 1e-9),
        max_iter=max_iterations,
    )
    if res.status is SolveStatus.OPTIMAL:
        if res.x is None:
            raise ValueError("optimal solve returned no solution vector")
        return np.asarray(res.x, dtype=np.float64), True, "Optimal (conic core)"
    message = f"{res.status.value}: {res.reason}".rstrip(": ")
    logger.warning("Force redistribution not solved (%s)", message)
    return np.zeros(total), False, message


def redistribute_forces(
    space: ForceNullSpace,
    weights: Array,
    constraints: ForceConstraints,
    *,
    reference: Array | None = None,
    tolerance: float = 1e-8,
    max_iterations: int = 300,
) -> RedistributionResult:
    """Minimize ||weights * (x - reference)||² within the dynamics null space.

    Weights are inverse physical scales (not squared cost coefficients).
    A physical reference may encode a previous force solution; this is not a
    trajectory smoothing solver. Returned feasibility is independently audited
    even if the conic solver reports success. No clipping or equilibrium repair
    occurs. Solved by the conic core (NSX-24); a non-optimal solver status
    yields ``converged=False`` and the particular solution, audited as is.
    tolerance applies to scaled equality residuals and the caller's constraint
    margins; callers must choose consistent row units for linear inequalities.
    """
    n = space.particular.size
    w = _array(weights, 1, "weights")
    target = _array(np.zeros(n) if reference is None else reference, 1, "reference")
    if w.shape != (n,) or np.any(w <= 0):
        raise ValueError("weights must be positive and match force dimension")
    if target.shape != (n,) or constraints.lower.shape != (n,):
        raise ValueError("reference and constraints must match force dimension")
    if not np.isfinite(tolerance) or tolerance <= 0 or max_iterations < 1:
        raise ValueError("tolerance and max_iterations must be positive")
    basis = space.basis
    if basis.shape[1]:
        z, converged, message = _solve_stacked(
            [space],
            [constraints],
            w,
            target[None, :],
            0.0,
            np.zeros(0),
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
    else:
        z = np.empty(0)
        converged, message = True, "Unique equality solution; constraints audited"
    x = space.particular + basis @ z
    residual = w * (x - target)
    finite = bool(np.isfinite(x).all())
    error = (
        float(np.max(np.abs((space.matrix @ x - space.rhs) / space.row_scale)))
        if finite
        else float("inf")
    )
    violation = _margin_violation(constraints.margins(x), finite=finite)
    return _finalize_redistribution(
        x,
        finite=finite,
        converged=converged,
        error=error,
        violation=violation,
        cost=float(residual @ residual),
        message=message,
        tolerance=tolerance,
    )


def redistribute_trajectory(
    spaces: Sequence[ForceNullSpace],
    weights: Array,
    constraints: Sequence[ForceConstraints],
    time_s: Array,
    *,
    smoothness_weight: float = 1.0,
    tolerance: float = 1e-8,
    max_iterations: int = 500,
) -> RedistributionResult:
    """Solve multi-frame null-space force trajectories penalizing physical rates.

    Objective:
      sum_k ||W x_k||² + lambda sum_k ||(x_{k+1} - x_k) / dt_k||²
    Subject to:
      A_k x_k = b_k  (embedded via x_k = x_{p,k} + N_k z_k)
      constraints[k].margins(x_k) >= 0

    Because the smoothing penalty is evaluated strictly on physical force/torque
    derivatives dx/dt rather than z derivatives, the formulation is strictly
    invariant to null-space basis column sign changes or basis rotations.
    Solved as one convex QP by the conic core (NSX-24).
    """
    t_count = len(spaces)
    if t_count == 0:
        raise ValueError("spaces must be nonempty")
    time = _array(time_s, 1, "time_s")
    if time.shape != (t_count,):
        raise ValueError("time_s must match number of spaces")
    if len(constraints) != t_count:
        raise ValueError("constraints must match number of spaces")
    n = spaces[0].particular.size
    if any(s.particular.size != n for s in spaces):
        raise ValueError("all spaces must have identical force dimensions")
    w = _array(weights, 1, "weights")
    if w.shape != (n,) or np.any(w <= 0):
        raise ValueError("weights must be positive and match force dimension")
    if not np.isfinite(tolerance) or tolerance <= 0 or max_iterations < 1:
        raise ValueError("tolerance and max_iterations must be positive")
    if smoothness_weight < 0 or not np.isfinite(smoothness_weight):
        raise ValueError("smoothness_weight must be non-negative and finite")

    dts = np.diff(time)
    if t_count > 1 and np.any(dts <= 0):
        raise ValueError("time_s must be strictly increasing")

    if sum(s.basis.shape[1] for s in spaces) > 0:
        z_all, converged, message = _solve_stacked(
            spaces,
            constraints,
            w,
            np.zeros((t_count, n)),
            smoothness_weight,
            dts,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
    else:
        z_all = np.empty(0)
        converged, message = True, "Unique trajectory solution; constraints audited"

    cuts = np.cumsum([0] + [s.basis.shape[1] for s in spaces])
    x_list = [
        spaces[k].particular + spaces[k].basis @ z_all[cuts[k] : cuts[k + 1]]
        for k in range(t_count)
    ]
    x_all = np.stack(x_list)
    finite = bool(np.isfinite(x_all).all())
    error = (
        float(
            max(
                np.max(
                    np.abs(
                        (spaces[k].matrix @ x_list[k] - spaces[k].rhs)
                        / spaces[k].row_scale
                    )
                )
                for k in range(t_count)
            )
        )
        if finite
        else float("inf")
    )
    all_margins = np.concatenate(
        [constraints[k].margins(x_list[k]) for k in range(t_count)]
    )
    cost = sum(float((w * x) @ (w * x)) for x in x_list)
    for k in range(t_count - 1):
        rate = (x_list[k + 1] - x_list[k]) / dts[k]
        cost += float(smoothness_weight * (rate @ rate))
    return _finalize_redistribution(
        x_all,
        finite=finite,
        converged=converged,
        error=error,
        violation=_margin_violation(all_margins, finite=finite),
        cost=cost,
        message=message,
        tolerance=tolerance,
    )
