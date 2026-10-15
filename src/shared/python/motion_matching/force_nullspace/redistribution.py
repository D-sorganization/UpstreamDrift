"""Force redistribution solvers on a null space (NSX-03 split from force_nullspace)."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.optimize import LinearConstraint, NonlinearConstraint

from ._common import Array, _array
from .constraints import ForceConstraints
from .space import ForceNullSpace


def _minimize(*args: Any, **kwargs: Any) -> Any:
    """Delegate to the package-level ``minimize``.

    The package attribute is the public patch point used by the existing tests,
    so the solver call is resolved through it at call time.
    """
    return importlib.import_module(__package__).minimize(*args, **kwargs)


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
    even if SLSQP reports success. No clipping or equilibrium repair occurs.
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
    offset = space.particular - target
    weighted_basis = w[:, None] * basis
    weighted_offset = w * offset

    def objective(z: Array) -> float:
        residual = weighted_offset + weighted_basis @ z
        return float(residual @ residual)

    def gradient(z: Array) -> Array:
        return 2 * weighted_basis.T @ (weighted_offset + weighted_basis @ z)

    if basis.shape[1]:
        initial = np.linalg.lstsq(weighted_basis, -weighted_offset, rcond=None)[0]
        cons_list: list[Any] = []
        lb_box = constraints.lower - space.particular
        ub_box = constraints.upper_bound - space.particular
        eq_mask = np.isclose(lb_box, ub_box, atol=1e-12)
        if np.any(eq_mask):
            cons_list.append(
                LinearConstraint(basis[eq_mask], lb_box[eq_mask], lb_box[eq_mask])
            )
        if np.any(~eq_mask):
            cons_list.append(
                LinearConstraint(basis[~eq_mask], lb_box[~eq_mask], ub_box[~eq_mask])
            )
        if (
            constraints.matrix is not None
            and constraints.upper is not None
            and constraints.matrix.shape[0] > 0
        ):
            cons_list.append(
                LinearConstraint(
                    constraints.matrix @ basis,
                    -np.inf,
                    constraints.upper - constraints.matrix @ space.particular,
                )
            )
        if constraints.cones:
            cons_list.append(
                NonlinearConstraint(
                    lambda z: np.concatenate(
                        [
                            cone.margins(space.particular + basis @ z)
                            for cone in constraints.cones
                        ]
                    ),
                    0.0,
                    np.inf,
                )
            )
        solve = _minimize(
            objective,
            initial,
            jac=gradient,
            method="SLSQP",
            constraints=cons_list,
            options={"ftol": max(tolerance, 1e-12), "maxiter": max_iterations},
        )
        z = solve.x
        converged = bool(solve.success)
        message = str(solve.message)
    else:
        z = np.empty(0)
        converged, message = True, "Unique equality solution; constraints audited"
    x = space.particular + basis @ z
    finite = bool(np.isfinite(x).all())
    error = (
        float(np.max(np.abs((space.matrix @ x - space.rhs) / space.row_scale)))
        if finite
        else float("inf")
    )
    margins = constraints.margins(x)
    violation = _margin_violation(margins, finite=finite)
    cost = objective(z)
    return _finalize_redistribution(
        x,
        finite=finite,
        converged=converged,
        error=error,
        violation=violation,
        cost=cost,
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

    slices: list[slice] = []
    offset = 0
    for s in spaces:
        d_k = s.basis.shape[1]
        slices.append(slice(offset, offset + d_k))
        offset += d_k
    total_dim = offset

    def unpack_x(z: Array) -> list[Array]:
        return [
            spaces[k].particular + spaces[k].basis @ z[slices[k]]
            for k in range(t_count)
        ]

    def objective(z: Array) -> float:
        xs = unpack_x(z)
        cost = 0.0
        for k in range(t_count):
            wx = w * xs[k]
            cost += float(wx @ wx)
        for k in range(t_count - 1):
            dt = dts[k]
            dx = (xs[k + 1] - xs[k]) / dt
            cost += float(smoothness_weight * (dx @ dx))
        return cost

    def gradient(z: Array) -> Array:
        xs = unpack_x(z)
        grad = np.zeros(total_dim)
        w_sq = w**2
        for k in range(t_count):
            if slices[k].stop > slices[k].start:
                grad[slices[k]] += 2.0 * spaces[k].basis.T @ (w_sq * xs[k])
        for k in range(t_count - 1):
            dt = dts[k]
            coeff = 2.0 * smoothness_weight / (dt**2)
            diff = xs[k + 1] - xs[k]
            if slices[k].stop > slices[k].start:
                grad[slices[k]] -= coeff * (spaces[k].basis.T @ diff)
            if slices[k + 1].stop > slices[k + 1].start:
                grad[slices[k + 1]] += coeff * (spaces[k + 1].basis.T @ diff)
        return grad

    if total_dim > 0:
        z_init = np.zeros(total_dim)
        for k in range(t_count):
            if slices[k].stop > slices[k].start:
                wb = w[:, None] * spaces[k].basis
                wp = w * spaces[k].particular
                z_init[slices[k]] = np.linalg.lstsq(wb, -wp, rcond=None)[0]

        a_blocks = np.zeros((t_count * n, total_dim))
        lb_list: list[Array] = []
        ub_list: list[Array] = []
        for k in range(t_count):
            d_k = spaces[k].basis.shape[1]
            if d_k > 0:
                a_blocks[k * n : (k + 1) * n, slices[k]] = spaces[k].basis
            lb_list.append(constraints[k].lower - spaces[k].particular)
            ub_list.append(constraints[k].upper_bound - spaces[k].particular)

        lb_all = np.concatenate(lb_list)
        ub_all = np.concatenate(ub_list)
        eq_mask_traj = np.isclose(lb_all, ub_all, atol=1e-12)

        cons_traj: list[Any] = []
        if np.any(eq_mask_traj):
            cons_traj.append(
                LinearConstraint(
                    a_blocks[eq_mask_traj],
                    lb_all[eq_mask_traj],
                    lb_all[eq_mask_traj],
                )
            )
        if np.any(~eq_mask_traj):
            cons_traj.append(
                LinearConstraint(
                    a_blocks[~eq_mask_traj],
                    lb_all[~eq_mask_traj],
                    ub_all[~eq_mask_traj],
                )
            )

        has_matrix = any(
            c.matrix is not None and c.upper is not None and c.matrix.shape[0] > 0
            for c in constraints
        )
        if has_matrix:
            g_rows = sum(
                c.matrix.shape[0]
                for c in constraints
                if c.matrix is not None and c.upper is not None
            )
            g_mat = np.zeros((g_rows, total_dim))
            g_ub: list[Array] = []
            curr_row = 0
            for k in range(t_count):
                c = constraints[k]
                if (
                    c.matrix is not None
                    and c.upper is not None
                    and c.matrix.shape[0] > 0
                ):
                    m_k = c.matrix.shape[0]
                    d_k = spaces[k].basis.shape[1]
                    if d_k > 0:
                        g_mat[curr_row : curr_row + m_k, slices[k]] = (
                            c.matrix @ spaces[k].basis
                        )
                    g_ub.append(c.upper - c.matrix @ spaces[k].particular)
                    curr_row += m_k
            cons_traj.append(LinearConstraint(g_mat, -np.inf, np.concatenate(g_ub)))

        has_cones = any(len(c.cones) > 0 for c in constraints)
        if has_cones:

            def cone_margins(z: Array) -> Array:
                xs = unpack_x(z)
                return np.concatenate(
                    [
                        np.concatenate(
                            [cone.margins(xs[k]) for cone in constraints[k].cones]
                        )
                        for k in range(t_count)
                        if constraints[k].cones
                    ]
                )

            cons_traj.append(NonlinearConstraint(cone_margins, 0.0, np.inf))

        solve = _minimize(
            objective,
            z_init,
            jac=gradient,
            method="SLSQP",
            constraints=cons_traj,
            options={"ftol": max(tolerance, 1e-12), "maxiter": max_iterations},
        )
        z_sol = solve.x
        converged = bool(solve.success)
        message = str(solve.message)
    else:
        z_sol = np.empty(0)
        converged, message = True, "Unique trajectory solution; constraints audited"

    x_list = unpack_x(z_sol)
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
    violation = _margin_violation(all_margins, finite=finite)
    cost = objective(z_sol)
    return _finalize_redistribution(
        x_all,
        finite=finite,
        converged=converged,
        error=error,
        violation=violation,
        cost=cost,
        message=message,
        tolerance=tolerance,
    )
