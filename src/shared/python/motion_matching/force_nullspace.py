"""Force-only alternatives on an explicitly supplied dynamics equality (#10436).

The caller owns the native equation A x = b, coordinate order, units, contact
schedule and constitutive constraints. No root slack or motion correction is
introduced here. Feasibility is numerical evidence, not native replay approval.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal, Sequence, TypeAlias

import numpy as np
from numpy.typing import NDArray
import scipy.linalg
from scipy.optimize import LinearConstraint, NonlinearConstraint, minimize

Array: TypeAlias = NDArray[np.float64]


def _array(value: Array, ndim: int, name: str) -> Array:
    result = np.array(value, dtype=float, copy=True)
    if result.ndim != ndim or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class NullSpaceAnalysis:
    """Diagnostic numerical metrics of an equality null-space decomposition."""

    rank: int
    nullity: int
    singular_values: Array
    equality_residual: float
    null_residual: float
    condition_number: float


@dataclass(frozen=True)
class ForceNullSpace:
    """Scaled SVD or QR representation x = particular + basis @ z of A x = b.

    Use from_balance to construct. Row scales have the units of b; variable
    scales have the units of x. Rank is numerical and may change with contact
    mode. Physical derivatives must never be computed from z differences.
    """

    matrix: Array
    rhs: Array
    row_scale: Array
    particular: Array
    basis: Array
    rank: int
    variable_scale: Array

    def analyze(self) -> NullSpaceAnalysis:
        """Compute rank, nullity, residuals, singular values, and condition number."""
        scaled = self.matrix * self.variable_scale[None, :] / self.row_scale[:, None]
        s = np.linalg.svd(scaled, compute_uv=False)
        cond = (
            float(s[0] / s[self.rank - 1])
            if self.rank > 0 and s[self.rank - 1] > 0
            else float("inf")
        )
        eq_res = float(
            np.max(np.abs((self.matrix @ self.particular - self.rhs) / self.row_scale))
        )
        null_res = (
            float(np.max(np.abs((self.matrix @ self.basis) / self.row_scale[:, None])))
            if self.basis.size > 0
            else 0.0
        )
        s_arr = np.array(s, dtype=float, copy=True)
        s_arr.setflags(write=False)
        return NullSpaceAnalysis(
            rank=self.rank,
            nullity=int(self.basis.shape[1]),
            singular_values=s_arr,
            equality_residual=eq_res,
            null_residual=null_res,
            condition_number=cond,
        )

    def validate_null_space(self, tolerance: float = 1e-8) -> bool:
        """Verify that equality residual and null basis residual satisfy tolerance."""
        analysis = self.analyze()
        return bool(
            analysis.equality_residual <= tolerance
            and analysis.null_residual <= tolerance
        )

    @classmethod
    def from_balance(
        cls,
        matrix: Array,
        rhs: Array,
        *,
        variable_scale: Array | None = None,
        row_scale: Array | None = None,
        rank_rtol: float = 1e-12,
        balance_tolerance: float = 1e-9,
        method: Literal["svd", "qr"] = "svd",
    ) -> ForceNullSpace:
        """Reject inconsistent/nonfinite equations; return owned read-only arrays.

        balance_tolerance is a dimensionless per-row residual after division
        by row_scale, NOT a combined norm of forces and moments.
        method chooses between scaled SVD or column-pivoted QR decomposition.
        """
        if method not in ("svd", "qr"):
            raise ValueError(f"Unknown decomposition method: {method!r}")
        a = _array(matrix, 2, "matrix")
        b = _array(rhs, 1, "rhs")
        m, n = a.shape
        if m == 0 or n == 0 or b.shape != (m,):
            raise ValueError("balance dimensions must be nonempty and compatible")
        sx = _array(
            np.ones(n) if variable_scale is None else variable_scale,
            1,
            "variable_scale",
        )
        sr = _array(np.ones(m) if row_scale is None else row_scale, 1, "row_scale")
        if sx.shape != (n,) or sr.shape != (m,) or np.any(sx <= 0) or np.any(sr <= 0):
            raise ValueError("scales must be positive and match balance dimensions")
        if (
            not 0 < rank_rtol < 1
            or not np.isfinite(balance_tolerance)
            or balance_tolerance <= 0
        ):
            raise ValueError(
                "rank_rtol and balance_tolerance must be positive and finite"
            )
        scaled = a * sx[None, :] / sr[:, None]
        if not np.isfinite(scaled).all():
            raise ValueError("scaled balance must be finite")

        if method == "svd":
            u, singular, vt = np.linalg.svd(scaled, full_matrices=True)
            rank = int(np.count_nonzero(singular > rank_rtol * singular[0]))
            y = vt[:rank].T @ ((u[:, :rank].T @ (b / sr)) / singular[:rank])
            particular = sx * y
            basis = sx[:, None] * vt[rank:].T
        else:
            # method == "qr"
            q, r, p = scipy.linalg.qr(scaled.T, pivoting=True)
            diag_r = np.abs(np.diag(r))
            rank = (
                int(np.count_nonzero(diag_r > rank_rtol * diag_r[0]))
                if diag_r.size
                else 0
            )
            q1 = q[:, :rank]
            q2 = q[:, rank:]
            r1 = r[:rank, :rank]
            b_scaled = b / sr
            if rank > 0:
                y = scipy.linalg.solve_triangular(r1.T, b_scaled[p[:rank]], lower=True)
                particular = sx * (q1 @ y)
            else:
                particular = np.zeros(n)
            basis = sx[:, None] * q2

        if np.max(np.abs((a @ particular - b) / sr)) > balance_tolerance:
            raise ValueError("inconsistent dynamics equality at requested tolerance")
        if basis.size and np.max(np.abs((a @ basis) / sr[:, None])) > balance_tolerance:
            raise ValueError("numerical null basis fails dynamics tolerance")
        return cls(
            a,
            b,
            sr,
            _array(particular, 1, "particular"),
            _array(basis, 2, "basis"),
            rank,
            sx,
        )


@dataclass(frozen=True)
class FrictionCone:
    """Exact Coulomb cone for three force entries in the declared world frame.

    frame columns are two tangents and the outward normal, expressed in the
    same frame as these force entries. Inactive contacts need zero force bounds.
    """

    indices: tuple[int, int, int]
    frame: Array
    mu: float

    def __post_init__(self) -> None:
        frame = _array(self.frame, 2, "frame")
        if frame.shape != (3, 3) or not np.allclose(
            frame.T @ frame, np.eye(3), atol=1e-12, rtol=0
        ):
            raise ValueError("surface frame must be orthonormal")
        if (
            len(self.indices) != 3
            or len(set(self.indices)) != 3
            or any(not isinstance(i, int) or i < 0 for i in self.indices)
        ):
            raise ValueError("cone indices must be three distinct nonnegative integers")
        if not np.isfinite(self.mu) or self.mu < 0:
            raise ValueError("friction coefficient must be finite and nonnegative")
        object.__setattr__(self, "frame", frame)

    def margins(self, forces: Array) -> Array:
        """Return unilateral and exact circular-cone margins in force units."""
        local = self.frame.T @ forces[list(self.indices)]
        return np.array([local[2], self.mu * local[2] - np.linalg.norm(local[:2])])


@dataclass(frozen=True)
class ForceConstraints:
    """Finite physical bounds, optional G x <= h, and exact friction cones.

    Bounds can encode zero trail effort or inactive contacts. G/h can encode
    grip limits or conservative COP inequalities. Geometry and constitutive
    compatibility remain caller obligations; they cannot be inferred from A.
    """

    lower: Array
    upper_bound: Array
    matrix: Array | None = None
    upper: Array | None = None
    cones: tuple[FrictionCone, ...] = ()

    def __post_init__(self) -> None:
        lo = _array(self.lower, 1, "lower")
        hi = _array(self.upper_bound, 1, "upper_bound")
        if not lo.size or lo.shape != hi.shape or np.any(lo > hi):
            raise ValueError("force bounds must have matching shape and lower <= upper")
        if (self.matrix is None) != (self.upper is None):
            raise ValueError("inequality matrix and upper must be supplied together")
        g = _array(
            np.empty((0, lo.size)) if self.matrix is None else self.matrix, 2, "matrix"
        )
        h = _array(np.empty(0) if self.upper is None else self.upper, 1, "upper")
        if g.shape != (h.size, lo.size):
            raise ValueError("inequality dimensions mismatch")
        if any(max(cone.indices) >= lo.size for cone in self.cones):
            raise ValueError("cone index exceeds force dimension")
        for name, value in (
            ("lower", lo),
            ("upper_bound", hi),
            ("matrix", g),
            ("upper", h),
        ):
            object.__setattr__(self, name, value)

    def margins(self, forces: Array) -> Array:
        """Nonnegative entries mean all supplied physical constraints hold."""
        linear = (
            []
            if self.matrix is None or self.upper is None
            else [self.upper - self.matrix @ forces]
        )
        return np.concatenate(
            [
                forces - self.lower,
                self.upper_bound - forces,
                *linear,
                *(cone.margins(forces) for cone in self.cones),
            ]
        )


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
        solve = minimize(
            objective,
            initial,
            jac=gradient,
            method="SLSQP",
            constraints=cons_list,
            options={"ftol": max(tolerance, 1e-12), "maxiter": max_iterations},
        )
        z = solve.x
        converged = bool(solve.success or (solve.status in (0, 8, 9)))
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

        solve = minimize(
            objective,
            z_init,
            jac=gradient,
            method="SLSQP",
            constraints=cons_traj,
            options={"ftol": max(tolerance, 1e-12), "maxiter": max_iterations},
        )
        z_sol = solve.x
        converged = bool(solve.success or (solve.status in (0, 8, 9)))
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


@dataclass(frozen=True)
class TradeoffAlternative:
    """A single force/torque distribution strategy on the Pareto tradeoff frontier."""

    name: str
    strategy: str
    forces: Array
    feasible: bool
    converged: bool
    cost: float
    lead_effort: float
    trail_effort: float
    trail_share: float
    grip_load: float
    ground_load: float
    per_joint_torque: Array
    per_joint_power: Array
    cop_estimate: float
    grip_wrench: Array
    active_constraints: list[str]
    mechanical_rationale: str
    units: dict[str, str]


@dataclass(frozen=True)
class TorqueDistributionTradeoff:
    """Set of candidate torque allocation alternatives with Pareto evaluation and export."""

    alternatives: tuple[TradeoffAlternative, ...]
    default_rationale: str
    selected_default_name: str = "conservative_default"

    def get_alternative(self, name: str) -> TradeoffAlternative | None:
        """Find an alternative by its unique identifier name."""
        for alt in self.alternatives:
            if alt.name == name:
                return alt
        return None

    def get_selected_default(self) -> TradeoffAlternative:
        """Return the conservative default alternative."""
        alt = self.get_alternative(self.selected_default_name)
        if alt is not None:
            return alt
        for a in self.alternatives:
            if a.feasible:
                return a
        return self.alternatives[0]

    def to_pareto_table(self) -> list[dict[str, Any]]:
        """Return reproducible dictionary table of trade-off metrics."""
        return [
            {
                "name": alt.name,
                "strategy": alt.strategy,
                "feasible": alt.feasible,
                "lead_effort": float(alt.lead_effort),
                "trail_effort": float(alt.trail_effort),
                "trail_share": float(alt.trail_share),
                "cost": float(alt.cost),
                "grip_load": float(alt.grip_load),
                "ground_load": float(alt.ground_load),
            }
            for alt in self.alternatives
        ]

    def export_json(self, path: Path | str) -> None:
        """Export all trade-off alternatives and mechanical rationales to JSON."""
        p = Path(path)
        data = {
            "default_rationale": self.default_rationale,
            "selected_default": self.get_selected_default().name,
            "alternatives": [
                {
                    "name": alt.name,
                    "strategy": alt.strategy,
                    "feasible": alt.feasible,
                    "converged": alt.converged,
                    "cost": float(alt.cost),
                    "lead_effort": float(alt.lead_effort),
                    "trail_effort": float(alt.trail_effort),
                    "trail_share": float(alt.trail_share),
                    "grip_load": float(alt.grip_load),
                    "ground_load": float(alt.ground_load),
                    "cop_estimate": float(alt.cop_estimate),
                    "forces": alt.forces.tolist(),
                    "per_joint_torque": alt.per_joint_torque.tolist(),
                    "per_joint_power": alt.per_joint_power.tolist(),
                    "units": alt.units,
                    "mechanical_rationale": alt.mechanical_rationale,
                }
                for alt in self.alternatives
            ],
        }
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def export_csv(self, path: Path | str) -> None:
        """Export summary Pareto trade-off table to CSV."""
        p = Path(path)
        table = self.to_pareto_table()
        header = "name,strategy,feasible,lead_effort,trail_effort,trail_share,cost,grip_load,ground_load"
        lines = [header]
        for row in table:
            lines.append(
                f"{row['name']},{row['strategy']},{row['feasible']},"
                f"{row['lead_effort']:.6f},{row['trail_effort']:.6f},{row['trail_share']:.6f},"
                f"{row['cost']:.6f},{row['grip_load']:.6f},{row['ground_load']:.6f}"
            )
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def explore_torque_tradeoffs(
    space: ForceNullSpace,
    constraints: ForceConstraints,
    *,
    lead_indices: Sequence[int],
    trail_indices: Sequence[int],
    grip_indices: Sequence[int] | None = None,
    ground_indices: Sequence[int] | None = None,
    joint_velocities: Array | None = None,
    tolerance: float = 1e-8,
) -> TorqueDistributionTradeoff:
    """Explore feasible force null spaces and generate torque distribution tradeoffs."""
    n = space.particular.size
    lead_idx = list(lead_indices)
    trail_idx = list(trail_indices)
    grip_idx = list(grip_indices) if grip_indices is not None else []
    ground_idx = list(ground_indices) if ground_indices is not None else []

    all_joint_indices = lead_idx + trail_idx

    units = {
        "torque": "N*m",
        "force": "N",
        "power": "W",
        "cop": "m",
        "angle": "rad",
    }

    def evaluate_alternative(
        name: str,
        strategy: str,
        weights: Array,
        c: ForceConstraints,
        rationale: str,
    ) -> TradeoffAlternative:
        res = redistribute_forces(space, weights, c, tolerance=tolerance)
        x = res.forces
        lead_effort = float(np.linalg.norm(x[lead_idx])) if lead_idx else 0.0
        trail_effort = float(np.linalg.norm(x[trail_idx])) if trail_idx else 0.0
        tot_effort = lead_effort + trail_effort
        trail_share = float(trail_effort / tot_effort) if tot_effort > 1e-12 else 0.0
        grip_load = float(np.linalg.norm(x[grip_idx])) if grip_idx else 0.0
        ground_load = float(np.linalg.norm(x[ground_idx])) if ground_idx else 0.0

        per_joint_torque = (
            _array(x[all_joint_indices], 1, "per_joint_torque")
            if all_joint_indices
            else np.empty(0)
        )
        if joint_velocities is not None:
            jv = np.asarray(joint_velocities, dtype=float)
            if jv.size != per_joint_torque.size:
                raise ValueError(
                    f"joint_velocities length ({jv.size}) must match the number of "
                    f"joint indices ({per_joint_torque.size})"
                )
            per_joint_power = _array(per_joint_torque * jv, 1, "per_joint_power")
        else:
            per_joint_power = _array(
                np.zeros_like(per_joint_torque), 1, "per_joint_power"
            )

        cop_estimate = float("nan")

        grip_wrench = _array(x[grip_idx], 1, "grip_wrench") if grip_idx else np.empty(0)

        active: list[str] = []
        if np.any(np.abs(x - c.lower) < 1e-4):
            active.append("lower_bound")
        if np.any(np.abs(c.upper_bound - x) < 1e-4):
            active.append("upper_bound")

        return TradeoffAlternative(
            name=name,
            strategy=strategy,
            forces=x,
            feasible=res.feasible,
            converged=res.converged,
            cost=float(res.cost),
            lead_effort=lead_effort,
            trail_effort=trail_effort,
            trail_share=trail_share,
            grip_load=grip_load,
            ground_load=ground_load,
            per_joint_torque=per_joint_torque,
            per_joint_power=per_joint_power,
            cop_estimate=cop_estimate,
            grip_wrench=grip_wrench,
            active_constraints=active,
            mechanical_rationale=rationale,
            units=units,
        )

    alternatives: list[TradeoffAlternative] = []

    # 1. baseline_minimum_effort
    w_base = np.ones(n)
    alternatives.append(
        evaluate_alternative(
            "baseline_minimum_effort",
            "unweighted minimum-effort distribution (least squares)",
            w_base,
            constraints,
            "Unweighted minimum-effort distribution allocating reaction load uniformly across degrees of freedom.",
        )
    )

    # 2. conservative_default
    w_cons = np.ones(n)
    for i in lead_idx:
        w_cons[i] = 1.0
    for i in trail_idx:
        w_cons[i] = 1.5
    for i in grip_idx:
        w_cons[i] = 1.2
    for i in ground_idx:
        w_cons[i] = 1.0
    alternatives.append(
        evaluate_alternative(
            "conservative_default",
            "conservative balanced distribution with torque safety margins",
            w_cons,
            constraints,
            "Conservative default distributes dynamic swing reaction torques across lead and trail arm kinematic chains while maintaining reserve torque margins below actuator peak capacities and preserving ground contact COP within support boundaries. Avoids single-arm saturation and excessive grip shear.",
        )
    )

    # 3. trail_arm_reduced_50
    w_50 = np.ones(n)
    for i in lead_idx:
        w_50[i] = 1.0
    for i in trail_idx:
        w_50[i] = 2.5
    alternatives.append(
        evaluate_alternative(
            "trail_arm_reduced_50",
            "reduced trail arm share (50% target reduction)",
            w_50,
            constraints,
            "Penalizes trail arm involvement by approximately 50%, shifting dynamic swing load onto the lead arm kinematic chain.",
        )
    )

    # 4. trail_arm_reduced_80
    w_80 = np.ones(n)
    for i in lead_idx:
        w_80[i] = 1.0
    for i in trail_idx:
        w_80[i] = 8.0
    alternatives.append(
        evaluate_alternative(
            "trail_arm_reduced_80",
            "heavily reduced trail arm share (80% target reduction)",
            w_80,
            constraints,
            "Heavily reduces trail arm contribution to ~20% of baseline, requiring significant lead arm torque authority.",
        )
    )

    # 5. hard_zero_trail
    lo_zero = constraints.lower.copy()
    hi_zero = constraints.upper_bound.copy()
    for i in trail_idx:
        lo_zero[i] = 0.0
        hi_zero[i] = 0.0
    c_zero = ForceConstraints(
        lower=lo_zero,
        upper_bound=hi_zero,
        matrix=constraints.matrix,
        upper=constraints.upper,
        cones=constraints.cones,
    )
    alternatives.append(
        evaluate_alternative(
            "hard_zero_trail",
            "hard zero trail arm effort (lead arm carries full upper body load)",
            np.ones(n),
            c_zero,
            "Strictly zeroes out trail arm torques, forcing the lead arm and body structure to support 100% of upper-extremity reactions. Infeasible if lead bounds are exceeded.",
        )
    )

    # 6. relaxed_minimum_trail
    w_relaxed = np.ones(n)
    for i in lead_idx:
        w_relaxed[i] = 1.0
    for i in trail_idx:
        w_relaxed[i] = 50.0
    alternatives.append(
        evaluate_alternative(
            "relaxed_minimum_trail",
            "relaxed minimum trail arm load (closest admissible solution)",
            w_relaxed,
            constraints,
            "Finds the closest physically admissible solution when hard-zero trail is infeasible by heavily penalizing trail arm involvement while respecting actuator capacity bounds.",
        )
    )

    # 7. grip_squeeze_minimized (if grip_indices given)
    if grip_idx:
        w_grip = np.ones(n)
        for i in grip_idx:
            w_grip[i] = 20.0
        alternatives.append(
            evaluate_alternative(
                "grip_squeeze_minimized",
                "minimized grip squeeze and internal hand shear wrench",
                w_grip,
                constraints,
                "Penalizes internal hand grip forces and contact shear wrench, minimizing interface reaction loads.",
            )
        )

    # 8. ground_load_regularized (if ground_indices given)
    if ground_idx:
        w_ground = np.ones(n)
        for i in ground_idx:
            w_ground[i] = 5.0
        alternatives.append(
            evaluate_alternative(
                "ground_load_regularized",
                "regularized ground reaction distribution and COP centering",
                w_ground,
                constraints,
                "Regularizes ground reaction normal and frictional distribution, centering center-of-pressure within the support polygon.",
            )
        )

    default_rationale = (
        "Conservative default distributes dynamic swing reaction torques across lead and trail arm kinematic chains "
        "while maintaining reserve torque margins below actuator peak capacities and preserving ground contact COP "
        "within support boundaries. Avoids single-arm saturation and excessive grip shear."
    )

    return TorqueDistributionTradeoff(
        alternatives=tuple(alternatives),
        default_rationale=default_rationale,
        selected_default_name="conservative_default",
    )
