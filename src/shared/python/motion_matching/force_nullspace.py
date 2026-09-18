"""Force-only alternatives on an explicitly supplied dynamics equality (#10436).

The caller owns the native equation A x = b, coordinate order, units, contact
schedule and constitutive constraints. No root slack or motion correction is
introduced here. Feasibility is numerical evidence, not native replay approval.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

Array = NDArray[np.float64]


def _array(value: Array, ndim: int, name: str) -> Array:
    result = np.array(value, dtype=float, copy=True)
    if result.ndim != ndim or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ForceNullSpace:
    """Scaled SVD representation x = particular + basis @ z of A x = b.

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
    ) -> ForceNullSpace:
        """Reject inconsistent/nonfinite equations; return owned read-only arrays.

        balance_tolerance is a dimensionless per-row residual after division
        by row_scale, NOT a combined norm of forces and moments.
        """
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
        u, singular, vt = np.linalg.svd(scaled, full_matrices=True)
        rank = int(np.count_nonzero(singular > rank_rtol * singular[0]))
        y = vt[:rank].T @ ((u[:, :rank].T @ (b / sr)) / singular[:rank])
        particular = sx * y
        basis = sx[:, None] * vt[rank:].T
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
        solve = minimize(
            objective,
            initial,
            jac=gradient,
            method="SLSQP",
            constraints={
                "type": "ineq",
                "fun": lambda z: constraints.margins(space.particular + basis @ z),
            },
            options={"ftol": tolerance**2, "maxiter": max_iterations},
        )
        z = solve.x
        converged, message = bool(solve.success), str(solve.message)
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
    violation = (
        max(0.0, float(-np.min(margins)))
        if finite and np.isfinite(margins).all()
        else float("inf")
    )
    cost = objective(z)
    feasible = (
        finite and np.isfinite(cost) and error <= tolerance and violation <= tolerance
    )
    x.setflags(write=False)
    return RedistributionResult(
        x, bool(feasible), converged, error, violation, cost, message
    )
