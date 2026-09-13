"""Diagnostic-only linear effort identification with weld reactions eliminated.

The caller supplies a smooth position/rate/acceleration closure-feasible path.
This helper checks acceleration compatibility only; it is not a trajectory
assembler, built-in inverse dynamics call, optimizer, or forward validator.
"""

from typing import TypeAlias
import numpy as np
from numpy.typing import NDArray

Array: TypeAlias = NDArray[np.float64]


def projected_system(
    mass: Array,
    bias: Array,
    acceleration: Array,
    jacobian: Array,
    constraint_bias: Array,
    effort_design: Array,
    *,
    rank_tolerance: float = 1e-10,
    closure_tolerance: float = 1e-7,
) -> tuple[Array, Array]:
    """Stack N.T E theta=N.T(M a+h), where J N=0 and h excludes actuation.

    E maps global polynomial coefficients to generalized primitive efforts.
    Bias h uses M a+h=E theta+J.T reaction. No reaction value is assumed zero.
    Units and row weighting are caller responsibilities; coefficients need not
    be identifiable. A least-squares residual does not imply forward acceptance.
    """
    mass, bias, acceleration, jacobian, constraint_bias, effort_design = (
        np.asarray(v, dtype=float)
        for v in (mass, bias, acceleration, jacobian, constraint_bias, effort_design)
    )
    if any(
        not np.isfinite(v).all()
        for v in (mass, bias, acceleration, jacobian, constraint_bias, effort_design)
    ):
        raise ValueError("All projected-system inputs must be finite")
    if not all(np.isfinite(v) and v > 0 for v in (rank_tolerance, closure_tolerance)):
        raise ValueError("Tolerances must be finite and positive")
    if mass.ndim != 3 or mass.shape[0] == 0 or mass.shape[1] != mass.shape[2]:
        raise ValueError("Mass must be sample-by-coordinate-by-coordinate")
    samples, n, _ = mass.shape
    if (
        bias.shape != (samples, n)
        or acceleration.shape != (samples, n)
        or jacobian.ndim != 3
        or jacobian.shape[0] != samples
        or jacobian.shape[2] != n
        or not 0 < jacobian.shape[1] < n
        or constraint_bias.shape != jacobian.shape[:2]
        or effort_design.ndim != 3
        or effort_design.shape[:2] != (samples, n)
        or effort_design.shape[2] == 0
    ):
        raise ValueError("Inconsistent projected-system dimensions")
    rows = []
    rhs = []
    for m, h, a, j, gamma, e in zip(
        mass, bias, acceleration, jacobian, constraint_bias, effort_design, strict=True
    ):
        _, singular, right = np.linalg.svd(j, full_matrices=True)
        rank = int(np.sum(singular > rank_tolerance * singular[0]))
        if rank != j.shape[0]:
            raise ValueError("Constraint Jacobian must have full row rank")
        if np.max(np.abs(j @ a + gamma)) > closure_tolerance:
            raise ValueError("Acceleration is not closure compatible")
        null = right[rank:].T
        rows.append(null.T @ e)
        rhs.append(null.T @ (m @ a + h))
    matrix = np.vstack(rows)
    target = np.concatenate(rhs)
    matrix.setflags(write=False)
    target.setflags(write=False)
    return matrix, target
