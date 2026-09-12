"""Scaled local least-squares diagnostics; not nonlinear fit acceptance."""

from typing import NamedTuple
import numpy as np
from numpy.typing import ArrayLike, NDArray


class LinearizedResidualAnalysis(NamedTuple):
    """Unbounded truncated-SVD proposal in physical parameter units."""

    singular_values: NDArray[np.float64]
    rank: int
    physical_step: NDArray[np.float64]
    linear_residual: NDArray[np.float64]


def analyze_linearized_residual(
    jacobian: ArrayLike,
    residual: ArrayLike,
    parameter_scales: ArrayLike,
    *,
    relative_cutoff: float = 1e-8,
) -> LinearizedResidualAnalysis:
    """Diagnose J diag(scales), returning an unconstrained local proposal.

    Residual weighting must already match the caller's objective. Scales must
    declare physical units per dimensionless parameter. The residual left after
    truncation is local evidence only, not a global model feasibility bound.
    Bounds, nonlinearities and early-retention gates require separate replay.
    """
    j = np.asarray(jacobian, dtype=float)
    r = np.asarray(residual, dtype=float)
    scales = np.asarray(parameter_scales, dtype=float)
    if (
        j.ndim != 2
        or min(j.shape) == 0
        or r.shape != (j.shape[0],)
        or scales.shape != (j.shape[1],)
        or not np.isfinite(j).all()
        or not np.isfinite(r).all()
        or not np.isfinite(scales).all()
        or np.any(scales <= 0)
        or not np.isfinite(relative_cutoff)
        or not 0 < relative_cutoff < 1
    ):
        raise ValueError("Invalid finite Jacobian, residual, scales or cutoff")
    u, singular, vt = np.linalg.svd(j * scales, full_matrices=False)
    keep = singular > relative_cutoff * singular[0]
    step = scales * (vt[keep].T @ (-(u[:, keep].T @ r) / singular[keep]))
    remaining = r + j @ step
    if not np.isfinite(step).all() or not np.isfinite(remaining).all():
        raise ValueError("Linearized proposal overflowed")
    for array in (singular, step, remaining):
        array.setflags(write=False)
    return LinearizedResidualAnalysis(singular, int(np.sum(keep)), step, remaining)
