"""Least-squares parameter uncertainty via Gauss-Newton / Laplace approximation (#11021).

Provides parameter covariance, standard errors, correlation, and identifiability
diagnostics for scipy.optimize.least_squares solutions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import OptimizeResult

from src.shared.python.core.contracts import require
from src.shared.python.estimation.multi_trial import shared_parameter_covariance


UncertaintyStatus = Literal[
    "estimated", "partial_at_bounds", "rank_deficient", "underdetermined"
]
_Stats = tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]


@dataclass(frozen=True)
class ParameterUncertainty:
    """Laplace approximation of parameter uncertainty for least-squares fits.

    Attributes:
        status: Estimation verdict ("estimated", "partial_at_bounds", "rank_deficient", "underdetermined").
        parameter_indices: Indices into optimum.x in output order.
        covariance: Square covariance matrix over parameter_indices, read-only.
        standard_errors: Standard errors for parameter_indices (NaN for parameters at active bounds).
        correlation: Correlation matrix over parameter_indices (NaN for parameters at active bounds).
        residual_variance: Estimated residual variance sigma^2 = 2 * cost / (m - n_free).
        n_residuals: Number of residual components m.
        n_free: Number of free parameters not held at active bounds.
        rank: Numerical rank of the Jacobian over free columns.
        condition_number: Condition number of J over free columns from singular values.
        at_bound: Requested parameter indices held at active bounds.
    """

    status: UncertaintyStatus
    parameter_indices: tuple[int, ...]
    covariance: np.ndarray | None
    standard_errors: np.ndarray | None
    correlation: np.ndarray | None
    residual_variance: float
    n_residuals: int
    n_free: int
    rank: int
    condition_number: float
    at_bound: tuple[int, ...]


def _requested_indices(
    parameter_indices: Sequence[int] | None, n: int
) -> tuple[int, ...]:
    """Validate and normalise the requested parameter indices."""
    if parameter_indices is None:
        return tuple(range(n))
    param_idx = tuple(int(i) for i in parameter_indices)
    require(len(param_idx) == len(set(param_idx)), "parameter_indices must be unique")
    for idx in param_idx:
        require(0 <= idx < n, f"parameter index {idx} out of range [0, {n})")
    return param_idx


def _validated_jacobian(optimum: OptimizeResult) -> np.ndarray:
    """Return the finite 2-D Jacobian whose width matches ``optimum.x``."""
    require(
        hasattr(optimum, "jac") and optimum.jac is not None,
        "optimum must have a jacobian",
    )
    require(
        hasattr(optimum, "cost") and optimum.cost is not None,
        "optimum must have cost",
    )
    require(
        hasattr(optimum, "x") and optimum.x is not None,
        "optimum must have parameter vector x",
    )
    x = np.asarray(optimum.x, dtype=float)
    require(x.ndim == 1, "optimum.x must be 1D")
    jac = np.asarray(optimum.jac, dtype=float)
    require(jac.ndim == 2, "optimum.jac must be 2D")
    require(bool(np.all(np.isfinite(jac))), "optimum.jac must be finite")
    require(
        jac.shape[1] == x.size,
        f"optimum.jac width {jac.shape[1]} must match optimum.x size {x.size}",
    )
    return jac


def _free_mask(optimum: OptimizeResult, n: int) -> np.ndarray:
    """Columns not held at an active bound (``active_mask == 0``)."""
    active_mask = getattr(optimum, "active_mask", None)
    if active_mask is None:
        return np.ones(n, dtype=bool)
    return np.asarray(active_mask) == 0


def _rank_and_condition(jac_free: np.ndarray, rank_rtol: float) -> tuple[int, float]:
    """Numerical rank and condition number of the free-column Jacobian."""
    if jac_free.shape[0] == 0 or jac_free.shape[1] == 0:
        return 0, float("inf")
    s = np.linalg.svd(jac_free, compute_uv=False)
    rank = int(np.count_nonzero(s > rank_rtol * s[0])) if s[0] > 0 else 0
    cond = float(s[0] / s[-1]) if s[-1] > 0 else float("inf")
    return rank, cond


def _free_covariance(jac_free: np.ndarray, residual_variance: float) -> np.ndarray:
    """Symmetric Gauss-Newton covariance over the free columns."""
    n_free = jac_free.shape[1]
    if n_free == 0 or residual_variance == 0.0:
        return np.zeros((n_free, n_free), dtype=float)
    cov = shared_parameter_covariance(
        jac_free, noise_variance=residual_variance, regularization=0.0
    )
    return 0.5 * (cov + cov.T)


def _marginal_statistics(
    cov_free: np.ndarray, free: Sequence[int], param_idx: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Covariance, standard errors and correlation over ``param_idx``.

    Rows/columns of parameters held at a bound stay NaN. Postcondition: all three
    arrays are read-only.
    """
    k = len(param_idx)
    cov_out = np.full((k, k), np.nan, dtype=float)
    se_out = np.full(k, np.nan, dtype=float)
    corr_out = np.full((k, k), np.nan, dtype=float)
    free_pos = {col: pos for pos, col in enumerate(free)}
    sel = np.array([i for i, idx in enumerate(param_idx) if idx in free_pos], int)
    if sel.size:
        pos = np.array([free_pos[param_idx[i]] for i in sel], dtype=int)
        cov_out[np.ix_(sel, sel)] = cov_free[np.ix_(pos, pos)]
        var = cov_out[sel, sel]
        nonneg = var >= 0.0
        se_out[sel[nonneg]] = np.sqrt(var[nonneg])
        positive = sel[np.isfinite(se_out[sel]) & (se_out[sel] > 0)]
        corr_out[np.ix_(positive, positive)] = cov_out[
            np.ix_(positive, positive)
        ] / np.outer(se_out[positive], se_out[positive])
        corr_out[sel, sel] = np.where(np.isnan(se_out[sel]), np.nan, 1.0)
    for arr in (cov_out, se_out, corr_out):
        arr.setflags(write=False)
    return cov_out, se_out, corr_out


def least_squares_parameter_uncertainty(
    optimum: OptimizeResult,
    *,
    parameter_indices: Sequence[int] | None = None,
    rank_rtol: float = 1e-10,
) -> ParameterUncertainty:
    """Compute parameter covariance via Gauss-Newton / Laplace approximation.

    Args:
        optimum: Result from scipy.optimize.least_squares with jac, cost, and x.
        parameter_indices: Parameter indices to extract (marginal block). Defaults to range(n).
        rank_rtol: Relative tolerance for numerical rank from singular values.

    Returns:
        ParameterUncertainty containing covariance, standard errors, and diagnostics.
    """
    require(rank_rtol > 0.0, "rank_rtol must be positive")
    jac = _validated_jacobian(optimum)
    m, n = jac.shape
    param_idx = _requested_indices(parameter_indices, n)
    free_mask = _free_mask(optimum, n)
    free = [i for i in range(n) if free_mask[i]]
    n_free = len(free)
    at_bound = tuple(idx for idx in param_idx if not free_mask[idx])
    jac_free = jac[:, free]
    rank, cond = _rank_and_condition(jac_free, rank_rtol)

    def verdict(
        status: UncertaintyStatus,
        residual_variance: float,
        stats: _Stats = (None, None, None),
    ) -> ParameterUncertainty:
        covariance, standard_errors, correlation = stats
        return ParameterUncertainty(
            status=status,
            parameter_indices=param_idx,
            covariance=covariance,
            standard_errors=standard_errors,
            correlation=correlation,
            residual_variance=residual_variance,
            n_residuals=m,
            n_free=n_free,
            rank=rank,
            condition_number=cond,
            at_bound=at_bound,
        )

    if m <= n_free:
        return verdict("underdetermined", float("nan"))
    residual_variance = float(2.0 * float(optimum.cost) / (m - n_free))
    if rank < n_free:
        return verdict("rank_deficient", residual_variance)
    stats = _marginal_statistics(
        _free_covariance(jac_free, residual_variance), free, param_idx
    )
    status: UncertaintyStatus = "partial_at_bounds" if at_bound else "estimated"
    return verdict(status, residual_variance, stats)


def fitted_uncertainty_or_none(
    optimum: OptimizeResult,
    *,
    parameter_indices: Sequence[int] | None = None,
) -> ParameterUncertainty | None:
    """Uncertainty for a fit's own report, or None when no Jacobian can support it.

    A diverged fit can end with a non-finite Jacobian. Its report must still be
    produced (and rejected on its own gates), so this returns None instead of
    letting the contract in :func:`least_squares_parameter_uncertainty` raise.
    """
    jac = getattr(optimum, "jac", None)
    if jac is None or getattr(optimum, "cost", None) is None:
        return None
    if not bool(np.all(np.isfinite(np.asarray(jac, dtype=float)))):
        return None
    return least_squares_parameter_uncertainty(
        optimum, parameter_indices=parameter_indices
    )
