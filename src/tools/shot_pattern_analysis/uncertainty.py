"""Monte Carlo uncertainty for paired shot-pattern variance comparisons."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _paired_arrays(
    curved: ArrayLike,
    straight: ArrayLike,
    resamples: int,
    seed: int,
    minimum_pairs: int = 10,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    a = np.asarray(curved, dtype=float)
    b = np.asarray(straight, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape or a.size < minimum_pairs:
        raise ValueError(
            f"inputs must be equal one-dimensional arrays of >={minimum_pairs} pairs"
        )
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("paired values must be finite")
    if not isinstance(resamples, int) or isinstance(resamples, bool) or resamples < 100:
        raise ValueError("resamples must be an integer >=100")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return a, b


def paired_hit_difference(
    curved: ArrayLike,
    straight: ArrayLike,
    *,
    resamples: int = 500,
    seed: int = 20261008,
) -> dict[str, float]:
    """Return curved-minus-straight hit fraction and paired 95% bootstrap interval.

    Inputs must be equal finite one-dimensional binary arrays of >=10 pairs.
    Resample matching shot indices, preserving correlated hits. The interval
    reflects Monte Carlo sampling only, excluding model uncertainty.
    Postcondition: finite estimates and bounds within [-1, 1].
    """
    a, b = _paired_arrays(curved, straight, resamples, seed)
    if not np.all(np.isin(a, [0, 1])) or not np.all(np.isin(b, [0, 1])):
        raise ValueError("hit arrays must be binary (zero or one)")
    return paired_mean_difference(a, b, resamples=resamples, seed=seed)


def paired_mean_difference(
    curved: ArrayLike,
    straight: ArrayLike,
    *,
    resamples: int = 500,
    seed: int = 20261008,
) -> dict[str, float]:
    """Estimate a finite paired mean difference and its sampling-only 95% CI.

    Requires at least two finite pairs; resamples preserve matching indices.
    """
    a, b = _paired_arrays(curved, straight, resamples, seed, minimum_pairs=2)
    differences = a - b
    if not np.all(np.isfinite(differences)):
        raise ValueError("paired differences must be finite")
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples)
    for i in range(resamples):
        indices = rng.integers(0, a.size, size=a.size)
        estimates[i] = np.mean(differences[indices])
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    result = {
        "estimate": float(np.mean(differences)),
        "lower_95": float(lower),
        "upper_95": float(upper),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("mean comparison produced nonfinite results")
    return result


def paired_variance_ratio(
    curved: ArrayLike,
    straight: ArrayLike,
    *,
    resamples: int = 500,
    seed: int = 20261008,
) -> dict[str, float]:
    """Return the variance ratio and paired percentile bootstrap 95% interval.

    Preconditions: equal finite one-dimensional arrays, at least ten pairs,
    positive reference variance, at least 100 resamples, and nonnegative seed.
    Each resample draws matching shot indices for both arrays, preserving
    their shared face errors. Postcondition: finite nonnegative estimates.
    The interval describes Monte Carlo sampling uncertainty only, excluding
    model discrepancy, uncertain parameters, and player-to-player variation.
    """
    a, b = _paired_arrays(curved, straight, resamples, seed)
    reference = float(np.var(b, ddof=1))
    if reference <= 0:
        raise ValueError("straight reference variance must be positive")
    rng = np.random.default_rng(seed)
    ratios = np.empty(resamples)
    for i in range(resamples):
        indices = rng.integers(0, a.size, size=a.size)
        denominator = float(np.var(b[indices], ddof=1))
        if denominator <= 0:
            raise ValueError(
                "bootstrap reference variance is zero; more pairs required"
            )
        ratios[i] = np.var(a[indices], ddof=1) / denominator
    lower, upper = np.quantile(ratios, [0.025, 0.975])
    result = {
        "estimate": float(np.var(a, ddof=1) / reference),
        "lower_95": float(lower),
        "upper_95": float(upper),
    }
    if not all(np.isfinite(value) and value >= 0 for value in result.values()):
        raise ValueError("variance comparison produced nonfinite results")
    return result
