"""Monte Carlo uncertainty for paired shot-pattern variance comparisons."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


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
    a = np.asarray(curved, dtype=float)
    b = np.asarray(straight, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape or a.size < 10:
        raise ValueError("inputs must be equal one-dimensional arrays of >=10 pairs")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("paired values must be finite")
    if not isinstance(resamples, int) or isinstance(resamples, bool) or resamples < 100:
        raise ValueError("resamples must be an integer >=100")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
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
