"""Sampled marginals against deterministic envelopes (NSX-70, #12588).

For affine quantities ``q(z) = const + coeff . z`` per frame, compare the
range reached by sampled trajectories with the static envelope of the same
feasible sets.  Containment is a consistency check (a sample is feasible, so
it must lie in the envelope); *reach* measures how much of each envelope the
samples cover and says nothing about physical plausibility.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.shared.python.loop_nullspace.envelopes import EnvelopeQuantity
from src.shared.python.loop_nullspace.evidence import Array

__all__ = ["MarginalReach", "marginal_reach"]


@dataclass(frozen=True, eq=False)
class MarginalReach:
    """Per-frame, per-quantity sample ranges (all arrays ``(T, Q)``).

    ``reach`` is ``(sample max - sample min) / (upper - lower)`` and
    ``central_reach`` uses the 5th-95th percentile range; both are NaN where
    the envelope has zero or non-finite width.  ``containment_violation`` is
    the largest amount by which any sample leaves its envelope (0 if none).
    """

    names: tuple[str, ...]
    sample_min: Array
    sample_max: Array
    sample_p05: Array
    sample_p95: Array
    reach: Array
    central_reach: Array
    containment_violation: float
    n_samples: int

    def mean_reach(self) -> Array:
        """Mean reach per quantity over frames with a defined reach."""
        return np.asarray(np.nanmean(self.reach, axis=0))


def marginal_reach(
    trajectories: Array,
    quantities: Sequence[Sequence[EnvelopeQuantity]],
    lower: Array,
    upper: Array,
) -> MarginalReach:
    """Compare sampled trajectories with envelope bounds.

    ``trajectories`` is ``(N, T, n_z)``; ``quantities[t]`` lists the Q
    quantities of frame ``t`` (same names every frame); ``lower``/``upper``
    are ``(T, Q)`` envelope bounds (NaN where unbounded).  Raises
    ``ValueError`` on shape or name mismatches or an empty sample.
    """
    z = np.asarray(trajectories, dtype=np.float64)
    if z.ndim != 3 or z.shape[0] < 1:
        raise ValueError("trajectories must be a non-empty (N, T, n_z) array")
    n, t_count, n_z = z.shape
    if len(quantities) != t_count:
        raise ValueError("need one quantity list per frame")
    names = tuple(q.name for q in quantities[0])
    if any(tuple(q.name for q in qs) != names for qs in quantities):
        raise ValueError("quantity names differ between frames")
    lo, hi = np.asarray(lower, float), np.asarray(upper, float)
    shape = (t_count, len(names))
    if lo.shape != shape or hi.shape != shape:
        raise ValueError(f"lower/upper must have shape {shape}")
    coeff = np.array([[q.coeff for q in qs] for qs in quantities])
    const = np.array([[q.const for q in qs] for qs in quantities])
    if coeff.shape[2] != n_z:
        raise ValueError("quantity coefficients do not match n_z")
    values = np.einsum("ntj,tqj->ntq", z, coeff) + const
    smin, smax = values.min(axis=0), values.max(axis=0)
    p05, p95 = np.percentile(values, [5.0, 95.0], axis=0)
    width = hi - lo
    ok = np.isfinite(width) & (width > 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        reach = np.where(ok, (smax - smin) / width, math.nan)
        central = np.where(ok, (p95 - p05) / width, math.nan)
    finite_lo, finite_hi = np.isfinite(lo), np.isfinite(hi)
    below = np.where(finite_lo, lo - smin, -np.inf)
    above = np.where(finite_hi, smax - hi, -np.inf)
    violation = max(0.0, float(np.max(below)), float(np.max(above)))
    return MarginalReach(names, smin, smax, p05, p95, reach, central, violation, n)
