"""Geometric equal-range diagnostic for existing aimed landing coordinates.

This rescales each landing radially to a common distance while retaining its
bearing. It does not recompute impact, spin, or aerodynamic flight.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def equal_range_endpoint(
    x_m: float, y_m: float, target_range_m: float
) -> tuple[float, float]:
    """Preserve landing bearing and set its radial distance to target_range_m."""
    if not all(math.isfinite(v) for v in (x_m, y_m, target_range_m)):
        raise ValueError("coordinates and target range must be finite")
    radius = math.hypot(x_m, y_m)
    if radius <= 0 or target_range_m <= 0:
        raise ValueError("actual and target range must be positive")
    scale = target_range_m / radius
    return x_m * scale, y_m * scale


def equal_range_summary(
    aimed_endpoints_m: Sequence[tuple[float, float]],
    target_range_m: float,
    target_radius_m: float,
) -> dict[str, float]:
    """Summarize dispersion after geometric radial normalization."""
    if not math.isfinite(target_radius_m) or target_radius_m < 0:
        raise ValueError("target_radius_m must be finite and non-negative")
    if len(aimed_endpoints_m) < 2:
        raise ValueError("at least two endpoints required for sample variance")
    normalized = np.asarray(
        [equal_range_endpoint(x, y, target_range_m) for x, y in aimed_endpoints_m],
        dtype=float,
    )
    lateral = normalized[:, 1]
    errors = np.hypot(normalized[:, 0] - target_range_m, lateral)
    radii = np.hypot(normalized[:, 0], normalized[:, 1])
    q5, q95 = np.quantile(lateral, [0.05, 0.95])
    return {
        "n": float(len(aimed_endpoints_m)),
        "lateral_mean_m": float(np.mean(lateral)),
        "lateral_sd_m": float(np.std(lateral, ddof=1)),
        "lateral_variance_m2": float(np.var(lateral, ddof=1)),
        "lateral_p05_p95_width_m": float(q95 - q5),
        "target_hit_fraction": float(np.mean(errors <= target_radius_m)),
        "target_rmse_m": float(np.sqrt(np.mean(errors**2))),
        "range_min_m": float(np.min(radii)),
        "range_max_m": float(np.max(radii)),
    }
