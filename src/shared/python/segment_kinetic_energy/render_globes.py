"""Scaling and colouring for segment kinetic-energy globes (#15389)."""

from __future__ import annotations

import math

__all__ = ["globe_rgba_for_ke", "radius_for_ke"]


def radius_for_ke(
    ke_j: float,
    *,
    reference_peak_j: float,
    max_radius_m: float = 0.22,
    min_radius_m: float = 0.015,
) -> float:
    if ke_j <= 0.0 or reference_peak_j <= 0.0:
        return min_radius_m
    scale = max_radius_m / math.sqrt(reference_peak_j)
    return max(min_radius_m, scale * math.sqrt(ke_j))


def globe_rgba_for_ke(
    ke_j: float,
    *,
    reference_peak_j: float,
    at_peak: bool,
) -> tuple[float, float, float, float]:
    t = 0.0 if reference_peak_j <= 0.0 else min(1.0, max(0.0, ke_j / reference_peak_j))
    r = 0.15 + 0.75 * t
    g = 0.35 + 0.45 * (1.0 - abs(t - 0.5) * 2.0)
    b = 0.95 - 0.8 * t
    alpha = 0.88 if at_peak else 0.42
    if at_peak:
        r = min(1.0, r + 0.15)
        g = min(1.0, g + 0.1)
    return (float(r), float(g), float(b), float(alpha))
