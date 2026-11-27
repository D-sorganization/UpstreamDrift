"""Scaling and colouring for segment kinetic-energy proxy globes (#15389, #15396)."""

from __future__ import annotations

import math
from dataclasses import replace

from src.shared.python.body_part_viz.scalar_shading import ScalarColorScale

from .shading import default_stored_ke_color_scale

__all__ = [
    "globe_rgba_for_ke",
    "ke_scalar_color_scale",
    "radius_for_ke",
    "sample_for_globe_ke",
]

_GLOBE_BASE_RGB = (0.15, 0.35, 0.95)


def ke_scalar_color_scale(reference_peak_j: float | None = None) -> ScalarColorScale:
    """Fixed cividis scale shared with stored-KE surface shading."""
    if reference_peak_j is None:
        scale = default_stored_ke_color_scale()
    else:
        peak = max(float(reference_peak_j), 0.0)
        if peak <= 0.0:
            peak = 1.0
        scale = default_stored_ke_color_scale(high_limit_j=peak)
    return replace(scale, enabled=True)


def sample_for_globe_ke(ke_j: float | None) -> float | None:
    """Normalize KE samples for scalar shading (0 J uses colormap zero)."""
    if ke_j is None:
        return None
    value = float(ke_j)
    if not math.isfinite(value):
        return None
    if value <= 0.0:
        return 0.0
    return value


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
    ke_j: float | None,
    *,
    reference_peak_j: float,
) -> tuple[float, float, float, float]:
    sample = sample_for_globe_ke(ke_j)
    if sample is None:
        return (*_GLOBE_BASE_RGB, 1.0)
    scale = ke_scalar_color_scale(reference_peak_j)
    shaded = scale.shade_rgb(sample, _GLOBE_BASE_RGB)
    if shaded is None:
        shaded = _GLOBE_BASE_RGB
    r, g, b = shaded
    return (float(r), float(g), float(b), 1.0)
