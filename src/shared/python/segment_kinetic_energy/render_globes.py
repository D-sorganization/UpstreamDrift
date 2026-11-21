"""Scaling and colouring for segment kinetic-energy globes (#15389)."""

from __future__ import annotations

import math

from src.shared.python.body_part_viz.scalar_shading import ScalarColorScale

__all__ = ["globe_rgba_for_ke", "ke_scalar_color_scale", "radius_for_ke"]

_GLOBE_BASE_RGB = (0.15, 0.35, 0.95)


def ke_scalar_color_scale(reference_peak_j: float) -> ScalarColorScale:
    """Sequential sqrt-mapped scale capped at the swing peak segment KE."""
    peak = max(float(reference_peak_j), 0.0)
    if peak <= 0.0:
        peak = 1.0
    return ScalarColorScale(
        enabled=True,
        scale_kind="sequential",
        units="J",
        sign_convention="unsigned",
        low_limit=0.0,
        high_limit=peak,
        display_transform="sqrt",
        opacity=1.0,
        blend_mode="replace",
        colormap_id="viridis",
    )


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
    scale = ke_scalar_color_scale(reference_peak_j)
    sample = ke_j if ke_j > 0.0 else None
    shaded = scale.shade_rgb(sample, _GLOBE_BASE_RGB)
    if shaded is None:
        r, g, b = _GLOBE_BASE_RGB
    else:
        r, g, b = shaded
    alpha = 0.88 if at_peak else 0.42
    if at_peak:
        r = min(1.0, r + 0.15)
        g = min(1.0, g + 0.1)
    return (float(r), float(g), float(b), float(alpha))