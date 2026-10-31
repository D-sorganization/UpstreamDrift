"""Non-planar undulating green topography, spatial slope gradients, and turf micro-roughness.

Part of the UpstreamDrift 3D Counterbalanced Putting Dynamics Suite (v4.0).
Implements:
- Zero-allocation @precondition signature caching (Issue D-11803)
- Non-planar multi-tier undulating green surface z(x, y) combining planar grade/break,
  Gaussian mounds/swales, and logistic two-tier step transitions
- Exact analytical spatial slope gradient grad_z(x, y) = (dz/dx, dz/dy)
- Topographic contour & slope vector grid generator (generate_topographic_grid)
- Spatial turf micro-roughness / Poa annua chatter perturbation acceleration

Architecture budgets enforced:
- <= 500 LOC per file
- <= 100 LOC per function
- <= 8 parameters per function
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    from .turf_properties import STANDARD_GRAVITY_M_S2, precondition
except ImportError:
    from turf_properties import STANDARD_GRAVITY_M_S2, precondition  # type: ignore[no-redef]


@dataclass(frozen=True)
class GaussianMound:
    """Localized Gaussian mound (amplitude_m > 0) or swale (amplitude_m < 0)."""

    x0_m: float
    y0_m: float
    amplitude_m: float
    sigma_x_m: float
    sigma_y_m: float


@dataclass(frozen=True)
class GreenUndulationProfile:
    """Immutable specification of a non-planar multi-tier putting green surface."""

    slope_x_pct: float = 0.0
    slope_y_pct: float = 2.2
    tier_height_m: float = 0.035
    tier_center_x_m: float = 1.85
    tier_width_m: float = 0.65
    mounds: Tuple[GaussianMound, ...] = (
        GaussianMound(x0_m=1.40, y0_m=-0.22, amplitude_m=0.018, sigma_x_m=0.85, sigma_y_m=0.55),
        GaussianMound(x0_m=2.65, y0_m=0.18, amplitude_m=-0.014, sigma_x_m=0.75, sigma_y_m=0.50),
    )
    roughness_rms_mm: float = 0.35


@precondition(
    lambda x_m, y_m, profile=GreenUndulationProfile(): math.isfinite(x_m) and math.isfinite(y_m),
    "Green coordinates (x_m, y_m) must be finite real numbers",
)
def evaluate_green_elevation_m(
    x_m: float,
    y_m: float,
    profile: GreenUndulationProfile = GreenUndulationProfile(),
) -> float:
    """Evaluate surface elevation z(x, y) [m] across planar grade, logistic tier, and mounds."""
    sx = 0.01 * profile.slope_x_pct
    sy = 0.01 * profile.slope_y_pct
    z_planar = sx * x_m + sy * y_m
    u_tier = (x_m - profile.tier_center_x_m) / max(profile.tier_width_m, 1e-3)
    u_clamped = max(-40.0, min(40.0, u_tier))
    z_tier = profile.tier_height_m / (1.0 + math.exp(-u_clamped))
    z_mounds = 0.0
    for m in profile.mounds:
        dx = (x_m - m.x0_m) / max(m.sigma_x_m, 1e-3)
        dy = (y_m - m.y0_m) / max(m.sigma_y_m, 1e-3)
        z_mounds += m.amplitude_m * math.exp(-0.5 * (dx * dx + dy * dy))
    return float(z_planar + z_tier + z_mounds)


@precondition(
    lambda x_m, y_m, profile=GreenUndulationProfile(): math.isfinite(x_m) and math.isfinite(y_m),
    "Green coordinates (x_m, y_m) must be finite real numbers",
)
def evaluate_green_gradient(
    x_m: float,
    y_m: float,
    profile: GreenUndulationProfile = GreenUndulationProfile(),
) -> Tuple[float, float]:
    """Return analytical spatial gradient (dz/dx, dz/dy) at (x_m, y_m)."""
    dz_dx = 0.01 * profile.slope_x_pct
    dz_dy = 0.01 * profile.slope_y_pct
    w = max(profile.tier_width_m, 1e-3)
    u_tier = max(-40.0, min(40.0, (x_m - profile.tier_center_x_m) / w))
    sig = 1.0 / (1.0 + math.exp(-u_tier))
    dz_dx += (profile.tier_height_m / w) * sig * (1.0 - sig)
    for m in profile.mounds:
        sx2 = max(m.sigma_x_m, 1e-3) ** 2
        sy2 = max(m.sigma_y_m, 1e-3) ** 2
        dx = x_m - m.x0_m
        dy = y_m - m.y0_m
        gauss = m.amplitude_m * math.exp(-0.5 * (dx * dx / sx2 + dy * dy / sy2))
        dz_dx += -gauss * (dx / sx2)
        dz_dy += -gauss * (dy / sy2)
    return float(dz_dx), float(dz_dy)


@precondition(
    lambda x_bounds_m, y_bounds_m, nx=25, ny=17, profile=GreenUndulationProfile(): (
        nx >= 2 and ny >= 2 and x_bounds_m[1] > x_bounds_m[0] and y_bounds_m[1] > y_bounds_m[0]
    ),
    "Grid dimensions must be >= 2 and bounds strictly increasing",
)
def generate_topographic_grid(
    x_bounds_m: Tuple[float, float],
    y_bounds_m: Tuple[float, float],
    nx: int = 25,
    ny: int = 17,
    profile: GreenUndulationProfile = GreenUndulationProfile(),
) -> Dict[str, NDArray[np.float64]]:
    """Generate 2D elevation and slope-gradient mesh arrays for topographic overlay rendering."""
    xs = np.linspace(x_bounds_m[0], x_bounds_m[1], nx, dtype=np.float64)
    ys = np.linspace(y_bounds_m[0], y_bounds_m[1], ny, dtype=np.float64)
    z_grid = np.zeros((ny, nx), dtype=np.float64)
    gx_grid = np.zeros((ny, nx), dtype=np.float64)
    gy_grid = np.zeros((ny, nx), dtype=np.float64)
    for j, y_val in enumerate(ys):
        for i, x_val in enumerate(xs):
            z_grid[j, i] = evaluate_green_elevation_m(float(x_val), float(y_val), profile)
            gx, gy = evaluate_green_gradient(float(x_val), float(y_val), profile)
            gx_grid[j, i] = gx
            gy_grid[j, i] = gy
    return {
        "x_m": xs,
        "y_m": ys,
        "z_m": z_grid,
        "dz_dx": gx_grid,
        "dz_dy": gy_grid,
        "fall_line_pct": 100.0 * np.hypot(gx_grid, gy_grid),
    }


@precondition(
    lambda x_m, y_m, speed_m_s, roughness_rms_mm=0.35, seed=0: (
        roughness_rms_mm >= 0.0 and speed_m_s >= 0.0
    ),
    "Surface roughness RMS and ball speed must be non-negative",
)
def sample_surface_roughness_acceleration(
    x_m: float,
    y_m: float,
    speed_m_s: float,
    roughness_rms_mm: float = 0.35,
    seed: int = 0,
) -> Tuple[float, float]:
    """Compute spatial Poa annua / bentgrass turf chatter acceleration (ax_m_s2, ay_m_s2)."""
    if roughness_rms_mm <= 1e-6 or speed_m_s <= 1e-4:
        return 0.0, 0.0
    amp_slope = 0.0042 * (roughness_rms_mm / 0.50)
    phase1 = 19.7 * x_m + 23.3 * y_m + 0.73 * seed
    phase2 = 37.1 * x_m - 29.9 * y_m + 1.37 * seed
    phase3 = 53.3 * (x_m + y_m) - 2.11 * seed
    delta_sx = amp_slope * (0.60 * math.sin(phase1) + 0.40 * math.cos(phase2))
    delta_sy = amp_slope * (0.60 * math.cos(phase1) + 0.40 * math.sin(phase3))
    g_eff = (5.0 / 7.0) * STANDARD_GRAVITY_M_S2
    chatter_gain = min(1.35, 0.45 + 0.55 / max(0.25, speed_m_s))
    return float(-g_eff * delta_sx * chatter_gain), float(-g_eff * delta_sy * chatter_gain)
