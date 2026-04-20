from __future__ import annotations

import functools

import numpy as np

from src.shared.python.contracts import check_positive, require
from src.shared.python.core.physics_constants import (
    GRAPHITE_DENSITY_KG_M3,
)

from ._shaft_data import ShaftMaterial, ShaftProperties

SHAFT_LENGTH_DRIVER = 1.1557  # [m] 45.5" tour-average driver shaft
SHAFT_LENGTH_IRON = 0.9398  # [m] 37" tour-average 7-iron shaft
STEEL_DENSITY = 7850  # [kg/m³]
GRAPHITE_DENSITY = int(GRAPHITE_DENSITY_KG_M3)  # [kg/m³] from physics_constants
STEEL_E = 200e9  # [Pa] Young's modulus for steel
GRAPHITE_E = 130e9  # [Pa] Young's modulus for graphite


@functools.lru_cache(maxsize=256)
def compute_section_inertia(
    outer_diameter: float,
    wall_thickness: float,
) -> float:
    """Compute second moment of area for hollow circular section. Cached for performance.

    I = π/64 * (D⁴ - d⁴)

    Args:
        outer_diameter: Outer diameter [m]
        wall_thickness: Wall thickness [m]

    Returns:
        Second moment of area [m⁴]
    """
    check_positive(outer_diameter, "outer_diameter")
    check_positive(wall_thickness, "wall_thickness")
    require(
        wall_thickness <= outer_diameter / 2,
        "wall_thickness must not exceed outer_diameter / 2",
        wall_thickness,
    )
    d_outer = outer_diameter
    d_inner = outer_diameter - 2 * wall_thickness
    d_inner = max(d_inner, 0.0)  # Ensure non-negative

    inertia = np.pi / 64 * (d_outer**4 - d_inner**4)
    return float(inertia)


@functools.lru_cache(maxsize=256)
def compute_section_area(
    outer_diameter: float,
    wall_thickness: float,
) -> float:
    """Compute cross-sectional area for hollow circular section. Cached for performance.

    A = π/4 * (D² - d²)

    Args:
        outer_diameter: Outer diameter [m]
        wall_thickness: Wall thickness [m]

    Returns:
        Cross-sectional area [m²]
    """
    check_positive(outer_diameter, "outer_diameter")
    check_positive(wall_thickness, "wall_thickness")
    require(
        wall_thickness <= outer_diameter / 2,
        "wall_thickness must not exceed outer_diameter / 2",
        wall_thickness,
    )
    d_outer = outer_diameter
    d_inner = outer_diameter - 2 * wall_thickness
    d_inner = max(d_inner, 0.0)

    A = np.pi / 4 * (d_outer**2 - d_inner**2)
    return float(A)


def compute_EI_profile(
    properties: ShaftProperties,
) -> np.ndarray:
    """Compute bending stiffness EI along shaft.

    EI = E * I(x) where I is the section inertia.

    Args:
        properties: Shaft properties

    Returns:
        EI values at each station [N·m²] (N,)
    """
    n_stations = len(properties.station_positions)
    EI = np.zeros(n_stations)

    for i in range(n_stations):
        inertia = compute_section_inertia(
            properties.outer_diameter[i], properties.wall_thickness[i]
        )
        EI[i] = properties.youngs_modulus * inertia

    return EI


def compute_mass_profile(
    properties: ShaftProperties,
) -> np.ndarray:
    """Compute mass per unit length along shaft.

    μ = ρ * A(x)

    Args:
        properties: Shaft properties

    Returns:
        Mass per length at each station [kg/m] (N,)
    """
    n_stations = len(properties.station_positions)
    mass_per_length = np.zeros(n_stations)

    for i in range(n_stations):
        A = compute_section_area(
            properties.outer_diameter[i], properties.wall_thickness[i]
        )
        mass_per_length[i] = properties.density * A

    return mass_per_length


def create_standard_shaft(
    material: ShaftMaterial = ShaftMaterial.GRAPHITE,
    length: float = SHAFT_LENGTH_DRIVER,
    n_stations: int = 11,
    tip_diameter: float = 0.0085,  # [m] 8.5mm tip
    butt_diameter: float = 0.015,  # [m] 15mm butt
    wall_thickness: float = 0.001,  # [m] 1mm wall
) -> ShaftProperties:
    """Create standard tapered golf shaft properties.

    Args:
        material: Shaft material
        length: Total shaft length [m]
        n_stations: Number of stations for property definition
        tip_diameter: Diameter at tip (head end) [m]
        butt_diameter: Diameter at butt (grip end) [m]
        wall_thickness: Wall thickness [m]

    Returns:
        ShaftProperties with linear taper
    """
    require(material is not None, "material must be provided", material)
    check_positive(length, "length")
    require(n_stations >= 2, "n_stations must be at least 2", n_stations)
    check_positive(tip_diameter, "tip_diameter")
    check_positive(butt_diameter, "butt_diameter")
    check_positive(wall_thickness, "wall_thickness")
    # Linear station positions from tip to butt
    stations = np.linspace(0, length, n_stations)

    # Linear taper in diameter
    diameters = np.linspace(tip_diameter, butt_diameter, n_stations)

    # Constant wall thickness (could be varied)
    wall = np.full(n_stations, wall_thickness)

    # Material properties
    if material == ShaftMaterial.STEEL:
        E = STEEL_E
        density = STEEL_DENSITY
    else:
        E = GRAPHITE_E
        density = GRAPHITE_DENSITY

    return ShaftProperties(
        length=length,
        outer_diameter=diameters,
        wall_thickness=wall,
        station_positions=stations,
        material=material,
        youngs_modulus=E,
        density=density,
    )


# Swing-speed thresholds (mph) for shaft-flex recommendation.
_FLEX_THRESHOLDS_MPH = {
    "Ladies": 60.0,
    "Senior": 72.0,
    "Regular": 84.0,
    "Stiff": 96.0,
}


def recommend_shaft_flex(
    swing_speed_mph: float,
    tempo_ms: float | None = None,
    release_timing_ms: float | None = None,
) -> str:
    """Return a recommended shaft flex based on swing parameters.

    Rule-based logic using industry-standard swing-speed breakpoints with
    optional corrections for slow tempo (shift one flex stiffer) or early
    release (shift one flex softer).

    Args:
        swing_speed_mph: Driver club-head speed at impact in mph.
        tempo_ms: Total swing duration in milliseconds. Slow tempo (>1400 ms)
            shifts recommendation one category stiffer; fast tempo (<1000 ms)
            shifts softer.
        release_timing_ms: Time from top-of-backswing to ball-strike in ms.
            Early release (< 200 ms) shifts recommendation one category softer.

    Returns:
        Flex category string: one of "Ladies", "Senior", "Regular",
        "Stiff", "X-Stiff".
    """
    if swing_speed_mph < 0:
        raise ValueError("swing_speed_mph must be non-negative")

    flex_order = ["Ladies", "Senior", "Regular", "Stiff", "X-Stiff"]

    base = "X-Stiff"
    for flex, threshold in _FLEX_THRESHOLDS_MPH.items():
        if swing_speed_mph < threshold:
            base = flex
            break

    idx = flex_order.index(base)

    if tempo_ms is not None:
        if tempo_ms > 1400:
            idx = min(idx + 1, len(flex_order) - 1)
        elif tempo_ms < 1000:
            idx = max(idx - 1, 0)

    if release_timing_ms is not None and release_timing_ms < 200:
        idx = max(idx - 1, 0)

    return flex_order[idx]
