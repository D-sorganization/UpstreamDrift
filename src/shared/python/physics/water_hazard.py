"""Hydrodynamic water-hazard entry kinematics for golf balls.

This module provides a minimal, analytic 1-D model of a golf ball entering
a water hazard. It estimates submersion depth, peak deceleration, whether
the ball will skip (bounce once) along the surface, and the kinetic energy
dissipated on entry.

References
----------
- Worthington, A. M. (1908). *A Study of Splashes*. Longmans, Green, and Co.
  Pioneering empirical study of liquid impact and water entry.
- Truscott, T. T., & Techet, A. H. (2009). "Water entry of spinning spheres."
  *Journal of Fluid Mechanics*, 625, 135-165. Skipping/bouncing criteria for
  spheres impacting a free water surface at shallow angles.

Notes
-----
The submersion depth uses the classical analytic solution for 1-D motion
under quadratic drag with constant gravity:

    d = (m / (rho * Cd * A)) * ln(1 + (rho * Cd * A * v_n^2) / (2 * m * g))

where ``v_n = v * sin(theta)`` is the velocity component normal to the
free surface. Buoyancy and added-mass effects are intentionally neglected
for this minimal model; the result is a useful first-order estimate, not a
high-fidelity simulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

GRAVITY_M_S2: float = 9.80665


@dataclass(frozen=True)
class WaterEntryResult:
    """Result of a water-entry kinematic estimate.

    Attributes
    ----------
    submersion_depth_m:
        Stopping depth from a 1-D quadratic-drag model, in metres.
    vertical_decel_peak_m_s2:
        Peak deceleration at the moment of impact, in m/s^2.
    bounces:
        Either 0 (ball submerges) or 1 (ball skips off the surface).
    energy_dissipated_j:
        Kinetic energy lost to the water during entry, in joules.
    """

    submersion_depth_m: float
    vertical_decel_peak_m_s2: float
    bounces: int
    energy_dissipated_j: float


def _validate_positive_finite(name: str, value: float) -> None:
    """Raise if ``value`` is not a finite, strictly positive number."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if float(value) <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value!r}")


def water_entry_kinematics(
    *,
    impact_velocity_m_s: float,
    impact_angle_deg: float,
    ball_mass_kg: float = 0.04593,
    ball_radius_m: float = 0.02135,
    water_density_kg_m3: float = 1000.0,
    drag_coefficient: float = 0.47,
) -> WaterEntryResult:
    """Estimate water-entry kinematics for a golf ball.

    Parameters
    ----------
    impact_velocity_m_s:
        Ball speed at the water surface (must be > 0 and finite).
    impact_angle_deg:
        Angle below horizontal in degrees, in the range [0, 90].
    ball_mass_kg:
        Ball mass in kilograms (default: USGA-spec 0.04593 kg).
    ball_radius_m:
        Ball radius in metres (default: USGA-spec 0.02135 m).
    water_density_kg_m3:
        Water density in kg/m^3 (default: 1000 for fresh water).
    drag_coefficient:
        Drag coefficient for a rigid sphere in water (default: 0.47).

    Returns
    -------
    WaterEntryResult
        Submersion depth, peak deceleration, bounce count, and dissipated
        kinetic energy.

    Raises
    ------
    TypeError
        If any argument is not a real number.
    ValueError
        If velocity, mass, or radius are non-positive or non-finite, or if
        ``impact_angle_deg`` is outside [0, 90] or non-finite.

    Notes
    -----
    Skipping criterion follows Truscott & Techet (2009): shallow, fast
    impacts (angle < 20 deg and speed > 15 m/s) yield a single bounce.
    """
    # ---- DbC: precondition validation -----------------------------------
    _validate_positive_finite("impact_velocity_m_s", impact_velocity_m_s)
    _validate_positive_finite("ball_mass_kg", ball_mass_kg)
    _validate_positive_finite("ball_radius_m", ball_radius_m)
    _validate_positive_finite("water_density_kg_m3", water_density_kg_m3)
    _validate_positive_finite("drag_coefficient", drag_coefficient)

    if not isinstance(impact_angle_deg, (int, float)) or isinstance(
        impact_angle_deg, bool
    ):
        raise TypeError(
            "impact_angle_deg must be a real number, "
            f"got {type(impact_angle_deg).__name__}"
        )
    if not math.isfinite(float(impact_angle_deg)):
        raise ValueError(f"impact_angle_deg must be finite, got {impact_angle_deg!r}")
    if not (0.0 <= float(impact_angle_deg) <= 90.0):
        raise ValueError(
            f"impact_angle_deg must be in [0, 90], got {impact_angle_deg!r}"
        )

    # ---- Geometry & drag prefactor --------------------------------------
    v = float(impact_velocity_m_s)
    theta = math.radians(float(impact_angle_deg))
    v_n = v * math.sin(theta)  # normal-to-surface velocity component
    m = float(ball_mass_kg)
    r = float(ball_radius_m)
    rho = float(water_density_kg_m3)
    cd = float(drag_coefficient)
    area = math.pi * r * r  # frontal area
    k = rho * cd * area  # quadratic-drag prefactor (kg/m)

    # ---- Submersion depth (analytic terminal-impact formula) ------------
    # d = (m / k) * ln(1 + k * v_n^2 / (2 * m * g))
    if v_n > 0.0:
        depth = (m / k) * math.log1p(k * v_n * v_n / (2.0 * m * GRAVITY_M_S2))
    else:
        depth = 0.0

    # ---- Peak deceleration at impact ------------------------------------
    # F_drag = 0.5 * rho * Cd * A * v_n^2; a = F/m
    decel_peak = 0.5 * k * v_n * v_n / m

    # ---- Skipping criterion (Truscott & Techet, 2009) -------------------
    bounces = 1 if (impact_angle_deg < 20.0 and impact_velocity_m_s > 15.0) else 0

    # ---- Energy dissipated ----------------------------------------------
    # On submersion the entire KE is eventually lost to the water; on a
    # bounce the normal component is dissipated, the tangential component
    # is retained. This matches the standard skipping idealisation.
    ke_total = 0.5 * m * v * v
    energy_dissipated = 0.5 * m * v_n * v_n if bounces == 1 else ke_total

    return WaterEntryResult(
        submersion_depth_m=depth,
        vertical_decel_peak_m_s2=decel_peak,
        bounces=bounces,
        energy_dissipated_j=energy_dissipated,
    )


__all__ = ["WaterEntryResult", "water_entry_kinematics", "GRAVITY_M_S2"]
