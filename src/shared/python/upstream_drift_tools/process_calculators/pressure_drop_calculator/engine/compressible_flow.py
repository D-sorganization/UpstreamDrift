"""Compressible-flow helpers for the pressure drop engine."""

from __future__ import annotations

import logging
import math

from ....utils.unit_constants import R_UNIVERSAL_KMOL

logger = logging.getLogger(__name__)

PI = math.pi
R_UNIVERSAL = R_UNIVERSAL_KMOL


def _iterate_compressible_pressure(
    p1: float,
    p2_initial: float,
    coeff: float,
    resistance: float,
    max_iterations: int = 50,
    tolerance: float = 1.0,
) -> tuple[float, bool]:
    """Iteratively solve the isothermal compressible flow equation for P2."""
    if p1 is None:
        raise ValueError("P1 must be provided")

    p2 = p2_initial
    for iteration in range(max_iterations):
        p2_old = p2
        ln_term = 2.0 * math.log(p1 / p2) if p2 > 0 and p1 > p2 else 0.0
        rhs = coeff * (resistance + ln_term)
        p2_squared = p1**2 - rhs

        if p2_squared <= 0:
            logger.warning(
                "Compressible flow calculation indicates choked flow condition"
            )
            return p2, True

        p2 = math.sqrt(p2_squared)
        if abs(p2 - p2_old) < tolerance:
            logger.debug("Compressible flow converged in %s iterations", iteration + 1)
            break

    return p2, False


def calculate_compressible_flow_correction(
    inlet_pressure: float,
    outlet_pressure: float,
    length: float,
    diameter: float,
    mass_flow_rate: float,
    temperature: float,
    molecular_weight: float,
    compressibility_factor: float,
    friction_factor: float,
    total_k_factor: float = 0.0,
) -> tuple[float, float]:
    """Calculate pressure drop accounting for compressibility effects."""
    if not (diameter > 0):
        raise ValueError(f"diameter must be positive, got {diameter}")
    if not (temperature > 0):
        raise ValueError(f"temperature must be positive (K), got {temperature}")
    if not (molecular_weight > 0):
        raise ValueError(f"molecular_weight must be positive, got {molecular_weight}")

    area = PI * (diameter**2) / 4.0
    mass_flux = mass_flow_rate / area
    resistance = friction_factor * (length / diameter) + total_k_factor
    coeff = (
        (mass_flux**2)
        * (compressibility_factor * R_UNIVERSAL * temperature)
        / molecular_weight
    )

    p2, is_choked = _iterate_compressible_pressure(
        inlet_pressure,
        outlet_pressure,
        coeff,
        resistance,
    )
    if is_choked:
        return inlet_pressure - outlet_pressure, outlet_pressure

    corrected_dp = inlet_pressure - p2
    pressure_ratio = p2 / inlet_pressure
    if pressure_ratio > 0:
        expansion_factor = math.sqrt(
            pressure_ratio * (1 - pressure_ratio**2) / (1 - pressure_ratio)
            if pressure_ratio < 1
            else 1.0
        )
    else:
        expansion_factor = 1.0

    logger.debug(
        "Compressible flow correction: ΔP_incomp=%.0f Pa, ΔP_comp=%.0f Pa, Y=%.3f",
        inlet_pressure - outlet_pressure,
        corrected_dp,
        expansion_factor,
    )
    return corrected_dp, p2


def calculate_expansion_factor(
    inlet_pressure: float,
    pressure_drop: float,
    friction_factor: float,
    length_over_diameter: float,
    gamma: float = 1.4,
) -> float:
    """Calculate gas expansion factor Y for compressible flow."""
    del friction_factor, length_over_diameter
    if inlet_pressure is None:
        raise ValueError("inlet_pressure must be provided")
    if inlet_pressure <= 0 or pressure_drop < 0:
        return 1.0

    pressure_ratio = (inlet_pressure - pressure_drop) / inlet_pressure
    if pressure_ratio <= 0:
        return 0.0
    if pressure_ratio >= 0.99:
        return 1.0

    try:
        numerator = (
            gamma
            * (pressure_ratio ** (2.0 / gamma))
            * (1.0 - pressure_ratio ** ((gamma - 1.0) / gamma))
        )
        denominator = (gamma - 1.0) * (1.0 - pressure_ratio)
        y_factor = 1.0 if denominator <= 0 else math.sqrt(numerator / denominator)
    except (ValueError, ZeroDivisionError):
        y_factor = 1.0 - pressure_drop / (3.0 * gamma * inlet_pressure)

    return max(0.0, min(y_factor, 1.0))
