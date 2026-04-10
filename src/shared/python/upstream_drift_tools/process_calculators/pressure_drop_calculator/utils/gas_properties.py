#!/usr/bin/env python3
"""Gas mixture property calculations for pressure drop analysis.

Comprehensive calculation of thermophysical properties for gas mixtures
including combustion and gasification gases.

References:
    - Reid, Prausnitz, Poling: "The Properties of Gases and Liquids", 5th Ed (2001)
    - Perry's Chemical Engineers' Handbook, 9th Edition
    - Poling, Prausnitz, O'Connell: "Properties of Gases and Liquids", 5th Ed
    - Chapman-Enskog theory for gas viscosity
    - Lucas method for gas viscosity
    - Lee-Kesler correlation for compressibility factor
"""

import logging

from ._component_database import (
    GAS_DATABASE,
    R_UNIVERSAL,
    SUTHERLAND_CONSTANTS,
    ComponentProperties,
)
from ._heat_capacity import (
    calculate_heat_capacity_ratio,
    calculate_ideal_gas_cp,
    calculate_mixture_cp,
    calculate_speed_of_sound,
)
from ._mixture_properties import (
    calculate_compressibility_factor,
    calculate_ideal_gas_density,
    calculate_mixture_molecular_weight,
    calculate_real_gas_density,
)
from ._viscosity import (
    _compute_pure_viscosities,
    _wilke_mixing_rule,
    calculate_mixture_viscosity_simple,
    calculate_mixture_viscosity_wilke,
    calculate_pure_gas_viscosity_lucas,
    calculate_pure_gas_viscosity_sutherland,
)

__all__ = [
    "ComponentProperties",
    "GAS_DATABASE",
    "R_UNIVERSAL",
    "SUTHERLAND_CONSTANTS",
    "calculate_ideal_gas_cp",
    "calculate_mixture_cp",
    "calculate_heat_capacity_ratio",
    "calculate_speed_of_sound",
    "calculate_mixture_molecular_weight",
    "calculate_ideal_gas_density",
    "calculate_compressibility_factor",
    "calculate_real_gas_density",
    "calculate_pure_gas_viscosity_lucas",
    "calculate_pure_gas_viscosity_sutherland",
    "_compute_pure_viscosities",
    "_wilke_mixing_rule",
    "calculate_mixture_viscosity_wilke",
    "calculate_mixture_viscosity_simple",
    "calculate_gas_properties",
]

logger = logging.getLogger(__name__)


def calculate_gas_properties(
    composition: dict[str, float],
    temperature: float,
    pressure: float,
    use_compressibility: bool = True,
) -> dict[str, float]:
    """Calculate complete set of gas mixture properties.

    Args:
        composition: Dictionary of {component: mole_fraction}
        temperature: Temperature (K)
        pressure: Pressure (Pa)
        use_compressibility: Whether to use real gas corrections

    Returns:
        Dictionary with properties:
            - molecular_weight (kg/kmol)
            - density (kg/m³)
            - viscosity (Pa·s)
            - compressibility_factor (dimensionless)
            - heat_capacity_ratio (γ = Cp/Cv)
            - speed_of_sound (m/s)
            - cp (J/(mol·K))

    Example:
        >>> comp = {'H2': 0.25, 'CO': 0.35, 'CO2': 0.15, 'N2': 0.25}
        >>> props = calculate_gas_properties(comp, 700, 5e5)
        >>> print(f"Density: {props['density']:.3f} kg/m³")
        >>> print(f"Gamma: {props['heat_capacity_ratio']:.3f}")
    """
    # Molecular weight
    if not (composition is not None):
        raise ValueError("composition must be provided")
    if not (composition is not None):
        raise ValueError("composition must be provided")
    mw = calculate_mixture_molecular_weight(composition)

    # Compressibility factor
    if use_compressibility:
        Z = calculate_compressibility_factor(composition, temperature, pressure)
        density = calculate_real_gas_density(mw, temperature, pressure, Z)
    else:
        Z = 1.0
        density = calculate_ideal_gas_density(mw, temperature, pressure)

    # Viscosity
    viscosity = calculate_mixture_viscosity_wilke(composition, temperature, pressure)

    # Heat capacity and gamma
    cp = calculate_mixture_cp(composition, temperature)
    gamma = calculate_heat_capacity_ratio(composition, temperature)

    # Speed of sound
    speed_of_sound = calculate_speed_of_sound(composition, temperature, mw)

    properties = {
        "molecular_weight": mw,
        "density": density,
        "viscosity": viscosity,
        "compressibility_factor": Z,
        "heat_capacity_ratio": gamma,
        "speed_of_sound": speed_of_sound,
        "cp": cp,
    }

    logger.info(f"Gas properties at T={temperature}K, P={pressure / 1e5:.1f}bar:")
    logger.info(f"  MW = {mw:.2f} kg/kmol")
    logger.info(f"  ρ = {density:.4f} kg/m³")
    logger.info(f"  μ = {viscosity:.6e} Pa·s")
    logger.info(f"  Z = {Z:.4f}")
    logger.info(f"  γ = {gamma:.4f}")
    logger.info(f"  a = {speed_of_sound:.1f} m/s")

    return properties


if __name__ == "__main__":
    # Demonstration
    logging.basicConfig(level=logging.INFO)

    logger.info("\n" + "=" * 80)
    logger.info("GAS MIXTURE PROPERTY CALCULATOR - EXAMPLES")
    logger.info("=" * 80)

    # Example 1: Syngas composition
    logger.info("\nExample 1: Syngas from coal gasification")
    logger.info("-" * 80)
    syngas = {
        "H2": 0.30,
        "CO": 0.40,
        "CO2": 0.15,
        "N2": 0.10,
        "CH4": 0.05,
    }
    T = 800  # K
    P = 25e5  # Pa (25 bar)

    props = calculate_gas_properties(syngas, T, P)
    logger.info(f"\nComposition: {syngas}")
    logger.info(f"Temperature: {T} K ({T - 273.15:.0f}°C)")
    logger.info(f"Pressure: {P / 1e5:.1f} bar")
    logger.info("\nCalculated Properties:")
    logger.info(f"  Molecular Weight: {props['molecular_weight']:.2f} kg/kmol")
    logger.info(f"  Density: {props['density']:.4f} kg/m³")
    logger.info(
        f"  Viscosity: {props['viscosity']:.6e} Pa·s ({props['viscosity'] * 1e6:.2f} µPa·s)"
    )
    logger.info(f"  Z-factor: {props['compressibility_factor']:.4f}")

    # Example 2: Air at different conditions
    logger.info("\n\nExample 2: Air at various temperatures")
    logger.info("-" * 80)
    air = {"Air": 1.0}
    for temp in [300, 500, 800, 1200]:
        props_air = calculate_gas_properties(air, temp, 1e5, use_compressibility=False)
        logger.info(
            f"T = {temp}K: ρ = {props_air['density']:.4f} kg/m³, "
            f"μ = {props_air['viscosity'] * 1e6:.2f} µPa·s"
        )
