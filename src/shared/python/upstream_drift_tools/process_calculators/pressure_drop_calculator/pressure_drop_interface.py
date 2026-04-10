#!/usr/bin/env python3
"""User-friendly Python interface for advanced pressure drop calculator.

This module provides a simplified API for performing pressure drop calculations
with support for various input units and gas compositions.

QUICK START:
    >>> from pressure_drop_calculator import calculate_pressure_drop, show_help
    >>> show_help()  # Display available options and examples

    >>> # Simple calculation with air
    >>> result = calculate_pressure_drop(
    ...     pipe_size="4",
    ...     pipe_schedule="40",
    ...     pipe_length=100,  # meters
    ...     flow_rate=1000,
    ...     flow_unit='kg/h',
    ...     pressure=10,  # bar
    ...     temperature=500  # K
    ... )
    >>> print(f"Pressure drop: {result['pressure_drop_bar']:.4f} bar")

AVAILABLE UNITS:
    - Temperature: K, C, F
    - Pressure: Pa, kPa, bar, psi, atm
    - Mass flow: kg/s, kg/h, lb/hr, ton/h
    - Molar flow: mol/s, kmol/h, lbmol/hr
    - Volumetric flow: m³/h, SCFM, ACFM, Nm³/h, CFM, L/s

FRICTION FACTOR METHODS:
    - 'colebrook': Most accurate, iterative (default)
    - 'swamee-jain': Explicit, within 1% of Colebrook
    - 'churchill': Covers all flow regimes
    - 'haaland': Simplified, within 1.5%

GAS COMPONENTS:
    H2, CO, CO2, CH4, C2H6, C2H4, N2, O2, H2O, Ar, H2S, NH3, Air
"""

from __future__ import annotations

import logging
from typing import Any

from ._pressure_drop_helpers import (
    compare_friction_methods,
    list_fittings,
    list_flow_units,
    list_gas_components,
    list_materials,
    list_pipe_sizes,
    show_help,
)
from ._pressure_drop_output import (
    _convert_pressure,
    _convert_temperature,
    _format_results,
    print_results,
)
from ._pressure_drop_validation import validate_inputs
from .engine.pressure_drop_calculation_engine import (
    PressureDropCalculationEngine,
)
from .models.pressure_drop_data_models import (
    GasComposition,
    PipeFitting,
    PressureDropInputs,
)
from .utils.fitting_loss_coefficients import FITTING_K_FACTORS
from .utils.flow_rate_converter import (
    convert_flow_rate_to_mass,
)
from .utils.gas_properties import (
    calculate_mixture_molecular_weight,
)
from .utils.pipe_database import (
    get_pipe_spec,
    get_roughness,
)

__all__ = [
    "calculate_pressure_drop",
    "calculate_pressure_drop_custom_gas",
    "calculate_pressure_drop_syngas",
    "compare_friction_methods",
    "list_fittings",
    "list_flow_units",
    "list_gas_components",
    "list_materials",
    "list_pipe_sizes",
    "main",
    "print_results",
    "show_help",
    "validate_inputs",
]

logger = logging.getLogger(__name__)


# ============================================================================
# CORE CALCULATION FUNCTIONS
# ============================================================================


# ============================================================================
# HIGH-LEVEL API FUNCTIONS
# ============================================================================


def _resolve_pipe_geometry(
    pipe_size: str | None,
    pipe_schedule: str | None,
    pipe_diameter: float | None,
    pipe_material: str,
    pipe_roughness: float | None,
) -> tuple[float, float]:
    """Resolve pipe diameter and roughness from user-supplied parameters.

    Returns:
        Tuple of (diameter_m, roughness_m).
    """
    if not (pipe_material is not None):
        raise ValueError("pipe_material must be provided")
    if not (pipe_material is not None):
        raise ValueError("pipe_material must be provided")
    if pipe_diameter is None:
        if pipe_size is None or pipe_schedule is None:
            raise ValueError(
                "Either provide pipe_diameter or both pipe_size and pipe_schedule"
            )
        pipe_spec = get_pipe_spec(pipe_size, pipe_schedule, pipe_material)
        pipe_diameter = pipe_spec.get_id_meters()
        logger.info(
            f'Using {pipe_size}" Schedule {pipe_schedule}: ID = {pipe_diameter * 1000:.2f} mm'
        )

    roughness = (
        pipe_roughness
        if pipe_roughness is not None
        else get_roughness(pipe_material, "m")
    )
    return pipe_diameter, roughness


def _resolve_gas_and_flow(
    flow_rate: float,
    flow_unit: str,
    gas_composition: dict[str, float] | None,
    temp_k: float,
    pressure_pa: float,
    compressibility_correction: bool,
    standard_condition: str,
) -> tuple[GasComposition, float]:
    """Normalize gas composition and convert flow rate to kg/s.

    Returns:
        Tuple of (composition, mass_flow_kg_s).
    """
    if not (flow_rate is not None):
        raise ValueError("flow_rate must be provided")
    if not (flow_rate is not None):
        raise ValueError("flow_rate must be provided")
    if gas_composition is None:
        gas_composition = {"Air": 1.0}
        logger.info("Using default gas composition: Air")

    composition = GasComposition(components=gas_composition)
    composition.normalize()
    molecular_weight = calculate_mixture_molecular_weight(composition.components)

    if flow_unit.upper() in ["ACFM", "CFM"]:
        from .utils.gas_properties import calculate_gas_properties

        props = calculate_gas_properties(
            composition.components, temp_k, pressure_pa, compressibility_correction
        )
        density = props["density"]
        from .utils.flow_rate_converter import volumetric_actual_to_mass

        mass_flow_kg_s = volumetric_actual_to_mass(
            flow_rate, flow_unit, density, "kg/s"
        )
    else:
        mass_flow_kg_s = convert_flow_rate_to_mass(
            flow_rate,
            flow_unit,
            molecular_weight,
            temperature=temp_k,
            pressure=pressure_pa,
            standard=standard_condition,
        )

    logger.info(f"Mass flow rate: {mass_flow_kg_s:.4f} kg/s ({flow_rate} {flow_unit})")
    return composition, mass_flow_kg_s


def _build_fitting_list(
    fittings: list[dict[str, str | int | float]] | None,
) -> list[PipeFitting]:
    """Convert raw fitting dicts into PipeFitting objects."""
    fitting_list: list[PipeFitting] = []
    if fittings:
        for fitting_dict in fittings:
            fitting_type = str(fitting_dict.get("type", ""))
            quantity = int(fitting_dict.get("quantity", 1))
            k_factor = float(
                fitting_dict.get("k_factor", FITTING_K_FACTORS.get(fitting_type, 0.0))
            )
            fitting_list.append(
                PipeFitting(
                    fitting_type=fitting_type, quantity=quantity, k_factor=k_factor
                )
            )
    return fitting_list


def calculate_pressure_drop(
    # Pipe geometry
    pipe_size: str | None = None,
    pipe_schedule: str | None = None,
    pipe_diameter: float | None = None,  # meters
    pipe_length: float = 100.0,  # meters
    pipe_material: str = "Commercial Steel",
    pipe_roughness: float | None = None,  # Override roughness (meters)
    elevation_change: float = 0.0,  # meters
    # Flow conditions
    flow_rate: float = 1000.0,
    flow_unit: str = "kg/h",
    pressure: float = 1.0,  # bar (absolute)
    pressure_unit: str = "bar",
    temperature: float = 288.15,  # K
    temperature_unit: str = "K",
    # Gas composition (default: air)
    gas_composition: dict[str, float] | None = None,
    # Fittings (optional)
    fittings: list[dict[str, str | int | float]] | None = None,
    # Calculation options
    friction_method: str = "colebrook",
    compressibility_correction: bool = True,
    standard_condition: str = "STP",
) -> dict[str, Any]:
    """Calculate pressure drop with flexible unit inputs.

    Args:
        pipe_size: Nominal pipe size (e.g., '4', '6', '8')
        pipe_schedule: Pipe schedule (e.g., '40', '80', 'STD')
        pipe_diameter: Pipe ID in meters (if pipe_size not provided)
        pipe_length: Pipe length (meters)
        pipe_material: Pipe material from MATERIAL_ROUGHNESS
        pipe_roughness: Explicit roughness in meters (overrides material)
        elevation_change: Elevation change (meters, + = upward)

        flow_rate: Flow rate value
        flow_unit: Flow rate unit (kg/h, lb/hr, SCFM, kmol/h, etc.)
        pressure: Inlet pressure
        pressure_unit: Pressure unit (bar, psi, Pa, atm)
        temperature: Inlet temperature
        temperature_unit: Temperature unit (K, C, F)

        gas_composition: Dict of {component: mole_fraction}
        fittings: List of fittings, e.g., [{'type': '90_elbow_std', 'quantity': 4}]

        friction_method: 'colebrook', 'swamee-jain', 'churchill', or 'haaland'
        compressibility_correction: Use real gas corrections
        standard_condition: 'STP', 'NTP', 'SCFM', etc. for volumetric flows

    Returns:
        Dictionary with results including pressure drop in various units

    Example:
        >>> result = calculate_pressure_drop(
        ...     pipe_size='4',
        ...     pipe_schedule='40',
        ...     pipe_length=100,
        ...     flow_rate=1500,
        ...     flow_unit='SCFM',
        ...     pressure=10,
        ...     temperature=500
        ... )
        >>> print(f"ΔP = {result['pressure_drop_bar']:.4f} bar")
    """
    if not (pipe_length is not None):
        raise ValueError("pipe_length must be provided")
    if not (pipe_length is not None):
        raise ValueError("pipe_length must be provided")
    temp_k = _convert_temperature(temperature, temperature_unit, "K")
    pressure_pa = _convert_pressure(pressure, pressure_unit, "Pa")

    pipe_diameter, roughness = _resolve_pipe_geometry(
        pipe_size, pipe_schedule, pipe_diameter, pipe_material, pipe_roughness
    )
    composition, mass_flow_kg_s = _resolve_gas_and_flow(
        flow_rate,
        flow_unit,
        gas_composition,
        temp_k,
        pressure_pa,
        compressibility_correction,
        standard_condition,
    )
    fitting_list = _build_fitting_list(fittings)

    inputs = PressureDropInputs(
        pipe_diameter=pipe_diameter,
        pipe_length=pipe_length,
        pipe_roughness=roughness,
        elevation_change=elevation_change,
        mass_flow_rate=mass_flow_kg_s,
        inlet_pressure=pressure_pa,
        inlet_temperature=temp_k,
        gas_composition=composition,
        fittings=fitting_list,
        compressibility_correction=compressibility_correction,
        friction_method=friction_method,
    )

    engine = PressureDropCalculationEngine()
    results = engine.calculate(inputs)
    return _format_results(results)


def calculate_pressure_drop_custom_gas(
    pipe_diameter: float,  # meters
    pipe_length: float,  # meters
    gas_composition: dict[str, float],
    flow_rate: float,
    flow_unit: str,
    pressure: float,  # bar
    temperature: float,  # K
    pipe_roughness: float = 0.000045,  # meters (default: commercial steel)
    elevation_change: float = 0.0,
    fittings: list[dict[str, Any]] | None = None,
    friction_method: str = "colebrook",
) -> dict[str, Any]:
    """Simplified API for custom gas composition.

    Args:
        pipe_diameter: Internal diameter (meters)
        pipe_length: Pipe length (meters)
        gas_composition: Dictionary of {component: mole_fraction}
        flow_rate: Flow rate value
        flow_unit: Flow rate unit
        pressure: Inlet pressure (bar, absolute)
        temperature: Inlet temperature (K)
        pipe_roughness: Absolute roughness (meters)
        elevation_change: Elevation change (meters)
        fittings: List of fittings
        friction_method: Friction factor correlation

    Returns:
        Results dictionary

    Example:
        >>> syngas = {'H2': 0.3, 'CO': 0.4, 'CO2': 0.2, 'N2': 0.1}
        >>> result = calculate_pressure_drop_custom_gas(
        ...     pipe_diameter=0.1543,  # 6" Schedule 40
        ...     pipe_length=50,
        ...     gas_composition=syngas,
        ...     flow_rate=2000,
        ...     flow_unit='kg/h',
        ...     pressure=25,
        ...     temperature=800
        ... )
    """
    return calculate_pressure_drop(
        pipe_diameter=pipe_diameter,
        pipe_length=pipe_length,
        pipe_roughness=pipe_roughness,
        elevation_change=elevation_change,
        flow_rate=flow_rate,
        flow_unit=flow_unit,
        pressure=pressure,
        pressure_unit="bar",
        temperature=temperature,
        temperature_unit="K",
        gas_composition=gas_composition,
        fittings=fittings,
        friction_method=friction_method,
    )


def calculate_pressure_drop_syngas(
    pipe_size: str,
    pipe_schedule: str,
    pipe_length: float,
    flow_rate: float,
    flow_unit: str,
    pressure: float,  # bar
    temperature: float,  # K
    H2_fraction: float = 0.30,
    CO_fraction: float = 0.40,
    CO2_fraction: float = 0.15,
    N2_fraction: float = 0.10,
    CH4_fraction: float = 0.05,
    **kwargs: Any,
) -> dict[str, Any]:
    """Convenience function for typical syngas calculations.

    Args:
        pipe_size: Nominal pipe size
        pipe_schedule: Pipe schedule
        pipe_length: Pipe length (meters)
        flow_rate: Flow rate value
        flow_unit: Flow rate unit
        pressure: Pressure (bar)
        temperature: Temperature (K)
        H2_fraction: Hydrogen mole fraction
        CO_fraction: CO mole fraction
        CO2_fraction: CO2 mole fraction
        N2_fraction: Nitrogen mole fraction
        CH4_fraction: Methane mole fraction
        **kwargs: Additional arguments passed to calculate_pressure_drop

    Returns:
        Results dictionary

    Example:
        >>> result = calculate_pressure_drop_syngas(
        ...     pipe_size='6',
        ...     pipe_schedule='40',
        ...     pipe_length=100,
        ...     flow_rate=5000,
        ...     flow_unit='kg/h',
        ...     pressure=20,
        ...     temperature=750
        ... )
    """
    # Create syngas composition
    if not (pipe_size is not None):
        raise ValueError("pipe_size must be provided")
    if not (pipe_size is not None):
        raise ValueError("pipe_size must be provided")
    syngas = {
        "H2": H2_fraction,
        "CO": CO_fraction,
        "CO2": CO2_fraction,
        "N2": N2_fraction,
        "CH4": CH4_fraction,
    }

    # Normalize
    total = sum(syngas.values())
    syngas = {k: v / total for k, v in syngas.items()}

    return calculate_pressure_drop(
        pipe_size=pipe_size,
        pipe_schedule=pipe_schedule,
        pipe_length=pipe_length,
        flow_rate=flow_rate,
        flow_unit=flow_unit,
        pressure=pressure,
        temperature=temperature,
        gas_composition=syngas,
        **kwargs,
    )


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================


def main() -> None:
    """Command line interface for pressure drop calculator."""
    logger.info("\n" + "=" * 80)
    logger.info("ADVANCED PRESSURE DROP CALCULATOR".center(80))
    logger.info("For Combustion and Gasification Gases".center(80))
    logger.info("=" * 80)

    logger.info("\n" + "-" * 80)
    logger.info('Example 1: Air in 4" Schedule 40 pipe')
    logger.info("-" * 80)

    result = calculate_pressure_drop(
        pipe_size="4",
        pipe_schedule="40",
        pipe_length=100,
        flow_rate=1000,
        flow_unit="SCFM",
        pressure=5,
        pressure_unit="bar",
        temperature=400,
        temperature_unit="K",
        fittings=[
            {"type": "90_elbow_std", "quantity": 4},
            {"type": "gate_valve_open", "quantity": 2},
        ],
    )
    print_results(result, "Example 1: Air Flow")

    logger.info("\n" + "-" * 80)
    logger.info('Example 2: Syngas in 6" Schedule 40 pipe')
    logger.info("-" * 80)

    result = calculate_pressure_drop_syngas(
        pipe_size="6",
        pipe_schedule="40",
        pipe_length=50,
        flow_rate=2000,
        flow_unit="kg/h",
        pressure=25,
        temperature=800,
        fittings=[
            {"type": "90_elbow_std", "quantity": 2},
            {"type": "tee_through_run", "quantity": 1},
        ],
    )
    print_results(result, "Example 2: Syngas Flow")


if __name__ == "__main__":
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s: %(message)s")
    main()
