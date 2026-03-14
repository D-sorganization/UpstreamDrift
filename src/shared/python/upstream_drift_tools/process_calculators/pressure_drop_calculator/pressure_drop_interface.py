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

    >>> # Calculation with custom gas composition
    >>> result = calculate_pressure_drop(
    ...     pipe_diameter=0.1023,  # meters
    ...     pipe_length=50,
    ...     gas_composition={'H2': 0.3, 'CO': 0.4, 'CO2': 0.3},
    ...     flow_rate=1500,
    ...     flow_unit='SCFM',
    ...     pressure=25,  # bar
    ...     temperature=800  # K
    ... )

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

import logging
from typing import Any

from .engine.pressure_drop_calculation_engine import (
    PressureDropCalculationEngine,
    friction_factor_churchill,
    friction_factor_colebrook,
    friction_factor_haaland,
    friction_factor_swamee_jain,
)
from .models.pressure_drop_data_models import (
    GasComposition,
    PipeFitting,
    PressureDropInputs,
)
from .utils.fitting_loss_coefficients import FITTING_K_FACTORS
from .utils.flow_rate_converter import (
    MASS_FLOW_CONVERSIONS,
    MOLAR_FLOW_CONVERSIONS,
    STANDARD_CONDITIONS,
    VOLUMETRIC_FLOW_CONVERSIONS_TO_M3_S,
    convert_flow_rate_to_mass,
)
from .utils.gas_properties import (
    GAS_DATABASE,
    calculate_mixture_molecular_weight,
)
from .utils.pipe_database import (
    MATERIAL_ROUGHNESS,
    get_pipe_spec,
    get_roughness,
    list_available_sizes,
    list_schedules_for_size,
)

logger = logging.getLogger(__name__)


# ============================================================================
# QUICK REFERENCE HELPERS
# ============================================================================


def show_help() -> None:
    """Display comprehensive help with available options and examples."""
    help_text = """
╔══════════════════════════════════════════════════════════════════════════════╗
║               ADVANCED PRESSURE DROP CALCULATOR - QUICK REFERENCE            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  BASIC USAGE:                                                                ║
║  ─────────────                                                               ║
║    result = calculate_pressure_drop(                                         ║
║        pipe_size="4", pipe_schedule="40",     # Use standard pipe OR         ║
║        pipe_diameter=0.1,                      # specify diameter (m)        ║
║        pipe_length=100,                        # meters                      ║
║        flow_rate=1000, flow_unit='kg/h',      # flow with units             ║
║        pressure=10, pressure_unit='bar',       # inlet pressure             ║
║        temperature=500, temperature_unit='K',  # inlet temperature          ║
║        gas_composition={'H2': 0.3, 'CO': 0.7}, # optional (default: air)    ║
║    )                                                                         ║
║                                                                              ║
║  AVAILABLE PIPE SIZES:                                                       ║
║    1/2, 3/4, 1, 1.5, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24 inches       ║
║                                                                              ║
║  AVAILABLE SCHEDULES:                                                        ║
║    5S, 10S, 40, STD, 80, XS, 120, 160, XXS                                   ║
║                                                                              ║
║  GAS COMPONENTS:                                                             ║
║    H2, CO, CO2, CH4, C2H6, C2H4, N2, O2, H2O, Ar, H2S, NH3, Air             ║
║                                                                              ║
║  FLOW RATE UNITS:                                                            ║
║    Mass:    kg/s, kg/h, lb/hr, ton/h, g/s                                   ║
║    Molar:   mol/s, kmol/h, lbmol/hr                                         ║
║    Volume:  m³/h, SCFM, CFM, Nm³/h, L/s, L/min, ft³/min                     ║
║                                                                              ║
║  FRICTION METHODS:                                                           ║
║    'colebrook'   - Most accurate (default)                                  ║
║    'swamee-jain' - Fast, ~1% of Colebrook                                   ║
║    'churchill'   - All flow regimes                                         ║
║    'haaland'     - Simple, ~1.5% accuracy                                   ║
║                                                                              ║
║  FITTING TYPES (examples):                                                   ║
║    90_elbow_std, 90_elbow_long, 45_elbow_std                                ║
║    tee_through_branch, tee_through_run                                       ║
║    gate_valve_open, globe_valve_open, ball_valve_open                       ║
║    check_valve_swing, butterfly_valve_open                                   ║
║    entrance_sharp, exit_sharp                                                ║
║                                                                              ║
║  HELPER FUNCTIONS:                                                           ║
║    show_help()           - Display this help                                ║
║    list_gas_components() - Show available gas components                    ║
║    list_fittings()       - Show available fittings with K-factors           ║
║    list_pipe_sizes()     - Show available pipe sizes                        ║
║    list_flow_units()     - Show available flow rate units                   ║
║    list_materials()      - Show pipe materials and roughness values         ║
║    compare_friction_methods() - Compare friction factor correlations        ║
║    validate_inputs()     - Validate inputs before calculation               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    logger.info(help_text)


def list_gas_components() -> dict[str, dict[str, Any]]:
    """List all available gas components with their properties.

    Returns:
        Dictionary of gas components with MW, Tc, Pc, and acentric factor
    """
    components = {}
    logger.info(
        "\n╔═══════════════════════════════════════════════════════════════════╗"
    )
    logger.info(
        "║                    AVAILABLE GAS COMPONENTS                        ║"
    )
    logger.info("╠═════════════╦═══════════╦══════════╦═══════════╦══════════════════╣")
    logger.info("║ Component   ║  MW       ║   Tc (K) ║  Pc (bar) ║ Acentric Factor  ║")
    logger.info("╠═════════════╬═══════════╬══════════╬═══════════╬══════════════════╣")

    for name, props in sorted(GAS_DATABASE.items()):
        logger.info(
            f"║ {name:11s} ║ {props.molecular_weight:9.3f} ║ {props.critical_temp:8.1f} ║"
            f" {props.critical_pressure / 1e5:9.2f} ║ {props.acentric_factor:16.3f} ║"
        )
        components[name] = {
            "molecular_weight": props.molecular_weight,
            "critical_temp": props.critical_temp,
            "critical_pressure": props.critical_pressure,
            "acentric_factor": props.acentric_factor,
        }

    logger.info("╚═════════════╩═══════════╩══════════╩═══════════╩══════════════════╝")
    return components


def list_fittings(category: str | None = None) -> dict[str, float]:
    """List available fittings with their K-factors.

    Args:
        category: Optional filter ('elbow', 'tee', 'valve', 'entrance', 'exit', 'bend')

    Returns:
        Dictionary of fitting types and K-factors
    """
    logger.info(
        "\n╔═══════════════════════════════════════════════════════════════════╗"
    )
    logger.info(
        "║                    AVAILABLE FITTINGS (K-factors)                  ║"
    )
    logger.info("╠══════════════════════════════════════════╦═════════╦══════════════╣")
    logger.info("║ Fitting Type                             ║ K-factor║  Category    ║")
    logger.info("╠══════════════════════════════════════════╬═════════╬══════════════╣")

    result = {}
    categories = {
        "elbow": ["elbow", "miter"],
        "tee": ["tee"],
        "valve": ["valve"],
        "entrance": ["entrance"],
        "exit": ["exit"],
        "bend": ["bend"],
        "reducer": ["reducer", "expander"],
    }

    for fitting_type, k_factor in sorted(FITTING_K_FACTORS.items()):
        # Determine category
        cat = "other"
        for cat_name, keywords in categories.items():
            if any(kw in fitting_type for kw in keywords):
                cat = cat_name
                break

        # Filter by category if specified
        if category and cat != category:
            continue

        result[fitting_type] = k_factor
        # Format name for display
        name = fitting_type.replace("_", " ").title()
        logger.info(f"║ {name:40s} ║ {k_factor:7.2f} ║ {cat:12s} ║")

    logger.info("╚══════════════════════════════════════════╩═════════╩══════════════╝")
    logger.info(
        "\nNote: K-factors are for fully turbulent flow in standard pipe sizes."
    )
    logger.info("      Use Two-K method for more accuracy in small pipes/low Re flows.")
    return result


def list_pipe_sizes() -> dict[str, list[str]]:
    """List available standard pipe sizes and schedules.

    Returns:
        Dictionary mapping pipe sizes to available schedules
    """
    sizes = list_available_sizes()
    result = {}

    logger.info(
        "\n╔═══════════════════════════════════════════════════════════════════╗"
    )
    logger.info(
        "║                    AVAILABLE PIPE SIZES (ASME B36.10M)             ║"
    )
    logger.info("╠═══════════════════════════════════════════════════════════════════╣")

    for size in sizes:
        schedules = list_schedules_for_size(size)
        result[size] = schedules
        sch_str = ", ".join(schedules)
        logger.info(f"║ NPS {size:5s} : {sch_str:56s}║")

    logger.info("╚═══════════════════════════════════════════════════════════════════╝")
    return result


def list_flow_units() -> dict[str, list[str]]:
    """List all available flow rate units.

    Returns:
        Dictionary of unit categories and available units
    """
    logger.info(
        "\n╔═══════════════════════════════════════════════════════════════════╗"
    )
    logger.info(
        "║                    AVAILABLE FLOW RATE UNITS                       ║"
    )
    logger.info("╠═══════════════════════════════════════════════════════════════════╣")

    logger.info(
        "║ MASS FLOW UNITS:                                                   ║"
    )
    mass_units = list(MASS_FLOW_CONVERSIONS.keys())
    logger.info(f"║   {', '.join(mass_units):63s}║")

    logger.info(
        "║                                                                    ║"
    )
    logger.info(
        "║ MOLAR FLOW UNITS:                                                  ║"
    )
    molar_units = list(MOLAR_FLOW_CONVERSIONS.keys())
    logger.info(f"║   {', '.join(molar_units):63s}║")

    logger.info(
        "║                                                                    ║"
    )
    logger.info(
        "║ VOLUMETRIC FLOW UNITS:                                             ║"
    )
    vol_units = list(VOLUMETRIC_FLOW_CONVERSIONS_TO_M3_S.keys())
    # Split into multiple lines if needed
    vol_str = ", ".join(vol_units)
    while len(vol_str) > 63:
        idx = vol_str[:63].rfind(",")
        logger.info(f"║   {vol_str[: idx + 1]:63s}║")
        vol_str = vol_str[idx + 2 :]
    logger.info(f"║   {vol_str:63s}║")

    logger.info(
        "║                                                                    ║"
    )
    logger.info(
        "║ STANDARD CONDITIONS FOR VOLUMETRIC FLOWS:                          ║"
    )
    for name, (T, P, desc) in STANDARD_CONDITIONS.items():
        logger.info(f"║   {name:6s}: T={T:.2f}K, P={P:.0f}Pa - {desc:34s}║")

    logger.info("╚═══════════════════════════════════════════════════════════════════╝")

    return {
        "mass": mass_units,
        "molar": molar_units,
        "volumetric": vol_units,
        "standard_conditions": list(STANDARD_CONDITIONS.keys()),
    }


def list_materials() -> dict[str, dict[str, float]]:
    """List available pipe materials with roughness values.

    Returns:
        Dictionary of materials with roughness values
    """
    logger.info(
        "\n╔═══════════════════════════════════════════════════════════════════╗"
    )
    logger.info(
        "║                    PIPE MATERIAL ROUGHNESS VALUES                  ║"
    )
    logger.info(
        "╠═══════════════════════════════════╦═════════════╦══════════════════╣"
    )
    logger.info(
        "║ Material                          ║  ε (mm)     ║  ε (m)           ║"
    )
    logger.info(
        "╠═══════════════════════════════════╬═════════════╬══════════════════╣"
    )

    result = {}
    for material, (roughness_mm, _roughness_ft, _desc) in sorted(
        MATERIAL_ROUGHNESS.items()
    ):
        result[material] = {
            "roughness_mm": roughness_mm,
            "roughness_m": roughness_mm / 1000,
        }
        logger.info(
            f"║ {material:33s} ║ {roughness_mm:11.4f} ║ {roughness_mm / 1000:16.6f} ║"
        )

    logger.info(
        "╚═══════════════════════════════════╩═════════════╩══════════════════╝"
    )
    return result


def compare_friction_methods(
    reynolds_number: float,
    relative_roughness: float = 0.0001,
) -> dict[str, float]:
    """Compare friction factor correlations for given conditions.

    Args:
        reynolds_number: Reynolds number
        relative_roughness: ε/D ratio (default 0.0001)

    Returns:
        Dictionary of friction factors from each method

    Example:
        >>> compare_friction_methods(100000, 0.001)
    """
    assert reynolds_number is not None, "reynolds_number must be provided"
    assert reynolds_number is not None, "reynolds_number must be provided"
    logger.info(
        "\n╔═══════════════════════════════════════════════════════════════════╗"
    )
    logger.info(
        "║                 FRICTION FACTOR METHOD COMPARISON                  ║"
    )
    logger.info(
        f"║  Re = {reynolds_number:.0f}, ε/D = {relative_roughness:.6f}".ljust(68) + "║"
    )
    logger.info(
        "╠═══════════════════════════════════╦═══════════╦═════════════════════╣"
    )
    logger.info(
        "║ Method                            ║ f         ║ Δ from Colebrook    ║"
    )
    logger.info(
        "╠═══════════════════════════════════╬═══════════╬═════════════════════╣"
    )

    results = {}

    # Colebrook (reference)
    f_colebrook = friction_factor_colebrook(reynolds_number, relative_roughness)
    results["colebrook"] = f_colebrook
    logger.info(
        f"║ Colebrook-White (iterative)       ║ {f_colebrook:.6f}  ║ (reference)         ║"
    )

    # Swamee-Jain
    f_swamee = friction_factor_swamee_jain(reynolds_number, relative_roughness)
    results["swamee-jain"] = f_swamee
    diff = (f_swamee / f_colebrook - 1) * 100
    logger.info(
        f"║ Swamee-Jain (explicit)            ║ {f_swamee:.6f}  ║ {diff:+.2f}%              ║"
    )

    # Churchill
    f_churchill = friction_factor_churchill(reynolds_number, relative_roughness)
    results["churchill"] = f_churchill
    diff = (f_churchill / f_colebrook - 1) * 100
    logger.info(
        f"║ Churchill (all regimes)           ║ {f_churchill:.6f}  ║ {diff:+.2f}%              ║"
    )

    # Haaland
    f_haaland = friction_factor_haaland(reynolds_number, relative_roughness)
    results["haaland"] = f_haaland
    diff = (f_haaland / f_colebrook - 1) * 100
    logger.info(
        f"║ Haaland (simplified)              ║ {f_haaland:.6f}  ║ {diff:+.2f}%              ║"
    )

    logger.info(
        "╚═══════════════════════════════════╩═══════════╩═════════════════════╝"
    )

    # Flow regime classification
    if reynolds_number < 2300:
        regime = "Laminar"
    elif reynolds_number < 4000:
        regime = "Transitional"
    else:
        regime = "Turbulent"
    logger.info(f"\nFlow regime: {regime}")

    if reynolds_number < 4000:
        logger.info("Note: For transitional flow, Churchill method is recommended.")

    return results


def _validate_pipe_params(
    pipe_size: str | None,
    pipe_schedule: str | None,
    pipe_diameter: float | None,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate pipe geometry parameters."""
    if pipe_diameter is None:
        if pipe_size is None or pipe_schedule is None:
            errors.append(
                "Must provide either pipe_diameter OR both pipe_size and pipe_schedule"
            )
        else:
            try:
                get_pipe_spec(pipe_size, pipe_schedule)
            except ValueError as e:
                available = list_available_sizes()
                errors.append(f"{e}. Available sizes: {', '.join(available)}")
    elif pipe_diameter <= 0:
        errors.append(f"pipe_diameter must be positive, got {pipe_diameter}")
    elif pipe_diameter > 2:
        warnings.append(
            f"Large diameter ({pipe_diameter}m). Did you mean mm? Use meters."
        )


def _validate_flow_params(
    flow_rate: float | None,
    flow_unit: str | None,
    errors: list[str],
) -> None:
    """Validate flow rate value and unit."""
    assert errors is not None, "errors must be provided"
    assert errors is not None, "errors must be provided"
    if flow_rate is not None:
        if flow_rate <= 0:
            errors.append(f"flow_rate must be positive, got {flow_rate}")
    else:
        errors.append("flow_rate is required")

    if flow_unit:
        all_units = (
            list(MASS_FLOW_CONVERSIONS.keys())
            + list(MOLAR_FLOW_CONVERSIONS.keys())
            + list(VOLUMETRIC_FLOW_CONVERSIONS_TO_M3_S.keys())
            + ["SCFM", "ACFM", "Nm3/h", "Nm³/h"]
        )
        if flow_unit not in all_units and flow_unit.upper() not in ["SCFM", "ACFM"]:
            similar = [u for u in all_units if flow_unit.lower() in u.lower()]
            errors.append(
                f"Unknown flow_unit '{flow_unit}'. "
                f"Did you mean: {', '.join(similar[:5]) if similar else 'see list_flow_units()'}"
            )


def _validate_conditions(
    pressure: float | None,
    temperature: float | None,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate pressure and temperature values."""
    assert errors is not None, "errors must be provided"
    assert errors is not None, "errors must be provided"
    if pressure is not None:
        if pressure <= 0:
            errors.append(f"pressure must be positive, got {pressure}")
        elif pressure > 1000:
            warnings.append(
                f"High pressure ({pressure}). Ensure units are correct (bar/psi/Pa)."
            )

    if temperature is not None:
        if temperature <= 0:
            errors.append(f"temperature must be positive (Kelvin), got {temperature}")
        elif temperature < 200:
            warnings.append(
                f"Low temperature ({temperature}K). Did you mean Celsius? Use temperature_unit='C'"
            )
        elif temperature > 2000:
            warnings.append(
                f"Very high temperature ({temperature}K). Verify this is correct."
            )


def _validate_composition_and_fittings(
    gas_composition: dict[str, float] | None,
    fittings: list[dict[str, Any]] | None,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate gas composition and fitting specifications."""
    assert errors is not None, "errors must be provided"
    assert errors is not None, "errors must be provided"
    if gas_composition:
        total = sum(gas_composition.values())
        if not (0.99 <= total <= 1.01):
            warnings.append(
                f"Gas composition sums to {total:.4f}, expected ~1.0. Will be auto-normalized."
            )

        unknown = [c for c in gas_composition if c not in GAS_DATABASE]
        if unknown:
            errors.append(
                f"Unknown gas components: {', '.join(unknown)}. "
                f"Available: {', '.join(GAS_DATABASE.keys())}"
            )

    if fittings:
        for i, fitting in enumerate(fittings):
            fitting_type = fitting.get("type", "")
            if fitting_type and fitting_type not in FITTING_K_FACTORS:
                similar = [f for f in FITTING_K_FACTORS if fitting_type in f]
                warnings.append(
                    f"Fitting[{i}] type '{fitting_type}' not in database. "
                    f"Similar: {', '.join(similar[:3]) if similar else 'see list_fittings()'}"
                )


def _log_validation_report(
    is_valid: bool, errors: list[str], warnings: list[str]
) -> None:
    """Log a formatted validation report."""
    assert is_valid is not None, "is_valid must be provided"
    assert is_valid is not None, "is_valid must be provided"
    logger.info(
        "\n╔═══════════════════════════════════════════════════════════════════╗"
    )
    logger.info(
        "║                        INPUT VALIDATION                            ║"
    )
    logger.info("╠═══════════════════════════════════════════════════════════════════╣")

    if errors:
        logger.error(
            "║ ERRORS (must fix):                                                ║"
        )
        for error in errors:
            for line in _wrap_text(error, 64):
                logger.info(f"║   ❌ {line:62s}║")

    if warnings:
        logger.warning(
            "║ WARNINGS (review):                                                ║"
        )
        for warning in warnings:
            for line in _wrap_text(warning, 64):
                logger.info(f"║   ⚠️  {line:61s}║")

    if is_valid:
        logger.info(
            "║   ✅ All inputs valid - ready to calculate                       ║"
        )

    logger.info("╚═══════════════════════════════════════════════════════════════════╝")


def validate_inputs(
    pipe_size: str | None = None,
    pipe_schedule: str | None = None,
    pipe_diameter: float | None = None,
    flow_rate: float | None = None,
    flow_unit: str | None = None,
    pressure: float | None = None,
    temperature: float | None = None,
    gas_composition: dict[str, float] | None = None,
    fittings: list[dict[str, Any]] | None = None,
) -> tuple[bool, list[str], list[str]]:
    """Validate inputs before calculation and provide helpful suggestions.

    Args:
        pipe_size: Nominal pipe size
        pipe_schedule: Pipe schedule
        pipe_diameter: Pipe diameter in meters
        flow_rate: Flow rate value
        flow_unit: Flow rate unit
        pressure: Inlet pressure
        temperature: Inlet temperature
        gas_composition: Gas composition dictionary
        fittings: List of fittings

    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    errors: list[str] = []
    warnings: list[str] = []

    _validate_pipe_params(pipe_size, pipe_schedule, pipe_diameter, errors, warnings)
    _validate_flow_params(flow_rate, flow_unit, errors)
    _validate_conditions(pressure, temperature, errors, warnings)
    _validate_composition_and_fittings(gas_composition, fittings, errors, warnings)

    is_valid = len(errors) == 0
    _log_validation_report(is_valid, errors, warnings)

    return is_valid, errors, warnings


def _wrap_text(text: str, width: int) -> list[str]:
    """Wrap text to specified width."""
    assert text is not None, "text must be provided"
    assert text is not None, "text must be provided"
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        if len(current_line) + len(word) + 1 <= width:
            current_line += (" " if current_line else "") + word
        else:
            if current_line:
                lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    return lines or [""]


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
    assert pipe_material is not None, "pipe_material must be provided"
    assert pipe_material is not None, "pipe_material must be provided"
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
    assert flow_rate is not None, "flow_rate must be provided"
    assert flow_rate is not None, "flow_rate must be provided"
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
    assert pipe_length is not None, "pipe_length must be provided"
    assert pipe_length is not None, "pipe_length must be provided"
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
    assert pipe_size is not None, "pipe_size must be provided"
    assert pipe_size is not None, "pipe_size must be provided"
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
# UTILITY FUNCTIONS
# ============================================================================


def _convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    """Convert temperature between units."""
    from_unit = from_unit.upper()
    to_unit = to_unit.upper()

    # Convert to Kelvin first
    if from_unit == "K":
        temp_k = value
    elif from_unit == "C":
        temp_k = value + 273.15
    elif from_unit == "F":
        temp_k = (value - 32) * 5 / 9 + 273.15
    else:
        raise ValueError(f"Unknown temperature unit: {from_unit}")

    # Convert from Kelvin to target
    if to_unit == "K":
        return temp_k
    if to_unit == "C":
        return temp_k - 273.15
    if to_unit == "F":
        return (temp_k - 273.15) * 9 / 5 + 32
    raise ValueError(f"Unknown temperature unit: {to_unit}")


def _convert_pressure(value: float, from_unit: str, to_unit: str) -> float:
    """Convert pressure between units."""
    # Conversion factors to Pa
    to_pa = {
        "Pa": 1.0,
        "kPa": 1000.0,
        "MPa": 1e6,
        "bar": 1e5,
        "mbar": 100.0,
        "atm": 101325.0,
        "psi": 6894.76,
        "psia": 6894.76,
        "psig": 6894.76,  # Note: gauge pressure, user should handle
    }

    if from_unit not in to_pa:
        raise ValueError(f"Unknown pressure unit: {from_unit}")
    if to_unit not in to_pa:
        raise ValueError(f"Unknown pressure unit: {to_unit}")

    pa = value * to_pa[from_unit]
    return pa / to_pa[to_unit]


def _format_results(results: Any) -> dict[str, Any]:
    """Format results into a comprehensive dictionary."""
    return {
        # Pressure drops
        "pressure_drop_pa": results.total_pressure_drop,
        "pressure_drop_bar": results.total_pressure_drop / 1e5,
        "pressure_drop_psi": results.total_pressure_drop / 6894.76,
        "pressure_drop_kpa": results.total_pressure_drop / 1000.0,
        # Pressure drop components
        "friction_loss_pa": results.friction_pressure_drop,
        "friction_loss_bar": results.friction_pressure_drop / 1e5,
        "fitting_loss_pa": results.fitting_pressure_drop,
        "fitting_loss_bar": results.fitting_pressure_drop / 1e5,
        "elevation_loss_pa": results.elevation_pressure_drop,
        # Outlet pressure
        "outlet_pressure_pa": results.outlet_pressure,
        "outlet_pressure_bar": results.outlet_pressure / 1e5,
        "outlet_pressure_psi": results.outlet_pressure / 6894.76,
        # Flow characteristics
        "friction_factor": results.friction_factor,
        "reynolds_number": results.flow_properties.reynolds_number,
        "flow_velocity_m_s": results.flow_properties.velocity,
        "flow_velocity_ft_s": results.flow_properties.velocity * 3.28084,
        "mach_number": results.flow_properties.mach_number,
        "flow_regime": results.flow_regime,
        # Gas properties
        "density_kg_m3": results.flow_properties.density,
        "viscosity_pa_s": results.flow_properties.viscosity,
        "compressibility_factor": results.flow_properties.compressibility_factor,
        "molecular_weight": results.flow_properties.molecular_weight,
        # Performance metrics
        "erosional_velocity_m_s": results.erosional_velocity,
        "erosion_ratio": results.erosion_ratio,
        "erosion_ratio_percent": results.erosion_ratio * 100,
        # Additional
        "pressure_drop_per_100ft_pa": results.pressure_drop_per_100ft,
        "velocity_pressure_pa": results.velocity_pressure,
        # Warnings
        "warnings": results.warnings,
    }


def _print_summary_section(results: dict[str, Any]) -> None:
    """Log the pressure-drop summary section."""
    logger.info("\n┌" + "─" * 78 + "┐")
    logger.info("│" + " SUMMARY ".center(78) + "│")
    logger.info("├" + "─" * 78 + "┤")
    logger.info(
        f"│  Total Pressure Drop:  {results['pressure_drop_bar']:10.4f} bar  "
        f"│  {results['pressure_drop_psi']:10.4f} psi  │  {results['pressure_drop_kpa']:10.2f} kPa  │"
    )
    logger.info(
        f"│  Outlet Pressure:      {results['outlet_pressure_bar']:10.4f} bar  "
        f"│  {results['outlet_pressure_psi']:10.4f} psi  │                 │"
    )
    logger.info("└" + "─" * 78 + "┘")


def _print_breakdown_section(results: dict[str, Any]) -> None:
    """Log the pressure-drop breakdown by component."""

    def safe_percent(num: float, denom: float) -> float:
        return (num / denom * 100) if denom != 0 else 0.0

    logger.info("\n┌" + "─" * 78 + "┐")
    logger.info("│" + " PRESSURE DROP BREAKDOWN ".center(78) + "│")
    logger.info("├" + "─" * 38 + "┬" + "─" * 19 + "┬" + "─" * 19 + "┤")
    logger.info(
        "│  Component                           │     Value (bar)   │    Percentage   │"
    )
    logger.info("├" + "─" * 38 + "┼" + "─" * 19 + "┼" + "─" * 19 + "┤")

    dp_total = results["pressure_drop_pa"]
    friction_pct = safe_percent(results["friction_loss_pa"], dp_total)
    fitting_pct = safe_percent(results["fitting_loss_pa"], dp_total)
    elevation_pct = safe_percent(results["elevation_loss_pa"], dp_total)

    logger.info(
        f"│  Friction (pipe wall)                │ {results['friction_loss_bar']:17.6f} │ {friction_pct:15.1f}% │"
    )
    logger.info(
        f"│  Fittings & valves                   │ {results['fitting_loss_bar']:17.6f} │ {fitting_pct:15.1f}% │"
    )
    if abs(results["elevation_loss_pa"]) > 0.1:
        logger.info(
            f"│  Elevation change                    │ {results['elevation_loss_pa'] / 1e5:17.6f} │ {elevation_pct:15.1f}% │"
        )
    logger.info("└" + "─" * 38 + "┴" + "─" * 19 + "┴" + "─" * 19 + "┘")


def _print_flow_and_gas_sections(results: dict[str, Any]) -> None:
    """Log flow characteristics and gas property sections."""
    logger.info("\n┌" + "─" * 78 + "┐")
    logger.info("│" + " FLOW CHARACTERISTICS ".center(78) + "│")
    logger.info("├" + "─" * 38 + "┬" + "─" * 39 + "┤")
    logger.info(
        f"│  Flow Velocity:     {results['flow_velocity_m_s']:10.2f} m/s   │  {results['flow_velocity_ft_s']:10.2f} ft/s                  │"
    )
    logger.info(
        f"│  Reynolds Number:   {results['reynolds_number']:10.0f}        │  Flow Regime: {results['flow_regime']:18s}   │"
    )
    logger.info(
        f"│  Mach Number:       {results['mach_number']:10.4f}        │  Friction Factor: {results['friction_factor']:14.6f}   │"
    )
    logger.info("└" + "─" * 38 + "┴" + "─" * 39 + "┘")

    logger.info("\n┌" + "─" * 78 + "┐")
    logger.info("│" + " GAS PROPERTIES ".center(78) + "│")
    logger.info("├" + "─" * 38 + "┬" + "─" * 39 + "┤")
    logger.info(
        f"│  Density:           {results['density_kg_m3']:10.4f} kg/m³  │  Molecular Weight: {results['molecular_weight']:12.2f} kg/kmol│"
    )
    logger.info(
        f"│  Viscosity:         {results['viscosity_pa_s'] * 1e6:10.4f} µPa·s  │  Compressibility (Z): {results['compressibility_factor']:10.4f}     │"
    )
    logger.info("└" + "─" * 38 + "┴" + "─" * 39 + "┘")


def _print_safety_section(results: dict[str, Any]) -> None:
    """Log the safety metrics section."""
    logger.info("\n┌" + "─" * 78 + "┐")
    logger.info("│" + " SAFETY METRICS ".center(78) + "│")
    logger.info("├" + "─" * 38 + "┬" + "─" * 39 + "┤")

    erosion_ratio = results["erosion_ratio_percent"]
    if erosion_ratio < 50:
        erosion_status = "✅ SAFE"
    elif erosion_ratio < 80:
        erosion_status = "⚠️  CAUTION"
    else:
        erosion_status = "❌ DANGER"

    logger.info(
        f"│  Erosional Velocity: {results['erosional_velocity_m_s']:9.2f} m/s   │  Status: {erosion_status:26s}  │"
    )
    logger.info(
        f"│  Erosion Ratio:      {erosion_ratio:9.1f} %     │  (Velocity/Erosional limit)         │"
    )
    logger.info("└" + "─" * 38 + "┴" + "─" * 39 + "┘")


def _print_warnings_and_recommendations(
    results: dict[str, Any], show_recommendations: bool
) -> None:
    """Log warnings and engineering recommendations."""
    assert results is not None, "results must be provided"
    assert results is not None, "results must be provided"
    if results.get("warnings"):
        warnings = results["warnings"]
        if isinstance(warnings, list) and len(warnings) > 0:
            logger.info("\n┌" + "─" * 78 + "┐")
            logger.warning("│" + " ⚠️  WARNINGS ".center(78) + "│")
            logger.info("├" + "─" * 78 + "┤")
            for warning in warnings:
                wrapped = _wrap_text(warning, 74)
                for line in wrapped:
                    logger.info(f"│  {line:74s}  │")
            logger.info("└" + "─" * 78 + "┘")

    if show_recommendations:
        recommendations = _generate_recommendations(results)
        if recommendations:
            logger.info("\n┌" + "─" * 78 + "┐")
            logger.info("│" + " 💡 RECOMMENDATIONS ".center(78) + "│")
            logger.info("├" + "─" * 78 + "┤")
            for rec in recommendations:
                wrapped = _wrap_text(rec, 74)
                for line in wrapped:
                    logger.info(f"│  {line:74s}  │")
            logger.info("└" + "─" * 78 + "┘")


def print_results(
    results: dict[str, Any],
    title: str = "PRESSURE DROP CALCULATION RESULTS",
    show_recommendations: bool = True,
) -> None:
    """Print results in a beautifully formatted table with recommendations.

    Args:
        results: Results dictionary from calculate_pressure_drop
        title: Title for the output
        show_recommendations: Whether to show engineering recommendations
    """
    assert results is not None, "results must be provided"
    assert results is not None, "results must be provided"
    logger.info("\n" + "═" * 80)
    logger.info(f"  {title}  ".center(80, "═"))
    logger.info("═" * 80)

    _print_summary_section(results)
    _print_breakdown_section(results)
    _print_flow_and_gas_sections(results)
    _print_safety_section(results)
    _print_warnings_and_recommendations(results, show_recommendations)

    logger.info("═" * 80 + "\n")


def _generate_recommendations(results: dict[str, Any]) -> list[str]:
    """Generate engineering recommendations based on calculation results."""
    recommendations = []

    # High pressure drop
    dp_ratio = results["pressure_drop_pa"] / (
        results["outlet_pressure_pa"] + results["pressure_drop_pa"]
    )
    if dp_ratio > 0.20:
        recommendations.append(
            f"High pressure drop ({dp_ratio * 100:.0f}% of inlet). Consider: larger pipe diameter, "
            "shorter pipe run, or fewer fittings."
        )

    # Erosion concerns
    erosion_ratio = results["erosion_ratio"]
    if erosion_ratio > 0.8:
        recommendations.append(
            "Velocity exceeds 80% of erosional limit. Consider larger pipe diameter to "
            "reduce velocity and extend pipe life."
        )
    elif erosion_ratio > 0.5:
        recommendations.append(
            "Velocity is 50-80% of erosional limit. Monitor pipe condition and consider "
            "velocity reduction for longer service life."
        )

    # Fitting losses
    if results["fitting_loss_pa"] > results["friction_loss_pa"]:
        recommendations.append(
            "Fitting losses exceed pipe friction. Consider using long-radius elbows, "
            "full-port valves, or reducing number of fittings."
        )

    # High Mach number
    if results["mach_number"] > 0.3:
        recommendations.append(
            f"High Mach number ({results['mach_number']:.3f}). Compressibility effects significant. "
            "Verify calculations and consider acoustic vibration analysis."
        )

    # Low Reynolds number
    if results["reynolds_number"] < 4000:
        recommendations.append(
            f"Low Reynolds number ({results['reynolds_number']:.0f}). Flow may be transitional "
            "or laminar - friction factor has higher uncertainty in this regime."
        )

    # Very high Reynolds number
    if results["reynolds_number"] > 1e7:
        recommendations.append(
            f"Very high Reynolds number ({results['reynolds_number']:.0e}). Ensure turbulent flow "
            "correlations are valid. Consider CFD analysis for critical applications."
        )

    return recommendations


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================


def main() -> None:
    """Command line interface for pressure drop calculator."""
    logger.info("\n" + "=" * 80)
    logger.info("ADVANCED PRESSURE DROP CALCULATOR".center(80))
    logger.info("For Combustion and Gasification Gases".center(80))
    logger.info("=" * 80)

    # Example 1: Standard pipe with air
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

    # Example 2: Syngas
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
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    main()
