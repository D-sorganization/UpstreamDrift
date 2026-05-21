"""Quick-reference helper functions for pressure drop interface."""

from __future__ import annotations

import logging
from typing import Any

from .engine.pressure_drop_calculation_engine import (
    friction_factor_churchill,
    friction_factor_colebrook,
    friction_factor_haaland,
    friction_factor_swamee_jain,
)
from .utils.fitting_loss_coefficients import FITTING_K_FACTORS
from .utils.flow_rate_converter import (
    MASS_FLOW_CONVERSIONS,
    MOLAR_FLOW_CONVERSIONS,
    STANDARD_CONDITIONS,
    VOLUMETRIC_FLOW_CONVERSIONS_TO_M3_S,
)
from .utils.gas_properties import (
    GAS_DATABASE,
)
from .utils.pipe_database import (
    MATERIAL_ROUGHNESS,
    list_available_sizes,
    list_schedules_for_size,
)

logger = logging.getLogger(__name__)


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
    if reynolds_number is None:
        raise ValueError("reynolds_number must be provided")
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


# ============================================================================
# SHARED TEXT UTILITY
# ============================================================================


def _wrap_text(text: str, width: int) -> list[str]:
    """Wrap text to specified width."""
    if text is None:
        raise ValueError("text must be provided")
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
