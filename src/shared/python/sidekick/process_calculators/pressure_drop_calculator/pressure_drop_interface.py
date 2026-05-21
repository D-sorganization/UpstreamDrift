# ruff: noqa: E501
#!/usr/bin/env python3
"""Thin public facade for advanced pressure drop calculations.

Refactored from a single 1407-line file into focused submodules (issue #1952):

    pressure_drop_units.py      — _convert_temperature, _convert_pressure
    pressure_drop_validation.py — validate_inputs, _validate_* helpers
    pressure_drop_results.py    — _format_results, print_results, _print_*,
                                   _generate_recommendations, _wrap_text

All public symbols remain importable from this module.

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

AVAILABLE UNITS:
    - Temperature: K, C, F
    - Pressure: Pa, kPa, bar, psi, atm
    - Mass flow: kg/s, kg/h, lb/hr, ton/h
    - Molar flow: mol/s, kmol/h, lbmol/hr
    - Volumetric flow: m3/h, SCFM, ACFM, Nm3/h, CFM, L/s

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
from typing import Any, cast

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

# Re-export submodule symbols so callers that import from here continue to work
from .pressure_drop_results import (  # noqa: F401
    _format_results,
    _generate_recommendations,
    _print_breakdown_section,
    _print_flow_and_gas_sections,
    _print_safety_section,
    _print_summary_section,
    _print_warnings_and_recommendations,
    _wrap_text,
    print_results,
)
from .pressure_drop_units import (  # noqa: F401
    _convert_pressure,
    _convert_temperature,
)
from .pressure_drop_validation import (  # noqa: F401
    _log_validation_report,
    _validate_composition_and_fittings,
    _validate_conditions,
    _validate_flow_params,
    _validate_pipe_params,
    validate_inputs,
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
\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557
\u2551               ADVANCED PRESSURE DROP CALCULATOR - QUICK REFERENCE            \u2551
\u2560\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2563
\u2551                                                                              \u2551
\u2551  BASIC USAGE:                                                                \u2551
\u2551  \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500                                                               \u2551
\u2551    result = calculate_pressure_drop(                                         \u2551
\u2551        pipe_size="4", pipe_schedule="40",     # Use standard pipe OR         \u2551
\u2551        pipe_diameter=0.1,                      # specify diameter (m)        \u2551
\u2551        pipe_length=100,                        # meters                      \u2551
\u2551        flow_rate=1000, flow_unit='kg/h',      # flow with units             \u2551
\u2551        pressure=10, pressure_unit='bar',       # inlet pressure             \u2551
\u2551        temperature=500, temperature_unit='K',  # inlet temperature          \u2551
\u2551        gas_composition={'H2': 0.3, 'CO': 0.7}, # optional (default: air)    \u2551
\u2551    )                                                                         \u2551
\u2551                                                                              \u2551
\u2551  AVAILABLE PIPE SIZES:                                                       \u2551
\u2551    1/2, 3/4, 1, 1.5, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24 inches       \u2551
\u2551                                                                              \u2551
\u2551  FRICTION METHODS:                                                           \u2551
\u2551    'colebrook'   - Most accurate (default)                                  \u2551
\u2551    'swamee-jain' - Fast, ~1% of Colebrook                                   \u2551
\u2551    'churchill'   - All flow regimes                                         \u2551
\u2551    'haaland'     - Simple, ~1.5% accuracy                                   \u2551
\u2551                                                                              \u2551
\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d
"""
    logger.info(help_text)


def list_gas_components() -> dict[str, dict[str, Any]]:
    """List all available gas components with their properties.

    Returns:
        Dictionary of gas components with MW, Tc, Pc, and acentric factor
    """
    components = {}
    logger.info(
        "\n\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557"
    )
    logger.info(
        "\u2551                    AVAILABLE GAS COMPONENTS                        \u2551"
    )
    logger.info(
        "\u2560\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2566\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2566\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2566\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2566\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2563"
    )
    logger.info(
        "\u2551 Component   \u2551  MW       \u2551   Tc (K) \u2551  Pc (bar) \u2551 Acentric Factor  \u2551"
    )
    logger.info(
        "\u2560\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u256c\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u256c\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u256c\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u256c\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2563"
    )

    for name, props in sorted(GAS_DATABASE.items()):
        logger.info(
            f"\u2551 {name:11s} \u2551 {props.molecular_weight:9.3f} \u2551 {props.critical_temp:8.1f} \u2551"
            f" {props.critical_pressure / 1e5:9.2f} \u2551 {props.acentric_factor:16.3f} \u2551"
        )
        components[name] = {
            "molecular_weight": props.molecular_weight,
            "critical_temp": props.critical_temp,
            "critical_pressure": props.critical_pressure,
            "acentric_factor": props.acentric_factor,
        }

    logger.info(
        "\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2569\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2569\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2569\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2569\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d"
    )
    return components


def list_fittings(category: str | None = None) -> dict[str, float]:
    """List available fittings with their K-factors.

    Args:
        category: Optional filter ('elbow', 'tee', 'valve', 'entrance', 'exit', 'bend')

    Returns:
        Dictionary of fitting types and K-factors
    """
    logger.info(
        "\n\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557"
    )
    logger.info(
        "\u2551                    AVAILABLE FITTINGS (K-factors)                  \u2551"
    )
    logger.info(
        "\u2560\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2566\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2566\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2563"
    )
    logger.info(
        "\u2551 Fitting Type                             \u2551 K-factor\u2551  Category    \u2551"
    )
    logger.info(
        "\u2560\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u256c\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u256c\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2563"
    )

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
        cat = "other"
        for cat_name, keywords in categories.items():
            if any(kw in fitting_type for kw in keywords):
                cat = cat_name
                break

        if category and cat != category:
            continue

        result[fitting_type] = k_factor
        name = fitting_type.replace("_", " ").title()
        logger.info(f"\u2551 {name:40s} \u2551 {k_factor:7.2f} \u2551 {cat:12s} \u2551")

    logger.info(
        "\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2569\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2569\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d"
    )
    return result


def list_pipe_sizes() -> dict[str, list[str]]:
    """List available standard pipe sizes and schedules.

    Returns:
        Dictionary mapping pipe sizes to available schedules
    """
    sizes = list_available_sizes()
    result = {}

    logger.info(
        "\n\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557"
    )
    logger.info(
        "\u2551                    AVAILABLE PIPE SIZES (ASME B36.10M)             \u2551"
    )
    logger.info(
        "\u2560\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2563"
    )

    for size in sizes:
        schedules = list_schedules_for_size(size)
        result[size] = schedules
        sch_str = ", ".join(schedules)
        logger.info(f"\u2551 NPS {size:5s} : {sch_str:56s}\u2551")

    logger.info(
        "\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d"
    )
    return result


def list_flow_units() -> dict[str, list[str]]:
    """List all available flow rate units.

    Returns:
        Dictionary of unit categories and available units
    """
    logger.info(
        "\n\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557"
    )
    logger.info(
        "\u2551                    AVAILABLE FLOW RATE UNITS                       \u2551"
    )
    logger.info(
        "\u2560\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2563"
    )
    logger.info(
        "\u2551 MASS FLOW UNITS:                                                   \u2551"
    )
    mass_units = list(MASS_FLOW_CONVERSIONS.keys())
    logger.info(f"\u2551   {', '.join(mass_units):63s}\u2551")

    logger.info(
        "\u2551                                                                    \u2551"
    )
    logger.info(
        "\u2551 MOLAR FLOW UNITS:                                                  \u2551"
    )
    molar_units = list(MOLAR_FLOW_CONVERSIONS.keys())
    logger.info(f"\u2551   {', '.join(molar_units):63s}\u2551")

    logger.info(
        "\u2551                                                                    \u2551"
    )
    logger.info(
        "\u2551 VOLUMETRIC FLOW UNITS:                                             \u2551"
    )
    vol_units = list(VOLUMETRIC_FLOW_CONVERSIONS_TO_M3_S.keys())
    vol_str = ", ".join(vol_units)
    while len(vol_str) > 63:
        idx = vol_str[:63].rfind(",")
        logger.info(f"\u2551   {vol_str[: idx + 1]:63s}\u2551")
        vol_str = vol_str[idx + 2 :]
    logger.info(f"\u2551   {vol_str:63s}\u2551")

    logger.info(
        "\u2551                                                                    \u2551"
    )
    logger.info(
        "\u2551 STANDARD CONDITIONS FOR VOLUMETRIC FLOWS:                          \u2551"
    )
    for name, (T, P, desc) in STANDARD_CONDITIONS.items():
        logger.info(f"\u2551   {name:6s}: T={T:.2f}K, P={P:.0f}Pa - {desc:34s}\u2551")

    logger.info(
        "\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d"
    )

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
        "\n\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557"
    )
    logger.info(
        "\u2551                    PIPE MATERIAL ROUGHNESS VALUES                  \u2551"
    )
    logger.info(
        "\u2560\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2566\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2566\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2563"
    )
    logger.info(
        "\u2551 Material                          \u2551  \u03b5 (mm)     \u2551  \u03b5 (m)           \u2551"
    )
    logger.info(
        "\u2560\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u256c\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u256c\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2563"
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
            f"\u2551 {material:33s} \u2551 {roughness_mm:11.4f} \u2551 {roughness_mm / 1000:16.6f} \u2551"
        )

    logger.info(
        "\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2569\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2569\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d"
    )
    return result


def compare_friction_methods(
    reynolds_number: float,
    relative_roughness: float = 0.0001,
) -> dict[str, float]:
    """Compare friction factor correlations for given conditions.

    Args:
        reynolds_number: Reynolds number
        relative_roughness: epsilon/D ratio (default 0.0001)

    Returns:
        Dictionary of friction factors from each method

    Example:
        >>> compare_friction_methods(100000, 0.001)
    """
    if reynolds_number is None:
        raise ValueError("reynolds_number must be provided")

    f_colebrook = friction_factor_colebrook(reynolds_number, relative_roughness)
    f_swamee = friction_factor_swamee_jain(reynolds_number, relative_roughness)
    f_churchill = friction_factor_churchill(reynolds_number, relative_roughness)
    f_haaland = friction_factor_haaland(reynolds_number, relative_roughness)

    results = {
        "colebrook": f_colebrook,
        "swamee-jain": f_swamee,
        "churchill": f_churchill,
        "haaland": f_haaland,
    }

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
    if gas_composition is None:
        gas_composition = {"Air": 1.0}
        logger.info("Using default gas composition: Air")

    composition = GasComposition(components=gas_composition)
    composition.normalize()
    molecular_weight = calculate_mixture_molecular_weight(composition.components)

    if flow_unit.upper() in ["ACFM", "CFM"]:
        from .utils.flow_rate_converter import volumetric_actual_to_mass
        from .utils.gas_properties import calculate_gas_properties

        props = calculate_gas_properties(
            composition.components, temp_k, pressure_pa, compressibility_correction
        )
        density = props["density"]
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


def _build_pressure_drop_inputs(
    pipe_size: str | None,
    pipe_schedule: str | None,
    pipe_diameter: float | None,
    pipe_length: float,
    pipe_material: str,
    pipe_roughness: float | None,
    elevation_change: float,
    flow_rate: float,
    flow_unit: str,
    gas_composition: dict[str, float] | None,
    fittings: list[dict[str, Any]] | None,
    friction_method: str,
    compressibility_correction: bool,
    standard_condition: str,
    temp_k: float,
    pressure_pa: float,
) -> PressureDropInputs:
    """Resolve all parameters and construct a PressureDropInputs object."""
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
    return PressureDropInputs(
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
        >>> result = calculate_pressure_drop(pipe_size='4', pipe_schedule='40',
        ...     pipe_length=100, flow_rate=1500, flow_unit='SCFM',
        ...     pressure=10, temperature=500)
    """
    assert pipe_length is not None, "pipe_length must be provided"
    temp_k = _convert_temperature(temperature, temperature_unit, "K")
    pressure_pa = _convert_pressure(pressure, pressure_unit, "Pa")
    inputs = _build_pressure_drop_inputs(
        pipe_size,
        pipe_schedule,
        pipe_diameter,
        pipe_length,
        pipe_material,
        pipe_roughness,
        elevation_change,
        flow_rate,
        flow_unit,
        gas_composition,
        fittings,
        friction_method,
        compressibility_correction,
        standard_condition,
        temp_k,
        pressure_pa,
    )
    engine = PressureDropCalculationEngine()
    results = engine.calculate(inputs)
    return cast(dict[str, Any], _format_results(results))


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
        ...     pipe_diameter=0.1543,
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
        ...     pipe_size='6', pipe_schedule='40', pipe_length=100,
        ...     flow_rate=5000, flow_unit='kg/h', pressure=20, temperature=750
        ... )
    """
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
# COMMAND LINE INTERFACE
# ============================================================================


def main() -> None:
    """Command line entrypoint with a representative example."""
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
    print_results(result, "Example: Air Flow")


__all__ = [
    "show_help",
    "list_gas_components",
    "list_fittings",
    "list_pipe_sizes",
    "list_flow_units",
    "list_materials",
    "compare_friction_methods",
    "validate_inputs",
    "calculate_pressure_drop",
    "calculate_pressure_drop_custom_gas",
    "calculate_pressure_drop_syngas",
    "print_results",
    "main",
]
