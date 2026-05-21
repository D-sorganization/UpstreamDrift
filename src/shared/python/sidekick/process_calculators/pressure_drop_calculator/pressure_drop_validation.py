# ruff: noqa: E501
"""Input validation helpers for the pressure drop calculator.

This module groups all ``_validate_*``, ``_log_validation_report``, and the
public ``validate_inputs`` function so they can be maintained and tested
independently of the main calculation interface.
"""

from __future__ import annotations

import logging
from typing import Any

from .pressure_drop_results import _wrap_text
from .utils.fitting_loss_coefficients import FITTING_K_FACTORS
from .utils.flow_rate_converter import (
    MASS_FLOW_CONVERSIONS,
    MOLAR_FLOW_CONVERSIONS,
    VOLUMETRIC_FLOW_CONVERSIONS_TO_M3_S,
)
from .utils.gas_properties import GAS_DATABASE
from .utils.pipe_database import get_pipe_spec, list_available_sizes

logger = logging.getLogger(__name__)


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
    if errors is None:
        raise ValueError("errors must be provided")
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
    if errors is None:
        raise ValueError("errors must be provided")
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
    if errors is None:
        raise ValueError("errors must be provided")
    if gas_composition:
        total = sum(gas_composition.values())
        if not (0.99 <= total <= 1.01):
            warnings.append(
                f"Gas composition sums to {total:.4f}, expected ~1.0. Will be auto-normalized."
            )

        unknown = [c for c in gas_composition.keys() if c not in GAS_DATABASE]
        if unknown:
            errors.append(
                f"Unknown gas components: {', '.join(unknown)}. "
                f"Available: {', '.join(GAS_DATABASE.keys())}"
            )

    if fittings:
        for i, fitting in enumerate(fittings):
            fitting_type = fitting.get("type", "")
            if fitting_type and fitting_type not in FITTING_K_FACTORS:
                similar = [f for f in FITTING_K_FACTORS.keys() if fitting_type in f]
                warnings.append(
                    f"Fitting[{i}] type '{fitting_type}' not in database. "
                    f"Similar: {', '.join(similar[:3]) if similar else 'see list_fittings()'}"
                )


def _log_validation_report(
    is_valid: bool, errors: list[str], warnings: list[str]
) -> None:
    """Log a formatted validation report."""
    if is_valid is None:
        raise ValueError("is_valid must be provided")
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
