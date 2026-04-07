"""Validation helpers for the pressure drop calculator facade."""

from __future__ import annotations

import logging
from typing import Any

from .utils.fitting_loss_coefficients import FITTING_K_FACTORS
from .utils.flow_rate_converter import (
    MASS_FLOW_CONVERSIONS,
    MOLAR_FLOW_CONVERSIONS,
    VOLUMETRIC_FLOW_CONVERSIONS_TO_M3_S,
)
from .utils.gas_properties import GAS_DATABASE
from .utils.pipe_database import get_pipe_spec, list_available_sizes

logger = logging.getLogger(__name__)


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
    """Validate inputs before calculation and provide helpful suggestions."""
    errors: list[str] = []
    warnings: list[str] = []
    _validate_pipe_params(pipe_size, pipe_schedule, pipe_diameter, errors, warnings)
    _validate_flow_params(flow_rate, flow_unit, errors)
    _validate_conditions(pressure, temperature, errors, warnings)
    _validate_composition_and_fittings(gas_composition, fittings, errors, warnings)
    is_valid = len(errors) == 0
    _log_validation_report(is_valid, errors, warnings)
    return is_valid, errors, warnings


def _validate_pipe_params(
    pipe_size: str | None,
    pipe_schedule: str | None,
    pipe_diameter: float | None,
    errors: list[str],
    warnings: list[str],
) -> None:
    if pipe_diameter is None:
        if pipe_size is None or pipe_schedule is None:
            errors.append(
                "Must provide either pipe_diameter OR both pipe_size and pipe_schedule"
            )
        else:
            try:
                get_pipe_spec(pipe_size, pipe_schedule)
            except ValueError as exc:
                available = list_available_sizes()
                errors.append(f"{exc}. Available sizes: {', '.join(available)}")
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
            similar = [unit for unit in all_units if flow_unit.lower() in unit.lower()]
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
    if gas_composition:
        total = sum(gas_composition.values())
        if not (0.99 <= total <= 1.01):
            warnings.append(
                f"Gas composition sums to {total:.4f}, expected ~1.0. Will be auto-normalized."
            )
        unknown = [
            component for component in gas_composition if component not in GAS_DATABASE
        ]
        if unknown:
            errors.append(
                f"Unknown gas components: {', '.join(unknown)}. "
                f"Available: {', '.join(GAS_DATABASE.keys())}"
            )
    if fittings:
        for index, fitting in enumerate(fittings):
            fitting_type = fitting.get("type", "")
            if fitting_type and fitting_type not in FITTING_K_FACTORS:
                similar = [
                    candidate
                    for candidate in FITTING_K_FACTORS
                    if fitting_type in candidate
                ]
                warnings.append(
                    f"Fitting[{index}] type '{fitting_type}' not in database. "
                    f"Similar: {', '.join(similar[:3]) if similar else 'see list_fittings()'}"
                )


def _wrap_text(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        proposed = word if not current else f"{current} {word}"
        if len(proposed) <= width:
            current = proposed
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _log_validation_report(
    is_valid: bool, errors: list[str], warnings: list[str]
) -> None:
    logger.info("Input validation result: valid=%s", is_valid)
    for error in errors:
        for line in _wrap_text(error, 80):
            logger.error(line)
    for warning in warnings:
        for line in _wrap_text(warning, 80):
            logger.warning(line)
