"""Functional helpers for unit conversion routines.

These helpers are intentionally stateless so that conversion math can be reused outside of
the service class and tested independently.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from ...utils.unit_constants import (
    CELSIUS_OFFSET,
    RANKINE_RATIO,
    SCFM_TO_CU_METER_PER_HOUR_AT_60F,
)
from .tables import StandardCondition


def _require_positive_finite(value: float, name: str) -> None:
    """Validate physical scalar inputs used in flow conversions."""
    if not math.isfinite(value) or value <= 0:
        msg = f"{name} must be positive and finite, got {value}"
        raise ValueError(msg)


def convert_via_table(
    value: float, from_unit: str, to_unit: str, table: Mapping[str, float]
) -> float:
    """Convert using a base-unit lookup table."""

    if not (value is not None):
        raise ValueError("value must be provided")
    if from_unit == to_unit:
        return value
    base_value = value * table[from_unit]
    return base_value / table[to_unit]


def convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    """Perform temperature conversions via Kelvin as the pivot scale."""

    if from_unit == to_unit:
        return value

    if from_unit == "K":
        kelvin = value
    elif from_unit == "C":
        kelvin = value + CELSIUS_OFFSET
    elif from_unit == "F":
        kelvin = (value - 32.0) * 5.0 / 9.0 + CELSIUS_OFFSET
    elif from_unit == "R":
        kelvin = value * RANKINE_RATIO
    else:
        msg = f"Unknown temperature unit: {from_unit}"
        raise ValueError(msg)

    if to_unit == "K":
        return kelvin
    if to_unit == "C":
        return kelvin - CELSIUS_OFFSET
    if to_unit == "F":
        return (kelvin - CELSIUS_OFFSET) * 9.0 / 5.0 + 32.0
    if to_unit == "R":
        return kelvin / RANKINE_RATIO

    msg = f"Unknown temperature unit: {to_unit}"
    raise ValueError(msg)


def standard_to_actual_flow(
    scfm_value: float,
    temperature_k: float,
    pressure_pa: float,
    standard: StandardCondition,
) -> float:
    """Translate a standard volumetric flow in SCFM to ACFM at the given conditions."""
    if not (scfm_value is not None):
        raise ValueError("scfm_value must be provided")
    _require_positive_finite(temperature_k, "temperature_k")
    _require_positive_finite(pressure_pa, "pressure_pa")
    std_temp, std_pressure_pa, _ = standard.value
    return scfm_value * (std_pressure_pa / pressure_pa) * (temperature_k / std_temp)


def actual_to_standard_flow(
    acfm_value: float,
    temperature_k: float,
    pressure_pa: float,
    standard: StandardCondition,
) -> float:
    """Translate an actual volumetric flow in ACFM back to SCFM at reference conditions."""
    if not (acfm_value is not None):
        raise ValueError("acfm_value must be provided")
    _require_positive_finite(temperature_k, "temperature_k")
    _require_positive_finite(pressure_pa, "pressure_pa")
    std_temp, std_pressure_pa, _ = standard.value
    return acfm_value * (pressure_pa / std_pressure_pa) * (std_temp / temperature_k)


def scfm_to_standard_m3_per_hour(
    scfm_value: float, standard: StandardCondition, reference_std: StandardCondition
) -> float:
    """Convert SCFM at a non-default standard condition into standard m^3/hr."""

    if not (scfm_value is not None):
        raise ValueError("scfm_value must be provided")
    m3_hr_std = scfm_value * SCFM_TO_CU_METER_PER_HOUR_AT_60F
    std_temp, std_pressure_pa, _ = standard.value
    ref_temp, ref_pressure_pa, _ = reference_std.value
    if std_temp != ref_temp or std_pressure_pa != ref_pressure_pa:
        return m3_hr_std * (ref_temp / std_temp) * (std_pressure_pa / ref_pressure_pa)
    return m3_hr_std


def standard_m3_per_hour_to_scfm(
    m3_hr_at_ref: float, reference_std: StandardCondition, standard: StandardCondition
) -> float:
    """Convert m³/hr at a reference standard condition to SCFM at the standard condition."""

    # First convert m³/hr at reference standard to m³/hr at SCFM standard condition
    if not (m3_hr_at_ref is not None):
        raise ValueError("m3_hr_at_ref must be provided")
    ref_temp, ref_pressure_pa, _ = reference_std.value
    std_temp, std_pressure_pa, _ = standard.value
    if ref_temp != std_temp or ref_pressure_pa != std_pressure_pa:
        m3_hr_at_scfm_std = (
            m3_hr_at_ref * (std_temp / ref_temp) * (ref_pressure_pa / std_pressure_pa)
        )
    else:
        m3_hr_at_scfm_std = m3_hr_at_ref
    # Then convert to SCFM
    return m3_hr_at_scfm_std / SCFM_TO_CU_METER_PER_HOUR_AT_60F
