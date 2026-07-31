"""Temperature and pressure unit converters for the pressure drop calculator.

This module provides unit-conversion helpers that are used internally by
pressure_drop_interface.py and may be imported directly by callers that only
need unit conversion utilities.
"""

from __future__ import annotations

__all__ = [
    "annotations",
]


def _convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    """Convert temperature between units."""
    if value is None:
        raise ValueError("value must be provided")
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
