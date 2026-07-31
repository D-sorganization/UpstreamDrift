"""Tar concentration conversion mixin for UnitConversionService.

Extracts tar/aerosol concentration (mg/Nm³, g/m³, ppm) conversion logic
from the main service class to improve single-responsibility adherence.
"""

from __future__ import annotations

import logging

from .tables import (
    NORMAL_REFERENCE_PRESSURE_PA,
    NORMAL_REFERENCE_TEMPERATURE_K,
)

__all__ = [
    "TarConcentrationConversionMixin",
]

_logger = logging.getLogger(__name__)

# Single authoritative Nm³ reference state, shared with the gas-flow mixin so
# the service cannot hold two contradictory definitions of "Nm³" (issue #3389).
_NORMAL_TEMPERATURE_K: float = NORMAL_REFERENCE_TEMPERATURE_K  # 273.15 K
_NORMAL_PRESSURE_KPA: float = NORMAL_REFERENCE_PRESSURE_PA / 1000.0  # 101.325 kPa

# The public ``tar_concentration`` default arguments are spelled as the literals
# 273.15 / 101.325 to keep the API-contract signature representation stable, but
# they MUST equal the shared normal-state constants. This guard makes any future
# drift fail loudly at import time rather than silently re-splitting the Nm³
# basis (issue #3389).
assert _NORMAL_TEMPERATURE_K == 273.15  # noqa: S101 - import-time invariant
assert _NORMAL_PRESSURE_KPA == 101.325  # noqa: S101 - import-time invariant

# Ideal-gas molar volume at the normal state (0 °C / 1 atm) [L/mol]. Because
# mg/Nm³ is anchored at the normal state, ppm-by-volume <-> mg/Nm³ MUST use the
# 0 °C molar volume — NOT 24.45 L/mol, which is the 25 °C value and produced
# results ~8.3% low while still being labelled mg/Nm³ (issue #3389).
_MOLAR_VOLUME_NORMAL_L_PER_MOL = 22.414


class TarConcentrationConversionMixin:
    """Mixin providing tar concentration conversions for UnitConversionService."""

    concentration_conversions: dict[
        str, float | None
    ]  # provided by UnitConversionService

    def tar_concentration(
        self,
        value: float,
        from_unit: str,
        to_unit: str,
        temperature: float = 273.15,  # == _NORMAL_TEMPERATURE_K (DIN 1343, #3389)
        pressure: float = 101.325,  # == _NORMAL_PRESSURE_KPA (DIN 1343, #3389)
        molecular_weight: float | None = None,
    ) -> float:
        """Convert tar concentration."""
        if value is None:
            raise ValueError("value must be provided")
        self._validate_tar_inputs(temperature, pressure)
        from_key = from_unit.lower()
        to_key = to_unit.lower()
        if from_key == to_key:
            return value
        molecular_weight = self._resolve_molecular_weight(
            from_key, to_key, molecular_weight
        )
        self._ensure_known_concentration_unit(from_key, from_unit)
        self._ensure_known_concentration_unit(to_key, to_unit)
        mg_nm3_value = self._tar_to_mg_nm3(
            value, from_key, from_unit, temperature, pressure, molecular_weight
        )
        return self._tar_from_mg_nm3(
            mg_nm3_value, to_key, to_unit, temperature, pressure, molecular_weight
        )

    def _validate_tar_inputs(self, temperature: float, pressure: float) -> None:
        """Validate temperature and pressure for tar concentration conversion."""
        if pressure <= 0:
            msg = f"pressure must be positive, got {pressure}"
            raise ValueError(msg)
        if temperature <= 0:
            msg = f"temperature must be positive, got {temperature}"
            raise ValueError(msg)

    def _resolve_molecular_weight(
        self, from_key: str, to_key: str, molecular_weight: float | None
    ) -> float | None:
        """Validate and return molecular weight when ppm conversions are used."""
        requires_molecular_weight = from_key == "ppm_mass" or to_key == "ppm_mass"
        if not requires_molecular_weight:
            return molecular_weight
        if molecular_weight is None:
            msg = "Molecular weight required for ppm conversion"
            raise ValueError(msg)
        self._require_positive_finite(molecular_weight, "Molecular weight")
        return molecular_weight

    def _ensure_known_concentration_unit(self, unit_key: str, raw_unit: str) -> None:
        """Validate concentration unit key."""
        if unit_key not in self.concentration_conversions:
            msg = f"Unknown concentration unit: {raw_unit}"
            raise ValueError(msg)

    def _tar_to_mg_nm3(
        self,
        value: float,
        from_key: str,
        from_unit: str,
        temperature: float,
        pressure: float,
        molecular_weight: float | None,
    ) -> float:
        """Convert source concentration unit to mg/Nm3."""
        factor = self.concentration_conversions[from_key]
        if factor is not None:
            return value * factor
        if from_key in {"mg/m³", "mg/m3"}:
            return (
                value
                * (temperature / _NORMAL_TEMPERATURE_K)
                * (_NORMAL_PRESSURE_KPA / pressure)
            )
        if from_key in {"g/m³", "g/m3"}:
            return (
                value
                * 1000.0
                * (temperature / _NORMAL_TEMPERATURE_K)
                * (_NORMAL_PRESSURE_KPA / pressure)
            )
        if from_key == "ppm_mass":
            assert molecular_weight is not None
            return value * molecular_weight / _MOLAR_VOLUME_NORMAL_L_PER_MOL
        msg = f"Conversion from {from_unit} not implemented"
        raise ValueError(msg)

    def _tar_from_mg_nm3(
        self,
        mg_nm3_value: float,
        to_key: str,
        to_unit: str,
        temperature: float,
        pressure: float,
        molecular_weight: float | None,
    ) -> float:
        """Convert mg/Nm3 concentration to target unit."""
        factor = self.concentration_conversions[to_key]
        if factor is not None:
            return mg_nm3_value / factor
        if to_key in {"mg/m³", "mg/m3"}:
            return (
                mg_nm3_value
                * (_NORMAL_TEMPERATURE_K / temperature)
                * (pressure / _NORMAL_PRESSURE_KPA)
            )
        if to_key in {"g/m³", "g/m3"}:
            return (
                mg_nm3_value
                / 1000.0
                * (_NORMAL_TEMPERATURE_K / temperature)
                * (pressure / _NORMAL_PRESSURE_KPA)
            )
        if to_key == "ppm_mass":
            assert molecular_weight is not None
            return mg_nm3_value * _MOLAR_VOLUME_NORMAL_L_PER_MOL / molecular_weight
        msg = f"Conversion to {to_unit} not implemented"
        raise ValueError(msg)

    # _require_positive_finite is provided by UnitConversionService
    @staticmethod
    def _require_positive_finite(value: float, name: str) -> None:
        """Validate positive scalar physical parameters."""
        import math

        if not math.isfinite(value) or value <= 0:
            msg = f"{name} must be positive and finite, got {value}"
            raise ValueError(msg)
