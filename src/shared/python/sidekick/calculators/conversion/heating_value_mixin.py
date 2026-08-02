"""Heating value conversion mixin for UnitConversionService.

Extracts heating value (MJ/kg, BTU/SCF, kWh/Nm³, etc.) conversion logic
from the main service class to improve single-responsibility adherence.
"""

from __future__ import annotations

import logging

__all__ = [
    "HeatingValueConversionMixin",
]

_logger = logging.getLogger(__name__)

# Volumetric heating-value reference bases differ by temperature:
#   * BTU/SCF is defined at 60 °F (standard cubic foot), and
#   * MJ/Nm³ and kWh/Nm³ are defined at 0 °C (normal cubic metre).
# The two pressure references (14.696 psia vs 101.325 kPa) are equal, so only
# the molar-volume (temperature) ratio remains. Per the ideal-gas law a colder
# Nm³ holds more gas — and therefore more energy — than a 60 °F m³ by exactly
# the ratio of absolute temperatures (issue #3388).
_T_SCF_KELVIN = 288.706  # 60 °F in K
_T_NORMAL_KELVIN = 273.15  # 0 °C in K
# Multiply an energy-per-m³(60 °F) figure by this to obtain energy-per-Nm³(0 °C).
_SCF60F_TO_NM3_VOLUME_RATIO = _T_SCF_KELVIN / _T_NORMAL_KELVIN  # ≈ 1.05695

# MJ per m³ at 60 °F per (BTU/SCF). Combined with the temperature ratio this
# yields the correct MJ/Nm³ basis.
_BTU_PER_SCF_TO_MJ_PER_M3_60F = 0.0372589


class HeatingValueConversionMixin:
    """Mixin providing heating value conversions for UnitConversionService."""

    heating_value_conversions: dict[
        str, float | None
    ]  # provided by UnitConversionService

    def heating_value(
        self,
        value: float,
        from_unit: str,
        to_unit: str,
        gas_density_stp: float | None = None,
    ) -> float:
        """Convert heating value."""
        if value is None:
            raise ValueError("value must be provided")
        if gas_density_stp is not None:
            self._require_positive_finite(gas_density_stp, "Gas density")
        from_key = from_unit.lower()
        to_key = to_unit.lower()
        if from_key == to_key:
            return value
        self._ensure_known_heating_unit(from_key, from_unit)
        self._ensure_known_heating_unit(to_key, to_unit)
        mj_per_kg = self._heating_to_mj_per_kg(
            value, from_key, from_unit, gas_density_stp
        )
        return self._heating_from_mj_per_kg(mj_per_kg, to_key, to_unit, gas_density_stp)

    def _ensure_known_heating_unit(self, unit_key: str, raw_unit: str) -> None:
        """Validate heating value unit key."""
        if unit_key not in self.heating_value_conversions:
            msg = f"Unknown heating value unit: {raw_unit}"
            raise ValueError(msg)

    def _heating_to_mj_per_kg(
        self,
        value: float,
        from_key: str,
        from_unit: str,
        gas_density_stp: float | None,
    ) -> float:
        """Convert heating value from source unit to MJ/kg."""
        factor = self.heating_value_conversions[from_key]
        if factor is not None:
            return value * factor
        density = self._require_gas_density(gas_density_stp, from_unit)
        # ``density`` is the gas density at the normal (0 °C) basis, so every
        # volumetric value is first expressed as MJ per Nm³ before dividing by
        # density to reach MJ/kg.
        if from_key in {"mj/nm³", "mj/nm3"}:
            return value / density
        if from_key == "btu/scf":
            mj_per_nm3 = (
                value * _BTU_PER_SCF_TO_MJ_PER_M3_60F * _SCF60F_TO_NM3_VOLUME_RATIO
            )
            return mj_per_nm3 / density
        if from_key in {"kwh/nm³", "kwh/nm3"}:
            return (value * 3.6) / density
        msg = f"Conversion from {from_unit} not implemented"
        raise ValueError(msg)

    def _heating_from_mj_per_kg(
        self,
        mj_per_kg: float,
        to_key: str,
        to_unit: str,
        gas_density_stp: float | None,
    ) -> float:
        """Convert MJ/kg heating value to target unit."""
        factor = self.heating_value_conversions[to_key]
        if factor is not None:
            return mj_per_kg / factor
        density = self._require_gas_density(gas_density_stp, to_unit)
        if to_key in {"mj/nm³", "mj/nm3"}:
            return mj_per_kg * density
        if to_key == "btu/scf":
            # MJ/kg -> MJ/Nm³ -> MJ/m³(60 °F) -> BTU/SCF.
            mj_per_nm3 = mj_per_kg * density
            mj_per_m3_60f = mj_per_nm3 / _SCF60F_TO_NM3_VOLUME_RATIO
            return mj_per_m3_60f / _BTU_PER_SCF_TO_MJ_PER_M3_60F
        if to_key in {"kwh/nm³", "kwh/nm3"}:
            return (mj_per_kg * density) / 3.6
        msg = f"Conversion to {to_unit} not implemented"
        raise ValueError(msg)

    def _require_gas_density(
        self, gas_density_stp: float | None, unit_name: str
    ) -> float:
        """Require gas density for volumetric heating value conversions."""
        if gas_density_stp is None:
            msg = f"Gas density required for {unit_name} conversion"
            raise ValueError(msg)
        return gas_density_stp

    # _require_positive_finite is provided by UnitConversionService
    @staticmethod
    def _require_positive_finite(value: float, name: str) -> None:
        """Validate positive scalar physical parameters."""
        import math

        if not math.isfinite(value) or value <= 0:
            msg = f"{name} must be positive and finite, got {value}"
            raise ValueError(msg)
