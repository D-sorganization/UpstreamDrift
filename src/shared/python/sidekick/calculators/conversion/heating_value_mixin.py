"""Heating value conversion mixin for UnitConversionService.

Extracts heating value (MJ/kg, BTU/SCF, kWh/Nm³, etc.) conversion logic
from the main service class to improve single-responsibility adherence.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


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
        assert value is not None, "value must be provided"
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
        if from_key in {"mj/nm³", "mj/nm3"}:
            return value / density
        if from_key == "btu/scf":
            return (value * 0.0372589) / density
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
            return (mj_per_kg * density) / 0.0372589
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
