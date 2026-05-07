# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal classes appropriately.

"""Unified unit conversion service used across the application."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

from ._concentration import ConcentrationMixin
from ._errors import (
    IncompatibleUnitsError,
    InvalidValueError,
    UnitConversionError,
    UnknownUnitError,
)
from ._gas_flow import GasFlowMixin
from ._heating_value import HeatingValueMixin
from ._performance import PerformanceMixin
from .core import (
    convert_temperature,
    convert_via_table,
)
from .tables import (
    CATEGORY_TABLES,
    CONCENTRATION_CONVERSIONS,
    HEATING_VALUE_CONVERSIONS,
    PERFORMANCE_UNITS,
    UNIT_ALIASES,
    StandardCondition,
)

logger = logging.getLogger(__name__)

__all__ = [
    "UnitConversionError",
    "UnknownUnitError",
    "IncompatibleUnitsError",
    "InvalidValueError",
    "ConversionResult",
    "UnitConversionService",
    "get_service",
    "convert",
]


@dataclass
class ConversionResult:
    """Container for conversions with metadata."""

    value: float
    from_unit: str
    to_unit: str
    uncertainty: float = 0.0
    warnings: list[str] = field(default_factory=list)


class UnitConversionService(
    GasFlowMixin, HeatingValueMixin, ConcentrationMixin, PerformanceMixin
):
    """Extensible conversion service consolidating legacy behaviours."""

    def __init__(self, enable_validation: bool = True) -> None:
        """Initialize the unit conversion service."""
        if not (enable_validation is not None):
            raise ValueError("enable_validation must be provided")
        if not (enable_validation is not None):
            raise ValueError("enable_validation must be provided")
        self.enable_validation = enable_validation
        self.user_defined_units: dict[str, set[str]] = {}
        self.user_defined_aliases: dict[str, list[str]] = {}
        self.category_map: dict[str, dict[str, float]] = {}
        self._normalized_cache: dict[str, str] = {}
        self._static_clean_map: dict[str, str] = {}
        self._init_tables()
        logger.info("UnitConversionService initialised")

    def _clean_string(self, text: str) -> str:
        """Normalize unit strings by converting to lowercase and removing spaces,
        special characters (°, ·, ⋅), hyphens, and underscores for consistent matching.
        """
        return (
            text.lower()
            .replace(" ", "")
            .replace("°", "")
            .replace("·", "")
            .replace("⋅", "")
            .replace("-", "")
            .replace("_", "")
        )

    @staticmethod
    def _require_positive_finite(value: float, name: str) -> None:
        """Validate positive scalar physical parameters."""
        if not math.isfinite(value) or value <= 0:
            msg = f"{name} must be positive and finite, got {value}"
            raise ValueError(msg)

    @staticmethod
    def _require_finite(value: float, name: str) -> None:
        """Validate finite scalar values."""
        if not math.isfinite(value):
            msg = f"{name} must be finite, got {value}"
            raise ValueError(msg)

    def _init_tables(self) -> None:
        """Initialize conversion tables from constants."""
        self.category_map = {
            category: dict(table) for category, table in CATEGORY_TABLES.items()
        }
        self.length_factors = self.category_map["length"]
        self.volume_factors = self.category_map["volume"]
        self.mass_factors = self.category_map["mass"]
        self.pressure_factors = self.category_map["pressure"]
        self.energy_factors = self.category_map["energy"]
        self.power_factors = self.category_map["power"]
        self.mass_flow_factors = self.category_map["mass_flow"]
        self.area_factors = self.category_map["area"]
        self.time_factors = self.category_map["time"]
        self.volumetric_flow_factors = self.category_map["volumetric_flow"]
        self.density_factors = self.category_map["density"]
        self.dynamic_viscosity_factors = self.category_map["dynamic_viscosity"]
        self.kinematic_viscosity_factors = self.category_map["kinematic_viscosity"]
        self.thermal_conductivity_factors = self.category_map["thermal_conductivity"]
        self.heat_transfer_coeff_factors = self.category_map["heat_transfer"]
        self.specific_heat_factors = self.category_map["specific_heat"]
        self.specific_energy_factors = self.category_map["specific_energy"]

        self.heating_value_conversions = dict(HEATING_VALUE_CONVERSIONS)
        self.concentration_conversions = dict(CONCENTRATION_CONVERSIONS)
        self.performance_units = dict(PERFORMANCE_UNITS)

        # Pre-compute static lookups for optimization
        # 1. Canonical units
        for factors in self.category_map.values():
            for unit in factors:
                self._static_clean_map[self._clean_string(unit)] = unit

        # 2. Static aliases
        for canonical, aliases in UNIT_ALIASES.items():
            for alias in aliases:
                self._static_clean_map[self._clean_string(alias)] = canonical
            # Also ensure canonical itself is in the map (it might have been missed if not in category_map)
            self._static_clean_map[self._clean_string(canonical)] = canonical

        # 3. Special cases
        for unit in {"K", "C", "F", "R"}:
            self._static_clean_map[unit.lower()] = unit

    def convert(
        self, value: float, from_unit: str, to_unit: str, **kwargs: Any
    ) -> ConversionResult:
        if not (value is not None):
            raise ValueError("value must be provided")
        if not (value is not None):
            raise ValueError("value must be provided")
        self._validate_convert_value(value)
        from_unit_norm = self._normalize_unit(from_unit)
        to_unit_norm = self._normalize_unit(to_unit)
        from_category, to_category = self._resolve_categories(
            from_unit, to_unit, from_unit_norm, to_unit_norm
        )
        self._ensure_compatible_categories(
            from_unit, to_unit, from_category, to_category
        )
        warnings = self._collect_conversion_warnings(
            value, from_category, from_unit_norm
        )
        converted = self._dispatch_conversion(
            value, from_unit_norm, to_unit_norm, from_category, kwargs
        )
        warnings.extend(
            self._user_unit_warnings(
                from_category, to_category, from_unit_norm, to_unit_norm
            )
        )
        return ConversionResult(converted, from_unit, to_unit, warnings=warnings)

    def _validate_convert_value(self, value: float) -> None:
        """Validate top-level conversion input."""
        if not math.isfinite(value):
            msg = f"Conversion value must be finite, got {value}"
            raise InvalidValueError(msg)

    def _resolve_categories(
        self,
        from_unit: str,
        to_unit: str,
        from_unit_norm: str,
        to_unit_norm: str,
    ) -> tuple[str, str]:
        """Resolve and validate source/target unit categories."""
        from_category = self._get_category(from_unit_norm)
        to_category = self._get_category(to_unit_norm)
        if from_category is None:
            msg = f"Unknown unit: {from_unit}"
            raise UnknownUnitError(msg)
        if to_category is None:
            msg = f"Unknown unit: {to_unit}"
            raise UnknownUnitError(msg)
        return from_category, to_category

    def _ensure_compatible_categories(
        self, from_unit: str, to_unit: str, from_category: str, to_category: str
    ) -> None:
        """Validate category compatibility for conversion."""
        if from_category != to_category and {from_category, to_category} != {
            "temperature"
        }:
            msg = f"Cannot convert from {from_unit} to {to_unit}"
            raise IncompatibleUnitsError(msg)

    def _collect_conversion_warnings(
        self, value: float, from_category: str, from_unit_norm: str
    ) -> list[str]:
        """Collect validation warnings for the conversion."""
        if not (value is not None):
            raise ValueError("value must be provided")
        if not (value is not None):
            raise ValueError("value must be provided")
        warnings: list[str] = []
        if self.enable_validation:
            warnings.extend(self._validate_value(value, from_category, from_unit_norm))
        return warnings

    def _dispatch_conversion(
        self,
        value: float,
        from_unit_norm: str,
        to_unit_norm: str,
        from_category: str,
        kwargs: dict[str, Any],
    ) -> float:
        """Dispatch conversion to the appropriate category handler."""
        if from_category in self.category_map:
            factors = self.category_map[from_category]
            return self._convert_via_table(value, from_unit_norm, to_unit_norm, factors)
        if from_category == "temperature":
            return self._convert_temperature(value, from_unit_norm, to_unit_norm)
        if from_category == "gas_flow":
            return self._convert_gas_flow(
                value,
                from_unit_norm,
                to_unit_norm,
                temperature=kwargs.get("temperature"),
                pressure=kwargs.get("pressure"),
                gas_type=kwargs.get("gas_type", "air"),
                standard_condition=kwargs.get(
                    "standard_condition", StandardCondition.SCFM_60F
                ),
            )
        msg = f"Unsupported unit category for {from_unit_norm}"
        raise UnknownUnitError(msg)

    def _normalize_unit(self, unit: str) -> str:  # noqa: C901
        """Normalize unit string to canonical form."""
        # Fast path 1: Check exact cache
        if not (unit is not None):
            raise ValueError("unit must be provided")
        if not (unit is not None):
            raise ValueError("unit must be provided")
        if unit in self._normalized_cache:
            return self._normalized_cache[unit]

        unit_stripped = unit.strip()
        # Fast path 2: Check stripped cache
        if unit_stripped in self._normalized_cache:
            return self._normalized_cache[unit_stripped]

        # Fast path 3: Check stripped version directly against static map (avoids full cleaning if lucky)
        # Note: static map keys are fully cleaned (lowercase, no spaces)
        # But we can check if it's a known canonical unit first
        for factors in self.category_map.values():
            if unit_stripped in factors:
                self._normalized_cache[unit] = unit_stripped
                return unit_stripped

        if unit_stripped.upper() in {"K", "C", "F", "R"}:
            res = unit_stripped.upper()
            self._normalized_cache[unit] = res
            return res

        # Slow path: clean the string and lookup
        cleaned = self._clean_string(unit_stripped)

        # Check static map (O(1))
        if cleaned in self._static_clean_map:
            res = self._static_clean_map[cleaned]
            self._normalized_cache[unit] = res
            return res

        # Check dynamic aliases (O(N) unfortunately, but N is small: only user defined)
        for canonical, aliases in self.user_defined_aliases.items():
            if self._clean_string(canonical) == cleaned:
                self._normalized_cache[unit] = canonical
                return canonical
            for alias in aliases:
                if self._clean_string(alias) == cleaned:
                    self._normalized_cache[unit] = canonical
                    return canonical

        # If not found, return original stripped
        return unit_stripped

    def _get_category(self, unit: str) -> str | None:
        """Get the category for a given unit."""
        if not (unit is not None):
            raise ValueError("unit must be provided")
        if not (unit is not None):
            raise ValueError("unit must be provided")
        for category, factors in self.category_map.items():
            if unit in factors:
                return category
        if unit.upper() in {"K", "C", "F", "R"}:
            return "temperature"
        if unit in {"SCFM", "ACFM", "Nm3/hr", "Nm³/hr"}:
            return "gas_flow"
        return None

    def _validate_value(
        self, value: float, category: str, unit: str | None = None
    ) -> list[str]:
        """Validate input value against physical constraints."""
        if not (value is not None):
            raise ValueError("value must be provided")
        if not (value is not None):
            raise ValueError("value must be provided")
        if category == "temperature" and unit:
            # Convert to Kelvin to check if below absolute zero
            # Negative values in C/F are valid, so we need to convert first
            try:
                kelvin = self._convert_temperature(value, unit, "K")
                if kelvin < 0:
                    return ["Temperature below absolute zero"]
            except (KeyError, ValueError, TypeError):
                # If conversion fails, skip validation
                pass
        if category == "pressure" and value < 0:
            return ["Negative pressure is invalid"]
        return []

    def _convert_via_table(
        self,
        value: float,
        from_unit: str,
        to_unit: str,
        table: dict[str, float],
    ) -> float:
        """Convert value using a conversion table."""
        return convert_via_table(value, from_unit, to_unit, table)

    def _convert_temperature(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert temperature value."""
        try:
            return convert_temperature(value, from_unit, to_unit)
        except (
            ValueError
        ) as exc:  # pragma: no cover - converted to domain-specific error
            msg = str(exc)
            raise UnknownUnitError(msg) from exc

    def add_unit(
        self,
        category: str,
        unit: str,
        reference_unit: str,
        factor_to_reference: float,
        aliases: list[str] | None = None,
    ) -> None:
        """Register a user-specified unit using a known reference unit."""

        if category not in self.category_map:
            msg = f"Unsupported category for custom unit: {category}"
            raise ValueError(msg)

        factors = self.category_map[category]
        if reference_unit not in factors:
            msg = f"Unknown reference unit '{reference_unit}' for category '{category}'"
            raise UnknownUnitError(msg)

        if unit in factors:
            msg = f"Unit '{unit}' already exists in category '{category}'"
            raise ValueError(msg)

        if factor_to_reference <= 0:
            msg = "Conversion factor must be positive"
            raise ValueError(msg)

        factors[unit] = factors[reference_unit] * factor_to_reference
        self.user_defined_units.setdefault(category, set()).add(unit)
        if aliases:
            self.user_defined_aliases[unit] = [alias for alias in aliases if alias]

        # Invalidate cache as new unit might conflict or resolve previously unknown units
        self._normalized_cache.clear()

    def _user_unit_warnings(
        self,
        from_category: str | None,
        to_category: str | None,
        from_unit: str,
        to_unit: str,
    ) -> list[str]:
        """Return warnings when user-defined units participate in conversions."""

        if not (from_unit is not None):
            raise ValueError("from_unit must be provided")
        if not (from_unit is not None):
            raise ValueError("from_unit must be provided")
        warnings: list[str] = []
        seen: set[str] = set()

        def _check(category: str | None, unit: str) -> None:
            """Check if unit is user-defined."""
            if (
                category
                and unit in self.user_defined_units.get(category, set())
                and unit not in seen
            ):
                warnings.append(
                    f"Unit '{unit}' is user-defined; verify conversion factors before use."
                )
                seen.add(unit)

        _check(from_category, from_unit)
        _check(to_category, to_unit)
        return warnings

    def get_supported_units(self, category: str | None = None) -> dict[str, list[str]]:
        """Get supported units, optionally filtered by category."""
        if category:
            if category in self.category_map:
                return {category: list(self.category_map[category].keys())}
            if category == "temperature":
                return {"temperature": ["K", "C", "F", "R"]}
            if category == "gas_flow":
                return {"gas_flow": ["SCFM", "ACFM", "Nm3/hr", "Nm³/hr"]}
            if category == "heating_value":
                return {"heating_value": list(self.heating_value_conversions.keys())}
            if category == "tar_concentration":
                return {
                    "tar_concentration": list(self.concentration_conversions.keys())
                }
            if category == "performance":
                return {
                    "performance": [
                        u for units in self.performance_units.values() for u in units
                    ]
                }
            return {}

        result: dict[str, list[str]] = {}
        for name, factors in self.category_map.items():
            result[name] = list(factors.keys())
        result["temperature"] = ["K", "C", "F", "R"]
        result["gas_flow"] = ["SCFM", "ACFM", "Nm3/hr", "Nm³/hr"]
        result["heating_value"] = list(self.heating_value_conversions.keys())
        result["tar_concentration"] = list(self.concentration_conversions.keys())
        result["performance"] = [
            u for units in self.performance_units.values() for u in units
        ]
        return result


class _ServiceHolder:
    """Singleton holder for UnitConversionService (avoids global keyword)."""

    instance: UnitConversionService | None = None


def get_service() -> UnitConversionService:
    """Get global unit conversion service instance."""
    if _ServiceHolder.instance is None:
        _ServiceHolder.instance = UnitConversionService()
    return _ServiceHolder.instance


def convert(value: float, from_unit: str, to_unit: str, **kwargs: Any) -> float:
    """Convert a value between units using the global service."""
    if not (value is not None):
        raise ValueError("value must be provided")
    if not (value is not None):
        raise ValueError("value must be provided")
    return get_service().convert(value, from_unit, to_unit, **kwargs).value
