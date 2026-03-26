"""Unified unit conversion service used across the application."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

from .core import (
    actual_to_standard_flow,
    convert_temperature,
    convert_via_table,
    scfm_to_standard_m3_per_hour,
    standard_m3_per_hour_to_scfm,
    standard_to_actual_flow,
)
from .tables import (
    CATEGORY_TABLES,
    CONCENTRATION_CONVERSIONS,
    GAS_DATABASE,
    HEATING_VALUE_CONVERSIONS,
    PERFORMANCE_UNITS,
    UNIT_ALIASES,
    StandardCondition,
)

logger = logging.getLogger(__name__)


class UnitConversionError(Exception):
    """Base class for conversion errors."""


class UnknownUnitError(UnitConversionError):
    """Raised when a unit is not recognized."""


class IncompatibleUnitsError(UnitConversionError):
    """Raised when attempting to convert between incompatible categories."""


class InvalidValueError(UnitConversionError):
    """Raised when an input value fails validation."""


@dataclass
class ConversionResult:
    """Container for conversions with metadata."""

    value: float
    from_unit: str
    to_unit: str
    uncertainty: float = 0.0
    warnings: list[str] = field(default_factory=list)


class UnitConversionService:
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
        self.category_map = {category: dict(table) for category, table in CATEGORY_TABLES.items()}
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
        self._ensure_compatible_categories(from_unit, to_unit, from_category, to_category)
        warnings = self._collect_conversion_warnings(value, from_category, from_unit_norm)
        converted = self._dispatch_conversion(
            value, from_unit_norm, to_unit_norm, from_category, kwargs
        )
        warnings.extend(
            self._user_unit_warnings(from_category, to_category, from_unit_norm, to_unit_norm)
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
        if from_category != to_category and {from_category, to_category} != {"temperature"}:
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
                standard_condition=kwargs.get("standard_condition", StandardCondition.SCFM_60F),
            )
        msg = f"Unsupported unit category for {from_unit_norm}"
        raise UnknownUnitError(msg)

    def _normalize_unit(self, unit: str) -> str:
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

    def _validate_value(self, value: float, category: str, unit: str | None = None) -> list[str]:
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
        except ValueError as exc:  # pragma: no cover - converted to domain-specific error
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

    def _convert_gas_flow(
        self,
        value: float,
        from_unit: str,
        to_unit: str,
        temperature: float | None = None,
        pressure: float | None = None,
        gas_type: str = "air",
        standard_condition: StandardCondition = StandardCondition.SCFM_60F,
    ) -> float:
        """Convert gas flow rate."""
        if not (value is not None):
            raise ValueError("value must be provided")
        if not (value is not None):
            raise ValueError("value must be provided")
        gas_props = GAS_DATABASE.get(gas_type.lower(), GAS_DATABASE["air"])
        self._ensure_acfm_inputs(from_unit, to_unit, temperature, pressure)
        m3_hr_std = self._gas_flow_to_standard_m3h(
            value,
            from_unit,
            gas_props.density_stp,
            temperature,
            pressure,
            standard_condition,
        )
        return self._standard_m3h_to_gas_flow(
            m3_hr_std,
            to_unit,
            gas_props.density_stp,
            temperature,
            pressure,
            standard_condition,
        )

    def _ensure_acfm_inputs(
        self,
        from_unit: str,
        to_unit: str,
        temperature: float | None,
        pressure: float | None,
    ) -> None:
        """Validate required inputs when ACFM is involved."""
        if (from_unit == "ACFM" or to_unit == "ACFM") and (temperature is None or pressure is None):
            msg = "Temperature and pressure are required for ACFM conversions"
            raise ValueError(msg)

    def _gas_flow_to_standard_m3h(
        self,
        value: float,
        from_unit: str,
        density_stp: float,
        temperature: float | None,
        pressure: float | None,
        standard_condition: StandardCondition,
    ) -> float:
        """Convert gas flow value to STP-normalized m³/hr."""
        if from_unit == "SCFM":
            return scfm_to_standard_m3_per_hour(value, standard_condition, StandardCondition.STP)
        if from_unit == "ACFM":
            if not (temperature is not None):
                raise ValueError("DbC Blocked: Precondition failed.")
            if not (pressure is not None):
                raise ValueError("DbC Blocked: Precondition failed.")
            scfm = actual_to_standard_flow(value, temperature, pressure, standard_condition)
            return scfm_to_standard_m3_per_hour(scfm, standard_condition, StandardCondition.STP)
        if from_unit in {"Nm3/hr", "Nm³/hr"}:
            return value
        if from_unit in self.mass_flow_factors:
            kg_s = value * self.mass_flow_factors[from_unit]
            return (kg_s * 3600.0) / density_stp
        msg = f"Unknown gas flow unit: {from_unit}"
        raise UnknownUnitError(msg)

    def _standard_m3h_to_gas_flow(
        self,
        m3_hr_std: float,
        to_unit: str,
        density_stp: float,
        temperature: float | None,
        pressure: float | None,
        standard_condition: StandardCondition,
    ) -> float:
        """Convert STP-normalized m³/hr to destination gas flow unit."""
        if to_unit == "SCFM":
            return standard_m3_per_hour_to_scfm(
                m3_hr_std, StandardCondition.STP, standard_condition
            )
        if to_unit == "ACFM":
            if not (temperature is not None):
                raise ValueError("DbC Blocked: Precondition failed.")
            if not (pressure is not None):
                raise ValueError("DbC Blocked: Precondition failed.")
            scfm = standard_m3_per_hour_to_scfm(
                m3_hr_std, StandardCondition.STP, standard_condition
            )
            return standard_to_actual_flow(scfm, temperature, pressure, standard_condition)
        if to_unit in {"Nm3/hr", "Nm³/hr"}:
            return m3_hr_std
        if to_unit in self.mass_flow_factors:
            kg_s = (m3_hr_std * density_stp) / 3600.0
            return kg_s / self.mass_flow_factors[to_unit]
        msg = f"Unknown gas flow unit: {to_unit}"
        raise UnknownUnitError(msg)

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

    def convert_gas_flow_scfm_acfm(
        self,
        value: float,
        from_unit: str,
        to_unit: str,
        gas_type: str = "air",
        actual_temp_K: float | None = None,
        actual_pressure_kPa: float | None = None,
        standard_condition: StandardCondition = StandardCondition.SCFM_60F,
        compressibility_factor: float = 1.0,
    ) -> float:
        """Convert gas flow between SCFM and ACFM."""
        if not (value is not None):
            raise ValueError("value must be provided")
        if not (value is not None):
            raise ValueError("value must be provided")
        std_temp, std_pressure_pa, _ = standard_condition.value
        temperature = actual_temp_K or std_temp
        pressure_pa = (
            actual_pressure_kPa * 1000.0 if actual_pressure_kPa is not None else std_pressure_pa
        )

        result = self._convert_gas_flow(
            value,
            from_unit.upper(),
            to_unit.upper(),
            temperature=temperature,
            pressure=pressure_pa,
            gas_type=gas_type,
            standard_condition=standard_condition,
        )

        if from_unit.upper() == "SCFM" and to_unit.upper() == "ACFM":
            return result * compressibility_factor
        if from_unit.upper() == "ACFM" and to_unit.upper() == "SCFM":
            if compressibility_factor <= 0:
                return result
            return result / compressibility_factor
        return result

    def heating_value(
        self,
        value: float,
        from_unit: str,
        to_unit: str,
        gas_density_stp: float | None = None,
    ) -> float:
        """Convert heating value."""
        if not (value is not None):
            raise ValueError("value must be provided")
        if not (value is not None):
            raise ValueError("value must be provided")
        if gas_density_stp is not None:
            self._require_positive_finite(gas_density_stp, "Gas density")
        from_key = from_unit.lower()
        to_key = to_unit.lower()
        if from_key == to_key:
            return value
        self._ensure_known_heating_unit(from_key, from_unit)
        self._ensure_known_heating_unit(to_key, to_unit)
        mj_per_kg = self._heating_to_mj_per_kg(value, from_key, from_unit, gas_density_stp)
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

    def _require_gas_density(self, gas_density_stp: float | None, unit_name: str) -> float:
        """Require gas density for volumetric heating value conversions."""
        if gas_density_stp is None:
            msg = f"Gas density required for {unit_name} conversion"
            raise ValueError(msg)
        return gas_density_stp

    def tar_concentration(
        self,
        value: float,
        from_unit: str,
        to_unit: str,
        temperature: float = 273.15,
        pressure: float = 101.325,
        molecular_weight: float | None = None,
    ) -> float:
        """Convert tar concentration."""
        if not (value is not None):
            raise ValueError("value must be provided")
        if not (value is not None):
            raise ValueError("value must be provided")
        self._validate_tar_inputs(temperature, pressure)
        from_key = from_unit.lower()
        to_key = to_unit.lower()
        if from_key == to_key:
            return value
        molecular_weight = self._resolve_molecular_weight(from_key, to_key, molecular_weight)
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
            return value * (temperature / 273.15) * (101.325 / pressure)
        if from_key in {"g/m³", "g/m3"}:
            return value * 1000.0 * (temperature / 273.15) * (101.325 / pressure)
        if from_key == "ppm_mass":
            if not (molecular_weight is not None):
                raise ValueError("DbC Blocked: Precondition failed.")
            return value * molecular_weight / 24.45
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
            return mg_nm3_value * (273.15 / temperature) * (pressure / 101.325)
        if to_key in {"g/m³", "g/m3"}:
            return mg_nm3_value / 1000.0 * (273.15 / temperature) * (pressure / 101.325)
        if to_key == "ppm_mass":
            if not (molecular_weight is not None):
                raise ValueError("DbC Blocked: Precondition failed.")
            return mg_nm3_value * 24.45 / molecular_weight
        msg = f"Conversion to {to_unit} not implemented"
        raise ValueError(msg)

    def syngas_composition(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert syngas composition units."""
        from_key = from_unit.lower()
        to_key = to_unit.lower()

        if from_key == to_key:
            return value

        if {from_key, to_key} <= {"mol%", "vol%"}:
            return value

        conversions = {
            ("ppm", "ppb"): 1000.0,
            ("ppb", "ppm"): 0.001,
            ("ppm", "%"): 0.0001,
            ("%", "ppm"): 10000.0,
            ("ppb", "%"): 0.0000001,
            ("%", "ppb"): 10000000.0,
            ("ppm", "mol%"): 0.0001,
            ("mol%", "ppm"): 10000.0,
            ("ppm", "vol%"): 0.0001,
            ("vol%", "ppm"): 10000.0,
        }

        key = (from_key, to_key)
        if key in conversions:
            return value * conversions[key]

        msg = f"Conversion from {from_unit} to {to_unit} not supported"
        raise ValueError(msg)

    def gasifier_performance(
        self,
        value: float,
        from_unit: str,
        to_unit: str,
        metric_type: str = "efficiency",
    ) -> float:
        """Convert gasifier performance metrics."""
        metric_type = metric_type.lower()
        from_key = from_unit.lower()
        to_key = to_unit.lower()

        if metric_type in {"efficiency", "carbon_conversion"}:
            if from_key == to_key:
                return value
            if from_key == "%" and to_key == "fraction":
                return value / 100.0
            if from_key == "fraction" and to_key == "%":
                return value * 100.0
            msg = f"Unknown conversion for {metric_type}"
            raise ValueError(msg)

        if metric_type == "specific_production":
            if from_key == to_key:
                return value
            if from_key in {"nm³/kg", "nm3/kg"} and to_key == "scf/lb":
                return value / 0.0624
            if from_key == "scf/lb" and to_key in {"nm³/kg", "nm3/kg"}:
                return value * 0.0624
            msg = "Unknown specific production conversion"
            raise ValueError(msg)

        msg = f"Unknown metric type: {metric_type}"
        raise ValueError(msg)

    def compressibility_factor(
        self,
        gas_type: str,
        temperature: float,
        pressure: float,
    ) -> float:
        """Calculate compressibility factor."""
        if not (gas_type is not None):
            raise ValueError("gas_type must be provided")
        if not (gas_type is not None):
            raise ValueError("gas_type must be provided")
        self._require_positive_finite(temperature, "temperature")
        self._require_positive_finite(pressure, "pressure")
        gas_props = GAS_DATABASE.get(gas_type.lower(), GAS_DATABASE["air"])
        Tr = temperature / gas_props.critical_temp
        Pr = pressure / gas_props.critical_pressure

        if 0.7 < Tr < 4 and Pr < 10:
            Z = 1 + (0.083 - 0.422 / Tr**1.6) * Pr + (0.139 - 0.172 / Tr**4.2) * Pr**2
            return float(max(Z, 0.1))
        return 1.0

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
                return {"tar_concentration": list(self.concentration_conversions.keys())}
            if category == "performance":
                return {
                    "performance": [u for units in self.performance_units.values() for u in units]
                }
            return {}

        result: dict[str, list[str]] = {}
        for name, factors in self.category_map.items():
            result[name] = list(factors.keys())
        result["temperature"] = ["K", "C", "F", "R"]
        result["gas_flow"] = ["SCFM", "ACFM", "Nm3/hr", "Nm³/hr"]
        result["heating_value"] = list(self.heating_value_conversions.keys())
        result["tar_concentration"] = list(self.concentration_conversions.keys())
        result["performance"] = [u for units in self.performance_units.values() for u in units]
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
