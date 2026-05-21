# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal classes appropriately.

"""
Steam Calculation Engine
========================

Pure Python steam property calculation engine, decoupled from UI.
Provides thermodynamic calculations using CoolProp, Cantera, or simplified correlations.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ._constants import (
    ANTOINE_A,
    ANTOINE_B,
    ANTOINE_C_CELSIUS,
    ANTOINE_C_KELVIN,
    BOILING_TEMPERATURE_WATER,
    BUCK_A,
    BUCK_B,
    BUCK_C,
    BUCK_D,
    CELSIUS_TO_FAHRENHEIT_SCALE,
    CRITICAL_PRESSURE_WATER,
    CRITICAL_TEMPERATURE_WATER,
    DEFAULT_DEW_POINT_TEMPERATURE_CELSIUS,
    DEFAULT_QUALITY,
    FAHRENHEIT_TO_CELSIUS_OFFSET,
    FAHRENHEIT_TO_CELSIUS_SCALE,
    FALLBACK_ATMOSPHERIC_PRESSURE,
    FALLBACK_BOILING_TEMPERATURE,
    KELVIN_TO_CELSIUS_OFFSET,
    KPA_TO_PA_FACTOR,
    LIQUID_WATER_PRESSURE_THRESHOLD,
    LIQUID_WATER_SPECIFIC_HEAT,
    MAX_TEMPERATURE_UI_K,
    MBAR_TO_KPA_FACTOR,
    MIN_TEMPERATURE_UI_K,
    MMHG_TO_PASCAL_FACTOR,
    NEWTON_RAPHSON_DERIVATIVE_TOLERANCE,
    NEWTON_RAPHSON_MAX_ITERATIONS,
    NEWTON_RAPHSON_STEP_SIZE,
    NEWTON_RAPHSON_TOLERANCE,
    PASCAL_TO_MMHG_FACTOR,
    SATURATED_FROM_PRESSURE_STATE,
    SATURATED_FROM_TEMP_STATE,
    SPECIFIC_GAS_CONSTANT_WATER,
    STANDARD_ATMOSPHERIC_PRESSURE,
    SUPERHEATED_STATE,
    TRIPLE_POINT_PRESSURE,
    TRIPLE_POINT_TEMPERATURE,
    VAPOR_ENTHALPY_REFERENCE,
    VAPOR_ENTHALPY_SLOPE,
    VAPOR_ENTROPY_REFERENCE,
    VAPOR_ENTROPY_SLOPE,
    VAPOR_SPECIFIC_HEAT_CP,
    VAPOR_SPECIFIC_HEAT_CV,
)
from ._models import SteamProperties
from ._property_backends import (
    CANTERA_AVAILABLE,
    COOLPROP_AVAILABLE,
    calculate_cantera_properties,
    calculate_coolprop_properties,
    calculate_saturated_cantera_from_pressure,
    calculate_saturated_cantera_from_temp,
    calculate_saturated_coolprop_from_pressure,
    calculate_saturated_coolprop_from_temp,
    calculate_saturated_simplified_from_pressure,
    calculate_saturated_simplified_from_temp,
    calculate_simplified_properties,
    get_saturation_pressure,
    get_saturation_temperature,
)
from ._vapor_pressure import calculate_dew_point, calculate_water_vapor_pressure

logger = logging.getLogger(__name__)

try:
    import cantera as ct
except ImportError:
    ct = None  # type: ignore[assignment]

__all__ = [
    "SteamProperties",
    "SteamCalculationEngine",
    # constants
    "STANDARD_ATMOSPHERIC_PRESSURE",
    "BOILING_TEMPERATURE_WATER",
    "LIQUID_WATER_PRESSURE_THRESHOLD",
    "SPECIFIC_GAS_CONSTANT_WATER",
    "LIQUID_WATER_SPECIFIC_HEAT",
    "VAPOR_ENTHALPY_REFERENCE",
    "VAPOR_ENTHALPY_SLOPE",
    "VAPOR_SPECIFIC_HEAT_CP",
    "VAPOR_SPECIFIC_HEAT_CV",
    "VAPOR_ENTROPY_REFERENCE",
    "VAPOR_ENTROPY_SLOPE",
    "FAHRENHEIT_TO_CELSIUS_OFFSET",
    "FAHRENHEIT_TO_CELSIUS_SCALE",
    "CELSIUS_TO_FAHRENHEIT_SCALE",
    "MIN_TEMPERATURE_UI_K",
    "MAX_TEMPERATURE_UI_K",
    "NEWTON_RAPHSON_TOLERANCE",
    "NEWTON_RAPHSON_DERIVATIVE_TOLERANCE",
    "NEWTON_RAPHSON_MAX_ITERATIONS",
    "NEWTON_RAPHSON_STEP_SIZE",
    "ANTOINE_A",
    "ANTOINE_B",
    "ANTOINE_C_CELSIUS",
    "ANTOINE_C_KELVIN",
    "BUCK_A",
    "BUCK_B",
    "BUCK_C",
    "BUCK_D",
    "MMHG_TO_PASCAL_FACTOR",
    "PASCAL_TO_MMHG_FACTOR",
    "MBAR_TO_KPA_FACTOR",
    "KPA_TO_PA_FACTOR",
    "KELVIN_TO_CELSIUS_OFFSET",
    "CRITICAL_TEMPERATURE_WATER",
    "CRITICAL_PRESSURE_WATER",
    "TRIPLE_POINT_TEMPERATURE",
    "TRIPLE_POINT_PRESSURE",
    "DEFAULT_DEW_POINT_TEMPERATURE_CELSIUS",
    "DEFAULT_QUALITY",
    "FALLBACK_ATMOSPHERIC_PRESSURE",
    "FALLBACK_BOILING_TEMPERATURE",
    "SUPERHEATED_STATE",
    "SATURATED_FROM_TEMP_STATE",
    "SATURATED_FROM_PRESSURE_STATE",
    "LIQUID_WATER_PRESSURE_THRESHOLD",
    "CANTERA_AVAILABLE",
    "COOLPROP_AVAILABLE",
]


class SteamCalculationEngine:
    """Core steam calculation engine using Cantera"""

    def __init__(self) -> None:
        """Initialize the steam calculation engine"""
        self.water: Any = None
        self.initialized = False
        self._initialize_cantera()

    def _initialize_cantera(self) -> None:
        """Initialize Cantera water object"""
        if not CANTERA_AVAILABLE:
            return

        try:
            self.water = ct.Water()
            self.initialized = True
            logger.info("Steam calculation engine initialized successfully")
        except (RuntimeError, ValueError, OSError) as e:
            logger.exception("Failed to initialize Cantera water: %s", e)
            self.initialized = False

    def _select_best_engine(self, engine: str) -> str:
        """
        Select the best available calculation engine based on preference and availability.

        Args:
            engine: Requested engine ("coolprop", "cantera", "simplified", "auto")

        Returns:
            Best available engine name in lowercase
        """
        if engine is None:
            raise ValueError("engine must be provided")
        if engine == "auto":
            if COOLPROP_AVAILABLE:
                return "coolprop"
            if CANTERA_AVAILABLE and self.water is not None:
                return "cantera"
            return "simplified"

        if engine == "coolprop" and COOLPROP_AVAILABLE:
            return "coolprop"
        if engine == "cantera" and CANTERA_AVAILABLE and self.water is not None:
            return "cantera"
        if engine == "simplified":
            return "simplified"

        logger.warning(
            "Requested engine '%s' not available, falling back to auto-selection",
            engine,
        )
        return self._select_best_engine("auto")

    def calculate_properties(
        self, temperature: float, pressure: float, engine: str = "auto"
    ) -> SteamProperties:
        """Calculate steam properties for given temperature and pressure.

        Args:
            temperature: Temperature in Kelvin (must be > 0)
            pressure: Pressure in Pa (must be > 0)
            engine: Calculation engine ('auto', 'coolprop', 'cantera', 'simplified')

        Returns:
            SteamProperties dataclass with all thermodynamic properties
        """
        # DbC preconditions
        if not (temperature > 0):
            raise ValueError(f"Temperature must be positive (K), got {temperature}")
        if not (pressure > 0):
            raise ValueError(f"Pressure must be positive (Pa), got {pressure}")

        try:
            selected_engine = self._select_best_engine(engine)

            if selected_engine == "coolprop":
                result = calculate_coolprop_properties(temperature, pressure)
            elif selected_engine == "cantera":
                result = calculate_cantera_properties(self.water, temperature, pressure)
            else:
                result = calculate_simplified_properties(temperature, pressure)

        except (RuntimeError, ValueError, TypeError) as e:
            logger.exception("Steam calculation failed: %s", e)
            result = calculate_simplified_properties(temperature, pressure)

        if not (np.isfinite(result.enthalpy)):
            raise ValueError(f"Enthalpy must be finite, got {result.enthalpy}")
        return result

    def calculate_saturated_properties_from_temperature(
        self, temperature: float, engine: str = "auto"
    ) -> SteamProperties:
        """
        Calculate saturated steam properties from temperature
        """
        try:
            selected_engine = self._select_best_engine(engine)

            if selected_engine == "coolprop":
                return calculate_saturated_coolprop_from_temp(temperature)
            if selected_engine == "cantera":
                return calculate_saturated_cantera_from_temp(self.water, temperature)
            return calculate_saturated_simplified_from_temp(temperature)

        except (RuntimeError, ValueError, TypeError) as e:
            logger.exception(
                "Saturated steam calculation from temperature failed: %s", e
            )
            return calculate_saturated_simplified_from_temp(temperature)

    def calculate_saturated_properties_from_pressure(
        self, pressure: float, engine: str = "auto"
    ) -> SteamProperties:
        """
        Calculate saturated steam properties from pressure
        """
        try:
            selected_engine = self._select_best_engine(engine)

            if selected_engine == "coolprop":
                return calculate_saturated_coolprop_from_pressure(pressure)
            if selected_engine == "cantera":
                return calculate_saturated_cantera_from_pressure(self.water, pressure)
            return calculate_saturated_simplified_from_pressure(pressure)

        except (RuntimeError, ValueError, TypeError) as e:
            logger.exception("Saturated steam calculation from pressure failed: %s", e)
            return calculate_saturated_simplified_from_pressure(pressure)

    def calculate_water_vapor_pressure(
        self, temperature: float, method: str = "buck"
    ) -> float:
        """
        Calculate water vapor pressure using various correlations
        """
        return calculate_water_vapor_pressure(temperature, method)

    def calculate_dew_point(
        self, partial_pressure_pa: float, total_pressure_pa: float
    ) -> float:
        """
        Calculate dew point temperature from partial pressure
        """
        return calculate_dew_point(partial_pressure_pa, total_pressure_pa)

    def get_saturation_pressure(self, temperature: float) -> float:
        """Get saturation pressure for given temperature"""
        return get_saturation_pressure(self.water, temperature)

    def get_saturation_temperature(self, pressure: float) -> float:
        """Get saturation temperature for given pressure"""
        return get_saturation_temperature(self.water, pressure)

    def _calculate_cantera_properties(
        self, temperature: float, pressure: float
    ) -> SteamProperties:
        """Calculate steam properties using Cantera"""
        return calculate_cantera_properties(self.water, temperature, pressure)

    def _calculate_coolprop_properties(
        self, temperature: float, pressure: float
    ) -> SteamProperties:
        """High-accuracy calculation using CoolProp"""
        return calculate_coolprop_properties(temperature, pressure)

    def _calculate_simplified_properties(
        self, temperature: float, pressure: float
    ) -> SteamProperties:
        """Simplified calculations based on ideal gas law and constant properties"""
        return calculate_simplified_properties(temperature, pressure)

    @staticmethod
    def _validate_coolprop_inputs(
        temperature: float,
        pressure: float,
    ) -> None:
        """Validate temperature and pressure for CoolProp calculations."""
        from ._property_backends import validate_coolprop_inputs

        validate_coolprop_inputs(temperature, pressure)

    @staticmethod
    def _compute_derived_properties(
        cp: float,
        cv: float,
        dynamic_viscosity: float,
        thermal_conductivity: float,
        pressure: float,
        specific_volume: float,
        temperature: float,
    ) -> dict[str, float | None]:
        """Compute derived thermo properties (Z, Pr, k)."""
        from ._property_backends import compute_derived_properties

        return compute_derived_properties(
            cp,
            cv,
            dynamic_viscosity,
            thermal_conductivity,
            pressure,
            specific_volume,
            temperature,
        )

    def _determine_phase_and_quality(
        self, temperature: float, pressure: float
    ) -> tuple[str, float]:
        """Determine phase and steam quality"""
        from ._property_backends import determine_phase_and_quality

        return determine_phase_and_quality(self.water, temperature, pressure)

    def _calculate_saturated_coolprop_from_temp(
        self, temperature: float
    ) -> SteamProperties:
        """Calculate saturated steam properties from temperature using CoolProp"""
        return calculate_saturated_coolprop_from_temp(temperature)

    def _calculate_saturated_coolprop_from_pressure(
        self, pressure: float
    ) -> SteamProperties:
        """Calculate saturated steam properties from pressure using CoolProp"""
        return calculate_saturated_coolprop_from_pressure(pressure)

    def _calculate_saturated_cantera_from_temp(
        self, temperature: float
    ) -> SteamProperties:
        """Calculate saturated steam properties from temperature using Cantera"""
        return calculate_saturated_cantera_from_temp(self.water, temperature)

    def _calculate_saturated_cantera_from_pressure(
        self, pressure: float
    ) -> SteamProperties:
        """Calculate saturated steam properties from pressure using Cantera"""
        return calculate_saturated_cantera_from_pressure(self.water, pressure)

    def _calculate_saturated_simplified_from_temp(
        self, temperature: float
    ) -> SteamProperties:
        """Calculate saturated steam properties from temperature using simplified correlations"""
        return calculate_saturated_simplified_from_temp(temperature)

    def _calculate_saturated_simplified_from_pressure(
        self, pressure: float
    ) -> SteamProperties:
        """Calculate saturated steam properties from pressure using simplified correlations"""
        return calculate_saturated_simplified_from_pressure(pressure)

    def _antoine_equation(self, temperature_c: float) -> float:
        """Antoine equation for water vapor pressure (valid 1-100°C)"""
        from ._vapor_pressure import _antoine_equation

        return _antoine_equation(temperature_c)

    def _buck_equation(self, temperature_c: float) -> float:
        """
        Buck equation for water vapor pressure (improved accuracy).
        """
        from ._vapor_pressure import _buck_equation

        return _buck_equation(temperature_c)

    def _iapws_equation(self, temperature_c: float) -> float:
        """IAPWS-IF97 formulation for high-accuracy vapor pressure"""
        from ._vapor_pressure import _iapws_equation

        return _iapws_equation(temperature_c)
