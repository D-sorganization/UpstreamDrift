"""
Steam Calculation Engine
========================

Pure Python steam property calculation engine, decoupled from UI.
Provides thermodynamic calculations using CoolProp, Cantera, or simplified correlations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

# Try to import Cantera
try:
    import cantera as ct

    CANTERA_AVAILABLE = True
except ImportError:
    CANTERA_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "Warning: Cantera not available. Steam calculator will use simplified calculations."
    )

# Try to import CoolProp (preferred high-accuracy backend)
try:
    from CoolProp.CoolProp import PhaseSI, PropsSI

    COOLPROP_AVAILABLE = True
except ImportError:
    COOLPROP_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "Warning: CoolProp not available. Falling back to Cantera or simplified correlations."
    )

logger = logging.getLogger(__name__)

# Water properties constants
STANDARD_ATMOSPHERIC_PRESSURE: float = 101325.0  # [Pa] Standard atmospheric pressure
BOILING_TEMPERATURE_WATER: float = 373.15  # [K] Boiling temperature of water at 1 atm
LIQUID_WATER_PRESSURE_THRESHOLD: float = (
    101325.0  # [Pa] Pressure threshold for liquid water
)
SPECIFIC_GAS_CONSTANT_WATER: float = (
    461.5  # [J/kg-K] Specific gas constant for water vapor
)
LIQUID_WATER_SPECIFIC_HEAT: float = 4186.0  # [J/kg-K] Specific heat of liquid water
VAPOR_ENTHALPY_REFERENCE: float = 2500.0  # [kJ/kg] Reference enthalpy for water vapor
VAPOR_ENTHALPY_SLOPE: float = 1.9  # [kJ/kg-K] Enthalpy slope for water vapor
VAPOR_SPECIFIC_HEAT_CP: float = (
    1.9  # [kJ/kg-K] Specific heat at constant pressure for vapor
)
VAPOR_SPECIFIC_HEAT_CV: float = (
    1.4  # [kJ/kg-K] Specific heat at constant volume for vapor
)

# Vapor entropy constants for simplified calculations
VAPOR_ENTROPY_REFERENCE: float = (
    8000.0  # [J/kg-K] Reference entropy for vapor calculations
)
VAPOR_ENTROPY_SLOPE: float = 2000.0  # [J/kg-K] Entropy slope for vapor calculations

# Temperature conversion constants
FAHRENHEIT_TO_CELSIUS_OFFSET: float = (
    32  # [°F] Temperature offset for Fahrenheit to Celsius
)
FAHRENHEIT_TO_CELSIUS_SCALE: float = 5 / 9  # [dimensionless] Scale factor for F to C
CELSIUS_TO_FAHRENHEIT_SCALE: float = 9 / 5  # [dimensionless] Scale factor for C to F

# UI range constants
MIN_TEMPERATURE_UI_K: float = 273.15  # [K] Minimum temperature for UI (0°C)
MAX_TEMPERATURE_UI_K: float = 647.15  # [K] Maximum temperature for UI (374°C)

# Numerical convergence parameters
NEWTON_RAPHSON_TOLERANCE: float = 1e-6  # [dimensionless] Tolerance for Newton-Raphson
NEWTON_RAPHSON_DERIVATIVE_TOLERANCE: float = 1e-10  # [dimensionless] Min derivative tol
NEWTON_RAPHSON_MAX_ITERATIONS: int = (
    10  # [dimensionless] Max iterations for Newton-Raphson
)
NEWTON_RAPHSON_STEP_SIZE: float = 0.1  # [°C] Step size for numerical derivative

# Antoine equation constants for water vapor pressure (valid 1-100°C)
# For saturation calculations, use different C value when temperature is in Kelvin
ANTOINE_A: float = 8.07131  # [dimensionless] Antoine equation constant A
ANTOINE_B: float = 1730.63  # [°C] Antoine equation constant B
ANTOINE_C_CELSIUS: float = 233.426  # [°C] Antoine C constant for temp in Celsius
ANTOINE_C_KELVIN: float = 39.724  # [K] Antoine C constant for temp in Kelvin

# Buck equation constants for water vapor pressure (improved accuracy)
BUCK_A: float = 6.1121  # [mbar] Buck equation constant A (or 0.61121 kPa)
BUCK_B: float = 18.678  # [dimensionless] Buck equation constant B
BUCK_C: float = 234.5  # [°C] Buck equation constant C
BUCK_D: float = 257.14  # [°C] Buck equation constant D

# Unit conversion constants
MMHG_TO_PASCAL_FACTOR: float = (
    133.322  # [Pa/mmHg] Conversion factor from mmHg to Pascal
)
PASCAL_TO_MMHG_FACTOR: float = (
    0.00750062  # [mmHg/Pa] Conversion factor from Pascal to mmHg
)
MBAR_TO_KPA_FACTOR: float = (
    10.0  # [mbar/kPa] Conversion factor from mbar to kPa (1 mbar = 0.1 kPa)
)
KPA_TO_PA_FACTOR: float = 1000.0  # [Pa/kPa] Conversion factor from kPa to Pascal
KELVIN_TO_CELSIUS_OFFSET: float = (
    273.15  # [K] Temperature offset for Kelvin to Celsius conversion
)

# Thermodynamic constants
CRITICAL_TEMPERATURE_WATER: float = 647.15  # [K] Critical temperature of water
CRITICAL_PRESSURE_WATER: float = 22.064e6  # [Pa] Critical pressure of water
TRIPLE_POINT_TEMPERATURE: float = 273.16  # [K] Triple point temperature of water
TRIPLE_POINT_PRESSURE: float = 611.657  # [Pa] Triple point pressure of water

# Default values
DEFAULT_DEW_POINT_TEMPERATURE_CELSIUS: float = (
    25.0  # [°C] Default dew point temperature
)
DEFAULT_QUALITY: float = 0.5  # [dimensionless] Default quality for two-phase region

# Fallback constants for error handling
FALLBACK_ATMOSPHERIC_PRESSURE: float = (
    101325.0  # [Pa] Standard atmospheric pressure (1 atm)
)
FALLBACK_BOILING_TEMPERATURE: float = (
    373.15  # [K] Boiling temperature of water at 1 atm (100°C)
)

# State selection constants
SUPERHEATED_STATE: int = 0  # Superheated steam (T & P) - dimensionless identifier
SATURATED_FROM_TEMP_STATE: int = (
    1  # Saturated steam from temperature - dimensionless identifier
)
SATURATED_FROM_PRESSURE_STATE: int = (
    2  # Saturated steam from pressure - dimensionless identifier
)


@dataclass
class SteamProperties:
    """Container for steam thermodynamic properties"""

    temperature: float  # K
    pressure: float  # Pa
    density: float  # kg/m³
    specific_volume: float  # m³/kg
    enthalpy: float  # J/kg
    entropy: float  # J/kg-K
    internal_energy: float  # J/kg
    cp: float  # J/kg-K (specific heat at constant pressure)
    cv: float  # J/kg-K (specific heat at constant volume)
    speed_of_sound: float  # m/s
    thermal_conductivity: float  # W/m-K
    dynamic_viscosity: float  # Pa·s
    kinematic_viscosity: float  # m²/s
    quality: float  # Steam quality (0-1, if applicable)
    phase: str  # 'liquid', 'vapor', 'two-phase'
    # --- new comprehensive fields ---
    compressibility_factor: float | None = None  # Dimensionless Z
    prandtl_number: float | None = None  # Dimensionless Pr
    specific_heat_ratio: float | None = None  # Cp/Cv (k)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for easy export"""
        return {
            "Temperature (K)": self.temperature,
            "Temperature (°C)": self.temperature - 273.15,
            "Pressure (Pa)": self.pressure,
            "Pressure (bar)": self.pressure / 1e5,
            "Density (kg/m³)": self.density,
            "Specific Volume (m³/kg)": self.specific_volume,
            "Enthalpy (J/kg)": self.enthalpy,
            "Enthalpy (kJ/kg)": self.enthalpy / 1000,
            "Entropy (J/kg-K)": self.entropy,
            "Internal Energy (J/kg)": self.internal_energy,
            "Cp (J/kg-K)": self.cp,
            "Cv (J/kg-K)": self.cv,
            "Speed of Sound (m/s)": self.speed_of_sound,
            "Thermal Conductivity (W/m-K)": self.thermal_conductivity,
            "Dynamic Viscosity (Pa·s)": self.dynamic_viscosity,
            "Kinematic Viscosity (m²/s)": self.kinematic_viscosity,
            "Quality": self.quality,
            "Phase": self.phase,
            "Compressibility Factor (Z)": self.compressibility_factor,
            "Prandtl Number": self.prandtl_number,
            "Cp/Cv (k)": self.specific_heat_ratio,
        }


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
            # Only log warning if strictly required? No, always warn.
            # But avoid spamming logs.
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
        if engine == "auto":
            if COOLPROP_AVAILABLE:
                return "coolprop"
            if CANTERA_AVAILABLE and self.water is not None:
                return "cantera"
            return "simplified"

        # Check if requested engine is available
        if engine == "coolprop" and COOLPROP_AVAILABLE:
            return "coolprop"
        if engine == "cantera" and CANTERA_AVAILABLE and self.water is not None:
            return "cantera"
        if engine == "simplified":
            return "simplified"

        # Fallback to best available if requested engine not available
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
        assert temperature > 0, f"Temperature must be positive (K), got {temperature}"
        assert pressure > 0, f"Pressure must be positive (Pa), got {pressure}"

        try:
            selected_engine = self._select_best_engine(engine)

            if selected_engine == "coolprop":
                result = self._calculate_coolprop_properties(temperature, pressure)
            elif selected_engine == "cantera":
                result = self._calculate_cantera_properties(temperature, pressure)
            else:
                result = self._calculate_simplified_properties(temperature, pressure)

        except (RuntimeError, ValueError, TypeError) as e:
            logger.exception("Steam calculation failed: %s", e)
            result = self._calculate_simplified_properties(temperature, pressure)

        # DbC postcondition: enthalpy and entropy should be finite
        assert np.isfinite(result.enthalpy), (
            f"Enthalpy must be finite, got {result.enthalpy}"
        )
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
                return self._calculate_saturated_coolprop_from_temp(temperature)
            if selected_engine == "cantera":
                return self._calculate_saturated_cantera_from_temp(temperature)
            return self._calculate_saturated_simplified_from_temp(temperature)

        except (RuntimeError, ValueError, TypeError) as e:
            logger.exception(
                "Saturated steam calculation from temperature failed: %s", e
            )
            return self._calculate_saturated_simplified_from_temp(temperature)

    def calculate_saturated_properties_from_pressure(
        self, pressure: float, engine: str = "auto"
    ) -> SteamProperties:
        """
        Calculate saturated steam properties from pressure
        """
        try:
            selected_engine = self._select_best_engine(engine)

            if selected_engine == "coolprop":
                return self._calculate_saturated_coolprop_from_pressure(pressure)
            if selected_engine == "cantera":
                return self._calculate_saturated_cantera_from_pressure(pressure)
            return self._calculate_saturated_simplified_from_pressure(pressure)

        except (RuntimeError, ValueError, TypeError) as e:
            logger.exception("Saturated steam calculation from pressure failed: %s", e)
            return self._calculate_saturated_simplified_from_pressure(pressure)

    def calculate_water_vapor_pressure(
        self, temperature: float, method: str = "buck"
    ) -> float:
        """
        Calculate water vapor pressure using various correlations
        """
        try:
            if method == "antoine":
                return self._antoine_equation(temperature)
            if method == "buck":
                return self._buck_equation(temperature)
            if method == "iapws":
                return self._iapws_equation(temperature)
            return self._buck_equation(temperature)  # Default to Buck
        except (RuntimeError, ValueError, TypeError) as e:
            logger.exception("Water vapor pressure calculation failed: %s", e)
            return self._antoine_equation(temperature)  # Fallback

    def _antoine_equation(self, temperature_c: float) -> float:
        """Antoine equation for water vapor pressure (valid 1-100°C)"""
        # Antoine equation: log10(P) = A - B/(C + T)
        # P in mmHg, T in °C
        log_p_mmhg = ANTOINE_A - ANTOINE_B / (ANTOINE_C_CELSIUS + temperature_c)
        p_mmhg = 10**log_p_mmhg

        # Convert to Pascal
        return p_mmhg * MMHG_TO_PASCAL_FACTOR

    def _buck_equation(self, temperature_c: float) -> float:
        """
        Buck equation for water vapor pressure (improved accuracy).
        """
        # Buck equation: P = a * exp((b - T/d) * T/(T + c))
        # P in kPa, T in °C
        # BUCK_A is stored in mbar, but Buck equation requires 'a' in kPa
        # Convert mbar to kPa by dividing by 10 (1 mbar = 0.1 kPa)
        a_kpa = BUCK_A / MBAR_TO_KPA_FACTOR
        p_kpa = a_kpa * np.exp(
            (BUCK_B - temperature_c / BUCK_D) * temperature_c / (temperature_c + BUCK_C)
        )

        # Convert kPa to Pascal
        return float(p_kpa * KPA_TO_PA_FACTOR)

    def _iapws_equation(self, temperature_c: float) -> float:
        """IAPWS-IF97 formulation for high-accuracy vapor pressure"""
        # Simplified IAPWS implementation
        # For high accuracy, use CoolProp if available
        if COOLPROP_AVAILABLE:
            try:
                temperature_k = temperature_c + 273.15
                return float(PropsSI("P", "T", temperature_k, "Q", 0, "Water"))
            except (ValueError, ZeroDivisionError, OverflowError, TypeError):
                # CoolProp not available, fall back to Buck equation
                pass  # Fall back to Buck equation

        # Fallback to Buck equation
        return self._buck_equation(temperature_c)

    def calculate_dew_point(
        self, partial_pressure_pa: float, total_pressure_pa: float
    ) -> float:
        """
        Calculate dew point temperature from partial pressure
        """
        try:
            # Use Newton-Raphson method to find temperature where vapor pressure equals
            # partial pressure
            def objective_function(T: float) -> float:
                """Objective function for dew point calculation."""
                return self.calculate_water_vapor_pressure(T) - partial_pressure_pa

            # Initial guess
            T_guess = DEFAULT_DEW_POINT_TEMPERATURE_CELSIUS

            # Simple Newton-Raphson iteration
            for _ in range(NEWTON_RAPHSON_MAX_ITERATIONS):
                f_val = objective_function(T_guess)
                if abs(f_val) < NEWTON_RAPHSON_TOLERANCE:
                    break

                # Numerical derivative
                f_plus = objective_function(T_guess + NEWTON_RAPHSON_STEP_SIZE)
                f_minus = objective_function(T_guess - NEWTON_RAPHSON_STEP_SIZE)
                df_dT = (f_plus - f_minus) / (2 * NEWTON_RAPHSON_STEP_SIZE)

                if abs(df_dT) < NEWTON_RAPHSON_DERIVATIVE_TOLERANCE:
                    break

                T_guess = T_guess - f_val / df_dT

                # Bounds check
                if T_guess < -50 or T_guess > 500:
                    break

            return T_guess

        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.exception("Dew point calculation failed: %s", e)
            return DEFAULT_DEW_POINT_TEMPERATURE_CELSIUS

    def _calculate_saturated_coolprop_from_temp(
        self, temperature: float
    ) -> SteamProperties:
        """Calculate saturated steam properties from temperature using CoolProp"""
        try:
            # Get saturation pressure from CoolProp
            pressure = PropsSI("P", "T", temperature, "Q", 1.0, "Water")

            # Calculate properties at saturation
            return self._calculate_coolprop_properties(temperature, pressure)

        except (RuntimeError, ValueError, TypeError) as e:
            logger.exception(
                "CoolProp saturated calculation from temperature failed: %s", e
            )
            return self._calculate_saturated_simplified_from_temp(temperature)

    def _calculate_saturated_coolprop_from_pressure(
        self, pressure: float
    ) -> SteamProperties:
        """Calculate saturated steam properties from pressure using CoolProp"""
        try:
            # Get saturation temperature from CoolProp
            temperature = PropsSI("T", "P", pressure, "Q", 1.0, "Water")

            # Calculate properties at saturation
            return self._calculate_coolprop_properties(temperature, pressure)

        except (RuntimeError, ValueError, TypeError) as e:
            logger.exception(
                "CoolProp saturated calculation from pressure failed: %s", e
            )
            return self._calculate_saturated_simplified_from_pressure(pressure)

    def _calculate_saturated_cantera_from_temp(
        self, temperature: float
    ) -> SteamProperties:
        """Calculate saturated steam properties from temperature using Cantera"""
        try:
            # Set state to saturated conditions at given temperature
            self.water.TQ = temperature, 1.0  # Saturated vapor

            # Get saturation pressure
            pressure = self.water.P

            # Calculate properties at saturation
            return self._calculate_cantera_properties(temperature, pressure)

        except (RuntimeError, ValueError, TypeError) as e:
            logger.exception(
                "Cantera saturated calculation from temperature failed: %s", e
            )
            return self._calculate_saturated_simplified_from_temp(temperature)

    def _calculate_saturated_cantera_from_pressure(
        self, pressure: float
    ) -> SteamProperties:
        """Calculate saturated steam properties from pressure using Cantera"""
        try:
            # Set state to saturated conditions at given pressure
            self.water.PQ = pressure, 1.0  # Saturated vapor

            # Get saturation temperature
            temperature = self.water.T

            # Calculate properties at saturation
            return self._calculate_cantera_properties(temperature, pressure)

        except (RuntimeError, ValueError, TypeError) as e:
            logger.exception(
                "Cantera saturated calculation from pressure failed: %s", e
            )
            return self._calculate_saturated_simplified_from_pressure(pressure)

    def _calculate_saturated_simplified_from_temp(
        self, temperature: float
    ) -> SteamProperties:
        """Calculate saturated steam properties from temperature using simplified correlations"""
        # Antoine equation for water vapor pressure (valid 1-100°C)
        # log10(P_mmHg) = A - B/(T_K - C) where C is for temperature in Kelvin
        temp_c = temperature - KELVIN_TO_CELSIUS_OFFSET

        if temp_c < 1.0:
            temp_c = 1.0
        elif temp_c > 374.0:  # Above critical temperature
            temp_c = 374.0

        # Calculate saturation pressure using Antoine equation
        log_p_mmhg = ANTOINE_A - ANTOINE_B / (temperature - ANTOINE_C_KELVIN)
        pressure_mmhg = 10**log_p_mmhg
        pressure = pressure_mmhg * MMHG_TO_PASCAL_FACTOR

        # Calculate properties at saturation
        return self._calculate_simplified_properties(temperature, pressure)

    def _calculate_saturated_simplified_from_pressure(
        self, pressure: float
    ) -> SteamProperties:
        """Calculate saturated steam properties from pressure using simplified correlations"""
        # Inverse Antoine equation to find temperature from pressure
        pressure_mmhg = pressure * PASCAL_TO_MMHG_FACTOR

        # Solve for temperature: T = B / (A - log10(P)) + C
        if pressure_mmhg <= 0:
            pressure_mmhg = 1.0

        log_p = np.log10(pressure_mmhg)
        temperature = ANTOINE_B / (ANTOINE_A - log_p) + ANTOINE_C_KELVIN

        # Ensure reasonable temperature range
        if temperature < 274.15:  # Below 1°C
            temperature = 274.15
        elif temperature > 647.15:  # Above critical temperature
            temperature = 647.15

        # Calculate properties at saturation
        return self._calculate_simplified_properties(temperature, pressure)

    def get_saturation_pressure(self, temperature: float) -> float:
        """Get saturation pressure for given temperature"""
        try:
            if CANTERA_AVAILABLE and self.water is not None and self.water:
                self.water.TQ = temperature, 1.0
                return float(self.water.P)
            # Use Antoine equation
            log_p_mmhg = ANTOINE_A - ANTOINE_B / (temperature - ANTOINE_C_KELVIN)
            pressure_mmhg = 10**log_p_mmhg
            return pressure_mmhg * MMHG_TO_PASCAL_FACTOR
        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.exception("Saturation pressure calculation failed: %s", e)
            return FALLBACK_ATMOSPHERIC_PRESSURE

    def get_saturation_temperature(self, pressure: float) -> float:
        """Get saturation temperature for given pressure"""
        try:
            if CANTERA_AVAILABLE and self.water is not None and self.water:
                self.water.PQ = pressure, 1.0
                return float(self.water.T)
            # Use inverse Antoine equation
            pressure_mmhg = pressure * PASCAL_TO_MMHG_FACTOR
            log_p = np.log10(pressure_mmhg)
            return float(ANTOINE_B / (ANTOINE_A - log_p) + ANTOINE_C_KELVIN)
        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.exception("Saturation temperature calculation failed: %s", e)
            return FALLBACK_BOILING_TEMPERATURE

    def _calculate_cantera_properties(
        self, temperature: float, pressure: float
    ) -> SteamProperties:
        """Calculate steam properties using Cantera"""
        if self.water is None:
            raise RuntimeError("Cantera water object is not initialized")

        try:
            # Set state
            self.water.TP = temperature, pressure

            # Calculate all properties
            density = self.water.density
            specific_volume = 1.0 / density
            enthalpy = self.water.enthalpy_mass
            entropy = self.water.entropy_mass
            internal_energy = self.water.int_energy_mass
            cp = self.water.cp_mass
            cv = self.water.cv_mass

            # Additional transport properties
            try:
                thermal_conductivity = self.water.thermal_conductivity
                dynamic_viscosity = self.water.viscosity
                kinematic_viscosity = dynamic_viscosity / density
            except (ValueError, ZeroDivisionError, OverflowError, TypeError):
                # Fallback values if transport properties not available
                thermal_conductivity = 0.6  # Approximate for water/steam
                dynamic_viscosity = 1e-6  # Approximate
                kinematic_viscosity = dynamic_viscosity / density

            # Speed of sound (approximate)
            try:
                speed_of_sound = np.sqrt(cp / cv * pressure / density)
            except (ValueError, ZeroDivisionError, OverflowError, TypeError):
                speed_of_sound = 1500.0  # Approximate for water

            # Determine phase and quality
            phase, quality = self._determine_phase_and_quality(temperature, pressure)

            # Derived advanced properties (approximations)
            R_specific = 461.5
            compressibility_factor = (
                pressure * (1 / density) / (R_specific * temperature)
            )
            specific_heat_ratio = cp / cv if cv else None
            prandtl_number = (
                (cp * dynamic_viscosity / thermal_conductivity)
                if thermal_conductivity
                else None
            )

            return SteamProperties(
                temperature=temperature,
                pressure=pressure,
                density=density,
                specific_volume=specific_volume,
                enthalpy=enthalpy,
                entropy=entropy,
                internal_energy=internal_energy,
                cp=cp,
                cv=cv,
                speed_of_sound=speed_of_sound,
                thermal_conductivity=thermal_conductivity,
                dynamic_viscosity=dynamic_viscosity,
                kinematic_viscosity=kinematic_viscosity,
                quality=quality,
                phase=phase,
                compressibility_factor=compressibility_factor,
                prandtl_number=prandtl_number,
                specific_heat_ratio=specific_heat_ratio,
            )

        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.exception("Cantera steam calculation failed: %s", e)
            return self._calculate_simplified_properties(temperature, pressure)

    def _calculate_coolprop_properties(
        self, temperature: float, pressure: float
    ) -> SteamProperties:
        """High-accuracy calculation using CoolProp"""
        try:
            self._validate_coolprop_inputs(temperature, pressure)

            density = PropsSI("D", "T", temperature, "P", pressure, "Water")
            specific_volume = 1.0 / density
            enthalpy = PropsSI("H", "T", temperature, "P", pressure, "Water")
            entropy = PropsSI("S", "T", temperature, "P", pressure, "Water")
            internal_energy = PropsSI("U", "T", temperature, "P", pressure, "Water")
            cp = PropsSI("Cpmass", "T", temperature, "P", pressure, "Water")
            cv = PropsSI("Cvmass", "T", temperature, "P", pressure, "Water")
            speed_of_sound = PropsSI("A", "T", temperature, "P", pressure, "Water")
            thermal_conductivity = PropsSI(
                "L", "T", temperature, "P", pressure, "Water"
            )
            dynamic_viscosity = PropsSI(
                "VISCOSITY", "T", temperature, "P", pressure, "Water"
            )
            kinematic_viscosity = dynamic_viscosity / density

            derived = self._compute_derived_properties(
                cp,
                cv,
                dynamic_viscosity,
                thermal_conductivity,
                pressure,
                specific_volume,
                temperature,
            )

            # Phase / quality determination via CoolProp
            try:
                phase_str = PhaseSI("T", temperature, "P", pressure, "Water")
            except (RuntimeError, ValueError):
                phase_str = "unknown"
            try:
                quality = PropsSI("Q", "T", temperature, "P", pressure, "Water")
                if np.isnan(quality):
                    quality = 0.0 if phase_str.lower() == "liquid" else 1.0
            except (ValueError, ZeroDivisionError, OverflowError, TypeError):
                quality = 0.0

            return SteamProperties(
                temperature=temperature,
                pressure=pressure,
                density=density,
                specific_volume=specific_volume,
                enthalpy=enthalpy,
                entropy=entropy,
                internal_energy=internal_energy,
                cp=cp,
                cv=cv,
                speed_of_sound=speed_of_sound,
                thermal_conductivity=thermal_conductivity,
                dynamic_viscosity=dynamic_viscosity,
                kinematic_viscosity=kinematic_viscosity,
                quality=quality,
                phase=phase_str,
                **derived,
            )
        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.exception("CoolProp steam calculation failed: %s", e)
            return self._calculate_simplified_properties(temperature, pressure)

    @staticmethod
    def _validate_coolprop_inputs(
        temperature: float,
        pressure: float,
    ) -> None:
        """Validate temperature and pressure for CoolProp calculations."""
        if temperature < TRIPLE_POINT_TEMPERATURE or temperature > 1000:
            msg = (
                f"Temperature {temperature} K is outside valid range "
                f"[{TRIPLE_POINT_TEMPERATURE}, 1000] K for CoolProp"
            )
            logger.error(msg)
            raise ValueError(msg)

        max_reasonable_pressure: float = 100e6
        if pressure < TRIPLE_POINT_PRESSURE or pressure > max_reasonable_pressure:
            msg = (
                f"Pressure {pressure} Pa is outside valid range "
                f"[{TRIPLE_POINT_PRESSURE}, {max_reasonable_pressure}] Pa for CoolProp. "
                f"Check unit conversion - this value seems too high."
            )
            logger.error(msg)
            raise ValueError(msg)

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
        r_specific = 461.5  # J/kg-K for water
        return {
            "compressibility_factor": (
                pressure * specific_volume / (r_specific * temperature)
            ),
            "prandtl_number": (
                (cp * dynamic_viscosity / thermal_conductivity)
                if thermal_conductivity
                else None
            ),
            "specific_heat_ratio": cp / cv if cv else None,
        }

    def _determine_phase_and_quality(
        self, temperature: float, pressure: float
    ) -> tuple[str, float]:
        """Determine phase and steam quality"""
        try:
            # Critical point properties for water
            T_critical = 647.1  # K
            P_critical = 22064000  # Pa (220.64 bar)

            if temperature > T_critical or pressure > P_critical:
                return "supercritical", 1.0

            # Saturation properties
            try:
                self.water.TQ = temperature, 0.0  # Saturated liquid
                P_sat = self.water.P

                if pressure > P_sat:
                    return "liquid", 0.0
                if abs(pressure - P_sat) / P_sat < 0.001:  # Close to saturation
                    return "two-phase", 0.5
                return "vapor", 1.0

            except (ValueError, ZeroDivisionError, OverflowError, TypeError):
                return "unknown", 0.0

        except (ValueError, ZeroDivisionError, OverflowError, TypeError):
            return "unknown", 0.0

    def _calculate_simplified_properties(
        self, temperature: float, pressure: float
    ) -> SteamProperties:
        """Simplified calculations based on ideal gas law and constant properties"""
        try:
            # Simple approximations for water/steam properties
            # These are rough estimates and should not be used for critical applications

            # Determine if liquid or vapor based on simple criteria
            T_sat_1atm = 373.15  # K
            if temperature < T_sat_1atm and pressure > 50000:  # Likely liquid
                density = 1000.0  # kg/m³
                enthalpy = 4186.0 * (
                    temperature - 273.15
                )  # Approximate liquid enthalpy
                entropy = 4186.0 * np.log(temperature / 273.15)  # Approximate
                cp = 4186.0  # J/kg-K
                cv = 4186.0  # J/kg-K
                phase = "liquid"
                quality = 0.0
            else:  # Likely vapor
                # Ideal gas approximation for steam
                density = pressure / (SPECIFIC_GAS_CONSTANT_WATER * temperature)
                enthalpy = VAPOR_ENTHALPY_REFERENCE + VAPOR_ENTHALPY_SLOPE * (
                    temperature - KELVIN_TO_CELSIUS_OFFSET
                )
                # Convert kJ/kg to J/kg
                enthalpy *= 1000

                # Simplified entropy
                entropy = VAPOR_ENTROPY_REFERENCE + VAPOR_ENTROPY_SLOPE * np.log(
                    temperature / 373.15
                )

                cp = VAPOR_SPECIFIC_HEAT_CP * 1000  # Convert to J/kg-K
                cv = VAPOR_SPECIFIC_HEAT_CV * 1000  # Convert to J/kg-K
                phase = "vapor"
                quality = 1.0

            specific_volume = 1.0 / density
            # Internal Energy: u = h - Pv
            internal_energy = enthalpy - pressure * specific_volume

            # Transport properties (approximate const)
            thermal_conductivity = 0.6 if phase == "liquid" else 0.025  # W/m-K
            dynamic_viscosity = 2.8e-4 if phase == "liquid" else 1.2e-5  # Pa·s
            kinematic_viscosity = dynamic_viscosity / density

            if phase == "liquid":
                speed_of_sound = 1500.0
            else:
                speed_of_sound = np.sqrt(1.3 * 461.5 * temperature)  # Steam gamma ~1.3

            return SteamProperties(
                temperature=temperature,
                pressure=pressure,
                density=density,
                specific_volume=specific_volume,
                enthalpy=enthalpy,
                entropy=entropy,
                internal_energy=internal_energy,
                cp=cp,
                cv=cv,
                speed_of_sound=speed_of_sound,
                thermal_conductivity=thermal_conductivity,
                dynamic_viscosity=dynamic_viscosity,
                kinematic_viscosity=kinematic_viscosity,
                quality=quality,
                phase=phase,
                compressibility_factor=pressure
                * specific_volume
                / (461.5 * temperature),
                prandtl_number=cp * dynamic_viscosity / thermal_conductivity,
                specific_heat_ratio=cp / cv if cv else None,
            )

        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            logger.exception("Simplified calculation failed: %s", e)
            # Return empty/zero properties on catastrophic failure
            return SteamProperties(
                temperature=temperature,
                pressure=pressure,
                density=0.0,
                specific_volume=0.0,
                enthalpy=0.0,
                entropy=0.0,
                internal_energy=0.0,
                cp=0.0,
                cv=0.0,
                speed_of_sound=0.0,
                thermal_conductivity=0.0,
                dynamic_viscosity=0.0,
                kinematic_viscosity=0.0,
                quality=0.0,
                phase="error",
            )
