"""
Syngas Water Calculator
=======================

Core logic for calculating water content in saturated syngas.
Provides calculation methods without GUI dependencies.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from .constants import (
    ANTOINE_WATER_A,
    ANTOINE_WATER_B,
    ANTOINE_WATER_C,
    BUCK_ABOVE_FREEZING_A,
    BUCK_ABOVE_FREEZING_B,
    BUCK_ABOVE_FREEZING_C,
    BUCK_ABOVE_FREEZING_D,
    CELSIUS_TO_KELVIN_OFFSET,
    IAPWS_COEFFICIENTS,
    IAPWS_CRITICAL_PRESSURE,
    IAPWS_CRITICAL_TEMP,
    IAPWS_TRIPLE_POINT_TEMP,
    KG_M3_TO_LB_FT3,
    MAGNUS_A,
    MAGNUS_B,
    MAGNUS_C,
    MMHG_TO_PA_CONV,
    MW_SYNGAS_TYPICAL_GMOL,
    MW_WATER_GMOL,
    NORMAL_PRESSURE_PA,
    NORMAL_TEMPERATURE_K,
    R_GAS_DENSITY,
    WATER_VAPOR_A,
    WATER_VAPOR_B,
    WATER_VAPOR_C,
    WATER_VAPOR_D,
)

# Maximum exponent for safe float64 exp() calls.  math.exp(709) is finite
# but math.exp(710) overflows.  We use 700 as a conservative upper bound.
_EXP_MAX_ARG: float = 700.0


def _safe_exp(x: float) -> float:
    """Compute exp(x) with clamping to prevent overflow.

    For x > _EXP_MAX_ARG the result is clamped to exp(_EXP_MAX_ARG) which is
    approximately 1.01e+304.  For x < -_EXP_MAX_ARG the result is clamped to
    exp(-_EXP_MAX_ARG) which is effectively 0.

    This avoids ``RuntimeWarning: overflow encountered in exp`` when extreme
    temperature values are passed through the Buck, Magnus, or IAPWS
    equations.

    Precondition:
        x must be a finite float (no NaN / inf).
    Postcondition:
        Return value is a finite, non-negative float.
    """
    clamped = max(-_EXP_MAX_ARG, min(x, _EXP_MAX_ARG))
    return math.exp(clamped)


logger = logging.getLogger(__name__)


@dataclass
class SyngasComposition:
    h2: float = 0.0
    co: float = 0.0
    co2: float = 0.0
    ch4: float = 0.0
    n2: float = 0.0
    ar: float = 0.0
    h2o: float = 0.0
    other: float = 0.0
    name: str = ""

    def normalize(self) -> SyngasComposition:
        """Normalize composition to sum to 1.0."""
        total = (
            self.h2
            + self.co
            + self.co2
            + self.ch4
            + self.n2
            + self.ar
            + self.h2o
            + self.other
        )
        if total > 0:
            return SyngasComposition(
                h2=self.h2 / total,
                co=self.co / total,
                co2=self.co2 / total,
                ch4=self.ch4 / total,
                n2=self.n2 / total,
                ar=self.ar / total,
                h2o=self.h2o / total,
                other=self.other / total,
                name=self.name,
            )
        return self

    def to_dict(self) -> dict[str, float]:
        """Convert composition to dictionary format."""
        return {
            "H2": self.h2,
            "CO": self.co,
            "CO2": self.co2,
            "CH4": self.ch4,
            "N2": self.n2,
            "AR": self.ar,
            "H2O": self.h2o,
            "Other": self.other,
        }

    @property
    def total(self) -> float:
        """Total mole fraction (should be 1.0 for dry basis)"""
        return (
            self.h2
            + self.co
            + self.co2
            + self.ch4
            + self.n2
            + self.ar
            + self.h2o
            + self.other
        )


@dataclass
class WaterContentResult:
    """Comprehensive water content calculation results"""

    # Input conditions
    temperature_c: float
    temperature_k: float
    pressure_bar: float
    pressure_pa: float
    gas_composition: str

    # Vapor pressure results
    vapor_pressure_pa: float
    vapor_pressure_bar: float
    saturation_temperature_c: float  # At given pressure

    # Water content in various units
    mole_fraction_water: float
    mass_fraction_water: float
    water_content_g_per_m3: float  # At actual conditions
    water_content_mg_per_nm3: float  # At normal conditions (0°C, 1 atm)
    water_content_ppmv: float  # Parts per million by volume
    water_content_lb_per_mmscf: float  # Pounds per million standard cubic feet

    # Dewpoint and related
    dew_point_c: float  # At given pressure
    dew_point_margin_c: float  # Temperature above dew point
    relative_humidity: float  # If below saturation

    # Additional info
    calculation_method: str
    timestamp: datetime = field(default_factory=datetime.now)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "input": {
                "temperature_c": self.temperature_c,
                "pressure_bar": self.pressure_bar,
                "gas_composition": self.gas_composition,
            },
            "results": {
                "water_mole_fraction": self.mole_fraction_water,
                "water_content_mg_per_nm3": self.water_content_mg_per_nm3,
                "water_content_ppmv": self.water_content_ppmv,
                "water_content_g_per_m3": self.water_content_g_per_m3,
                "water_content_lb_per_mmscf": self.water_content_lb_per_mmscf,
                "dew_point_c": self.dew_point_c,
                "dew_point_margin_c": self.dew_point_margin_c,
            },
            "vapor_pressure": {
                "pa": self.vapor_pressure_pa,
                "bar": self.vapor_pressure_bar,
            },
            "method": self.calculation_method,
            "warnings": self.warnings,
        }


# Predefined syngas compositions
SYNGAS_PRESETS = {
    "typical_syngas": SyngasComposition(
        h2=0.30, co=0.30, co2=0.15, ch4=0.05, n2=0.18, ar=0.02, name="Typical Syngas"
    ),
    "biomass_syngas": SyngasComposition(
        h2=0.20,
        co=0.20,
        co2=0.25,
        ch4=0.08,
        n2=0.25,
        ar=0.02,
        name="Biomass Gasification",
    ),
    "coal_syngas": SyngasComposition(
        h2=0.35, co=0.40, co2=0.10, ch4=0.02, n2=0.12, ar=0.01, name="Coal Gasification"
    ),
    "natural_gas_reforming": SyngasComposition(
        h2=0.75,
        co=0.15,
        co2=0.08,
        ch4=0.01,
        n2=0.01,
        ar=0.00,
        name="Natural Gas Reforming",
    ),
    "custom": SyngasComposition(name="Custom Composition"),
}


class SyngasWaterCalculator:
    """
    Core calculator for water content in syngas

    This class provides all calculation methods without GUI dependencies.
    Can be used programmatically for process simulations and integrations.
    """

    def __init__(self) -> None:
        """Initialize the calculator with correlation constants"""
        # Antoine equation constants for water
        self.antoine_constants = {
            "A": ANTOINE_WATER_A,
            "B": ANTOINE_WATER_B,
            "C": ANTOINE_WATER_C,
        }

        # Buck equation constants (improved accuracy)
        self.buck_constants = {
            "a": BUCK_ABOVE_FREEZING_A,
            "b": BUCK_ABOVE_FREEZING_B,
            "c": BUCK_ABOVE_FREEZING_C,
            "d": BUCK_ABOVE_FREEZING_D,
        }

        # IAPWS-IF97 constants for high accuracy
        self.iapws_constants = {
            "Tc": IAPWS_CRITICAL_TEMP,  # Critical temperature (K)
            "Pc": IAPWS_CRITICAL_TEMP / 29.356,  # Critical pressure (MPa) - derived
            "coefficients": IAPWS_COEFFICIENTS,
        }

        # Molecular weights
        self.mw_water = MW_WATER_GMOL  # g/mol
        self.mw_syngas_typical = MW_SYNGAS_TYPICAL_GMOL  # g/mol (approximate)

        # Fast interpolation table for vapor pressure
        self._init_vapor_pressure_table()

        logger.info("SyngasWaterCalculator initialized")

    def calculate_vapor_pressure(
        self, temperature_c: float, method: str = "buck"
    ) -> tuple[float, str]:
        """
        Calculate water vapor pressure using specified method

        Args:
            temperature_c: Temperature in Celsius
            method: 'antoine', 'buck', 'iapws', or 'magnus'

        Returns:
            Tuple of (vapor_pressure_pa, method_used)
        """
        if method == "antoine":
            return self._antoine_equation(temperature_c), "Antoine Equation"
        if method == "buck":
            return self._buck_equation(temperature_c), "Buck Equation"
        if method == "iapws":
            return self._iapws_equation(temperature_c), "IAPWS-IF97"
        if method == "magnus":
            return self._magnus_equation(temperature_c), "Magnus Equation"
        # Auto-select based on temperature
        if 0 <= temperature_c <= 100:
            return self._magnus_equation(temperature_c), "Magnus Equation (auto)"
        if -20 <= temperature_c <= 100:
            return self._buck_equation(temperature_c), "Buck Equation (auto)"
        if 100 < temperature_c <= 374:
            return self._iapws_equation(temperature_c), "IAPWS-IF97 (auto)"
        return self._antoine_equation(temperature_c), "Antoine Equation (auto)"

    def _antoine_equation(self, temperature_c: float) -> float:
        """Antoine equation for vapor pressure"""
        A, B, C = (
            self.antoine_constants["A"],
            self.antoine_constants["B"],
            self.antoine_constants["C"],
        )
        log10_p_mmhg = A - B / (C + temperature_c)
        # Convert log10 to natural log and use safe exp to prevent overflow.
        p_mmhg = _safe_exp(log10_p_mmhg * math.log(10))
        return p_mmhg * MMHG_TO_PA_CONV  # Convert to Pa

    def _buck_equation(self, temperature_c: float) -> float:
        """Buck equation for improved accuracy at moderate temperatures"""
        if temperature_c >= 0:
            # Above freezing
            a, b, c, d = (
                BUCK_ABOVE_FREEZING_A,
                BUCK_ABOVE_FREEZING_B,
                BUCK_ABOVE_FREEZING_C,
                BUCK_ABOVE_FREEZING_D,
            )
        else:
            # Below freezing
            a, b, c, d = WATER_VAPOR_A, WATER_VAPOR_B, WATER_VAPOR_C, WATER_VAPOR_D

        exponent = (b - temperature_c / d) * temperature_c / (c + temperature_c)
        p_kpa = a * _safe_exp(exponent)
        return p_kpa * 1000  # Convert to Pa

    def _iapws_equation(self, temperature_c: float) -> float:
        """Calculate vapor pressure using IAPWS-IF97 formulation.

        High-accuracy vapor pressure calculation using the International
        Association for the Properties of Water and Steam formulation.

        Args:
            temperature_c: Temperature in Celsius

        Returns:
            Vapor pressure in Pa

        Raises:
            ValueError: If temperature is outside valid range (0.01°C to 373.946°C)
        """
        # Use the IAPWS-IF97 formulation for high-accuracy vapor pressure
        T = temperature_c + CELSIUS_TO_KELVIN_OFFSET
        Tc = IAPWS_CRITICAL_TEMP
        Pc = IAPWS_CRITICAL_PRESSURE
        if T < IAPWS_TRIPLE_POINT_TEMP or Tc < T:
            msg = "Temperature out of IAPWS-IF97 range"
            raise ValueError(msg)
        theta = 1 - T / Tc
        a = IAPWS_COEFFICIENTS
        lnP = (
            Tc
            / T
            * (
                a[0] * theta
                + a[1] * theta**1.5
                + a[2] * theta**3
                + a[3] * theta**3.5
                + a[4] * theta**4
                + a[5] * theta**7.5
            )
        )
        return Pc * _safe_exp(lnP)

    def _magnus_equation(self, temperature_c: float) -> float:
        """Magnus equation for vapor pressure (very accurate for 0-100°C)"""
        # Magnus equation: P = 6.1094 * exp(17.625 * T / (T + 243.04))
        # P in hPa, T in °C
        # This is very accurate for temperatures 0-100°C

        if temperature_c < 0 or temperature_c > 100:
            msg = f"Magnus equation valid for 0°C to 100°C, got {temperature_c}°C"
            raise ValueError(msg)

        exponent = MAGNUS_B * temperature_c / (temperature_c + MAGNUS_C)
        p_hpa = MAGNUS_A * _safe_exp(exponent)

        # Convert hPa to Pa
        return p_hpa * 100

    def _init_vapor_pressure_table(self) -> None:
        """Init Vapor Pressure Table method."""
        T_range = np.linspace(273.15, 647.0, 1000)
        P_sat_range = []
        for T in T_range:
            try:
                P_sat = self._iapws_equation(T - 273.15)
                P_sat_range.append(P_sat)
            except (ValueError, OverflowError) as vapor_pressure_error:
                logger.debug(
                    "Skipping vapor pressure calculation at %s K: %s",
                    T,
                    vapor_pressure_error,
                )
                P_sat_range.append(np.nan)
        self.vapor_pressure_table = interp1d(
            T_range, P_sat_range, bounds_error=False, fill_value=np.nan
        )

    def vapor_pressure_fast(self, temperature_k: float) -> float:
        """Fast vapor pressure lookup using pre-computed interpolation table.

        Args:
            temperature_k: Temperature in Kelvin

        Returns:
            Vapor pressure in Pa
        """
        if not hasattr(self, "vapor_pressure_table"):
            self._init_vapor_pressure_table()
        return float(self.vapor_pressure_table(temperature_k))

    def calculate_dew_point(
        self, partial_pressure_pa: float, total_pressure_pa: float
    ) -> float:
        """
        Calculate dew point temperature

        Args:
            partial_pressure_pa: Water partial pressure in Pa
            total_pressure_pa: Total pressure in Pa

        Returns:
            Dew point temperature in Celsius
        """
        # Use Buck equation inverse
        p_kpa = partial_pressure_pa / 1000

        # Newton-Raphson iteration for dew point
        T_guess = 20.0  # Initial guess
        for _ in range(10):
            p_calc = self._buck_equation(T_guess) / 1000
            if abs(p_calc - p_kpa) < 0.001:
                break

            # Derivative approximation
            dp_dT = (
                (
                    self._buck_equation(T_guess + 0.1)
                    - self._buck_equation(T_guess - 0.1)
                )
                / 0.2
                / 1000
            )

            if dp_dT == 0:
                break

            T_guess = T_guess - (p_calc - p_kpa) / dp_dT

        return T_guess

    def calculate_water_content(
        self,
        temperature_c: float,
        pressure_bar: float,
        gas_composition: str | SyngasComposition = "typical_syngas",
        method: str = "auto",
    ) -> WaterContentResult:
        """
        Calculate comprehensive water content in syngas

        Args:
            temperature_c: Gas temperature in Celsius
            pressure_bar: Total pressure in bar
            gas_composition: Preset name or SyngasComposition object
            method: Vapor pressure calculation method

        Returns:
            WaterContentResult with all calculated values
        """
        # Get composition
        if isinstance(gas_composition, str):
            comp = SYNGAS_PRESETS.get(gas_composition, SYNGAS_PRESETS["typical_syngas"])
            comp_name = gas_composition
        else:
            comp = gas_composition
            comp_name = comp.name or "custom"

        # Convert units
        temperature_k = temperature_c + CELSIUS_TO_KELVIN_OFFSET
        pressure_pa = pressure_bar * 1e5

        # Calculate vapor pressure
        vapor_pressure_pa, method_used = self.calculate_vapor_pressure(
            temperature_c, method
        )
        vapor_pressure_bar = vapor_pressure_pa / 1e5

        # Check for warnings
        warnings = []
        if vapor_pressure_pa > pressure_pa:
            warnings.append(
                "Vapor pressure exceeds total pressure - condensation will occur"
            )
            vapor_pressure_pa = pressure_pa

        # Calculate water mole fraction
        y_water = vapor_pressure_pa / pressure_pa

        # Calculate mass fraction
        mw_dry_gas = self._calculate_mixture_mw(comp)
        x_water = (y_water * self.mw_water) / (
            y_water * self.mw_water + (1 - y_water) * mw_dry_gas
        )

        # Convert to various unit systems
        units = self._convert_water_content_units(
            y_water,
            vapor_pressure_pa,
            temperature_k,
        )

        # Calculate dew point
        dew_point_c = self.calculate_dew_point(vapor_pressure_pa, pressure_pa)
        dew_point_margin_c = temperature_c - dew_point_c

        # Relative humidity
        relative_humidity = min(
            (vapor_pressure_pa / self.calculate_vapor_pressure(temperature_c)[0]) * 100,
            100,
        )

        return WaterContentResult(
            temperature_c=temperature_c,
            temperature_k=temperature_k,
            pressure_bar=pressure_bar,
            pressure_pa=pressure_pa,
            gas_composition=comp_name,
            vapor_pressure_pa=vapor_pressure_pa,
            vapor_pressure_bar=vapor_pressure_bar,
            saturation_temperature_c=temperature_c,
            mole_fraction_water=y_water,
            mass_fraction_water=x_water,
            water_content_g_per_m3=units["g_m3"],
            water_content_mg_per_nm3=units["mg_nm3"],
            water_content_ppmv=units["ppmv"],
            water_content_lb_per_mmscf=units["lb_mmscf"],
            dew_point_c=dew_point_c,
            dew_point_margin_c=dew_point_margin_c,
            relative_humidity=relative_humidity,
            calculation_method=method_used,
            warnings=warnings,
        )

    def _convert_water_content_units(
        self,
        y_water: float,
        vapor_pressure_pa: float,
        temperature_k: float,
    ) -> dict[str, float]:
        """Convert water mole fraction to various engineering unit systems."""
        # Water content at actual conditions (g/m³)
        water_content_g_m3 = (
            vapor_pressure_pa * self.mw_water / (R_GAS_DENSITY * temperature_k)
        )

        # Water content at normal conditions (mg/Nm³) — 0°C, 1.01325 bar
        water_content_mg_nm3 = (
            y_water
            * self.mw_water
            * NORMAL_PRESSURE_PA
            / (R_GAS_DENSITY * NORMAL_TEMPERATURE_K)
            * 1e6
        )

        # Parts per million by volume
        water_content_ppmv = y_water * 1e6

        # Pounds per million standard cubic feet (US units) — 60°F, 14.696 psia
        water_content_lb_mmscf = water_content_mg_nm3 * KG_M3_TO_LB_FT3 / 1000

        return {
            "g_m3": water_content_g_m3,
            "mg_nm3": water_content_mg_nm3,
            "ppmv": water_content_ppmv,
            "lb_mmscf": water_content_lb_mmscf,
        }

    def _calculate_mixture_mw(self, composition: SyngasComposition) -> float:
        """Calculate molecular weight of dry gas mixture"""
        mw_components = {
            "h2": 2.016,
            "co": 28.01,
            "co2": 44.01,
            "ch4": 16.04,
            "n2": 28.014,
            "ar": 39.948,
        }

        mw = (
            composition.h2 * mw_components["h2"]
            + composition.co * mw_components["co"]
            + composition.co2 * mw_components["co2"]
            + composition.ch4 * mw_components["ch4"]
            + composition.n2 * mw_components["n2"]
            + composition.ar * mw_components["ar"]
        )

        return mw if mw > 0 else self.mw_syngas_typical

    def generate_water_content_curve(
        self,
        pressure_bar: float,
        temp_range: tuple[float, float] = (-20, 100),
        num_points: int = 50,
    ) -> pd.DataFrame:
        """
        Generate water content curve data for plotting

        Args:
            pressure_bar: System pressure in bar
            temp_range: Temperature range (min, max) in Celsius
            num_points: Number of calculation points

        Returns:
            DataFrame with temperature and water content data
        """
        temperatures = np.linspace(temp_range[0], temp_range[1], num_points)
        results = []

        for temp in temperatures:
            result = self.calculate_water_content(temp, pressure_bar)
            results.append(
                {
                    "temperature_c": temp,
                    "water_content_mg_nm3": result.water_content_mg_per_nm3,
                    "water_content_ppmv": result.water_content_ppmv,
                    "water_mole_fraction": result.mole_fraction_water,
                    "vapor_pressure_bar": result.vapor_pressure_bar,
                }
            )

        return pd.DataFrame(results)


# Convenience functions for quick calculations
def quick_water_content(temperature_c: float, pressure_bar: float) -> dict[str, float]:
    """
    Quick calculation of water content in typical syngas

    Args:
        temperature_c: Temperature in Celsius
        pressure_bar: Pressure in bar

    Returns:
        Dictionary with key results
    """
    calc = SyngasWaterCalculator()
    result = calc.calculate_water_content(temperature_c, pressure_bar)

    return {
        "water_content_mg_nm3": result.water_content_mg_per_nm3,
        "water_content_ppmv": result.water_content_ppmv,
        "dew_point_c": result.dew_point_c,
        "mole_fraction": result.mole_fraction_water,
    }


def estimate_condensation_risk(
    temperature_c: float, pressure_bar: float, safety_margin_c: float = 10.0
) -> dict[str, float | bool | str]:
    """
    Estimate risk of water condensation in syngas system

    Args:
        temperature_c: Operating temperature in Celsius
        pressure_bar: Operating pressure in bar
        safety_margin_c: Required temperature margin above dew point

    Returns:
        Dictionary with risk assessment
    """
    calc = SyngasWaterCalculator()
    result = calc.calculate_water_content(temperature_c, pressure_bar)

    risk_level = "Low"
    if result.dew_point_margin_c < 0:
        risk_level = "Critical - Condensation occurring"
    elif result.dew_point_margin_c < safety_margin_c:
        risk_level = "High"
    elif result.dew_point_margin_c < safety_margin_c * 2:
        risk_level = "Medium"

    return {
        "dew_point_c": result.dew_point_c,
        "temperature_margin_c": result.dew_point_margin_c,
        "condensation_risk": risk_level,
        "condensation_occurring": result.dew_point_margin_c < 0,
        "recommended_temperature_c": result.dew_point_c + safety_margin_c,
    }
