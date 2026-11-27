# TRACKED_TASK: see #2310 — architecture debt extraction schedule

"""
Syngas Water Calculator
=======================

Core logic for calculating water content in saturated syngas.
Provides calculation methods without GUI dependencies.
"""

from __future__ import annotations  # noqa: E402, F404

import logging  # noqa: E402
import math  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from datetime import datetime  # noqa: E402
from typing import Any, Final  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.shared.python.contracts import require, require_positive  # noqa: E402
from src.shared.python.sidekick.utils.unit_constants import R_UNIVERSAL  # noqa: E402

from .constants import (  # noqa: E402
    ANTOINE_WATER_A,
    ANTOINE_WATER_B,
    ANTOINE_WATER_C,
    BUCK_ABOVE_FREEZING_A,
    BUCK_ABOVE_FREEZING_B,
    BUCK_ABOVE_FREEZING_C,
    BUCK_ABOVE_FREEZING_D,
    CELSIUS_TO_KELVIN_OFFSET,
    IAPWS_COEFFICIENTS,
    IAPWS_CRITICAL_TEMP,
    MW_SYNGAS_TYPICAL_GMOL,
    MW_WATER_GMOL,
    NORMAL_PRESSURE_PA,
    NORMAL_TEMPERATURE_K,
    WATER_VAPOR_A,
    WATER_VAPOR_B,
    WATER_VAPOR_C,
    WATER_VAPOR_D,
)
from .water_vapor_pressure import (  # noqa: E402
    _EXP_MAX_ARG,
    antoine_pressure_pa,
    buck_pressure_pa,
    iapws_pressure_pa,
    magnus_pressure_pa,
    safe_exp,
)

STANDARD_BASIS: Final[str] = "60F_14.696psia"
STANDARD_MOLAR_VOLUME_SCF_LBMOL: Final[float] = 379.49

__all__ = [
    "SYNGAS_PRESETS",
    "SyngasComposition",
    "SyngasWaterCalculator",
    "WaterContentResult",
    "estimate_condensation_risk",
    "quick_water_content",
]

# Backward-compatible private aliases.  ``_safe_exp`` and ``_EXP_MAX_ARG`` were
# historically imported from this module by callers/tests before the
# correlations were lifted into the shared water_vapor_pressure kernel.
_safe_exp = safe_exp
_exp_max_arg = _EXP_MAX_ARG

_logger = logging.getLogger(__name__)


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


@dataclass(frozen=True)
class WaterContentUnits:
    """Water content across engineering unit systems."""

    ppmv: float
    g_m3: float
    mg_nm3: float
    lb_mmscf: float
    standard_basis: str = STANDARD_BASIS

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ppmv": self.ppmv,
            "g_m3": self.g_m3,
            "mg_nm3": self.mg_nm3,
            "lb_mmscf": self.lb_mmscf,
            "standard_basis": self.standard_basis,
        }


def water_content_units(
    y_water: float,
    p_pa: float,
    t_k: float,
    mw_water: float = MW_WATER_GMOL,
) -> WaterContentUnits:
    """Convert water mole fraction to various engineering unit systems.

    Physics & Unit Formulas (#6004):
    - ppmv = y_water * 1e6
    - water_content_g_m3 = (y_water * p_pa * mw_water) / (R_UNIVERSAL * t_k)
    - water_content_mg_nm3 = (y_water * NORMAL_PRESSURE_PA * mw_water)
      / (R_UNIVERSAL * NORMAL_TEMPERATURE_K) * 1000.0
    - water_content_lb_mmscf = ppmv * mw_water / STANDARD_MOLAR_VOLUME_SCF_LBMOL
      (on the 60F_14.696psia standard gas molar volume basis 379.49 scf/lbmol)
    """
    if not (
        isinstance(y_water, (int, float))
        and math.isfinite(y_water)
        and 0.0 <= y_water <= 1.0
    ):
        raise ValueError("y_water must be a finite float between 0.0 and 1.0")
    if not (isinstance(p_pa, (int, float)) and math.isfinite(p_pa) and p_pa > 0.0):
        raise ValueError("p_pa must be a positive finite float")
    if not (isinstance(t_k, (int, float)) and math.isfinite(t_k) and t_k > 0.0):
        raise ValueError("t_k must be a positive finite float")
    if not (
        isinstance(mw_water, (int, float))
        and math.isfinite(mw_water)
        and mw_water > 0.0
    ):
        raise ValueError("mw_water must be a positive finite float")

    ppmv = y_water * 1e6
    g_m3 = (y_water * p_pa * mw_water) / (R_UNIVERSAL * t_k)
    mg_nm3 = (
        (y_water * NORMAL_PRESSURE_PA * mw_water)
        / (R_UNIVERSAL * NORMAL_TEMPERATURE_K)
        * 1000.0
    )
    lb_mmscf = ppmv * mw_water / STANDARD_MOLAR_VOLUME_SCF_LBMOL
    return WaterContentUnits(
        ppmv=ppmv,
        g_m3=g_m3,
        mg_nm3=mg_nm3,
        lb_mmscf=lb_mmscf,
        standard_basis=STANDARD_BASIS,
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
    standard_basis: str = STANDARD_BASIS
    timestamp: datetime = field(default_factory=datetime.now)
    warnings: list[str] = field(default_factory=list)

    @property
    def water_fraction(self) -> float:
        """Compatibility alias for legacy consumers of the result object."""
        return self.mole_fraction_water

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
                "standard_basis": self.standard_basis,
                "dew_point_c": self.dew_point_c,
                "dew_point_margin_c": self.dew_point_margin_c,
            },
            "vapor_pressure": {
                "pa": self.vapor_pressure_pa,
                "bar": self.vapor_pressure_bar,
            },
            "method": self.calculation_method,
            "standard_basis": self.standard_basis,
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

        _logger.info("SyngasWaterCalculator initialized")

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
        if temperature_c is None:
            raise ValueError("temperature_c must be provided")
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
        """Antoine equation for vapor pressure (delegates to shared kernel)."""
        return float(
            antoine_pressure_pa(
                self.antoine_constants["A"],
                self.antoine_constants["B"],
                self.antoine_constants["C"],
                temperature_c,
            )
        )

    def _buck_equation(self, temperature_c: float) -> float:
        """Buck equation for improved accuracy at moderate temperatures."""
        if temperature_c is None:
            raise ValueError("temperature_c must be provided")
        if temperature_c >= 0:
            # Above freezing. buck_pressure_pa evaluates
            # a * exp((b - t/d) * t / (c + t)); the Buck (1981) reference
            # curve these constants are sourced from is
            # a * exp((b - t/c) * t / (d + t)), so c and d must be swapped
            # at this call site (matching calculators.thermo.steam_engine's
            # call to the same kernel) to get the correct curve (issue #3867).
            a, b, c, d = (
                BUCK_ABOVE_FREEZING_A,
                BUCK_ABOVE_FREEZING_B,
                BUCK_ABOVE_FREEZING_D,
                BUCK_ABOVE_FREEZING_C,
            )
        else:
            # Below freezing
            a, b, c, d = WATER_VAPOR_A, WATER_VAPOR_B, WATER_VAPOR_C, WATER_VAPOR_D
        return float(buck_pressure_pa(a, b, c, d, temperature_c))

    def _iapws_equation(self, temperature_c: float) -> float:
        """Calculate vapor pressure using IAPWS-IF97 formulation.

        High-accuracy vapor pressure calculation using the International
        Association for the Properties of Water and Steam formulation.
        Delegates to the shared :func:`iapws_pressure_pa` kernel.

        Args:
            temperature_c: Temperature in Celsius

        Returns:
            Vapor pressure in Pa

        Raises:
            ValueError: If temperature is outside valid range (0.01°C to 373.946°C)
        """
        return float(iapws_pressure_pa(temperature_c))

    def _magnus_equation(self, temperature_c: float) -> float:
        """Magnus equation for vapor pressure (very accurate for 0-100°C)."""
        return float(magnus_pressure_pa(temperature_c))

    def _init_vapor_pressure_table(self) -> None:
        """Init Vapor Pressure Table method."""
        T_range = np.linspace(273.15, 647.0, 1000)
        P_sat_range = []
        for T in T_range:
            try:
                P_sat = self._iapws_equation(T - 273.15)
                P_sat_range.append(P_sat)
            except (ValueError, OverflowError) as vapor_pressure_error:
                _logger.debug(
                    "Skipping vapor pressure calculation at %s K: %s",
                    T,
                    vapor_pressure_error,
                )
                P_sat_range.append(np.nan)
        from scipy.interpolate import interp1d  # lazy import

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
        if temperature_k is None:
            raise ValueError("temperature_k must be provided")
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
        require_positive(partial_pressure_pa, "partial_pressure_pa")
        require_positive(total_pressure_pa, "total_pressure_pa")
        p_kpa = partial_pressure_pa / 1000

        # Use Magnus equation inverse for accurate initial guess across all
        # temperature regimes.
        gamma = math.log(max(1e-12, partial_pressure_pa) / 610.94)
        denom = 17.625 - gamma
        T_guess = (243.04 * gamma / denom) if abs(denom) > 1e-6 else 100.0

        # Newton-Raphson iteration for dew point using Buck equation
        for _ in range(15):
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

            step = (p_calc - p_kpa) / dp_dT
            # Damped Newton step to avoid wild overshoots
            if abs(step) > 20.0:
                step = 20.0 if step > 0 else -20.0
            T_guess = T_guess - step

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
        # DbC preconditions
        require_positive(pressure_bar, "pressure_bar")
        require(
            temperature_c + CELSIUS_TO_KELVIN_OFFSET > 0,
            "temperature must yield a positive Kelvin value",
            temperature_c,
        )
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
            y_water=y_water,
            vapor_pressure_pa=vapor_pressure_pa,
            temperature_k=temperature_k,
            pressure_pa=pressure_pa,
        )

        # Calculate dew point
        if vapor_pressure_pa >= pressure_pa:
            dew_point_c = self.calculate_dew_point(pressure_pa, pressure_pa)
            dew_point_margin_c = temperature_c - dew_point_c
        else:
            dew_point_c = temperature_c
            dew_point_margin_c = 0.0

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
            water_content_g_per_m3=float(units["g_m3"]),
            water_content_mg_per_nm3=float(units["mg_nm3"]),
            water_content_ppmv=float(units["ppmv"]),
            water_content_lb_per_mmscf=float(units["lb_mmscf"]),
            dew_point_c=dew_point_c,
            dew_point_margin_c=dew_point_margin_c,
            relative_humidity=relative_humidity,
            calculation_method=method_used,
            standard_basis=str(units.get("standard_basis", STANDARD_BASIS)),
            warnings=warnings,
        )

    def _convert_water_content_units(
        self,
        y_water: float,
        vapor_pressure_pa: float,
        temperature_k: float,
        pressure_pa: float | None = None,
    ) -> dict[str, Any]:
        """Convert water mole fraction to various engineering unit systems."""
        if y_water is None:
            raise ValueError("y_water must be provided")
        p_pa = (
            pressure_pa
            if pressure_pa is not None
            else (
                (vapor_pressure_pa / y_water) if y_water > 0.0 else NORMAL_PRESSURE_PA
            )
        )
        u = water_content_units(
            y_water=y_water,
            p_pa=p_pa,
            t_k=temperature_k,
            mw_water=self.mw_water,
        )
        return {
            "g_m3": u.g_m3,
            "mg_nm3": u.mg_nm3,
            "ppmv": u.ppmv,
            "lb_mmscf": u.lb_mmscf,
            "standard_basis": u.standard_basis,
        }

    def _calculate_mixture_mw(self, composition: SyngasComposition) -> float:
        """Calculate molecular weight of dry gas mixture"""
        if composition is None:
            raise ValueError("composition must be provided")
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
        if pressure_bar is None:
            raise ValueError("pressure_bar must be provided")
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
    if temperature_c is None:
        raise ValueError("temperature_c must be provided")
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
    if temperature_c is None:
        raise ValueError("temperature_c must be provided")
    calc = SyngasWaterCalculator()
    result = calc.calculate_water_content(temperature_c, pressure_bar)

    risk_level = "Low"
    if result.dew_point_margin_c < 0 or temperature_c < 0:
        risk_level = "Critical - Condensation occurring"
    elif result.dew_point_margin_c < safety_margin_c:
        risk_level = "High"
    elif result.dew_point_margin_c < safety_margin_c * 2:
        risk_level = "Medium"

    return {
        "dew_point_c": result.dew_point_c,
        "temperature_margin_c": result.dew_point_margin_c,
        "condensation_risk": risk_level,
        "condensation_occurring": result.dew_point_margin_c < 0 or temperature_c < 0,
        "recommended_temperature_c": result.dew_point_c + safety_margin_c,
    }
