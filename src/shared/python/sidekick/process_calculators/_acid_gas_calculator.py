from __future__ import annotations

import logging

import numpy as np
import pandas as pd

try:
    import thermo

    THERMO_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    thermo = None
    THERMO_AVAILABLE = False

try:
    from CoolProp.CoolProp import PropsSI

    COOLPROP_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    PropsSI = None
    COOLPROP_AVAILABLE = False

from ._acid_gas_models import AcidGasComposition, DewpointResult
from .constants import (
    ANTOINE_H2S_A,
    ANTOINE_H2S_B,
    ANTOINE_H2S_C,
    ANTOINE_HCL_A,
    ANTOINE_HCL_B,
    ANTOINE_HCL_C,
    ANTOINE_HF_A,
    ANTOINE_HF_B,
    ANTOINE_HF_C,
    ANTOINE_WATER_A,
    ANTOINE_WATER_B,
    ANTOINE_WATER_C,
    ANTOINE_WATER_HIGH_A,
    ANTOINE_WATER_HIGH_B,
    ANTOINE_WATER_HIGH_C,
    BAR_TO_PA,
    CELSIUS_TO_KELVIN_OFFSET,
    MMHG_TO_PA_CONV,
)

logger = logging.getLogger(__name__)


class AcidGasDewpointCalculator:
    """
    Core calculator for acid gas dewpoint predictions

    Based on established thermodynamic correlations and literature sources:
    - Perry's Chemical Engineers' Handbook (8th Ed.)
    - NIST Chemistry WebBook
    - CRC Handbook of Chemistry and Physics
    - Journal of Chemical & Engineering Data
    """

    def __init__(self) -> None:
        """Initialize calculator with thermodynamic constants"""

        # Antoine equation constants for acid gases
        # Source: Perry's Chemical Engineers' Handbook, 8th Ed.
        self.antoine_constants = {
            "H2O": {"A": ANTOINE_WATER_A, "B": ANTOINE_WATER_B, "C": ANTOINE_WATER_C},
            "HF": {"A": ANTOINE_HF_A, "B": ANTOINE_HF_B, "C": ANTOINE_HF_C},
            "HCl": {"A": ANTOINE_HCL_A, "B": ANTOINE_HCL_B, "C": ANTOINE_HCL_C},
            "H2S": {"A": ANTOINE_H2S_A, "B": ANTOINE_H2S_B, "C": ANTOINE_H2S_C},
        }

        # Literature sources for validation
        self.literature_sources = {
            "H2O": [
                "Perry's Chemical Engineers' Handbook, 8th Ed.",
                "NIST Chemistry WebBook",
                "IAPWS-IF97 Formulation",
            ],
            "HF": [
                "Perry's Chemical Engineers' Handbook, 8th Ed.",
                "CRC Handbook of Chemistry and Physics",
                "Journal of Chemical & Engineering Data, 2001",
            ],
            "HCl": [
                "Perry's Chemical Engineers' Handbook, 8th Ed.",
                "NIST Chemistry WebBook",
                "Industrial & Engineering Chemistry Research, 1995",
            ],
            "H2S": [
                "Perry's Chemical Engineers' Handbook, 8th Ed.",
                "NIST Chemistry WebBook",
                "Journal of Chemical & Engineering Data, 2003",
            ],
        }

        # Temperature and pressure limits for correlations
        self.validity_limits = {
            "H2O": {"T_min": -20, "T_max": 374, "P_max": 220},
            "HF": {"T_min": -83, "T_max": 19, "P_max": 65},
            "HCl": {"T_min": -85, "T_max": 51, "P_max": 83},
            "H2S": {"T_min": -85, "T_max": 100, "P_max": 89},
        }

        # Component names for external libraries
        self.thermo_names = {
            "H2O": "water",
            "HF": "hydrogen fluoride",
            "HCl": "hydrogen chloride",
            "H2S": "hydrogen sulfide",
        }

        self.coolprop_names = {"H2O": "Water", "HF": "HF", "HCl": "HCl", "H2S": "H2S"}

    def calculate_vapor_pressure(  # noqa: C901
        self, temperature_c: float, component: str, method: str = "antoine"
    ) -> float:
        """Calculate vapor pressure using different methods.

        Args:
            temperature_c: Temperature in Celsius
            component: Component name ('H2O', 'HF', 'HCl', 'H2S')
            method: Calculation method ('antoine', 'extended_antoine',
                'thermo', 'coolprop')

        Returns:
            Vapor pressure in Pa
        """
        # DbC preconditions
        assert isinstance(temperature_c, (int, float)), (
            f"temperature_c must be numeric, got {type(temperature_c).__name__}"
        )
        assert isinstance(component, str) and len(component) > 0, (
            "component must be a non-empty string"
        )

        if component not in self.antoine_constants:
            msg = f"Unknown component: {component}"
            raise ValueError(msg)

        T = temperature_c + CELSIUS_TO_KELVIN_OFFSET  # Convert to Kelvin

        if method == "antoine":
            A, B, C = (
                self.antoine_constants[component]["A"],
                self.antoine_constants[component]["B"],
                self.antoine_constants[component]["C"],
            )

            # Antoine equation: log10(P) = A - B/(C + T)
            log_p = A - B / (C + temperature_c)
            p_mmhg = 10**log_p
            return p_mmhg * MMHG_TO_PA_CONV  # Convert mmHg to Pa

        if method == "extended_antoine":
            # Extended Antoine equation for wider temperature range
            # Source: Perry's Chemical Engineers' Handbook
            if component == "H2O":
                if temperature_c <= 100:
                    A, B, C = ANTOINE_WATER_A, ANTOINE_WATER_B, ANTOINE_WATER_C
                else:
                    A, B, C = (
                        ANTOINE_WATER_HIGH_A,
                        ANTOINE_WATER_HIGH_B,
                        ANTOINE_WATER_HIGH_C,
                    )
            else:
                A, B, C = (
                    self.antoine_constants[component]["A"],
                    self.antoine_constants[component]["B"],
                    self.antoine_constants[component]["C"],
                )

            log_p = A - B / (C + temperature_c)
            p_mmhg = 10**log_p
            return p_mmhg * MMHG_TO_PA_CONV

        if method == "thermo":
            if not THERMO_AVAILABLE:
                msg = "Thermo library not available"
                raise RuntimeError(msg)
            try:
                from thermo import Chemical

                name = self.thermo_names.get(component, component)
                chem = Chemical(name, T=T)
                return float(chem.Psat)
            except ImportError as e:  # pragma: no cover - fallback
                logger.warning("Thermo vapor pressure failed: %s; using Antoine", e)
                return self.calculate_vapor_pressure(
                    temperature_c, component, "antoine"
                )

        elif method == "coolprop":
            if not COOLPROP_AVAILABLE or PropsSI is None:
                msg = "CoolProp library not available"
                raise RuntimeError(msg)
            try:
                fluid = self.coolprop_names.get(component, component)
                return float(PropsSI("P", "T", T, "Q", 0, fluid))
            except (
                ValueError,
                ZeroDivisionError,
                OverflowError,
                TypeError,
            ) as e:  # pragma: no cover - fallback
                logger.warning("CoolProp vapor pressure failed: %s; using Antoine", e)
                return self.calculate_vapor_pressure(
                    temperature_c, component, "antoine"
                )

        else:
            msg = f"Unknown method: {method}"
            raise ValueError(msg)

    def calculate_dewpoint(
        self, partial_pressure_pa: float, component: str, total_pressure_pa: float = 0.0
    ) -> float:
        """
        Calculate dewpoint temperature using the inverse Antoine equation.

        This method is more efficient and accurate than numerical solving methods.

        Args:
            partial_pressure_pa: Partial pressure in Pa
            component: Component name
            total_pressure_pa: Total system pressure in Pa (optional, for future use)

        Returns:
            Dewpoint temperature in Celsius
        """
        if partial_pressure_pa <= 0:
            raise ValueError(
                f"partial_pressure_pa must be > 0, got {partial_pressure_pa}"
            )

        if component not in self.antoine_constants:
            raise ValueError(
                f"unknown component: {component!r}, "
                f"expected one of {list(self.antoine_constants.keys())}"
            )

        # Convert partial pressure to mmHg for the Antoine equation
        p_mmHg = partial_pressure_pa / MMHG_TO_PA_CONV

        if p_mmHg <= 0:
            raise ValueError(
                f"partial pressure in mmHg must be > 0, got {p_mmHg} "
                f"(from {partial_pressure_pa} Pa)"
            )

        A = self.antoine_constants[component]["A"]
        B = self.antoine_constants[component]["B"]
        C = self.antoine_constants[component]["C"]

        # Inverse Antoine equation: T = B / (A - log10(P)) - C
        denominator = A - np.log10(p_mmHg)
        if denominator == 0:
            raise ValueError(
                f"Antoine inverse calculation has zero denominator for "
                f"component={component!r}, partial_pressure_pa={partial_pressure_pa}"
            )
        return float(B / denominator - C)

    def _calculate_partial_pressures(
        self, pressure_pa: float, composition: AcidGasComposition
    ) -> dict[str, float]:
        """Calculate partial pressures for all components."""
        return {
            "H2O": composition.h2o * pressure_pa,
            "HF": composition.hf * pressure_pa,
            "HCl": composition.hcl * pressure_pa,
            "H2S": composition.h2s * pressure_pa,
        }

    def _calculate_all_individual_dewpoints(
        self, partial_pressures: dict[str, float], total_pressure_pa: float
    ) -> dict[str, float]:
        """Calculate dewpoints for each component in the mixture."""
        assert partial_pressures is not None, "partial_pressures must be provided"
        assert partial_pressures is not None, "partial_pressures must be provided"
        dewpoints = {}
        for component, partial_pa in partial_pressures.items():
            if partial_pa > 0:
                dewpoints[component] = self.calculate_dewpoint(
                    partial_pa, component, total_pressure_pa
                )
            else:
                dewpoints[component] = np.nan
        return dewpoints

    def _assess_condensation_risk(self, margin: float) -> str:
        """Categorize condensation risk based on safety margin."""
        assert margin is not None, "margin must be provided"
        assert margin is not None, "margin must be provided"
        if np.isnan(margin):
            return "Unknown"
        if margin < 0:
            return "HIGH - Condensation occurring"
        if margin < 10:
            return "MEDIUM - Within 10\u00b0C of dewpoint"
        if margin < 30:
            return "LOW - Safe margin"
        return "VERY LOW - Large safety margin"

    def calculate_dewpoint_mixture(
        self,
        temperature_c: float,
        pressure_bar: float,
        composition: AcidGasComposition,
        method: str = "antoine",
    ) -> DewpointResult:
        """
        Calculate dewpoint for acid gas mixture

        Args:
            temperature_c: System temperature in Celsius
            pressure_bar: System pressure in bar
            composition: Acid gas composition
            method: Vapor pressure calculation method
                ('antoine', 'extended_antoine', 'thermo', 'coolprop')

        Returns:
            Comprehensive dewpoint results
        """
        if pressure_bar <= 0:
            raise ValueError(f"pressure_bar must be > 0, got {pressure_bar}")
        if temperature_c + CELSIUS_TO_KELVIN_OFFSET <= 0:
            raise ValueError(
                f"temperature must yield a positive Kelvin value, "
                f"got {temperature_c} C ({temperature_c + CELSIUS_TO_KELVIN_OFFSET} K)"
            )

        # Convert units
        pressure_pa = pressure_bar * BAR_TO_PA
        temperature_k = temperature_c + CELSIUS_TO_KELVIN_OFFSET

        # Validate conditions
        warnings = []
        if not (-100 <= temperature_c <= 400):
            warnings.append(
                "Temperature outside recommended range (-100 to 400\u00b0C)"
            )
        if not (0.1 <= pressure_bar <= 300):
            warnings.append("Pressure outside recommended range (0.1 to 300 bar)")

        # 1. Partial & Vapor pressures
        partials = self._calculate_partial_pressures(pressure_pa, composition)
        vapors = {
            comp: self.calculate_vapor_pressure(temperature_c, comp, method)
            for comp in ["H2O", "HF", "HCl", "H2S"]
        }

        # 2. Individual dewpoints
        dewpoints = self._calculate_all_individual_dewpoints(partials, pressure_pa)

        # 3. Overall dewpoint determination
        valid_dewpoints = {k: v for k, v in dewpoints.items() if not np.isnan(v)}
        if valid_dewpoints:
            limiting_component = max(
                valid_dewpoints.keys(), key=lambda k: valid_dewpoints[k]
            )
            overall_dewpoint = valid_dewpoints[limiting_component]
        else:
            overall_dewpoint = np.nan
            limiting_component = "Unknown"
            warnings.append("Could not calculate dewpoint for any component")

        # 4. Risk assessment
        margin = (
            temperature_c - overall_dewpoint
            if not np.isnan(overall_dewpoint)
            else np.nan
        )
        condensation_risk = self._assess_condensation_risk(margin)

        # 5. Compile sources
        comp_dict = composition.to_dict()
        sources = set()
        for comp, fraction in comp_dict.items():
            if fraction > 0 and comp in self.literature_sources:
                sources.update(self.literature_sources[comp])

        return DewpointResult(
            temperature_c=temperature_c,
            temperature_k=temperature_k,
            pressure_bar=pressure_bar,
            pressure_pa=pressure_pa,
            composition=composition,
            h2o_dewpoint_c=dewpoints["H2O"],
            hf_dewpoint_c=dewpoints["HF"],
            hcl_dewpoint_c=dewpoints["HCl"],
            h2s_dewpoint_c=dewpoints["H2S"],
            overall_dewpoint_c=overall_dewpoint,
            limiting_component=limiting_component,
            h2o_vapor_pressure_pa=vapors["H2O"],
            hf_vapor_pressure_pa=vapors["HF"],
            hcl_vapor_pressure_pa=vapors["HCl"],
            h2s_vapor_pressure_pa=vapors["H2S"],
            h2o_partial_pressure_pa=partials["H2O"],
            hf_partial_pressure_pa=partials["HF"],
            hcl_partial_pressure_pa=partials["HCl"],
            h2s_partial_pressure_pa=partials["H2S"],
            dewpoint_margin_c=margin,
            condensation_risk=condensation_risk,
            calculation_method=method,
            warnings=warnings,
            sources=list(sources),
        )

    def generate_dewpoint_curves(
        self,
        pressure_bar: float,
        composition: AcidGasComposition,
        temp_range: tuple[float, float] = (-50, 200),
        num_points: int = 100,
    ) -> pd.DataFrame:
        """
        Generate dewpoint curves for analysis

        Args:
            pressure_bar: System pressure
            composition: Acid gas composition
            temp_range: Temperature range (min, max) in Celsius
            num_points: Number of calculation points

        Returns:
            DataFrame with temperature and dewpoint data
        """
        assert pressure_bar is not None, "pressure_bar must be provided"
        assert pressure_bar is not None, "pressure_bar must be provided"
        temperatures = np.linspace(temp_range[0], temp_range[1], num_points)
        results = []

        for T in temperatures:
            result = self.calculate_dewpoint_mixture(T, pressure_bar, composition)
            results.append(
                {
                    "Temperature_C": T,
                    "H2O_Dewpoint_C": result.h2o_dewpoint_c,
                    "HF_Dewpoint_C": result.hf_dewpoint_c,
                    "HCl_Dewpoint_C": result.hcl_dewpoint_c,
                    "H2S_Dewpoint_C": result.h2s_dewpoint_c,
                    "Overall_Dewpoint_C": result.overall_dewpoint_c,
                    "Limiting_Component": result.limiting_component,
                    "Condensation_Risk": result.condensation_risk,
                }
            )

        return pd.DataFrame(results)
