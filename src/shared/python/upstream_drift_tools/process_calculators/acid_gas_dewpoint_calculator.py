#!/usr/bin/env python3
"""
Acid Gas Dewpoint Calculator for Syngas Applications
===================================================

A comprehensive calculator for predicting dewpoint temperatures of acid gases
(HF, HCl, H2S) in syngas/water vapor mixtures.

Key Features:
- Multi-component acid gas dewpoint calculations
- Literature-based thermodynamic correlations
- Support for HF, HCl, and H2S
- Temperature and pressure range validation
- Comprehensive documentation with sources
- Modern GUI interface

Literature Sources:
- Perry's Chemical Engineers' Handbook (8th Ed.)
- NIST Chemistry WebBook
- CRC Handbook of Chemistry and Physics
- Journal of Chemical & Engineering Data
- Industrial & Engineering Chemistry Research

Example Usage:
    from acid_gas_dewpoint_calculator import AcidGasDewpointCalculator

    calc = AcidGasDewpointCalculator()
    result = calc.calculate_dewpoint(
        temperature_c=150,
        pressure_bar=30,
        composition={'H2O': 0.15, 'HF': 0.001, 'HCl': 0.002, 'H2S': 0.005}
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

# Optional thermodynamic libraries for more accurate vapor pressure
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

# Try to import PyQt6 for GUI, but make it optional
try:
    from PyQt6.QtCore import QTimer, pyqtSignal
    from PyQt6.QtGui import QFont
    from PyQt6.QtWidgets import (
        QDoubleSpinBox,
        QGridLayout,
        QGroupBox,
        QLabel,
        QPushButton,
        QSplitter,
        QTableWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )

    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False

# Import BaseCalculatorWidget for state management
try:
    from ..ui.widgets.base_calculator_widget import BaseCalculatorWidget

    BASE_CALCULATOR_AVAILABLE = True
except ImportError:
    BASE_CALCULATOR_AVAILABLE = False

    # Fallback to QWidget if BaseCalculatorWidget is not available
    class BaseCalculatorWidget(QWidget):  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            QWidget.__init__(self, *args, **kwargs)


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


@dataclass
class AcidGasComposition:
    """Composition of acid gases and water vapor in syngas"""

    h2o: float = 0.0  # Water vapor mole fraction
    hf: float = 0.0  # Hydrogen fluoride mole fraction
    hcl: float = 0.0  # Hydrogen chloride mole fraction
    h2s: float = 0.0  # Hydrogen sulfide mole fraction
    other: float = 0.0  # Other components (H2, CO, CO2, etc.)
    name: str = ""

    def normalize(self) -> AcidGasComposition:
        """Normalize composition to sum to 1.0"""
        total = self.h2o + self.hf + self.hcl + self.h2s + self.other
        if total > 0:
            return AcidGasComposition(
                h2o=self.h2o / total,
                hf=self.hf / total,
                hcl=self.hcl / total,
                h2s=self.h2s / total,
                other=self.other / total,
                name=self.name,
            )
        return self

    def to_dict(self) -> dict[str, float]:
        """Convert composition to dictionary format.

        Returns:
            Dictionary with component names as keys and mole fractions as values.
        """
        return {
            "H2O": self.h2o,
            "HF": self.hf,
            "HCl": self.hcl,
            "H2S": self.h2s,
            "Other": self.other,
        }

    @property
    def total(self) -> float:
        """Total mole fraction (should be 1.0 for normalized composition).

        Returns:
            Sum of all component mole fractions.
        """
        return self.h2o + self.hf + self.hcl + self.h2s + self.other


@dataclass
class DewpointResult:
    """Comprehensive dewpoint calculation results"""

    # Input conditions
    temperature_c: float
    temperature_k: float
    pressure_bar: float
    pressure_pa: float
    composition: AcidGasComposition

    # Individual acid gas dewpoints
    h2o_dewpoint_c: float
    hf_dewpoint_c: float
    hcl_dewpoint_c: float
    h2s_dewpoint_c: float

    # Overall dewpoint (highest among all components)
    overall_dewpoint_c: float
    limiting_component: str

    # Vapor pressures at current conditions
    h2o_vapor_pressure_pa: float
    hf_vapor_pressure_pa: float
    hcl_vapor_pressure_pa: float
    h2s_vapor_pressure_pa: float

    # Partial pressures
    h2o_partial_pressure_pa: float
    hf_partial_pressure_pa: float
    hcl_partial_pressure_pa: float
    h2s_partial_pressure_pa: float

    # Safety margins
    dewpoint_margin_c: float
    condensation_risk: str

    # Additional info
    calculation_method: str
    timestamp: datetime = field(default_factory=datetime.now)
    warnings: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for export.

        Returns:
            Dictionary containing all result data in exportable format.
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "input": {
                "temperature_c": self.temperature_c,
                "pressure_bar": self.pressure_bar,
                "composition": self.composition.to_dict(),
            },
            "dewpoints": {
                "H2O": self.h2o_dewpoint_c,
                "HF": self.hf_dewpoint_c,
                "HCl": self.hcl_dewpoint_c,
                "H2S": self.h2s_dewpoint_c,
                "overall": self.overall_dewpoint_c,
                "limiting_component": self.limiting_component,
            },
            "vapor_pressures_pa": {
                "H2O": self.h2o_vapor_pressure_pa,
                "HF": self.hf_vapor_pressure_pa,
                "HCl": self.hcl_vapor_pressure_pa,
                "H2S": self.h2s_vapor_pressure_pa,
            },
            "safety": {
                "dewpoint_margin_c": self.dewpoint_margin_c,
                "condensation_risk": self.condensation_risk,
            },
            "method": self.calculation_method,
            "sources": self.sources,
            "warnings": self.warnings,
        }


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

    def calculate_vapor_pressure(
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
            return "MEDIUM - Within 10°C of dewpoint"
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
            warnings.append("Temperature outside recommended range (-100 to 400°C)")
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


# Predefined acid gas compositions for common scenarios
ACID_GAS_PRESETS = {
    "typical_syngas": AcidGasComposition(
        h2o=0.15,
        hf=0.0001,
        hcl=0.0002,
        h2s=0.001,
        name="Typical Syngas with Acid Gases",
    ),
    "high_acid_content": AcidGasComposition(
        h2o=0.20, hf=0.001, hcl=0.002, h2s=0.005, name="High Acid Gas Content"
    ),
    "coal_gasification": AcidGasComposition(
        h2o=0.12, hf=0.0005, hcl=0.001, h2s=0.003, name="Coal Gasification"
    ),
    "biomass_gasification": AcidGasComposition(
        h2o=0.18, hf=0.0002, hcl=0.0005, h2s=0.002, name="Biomass Gasification"
    ),
    "custom": AcidGasComposition(name="Custom Composition"),
}


# Quick calculation functions
def quick_dewpoint_calculation(
    temperature_c: float,
    pressure_bar: float,
    h2o_fraction: float,
    hf_fraction: float = 0.0,
    hcl_fraction: float = 0.0,
    h2s_fraction: float = 0.0,
    method: str = "antoine",
) -> dict[str, float | str]:
    """
    Quick dewpoint calculation for common use cases

    Args:
        temperature_c: System temperature in Celsius
        pressure_bar: System pressure in bar
        h2o_fraction: Water vapor mole fraction
        hf_fraction: HF mole fraction
        hcl_fraction: HCl mole fraction
        h2s_fraction: H2S mole fraction
        method: Vapor pressure method

    Returns:
        Dictionary with key results
    """
    assert temperature_c is not None, "temperature_c must be provided"
    assert temperature_c is not None, "temperature_c must be provided"
    calc = AcidGasDewpointCalculator()
    composition = AcidGasComposition(
        h2o=h2o_fraction, hf=hf_fraction, hcl=hcl_fraction, h2s=h2s_fraction
    )

    result = calc.calculate_dewpoint_mixture(
        temperature_c, pressure_bar, composition, method
    )

    return {
        "overall_dewpoint_c": result.overall_dewpoint_c,
        "limiting_component": result.limiting_component,
        "condensation_risk": result.condensation_risk,
        "dewpoint_margin_c": result.dewpoint_margin_c,
    }


def estimate_condensation_risk(
    temperature_c: float,
    pressure_bar: float,
    composition: AcidGasComposition,
    safety_margin_c: float = 10.0,
    method: str = "antoine",
) -> dict[str, float | str]:
    """
    Estimate condensation risk for acid gas mixture

    Args:
        temperature_c: System temperature
        pressure_bar: System pressure
        composition: Acid gas composition
        safety_margin_c: Required safety margin in Celsius
        method: Vapor pressure method

    Returns:
        Risk assessment dictionary
    """
    assert temperature_c is not None, "temperature_c must be provided"
    assert temperature_c is not None, "temperature_c must be provided"
    calc = AcidGasDewpointCalculator()
    result = calc.calculate_dewpoint_mixture(
        temperature_c, pressure_bar, composition, method
    )

    margin = result.dewpoint_margin_c

    if np.isnan(margin):
        risk_level = "Unknown"
        recommendation = "Check input parameters and component validity"
    elif margin < 0:
        risk_level = "Critical"
        recommendation = "Immediate action required - condensation occurring"
    elif margin < safety_margin_c:
        risk_level = "High"
        recommendation = "Increase temperature or reduce acid gas content"
    elif margin < 2 * safety_margin_c:
        risk_level = "Medium"
        recommendation = "Monitor closely and consider preventive measures"
    else:
        risk_level = "Low"
        recommendation = "Safe operating conditions"

    return {
        "risk_level": risk_level,
        "current_margin_c": margin,
        "required_margin_c": safety_margin_c,
        "recommendation": recommendation,
        "limiting_component": result.limiting_component,
    }


# --- GUI Widget for Acid Gas Dewpoint Calculator ---
if GUI_AVAILABLE:
    # Handle dynamic base class based on availability
    BaseClass = BaseCalculatorWidget if BASE_CALCULATOR_AVAILABLE else QWidget

    class AcidGasDewpointCalculatorWidget(BaseClass):  # type: ignore[valid-type, misc]
        """Acid gas dewpoint calculator widget"""

        calculation_completed = pyqtSignal(dict)

        def __init__(self, parent: QWidget | None = None) -> None:
            """Initialize the class."""
            if BASE_CALCULATOR_AVAILABLE:
                super().__init__(calculator_name="AcidGasDewpoint", parent=parent)
            else:
                super().__init__(parent)

            self.calculator = AcidGasDewpointCalculator()
            self.current_result = None
            self.setup_ui()
            self.setup_connections()
            self.set_default_values()

            if BASE_CALCULATOR_AVAILABLE:
                QTimer.singleShot(0, self.setup_state_management)

        def setup_connections(self) -> None:
            """Setup signal connections"""

        def set_default_values(self) -> None:
            """Set default values for input widgets"""

        def setup_state_management(self) -> None:
            """Setup state management for the calculator"""
            if not BASE_CALCULATOR_AVAILABLE:
                return

            # Find and register splitters
            for child_splitter in self.findChildren(QSplitter):
                self.register_splitter(child_splitter, "main_splitter")

            # Register result widgets for copy/paste
            for child_label in self.findChildren((QLabel, QTextEdit, QTableWidget)):
                if hasattr(child_label, "text") and child_label.text().strip():
                    self.register_copyable_widget(child_label, "label")
            for child_table in self.findChildren(QTableWidget):
                self.register_copyable_widget(child_table, "table")
            for child_text in self.findChildren(QTextEdit):
                self.register_copyable_widget(child_text, "text")

        def closeEvent(self, event: Any) -> None:
            """Save state when tab is closed"""
            if BASE_CALCULATOR_AVAILABLE:
                self.save_state()
            super().closeEvent(event)

        def setup_ui(self) -> None:
            """Setup the user interface"""
            layout = QVBoxLayout(self)
            title = QLabel("Acid Gas Dewpoint Calculator")
            title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
            layout.addWidget(title)

            # Input fields
            input_group = QGroupBox("Input Conditions")
            input_layout = QGridLayout(input_group)
            self.temp_input = QDoubleSpinBox()
            self.temp_input.setRange(-100, 400)
            self.temp_input.setValue(150)
            self.temp_input.setSuffix(" °C")
            input_layout.addWidget(QLabel("Temperature:"), 0, 0)
            input_layout.addWidget(self.temp_input, 0, 1)

            self.pressure_input = QDoubleSpinBox()
            self.pressure_input.setRange(0.1, 300)
            self.pressure_input.setValue(30)
            self.pressure_input.setSuffix(" bar")
            input_layout.addWidget(QLabel("Pressure:"), 1, 0)
            input_layout.addWidget(self.pressure_input, 1, 1)

            # Acid gas composition
            self.h2o_input = QDoubleSpinBox()
            self.h2o_input.setRange(0, 1)
            self.h2o_input.setValue(0.15)
            self.hf_input = QDoubleSpinBox()
            self.hf_input.setRange(0, 1)
            self.hf_input.setValue(0.001)
            self.hcl_input = QDoubleSpinBox()
            self.hcl_input.setRange(0, 1)
            self.hcl_input.setValue(0.002)
            self.h2s_input = QDoubleSpinBox()
            self.h2s_input.setRange(0, 1)
            self.h2s_input.setValue(0.005)

            input_layout.addWidget(QLabel("H2O mole fraction:"), 2, 0)
            input_layout.addWidget(self.h2o_input, 2, 1)
            input_layout.addWidget(QLabel("HF mole fraction:"), 3, 0)
            input_layout.addWidget(self.hf_input, 3, 1)
            input_layout.addWidget(QLabel("HCl mole fraction:"), 4, 0)
            input_layout.addWidget(self.hcl_input, 4, 1)
            input_layout.addWidget(QLabel("H2S mole fraction:"), 5, 0)
            input_layout.addWidget(self.h2s_input, 5, 1)

            layout.addWidget(input_group)

            # Calculate button
            self.calc_btn = QPushButton("Calculate Dewpoint")
            self.calc_btn.clicked.connect(self.calculate)
            layout.addWidget(self.calc_btn)

            # Output area
            self.result_area = QTextEdit()
            self.result_area.setReadOnly(True)
            layout.addWidget(self.result_area)

        def calculate(self) -> None:
            """Collect inputs and run calculation."""
            temp = self.temp_input.value()
            pressure = self.pressure_input.value()
            comp = AcidGasComposition(
                h2o=self.h2o_input.value(),
                hf=self.hf_input.value(),
                hcl=self.hcl_input.value(),
                h2s=self.h2s_input.value(),
            )
            result = self.calculator.calculate_dewpoint_mixture(temp, pressure, comp)
            self.display_result(result)

        def display_result(self, result: DewpointResult) -> None:
            """Format and display results in the UI."""
            assert result is not None, "result must be provided"
            assert result is not None, "result must be provided"
            text = (
                f"<b>Input:</b> T = {result.temperature_c:.2f} °C, "
                f"P = {result.pressure_bar:.2f} bar<br>"
            )
            text += (
                f"<b>Composition:</b> H2O={result.composition.h2o:.4f}, "
                f"HF={result.composition.hf:.4f}, "
                f"HCl={result.composition.hcl:.4f}, "
                f"H2S={result.composition.h2s:.4f}<br>"
            )
            text += (
                f"<b>Dewpoints (°C):</b> H2O={result.h2o_dewpoint_c:.2f}, "
                f"HF={result.hf_dewpoint_c:.2f}, HCl={result.hcl_dewpoint_c:.2f}, "
                f"H2S={result.h2s_dewpoint_c:.2f}<br>"
            )
            text += (
                f"<b>Overall Dewpoint:</b> {result.overall_dewpoint_c:.2f} °C "
                f"({result.limiting_component})<br>"
            )
            text += f"<b>Dewpoint Margin:</b> {result.dewpoint_margin_c:.2f} °C<br>"
            text += f"<b>Condensation Risk:</b> {result.condensation_risk}<br>"
            if result.warnings:
                text += f"<b>Warnings:</b> {'; '.join(result.warnings)}<br>"
            self.result_area.setHtml(text)
