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

from ._acid_gas_calculator import AcidGasDewpointCalculator
from ._acid_gas_models import AcidGasComposition, DewpointResult
from ._acid_gas_utils import (
    ACID_GAS_PRESETS,
    estimate_condensation_risk,
    quick_dewpoint_calculation,
)
from ._acid_gas_widget import GUI_AVAILABLE

if GUI_AVAILABLE:
    from ._acid_gas_widget import AcidGasDewpointCalculatorWidget

__all__ = [
    "AcidGasDewpointCalculator",
    "AcidGasComposition",
    "DewpointResult",
    "ACID_GAS_PRESETS",
    "quick_dewpoint_calculation",
    "estimate_condensation_risk",
    "GUI_AVAILABLE",
]

if GUI_AVAILABLE:
    __all__ += ["AcidGasDewpointCalculatorWidget"]
