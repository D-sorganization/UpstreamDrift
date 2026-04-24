#!/usr/bin/env python3
"""Advanced Pressure Drop Calculator for Combustion and Gasification Gases.

This package provides comprehensive pressure drop calculations with support for:
- Variable gas compositions
- Multiple friction factor correlations
- Standard and custom pipe sizes
- Extensive unit conversions
- Compressible flow corrections
- Fitting and valve losses

Main Components:
    - PressureDropCalculationEngine: Core calculation engine
    - PressureDropInputs/Results: Data models
    - Gas property calculations
    - Pipe and fitting databases
    - Flow rate conversions

Example:
    >>> from calculators.pressure_drop_calculator import PressureDropCalculationEngine
    >>> from calculators.pressure_drop_calculator.models import (
    ...     PressureDropInputs, GasComposition
    ... )
    >>>
    >>> # Define gas composition
    >>> composition = GasComposition(components={'H2': 0.3, 'CO': 0.4, 'CO2': 0.3})
    >>>
    >>> # Create inputs
    >>> inputs = PressureDropInputs(
    ...     pipe_diameter=0.1023,  # 4" Schedule 40
    ...     pipe_length=50.0,
    ...     pipe_roughness=0.000045,
    ...     mass_flow_rate=1.0,
    ...     inlet_pressure=10e5,
    ...     inlet_temperature=700,
    ...     gas_composition=composition
    ... )
    >>>
    >>> # Calculate
    >>> engine = PressureDropCalculationEngine()
    >>> results = engine.calculate(inputs)
    >>> print(f"Pressure drop: {results.total_pressure_drop/1e5:.4f} bar")

References:
    - Crane Technical Paper No. 410
    - Perry's Chemical Engineers' Handbook, 9th Edition
    - API RP 14E
    - ASME B36.10M pipe standards
    - Reid, Prausnitz, Poling: Properties of Gases and Liquids, 5th Ed

Author: Advanced Pressure Drop Calculator System
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Gasification Model Team"

# Core engine
from .engine.pressure_drop_calculation_engine import PressureDropCalculationEngine

# Data models
from .models.pressure_drop_data_models import (
    FlowProperties,
    FlowRateInput,
    GasComposition,
    PipeFitting,
    PipeSpecification,
    PressureDropInputs,
    PressureDropResults,
)

# High-level interface functions
from .pressure_drop_interface import (
    calculate_pressure_drop,
    calculate_pressure_drop_custom_gas,
    calculate_pressure_drop_syngas,
    print_results,
)
from .utils.fitting_loss_coefficients import (
    calculate_two_k_factor,
    get_fitting_k_factor,
    get_multiple_fittings_k,
    list_available_fittings,
)
from .utils.flow_rate_converter import (
    STANDARD_CONDITIONS,
    acfm_to_scfm,
    convert_flow_rate_to_mass,
    mass_to_mass,
    molar_to_mass,
    scfm_to_acfm,
)
from .utils.gas_properties import (
    GAS_DATABASE,
    calculate_gas_properties,
    calculate_mixture_molecular_weight,
    calculate_mixture_viscosity_wilke,
)

# Utility functions
from .utils.pipe_database import (
    MATERIAL_ROUGHNESS,
    create_custom_pipe,
    get_pipe_spec,
    get_roughness,
    list_available_sizes,
    list_schedules_for_size,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Core engine
    "PressureDropCalculationEngine",
    # Data models
    "PressureDropInputs",
    "PressureDropResults",
    "GasComposition",
    "PipeFitting",
    "PipeSpecification",
    "FlowProperties",
    "FlowRateInput",
    # Pipe database
    "get_pipe_spec",
    "get_roughness",
    "list_available_sizes",
    "list_schedules_for_size",
    "create_custom_pipe",
    "MATERIAL_ROUGHNESS",
    # Fitting losses
    "get_fitting_k_factor",
    "get_multiple_fittings_k",
    "calculate_two_k_factor",
    "list_available_fittings",
    # Gas properties
    "calculate_gas_properties",
    "calculate_mixture_molecular_weight",
    "calculate_mixture_viscosity_wilke",
    "GAS_DATABASE",
    # Flow rate conversions
    "convert_flow_rate_to_mass",
    "mass_to_mass",
    "molar_to_mass",
    "scfm_to_acfm",
    "acfm_to_scfm",
    "STANDARD_CONDITIONS",
    # High-level interface
    "calculate_pressure_drop",
    "calculate_pressure_drop_custom_gas",
    "calculate_pressure_drop_syngas",
    "print_results",
    # Legacy API (backwards compatibility)
    "PIPE_DIMENSIONS_SCH40",
    "ROUGHNESS_VALUES",
    "PressureDropCalculator",
    "PressureDropResult",
]


# =============================================================================
# LEGACY API - Backwards Compatibility
# =============================================================================
# These exports maintain compatibility with code using the original
# pressure_drop_calculator.py module interface.

import math
from dataclasses import dataclass
from typing import Final

from upstream_drift_tools.utils.unit_constants import R_UNIVERSAL

# Standard Pipe Dimensions (Schedule 40) - Inner Diameter in meters
PIPE_DIMENSIONS_SCH40: Final[dict[str, float]] = {
    '1/2"': 0.01575,
    '3/4"': 0.02093,
    '1"': 0.02664,
    '1-1/4"': 0.03505,
    '1-1/2"': 0.04089,
    '2"': 0.05250,
    '2-1/2"': 0.06271,
    '3"': 0.07793,
    '4"': 0.10226,
    '6"': 0.15405,
    '8"': 0.20272,
    '10"': 0.25450,
    '12"': 0.30328,
    '14"': 0.33655,
    '16"': 0.38735,
    '18"': 0.43815,
    '20"': 0.48895,
    '24"': 0.59055,
}

# Standard Roughness Values in meters
ROUGHNESS_VALUES: Final[dict[str, float]] = {
    "Commercial Steel": 0.000045,
    "Drawn Tubing": 0.0000015,
    "Stainless Steel": 0.000015,
    "Cast Iron": 0.00026,
    "Concrete": 0.001,
}


@dataclass
class PressureDropResult:
    """Pressure drop calculation result (legacy API)."""

    pressure_drop_pa: float
    reynolds_number: float
    friction_factor: float
    velocity: float  # m/s
    flow_regime: str
    density: float  # kg/m^3
    viscosity: float  # Pa*s


class PressureDropCalculator:
    """Core pressure drop calculation engine (legacy API)."""

    def __init__(self) -> None:
        """Initialize the calculator."""

    def calculate_pressure_drop(
        self,
        pipe_diameter_m: float,
        pipe_length_m: float,
        roughness_m: float,
        flow_rate_kg_s: float,
        temperature_k: float,
        pressure_pa: float,
        molecular_weight_kg_mol: float,
    ) -> PressureDropResult:
        """Calculate pressure drop using Darcy-Weisbach equation."""
        # Calculate gas properties (Z=1.0 assumption - Ideal Gas)
        if pipe_diameter_m is None:
            raise ValueError("pipe_diameter_m must be provided")
        Z = 1.0
        density = (pressure_pa * molecular_weight_kg_mol) / (
            Z * R_UNIVERSAL * temperature_k
        )

        # Estimate viscosity using Sutherland's formula
        T_ref = 291.15  # K
        mu_ref = 1.827e-5  # Pa·s
        S = 120  # K
        viscosity = (
            mu_ref
            * ((T_ref + S) / (temperature_k + S))
            * (temperature_k / T_ref) ** 1.5
        )

        # Calculate volumetric flow and velocity
        vol_flow = flow_rate_kg_s / density if density > 0 else 0.0
        area = math.pi * (pipe_diameter_m / 2) ** 2
        velocity = vol_flow / area if area > 0 else 0.0

        # Calculate Reynolds number
        Re = (
            (density * velocity * pipe_diameter_m) / viscosity if viscosity > 0 else 0.0
        )

        # Calculate friction factor
        rel_roughness = roughness_m / pipe_diameter_m if pipe_diameter_m > 0 else 0.0

        if Re > 4000:
            # Swamee-Jain explicit approximation
            A = rel_roughness / 3.7
            B = 5.74 / (Re**0.9) if Re > 0 else 0.01
            try:
                friction_factor = 0.25 / (math.log10(A + B) ** 2)
            except ValueError:
                friction_factor = 0.02  # Fallback
        elif Re > 2300:
            friction_factor = 0.03  # Transition
        else:
            friction_factor = 64 / Re if Re > 0 else 0.05

        # Calculate pressure drop (Darcy-Weisbach)
        if pipe_diameter_m > 0:
            pressure_drop_pa = (
                friction_factor
                * (pipe_length_m / pipe_diameter_m)
                * (density * velocity**2 / 2)
            )
        else:
            pressure_drop_pa = 0.0

        # Determine flow regime
        if Re < 2300:
            flow_regime = "Laminar"
        elif Re < 4000:
            flow_regime = "Transitional"
        else:
            flow_regime = "Turbulent"

        return PressureDropResult(
            pressure_drop_pa=pressure_drop_pa,
            reynolds_number=Re,
            friction_factor=friction_factor,
            velocity=velocity,
            flow_regime=flow_regime,
            density=density,
            viscosity=viscosity,
        )
