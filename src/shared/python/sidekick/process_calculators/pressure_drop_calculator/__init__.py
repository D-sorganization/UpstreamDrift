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
    >>> logger.debug(f"Pressure drop: {results.total_pressure_drop/1e5:.4f} bar")

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
    "FlowProperties",
    "FlowRateInput",
    "GAS_DATABASE",
    "GasComposition",
    "MATERIAL_ROUGHNESS",
    "PIPE_DIMENSIONS_SCH40",
    "PipeFitting",
    "PipeSpecification",
    "PressureDropCalculationEngine",
    "PressureDropCalculator",
    "PressureDropInputs",
    "PressureDropResult",
    "PressureDropResults",
    "ROUGHNESS_VALUES",
    "STANDARD_CONDITIONS",
    "acfm_to_scfm",
    "calculate_gas_properties",
    "calculate_mixture_molecular_weight",
    "calculate_mixture_viscosity_wilke",
    "calculate_pressure_drop",
    "calculate_pressure_drop_custom_gas",
    "calculate_pressure_drop_syngas",
    "calculate_two_k_factor",
    "convert_flow_rate_to_mass",
    "create_custom_pipe",
    "get_fitting_k_factor",
    "get_multiple_fittings_k",
    "get_pipe_spec",
    "get_roughness",
    "list_available_fittings",
    "list_available_sizes",
    "list_schedules_for_size",
    "mass_to_mass",
    "molar_to_mass",
    "print_results",
    "scfm_to_acfm",
]


# =============================================================================
# LEGACY API - Backwards Compatibility
# =============================================================================
# Calculation logic extracted to _legacy.py (issue #1696 — god module refactor).
# These re-exports maintain the original public API surface.
from ._legacy import (
    PIPE_DIMENSIONS_SCH40,
    ROUGHNESS_VALUES,
    PressureDropCalculator,
    PressureDropResult,
)
