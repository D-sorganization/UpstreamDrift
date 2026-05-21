"""Pressure drop calculator utility functions.

This module provides utility functions for:
- Fitting loss coefficient calculations
- Flow rate conversions
- Gas property calculations
- Pipe database access
"""

from .fitting_loss_coefficients import (
    calculate_two_k_factor,
    get_fitting_k_factor,
    get_multiple_fittings_k,
    list_available_fittings,
)
from .flow_rate_converter import (
    STANDARD_CONDITIONS,
    acfm_to_scfm,
    convert_flow_rate_to_mass,
    mass_to_mass,
    molar_to_mass,
    scfm_to_acfm,
)
from .gas_properties import (
    GAS_DATABASE,
    calculate_gas_properties,
    calculate_mixture_molecular_weight,
    calculate_mixture_viscosity_wilke,
)
from .pipe_database import (
    MATERIAL_ROUGHNESS,
    create_custom_pipe,
    get_pipe_spec,
    get_roughness,
    list_available_sizes,
    list_schedules_for_size,
)

__all__ = [
    # Fitting coefficients
    "get_fitting_k_factor",
    "get_multiple_fittings_k",
    "calculate_two_k_factor",
    "list_available_fittings",
    # Flow rate conversion
    "convert_flow_rate_to_mass",
    "mass_to_mass",
    "molar_to_mass",
    "scfm_to_acfm",
    "acfm_to_scfm",
    "STANDARD_CONDITIONS",
    # Gas properties
    "calculate_gas_properties",
    "calculate_mixture_molecular_weight",
    "calculate_mixture_viscosity_wilke",
    "GAS_DATABASE",
    # Pipe database
    "get_pipe_spec",
    "get_roughness",
    "list_available_sizes",
    "list_schedules_for_size",
    "create_custom_pipe",
    "MATERIAL_ROUGHNESS",
]
