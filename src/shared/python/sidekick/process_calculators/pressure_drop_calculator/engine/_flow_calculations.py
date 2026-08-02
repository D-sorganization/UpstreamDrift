"""Flow property calculations and pressure drop components.

Provides functions for computing:
- Flow properties (density, velocity, Reynolds, Mach)
- Frictional, fitting, elevation pressure drops
- Compressible flow corrections
- Erosional velocity limits

References:
    - Darcy-Weisbach equation for pipe friction
    - Crane TP-410 for fitting losses
    - API RP 14E for erosional velocity
    - Perry's Chemical Engineers' Handbook, 9th Edition
"""

import logging
import math

from ....utils.unit_constants import R_UNIVERSAL_KMOL, STANDARD_GRAVITY
from ...constants import (
    API_14E_C_CONTINUOUS,
    API_14E_C_INTERMITTENT,
    FT_S_TO_M_S,
    KG_M3_TO_LB_FT3,
    RE_LAMINAR_UPPER,
    RE_TURBULENT_LOWER,
)
from ..models.pressure_drop_data_models import (
    FlowProperties,
    PipeFitting,
    PressureDropInputs,
)
from ..utils.fitting_loss_coefficients import (
    calculate_two_k_factor,
    get_fitting_k_factor,
)
from ..utils.gas_properties import calculate_gas_properties

# Single source of truth for the compressible-flow solver (issue #3103 F1).
# The near-verbatim duplicate that previously lived here (with a malformed
# expansion-factor expression) has been removed; import from the canonical
# module instead so a fix reaches every caller.
from .compressible_flow import (  # noqa: F401
    calculate_compressible_flow_correction,
    calculate_expansion_factor,
)

__all__ = [
    "GRAVITY",
    "PI",
    "R_UNIVERSAL",
    "calculate_compressible_flow_correction",
    "calculate_elevation_pressure_drop",
    "calculate_erosional_velocity",
    "calculate_expansion_factor",
    "calculate_fitting_pressure_drop",
    "calculate_flow_properties",
    "calculate_frictional_pressure_drop",
    "classify_flow_regime",
]

_logger = logging.getLogger(__name__)

GRAVITY = STANDARD_GRAVITY  # m/s²
R_UNIVERSAL = R_UNIVERSAL_KMOL  # J/(kmol·K)
PI = math.pi


def calculate_flow_properties(inputs: PressureDropInputs) -> FlowProperties:
    """Calculate comprehensive flow properties for the gas mixture.

    Args:
        inputs: Pressure drop input parameters

    Returns:
        FlowProperties object with all calculated properties

    Raises:
        ValueError: If calculations fail
    """
    if not (inputs.pipe_diameter > 0):
        raise ValueError(f"Pipe diameter must be positive, got {inputs.pipe_diameter}")
    if not (inputs.mass_flow_rate > 0):
        raise ValueError(
            f"Mass flow rate must be positive, got {inputs.mass_flow_rate}"
        )
    if not (inputs.inlet_temperature > 0):
        raise ValueError(
            f"Inlet temperature must be positive (K), got {inputs.inlet_temperature}"
        )
    if not (inputs.inlet_pressure > 0):
        raise ValueError(
            f"Inlet pressure must be positive, got {inputs.inlet_pressure}"
        )

    gas_props = calculate_gas_properties(
        inputs.gas_composition.components,
        inputs.inlet_temperature,
        inputs.inlet_pressure,
        inputs.compressibility_correction,
    )

    density = gas_props["density"]
    viscosity = gas_props["viscosity"]
    molecular_weight = gas_props["molecular_weight"]
    compressibility_factor = gas_props["compressibility_factor"]
    speed_of_sound = gas_props["speed_of_sound"]
    heat_capacity_ratio = gas_props["heat_capacity_ratio"]

    if not (density > 0):
        raise ValueError(f"Gas density must be positive, got {density}")
    if not (viscosity > 0):
        raise ValueError(f"Gas viscosity must be positive, got {viscosity}")
    if not (speed_of_sound > 0):
        raise ValueError(f"Speed of sound must be positive, got {speed_of_sound}")

    pipe_area = PI * (inputs.pipe_diameter**2) / 4.0
    velocity = inputs.mass_flow_rate / (density * pipe_area)
    mass_flux = inputs.mass_flow_rate / pipe_area

    reynolds_number = (density * velocity * inputs.pipe_diameter) / viscosity
    mach_number = velocity / speed_of_sound
    volumetric_flow_rate = inputs.mass_flow_rate / density

    flow_props = FlowProperties(
        density=density,
        viscosity=viscosity,
        velocity=velocity,
        reynolds_number=reynolds_number,
        mach_number=mach_number,
        compressibility_factor=compressibility_factor,
        molecular_weight=molecular_weight,
        mass_flux=mass_flux,
        volumetric_flow_rate=volumetric_flow_rate,
    )

    if not (flow_props.velocity > 0):
        raise ValueError(f"Flow velocity must be positive, got {flow_props.velocity}")
    if not (flow_props.reynolds_number > 0):
        raise ValueError(
            f"Reynolds number must be positive, got {flow_props.reynolds_number}"
        )
    if not (0 <= flow_props.mach_number < 50):
        raise ValueError(
            f"Mach number out of physical range, got {flow_props.mach_number}"
        )

    _logger.info("Flow properties calculated:")
    _logger.info(f"  Velocity: {velocity:.2f} m/s")
    _logger.info(f"  Reynolds: {reynolds_number:.0f}")
    _logger.info(f"  Mach: {mach_number:.4f}")
    _logger.info(f"  Density: {density:.4f} kg/m3")
    _logger.info(f"  gamma (Cp/Cv): {heat_capacity_ratio:.3f}")
    _logger.info(f"  Speed of sound: {speed_of_sound:.1f} m/s")

    return flow_props


def classify_flow_regime(reynolds_number: float) -> str:
    """Classify flow regime based on Reynolds number.

    Args:
        reynolds_number: Reynolds number

    Returns:
        Flow regime: 'laminar', 'transitional', or 'turbulent'

    Reference:
        - Re < 2300: Laminar
        - 2300 < Re < 4000: Transitional
        - Re > 4000: Turbulent
    """
    if reynolds_number < RE_LAMINAR_UPPER:
        return "laminar"
    if reynolds_number < RE_TURBULENT_LOWER:
        return "transitional"
    return "turbulent"


def calculate_frictional_pressure_drop(
    friction_factor: float,
    length: float,
    diameter: float,
    density: float,
    velocity: float,
) -> float:
    """Calculate frictional pressure drop using Darcy-Weisbach equation.

    dP_friction = f * (L/D) * (rho*V^2/2)

    Args:
        friction_factor: Darcy friction factor (must be positive)
        length: Pipe length in m (must be positive)
        diameter: Pipe diameter in m (must be positive)
        density: Fluid density in kg/m3 (must be positive)
        velocity: Flow velocity in m/s (must be positive)

    Returns:
        Frictional pressure drop in Pa (non-negative)

    Reference:
        Darcy, H. (1857), Weisbach, J. (1845): Pipe flow friction equation
    """
    if not (friction_factor > 0):
        raise ValueError(f"friction_factor must be positive, got {friction_factor}")
    if not (length > 0):
        raise ValueError(f"length must be positive, got {length}")
    if not (diameter > 0):
        raise ValueError(f"diameter must be positive, got {diameter}")
    if not (density > 0):
        raise ValueError(f"density must be positive, got {density}")
    if not (velocity > 0):
        raise ValueError(f"velocity must be positive, got {velocity}")

    velocity_head = 0.5 * density * (velocity**2)
    dp_friction = friction_factor * (length / diameter) * velocity_head

    if not (dp_friction >= 0):
        raise ValueError(f"Pressure drop must be non-negative, got {dp_friction}")

    _logger.debug(
        f"Darcy-Weisbach: f={friction_factor:.6f}, "
        f"L/D={length / diameter:.1f}, dP={dp_friction:.1f} Pa"
    )
    return dp_friction


def calculate_fitting_pressure_drop(
    fittings: list[PipeFitting],
    density: float,
    velocity: float,
    reynolds_number: float,
    diameter_inches: float,
) -> float:
    """Calculate total pressure drop across fittings and valves.

    dP_fitting = sum(K_i) * (rho*V^2/2)

    Args:
        fittings: List of PipeFitting objects
        density: Fluid density (kg/m3)
        velocity: Flow velocity (m/s)
        reynolds_number: Reynolds number (for Two-K method)
        diameter_inches: Pipe diameter (inches) (for Two-K method)

    Returns:
        Total fitting pressure drop (Pa)

    Reference:
        Crane TP-410, Chapter 2: Resistance of Valves and Fittings
    """
    if fittings is None:
        raise ValueError("fittings must be provided")
    total_k = 0.0
    velocity_head = 0.5 * density * (velocity**2)

    for fitting in fittings:
        fitting_type_2k = fitting.fitting_type + "_2k"

        try:
            k_factor = calculate_two_k_factor(
                fitting_type_2k, reynolds_number, diameter_inches
            )
            _logger.debug(
                f"Using Two-K method for {fitting.fitting_type}: K = {k_factor:.3f}"
            )
        except (ValueError, KeyError):
            try:
                k_factor = get_fitting_k_factor(fitting.fitting_type)
                _logger.debug(
                    f"Using standard K for {fitting.fitting_type}: K = {k_factor:.3f}"
                )
            except ValueError:
                k_factor = fitting.k_factor
                _logger.warning(
                    f"Using provided K-factor for {fitting.fitting_type}: "
                    f"K = {k_factor:.3f}"
                )

        total_k += k_factor * fitting.quantity

    dp_fitting = total_k * velocity_head

    _logger.info(f"Fitting losses: Total K = {total_k:.1f}, dP = {dp_fitting:.1f} Pa")
    return dp_fitting


def calculate_elevation_pressure_drop(density: float, elevation_change: float) -> float:
    """Calculate hydrostatic pressure change due to elevation.

    dP_elevation = rho * g * dh

    Args:
        density: Fluid density (kg/m3)
        elevation_change: Elevation change (m, positive = upward flow)

    Returns:
        Elevation pressure drop (Pa, positive = pressure loss)

    Note:
        Positive elevation_change (upward flow) results in positive pressure drop
        (loss). Negative elevation_change (downward flow) results in negative
        pressure drop (gain).
    """
    if density is None:
        raise ValueError("density must be provided")
    dp_elevation = density * GRAVITY * elevation_change

    _logger.debug(f"Elevation: dh={elevation_change:.1f}m, dP={dp_elevation:.1f} Pa")
    return float(dp_elevation)


def calculate_erosional_velocity(
    density: float, service_type: str = "continuous"
) -> float:
    """Calculate erosional velocity limit using API RP 14E correlation.

    V_erosion = C / sqrt(rho)

    Args:
        density: Gas density (kg/m3)
        service_type: 'continuous' (C=100) or 'intermittent' (C=125-150)

    Returns:
        Erosional velocity (m/s)

    Reference:
        API RP 14E: "Recommended Practice for Design and Installation of
        Offshore Production Platform Piping Systems"

    Note:
        C values:
        - Continuous service: C = 100 (conservative)
        - Non-corrosive service: C = 100-125
        - Intermittent service: C = 125-150
        - Solid-free service: C = 150-200
    """
    if density is None:
        raise ValueError("density must be provided")
    if service_type == "continuous":
        C = API_14E_C_CONTINUOUS
    elif service_type in ("intermittent", "non_corrosive"):
        C = API_14E_C_INTERMITTENT
    else:
        C = API_14E_C_CONTINUOUS  # Conservative default

    V_erosion = C / math.sqrt(density * KG_M3_TO_LB_FT3)
    V_erosion_si = V_erosion * FT_S_TO_M_S

    _logger.debug(f"Erosional velocity: {V_erosion_si:.2f} m/s (C={C})")
    return V_erosion_si
