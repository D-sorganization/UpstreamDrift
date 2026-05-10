"""Flow property and pressure drop calculation functions.

Extracted from pressure_drop_calculation_engine.py.
Import via pressure_drop_calculation_engine module.
"""

from __future__ import annotations

import logging
import math

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
from ._friction_factors import (
    GRAVITY,
    PI,
    R_UNIVERSAL,
)

logger = logging.getLogger(__name__)


def calculate_flow_properties(
    inputs: PressureDropInputs,
) -> FlowProperties:  # noqa: C901
    """Calculate comprehensive flow properties for the gas mixture.

    Args:
        inputs: Pressure drop input parameters

    Returns:
        FlowProperties object with all calculated properties

    Raises:
        ValueError: If calculations fail
    """
    # DbC preconditions
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

    # Calculate gas mixture properties (now includes gamma and speed of sound)
    gas_props = calculate_gas_properties(
        inputs.gas_composition.components,
        inputs.inlet_temperature,
        inputs.inlet_pressure,
        inputs.compressibility_correction,
    )

    # Extract properties
    density = gas_props["density"]
    viscosity = gas_props["viscosity"]
    molecular_weight = gas_props["molecular_weight"]
    compressibility_factor = gas_props["compressibility_factor"]
    speed_of_sound = gas_props["speed_of_sound"]  # Now calculated dynamically
    heat_capacity_ratio = gas_props["heat_capacity_ratio"]  # γ = Cp/Cv

    # DbC: intermediate invariants on physical properties
    if not (density > 0):
        raise ValueError(f"Gas density must be positive, got {density}")
    if not (viscosity > 0):
        raise ValueError(f"Gas viscosity must be positive, got {viscosity}")
    if not (speed_of_sound > 0):
        raise ValueError(f"Speed of sound must be positive, got {speed_of_sound}")

    # Calculate flow velocity
    pipe_area = PI * (inputs.pipe_diameter**2) / 4.0
    velocity = inputs.mass_flow_rate / (density * pipe_area)
    mass_flux = inputs.mass_flow_rate / pipe_area

    # Reynolds number
    reynolds_number = (density * velocity * inputs.pipe_diameter) / viscosity

    # Mach number using dynamically calculated speed of sound
    mach_number = velocity / speed_of_sound

    # Volumetric flow rate
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

    # DbC postconditions
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

    logger.info("Flow properties calculated:")
    logger.info(f"  Velocity: {velocity:.2f} m/s")
    logger.info(f"  Reynolds: {reynolds_number:.0f}")
    logger.info(f"  Mach: {mach_number:.4f}")
    logger.info(f"  Density: {density:.4f} kg/m³")
    logger.info(f"  γ (Cp/Cv): {heat_capacity_ratio:.3f}")
    logger.info(f"  Speed of sound: {speed_of_sound:.1f} m/s")

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


# ============================================================================
# PRESSURE DROP CALCULATIONS
# ============================================================================


def calculate_frictional_pressure_drop(
    friction_factor: float,
    length: float,
    diameter: float,
    density: float,
    velocity: float,
) -> float:
    """Calculate frictional pressure drop using Darcy-Weisbach equation.

    ΔP_friction = f × (L/D) × (ρV²/2)

    Args:
        friction_factor: Darcy friction factor (must be positive)
        length: Pipe length in m (must be positive)
        diameter: Pipe diameter in m (must be positive)
        density: Fluid density in kg/m³ (must be positive)
        velocity: Flow velocity in m/s (must be positive)

    Returns:
        Frictional pressure drop in Pa (non-negative)

    Reference:
        Darcy, H. (1857), Weisbach, J. (1845): Pipe flow friction equation
    """
    # DbC preconditions
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

    # DbC postcondition
    if not (dp_friction >= 0):
        raise ValueError(f"Pressure drop must be non-negative, got {dp_friction}")

    logger.debug(
        f"Darcy-Weisbach: f={friction_factor:.6f}, L/D={length / diameter:.1f}, ΔP={dp_friction:.1f} Pa"
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

    ΔP_fitting = Σ(K_i) × (ρV²/2)

    Args:
        fittings: List of PipeFitting objects
        density: Fluid density (kg/m³)
        velocity: Flow velocity (m/s)
        reynolds_number: Reynolds number (for Two-K method)
        diameter_inches: Pipe diameter (inches) (for Two-K method)

    Returns:
        Total fitting pressure drop (Pa)

    Reference:
        Crane TP-410, Chapter 2: Resistance of Valves and Fittings
    """
    if not (fittings is not None):
        raise ValueError("fittings must be provided")
    if not (fittings is not None):
        raise ValueError("fittings must be provided")
    total_k = 0.0
    velocity_head = 0.5 * density * (velocity**2)

    for fitting in fittings:
        # Try to use Two-K method if available
        fitting_type_2k = fitting.fitting_type + "_2k"

        try:
            # Use Two-K method for better accuracy
            k_factor = calculate_two_k_factor(
                fitting_type_2k, reynolds_number, diameter_inches
            )
            logger.debug(
                f"Using Two-K method for {fitting.fitting_type}: K = {k_factor:.3f}"
            )
        except (ValueError, KeyError):
            # Fall back to standard K-factor
            try:
                k_factor = get_fitting_k_factor(fitting.fitting_type)
                logger.debug(
                    f"Using standard K for {fitting.fitting_type}: K = {k_factor:.3f}"
                )
            except ValueError:
                # Use provided K-factor
                k_factor = fitting.k_factor
                logger.warning(
                    f"Using provided K-factor for {fitting.fitting_type}: K = {k_factor:.3f}"
                )

        total_k += k_factor * fitting.quantity

    dp_fitting = total_k * velocity_head

    logger.info(f"Fitting losses: Total K = {total_k:.1f}, ΔP = {dp_fitting:.1f} Pa")
    return dp_fitting


def calculate_elevation_pressure_drop(density: float, elevation_change: float) -> float:
    """Calculate hydrostatic pressure change due to elevation.

    ΔP_elevation = ρ × g × Δh

    Args:
        density: Fluid density (kg/m³)
        elevation_change: Elevation change (m, positive = upward flow)

    Returns:
        Elevation pressure drop (Pa, positive = pressure loss)

    Note:
        Positive elevation_change (upward flow) results in positive pressure drop (loss).
        Negative elevation_change (downward flow) results in negative pressure drop (gain).
    """
    if not (density is not None):
        raise ValueError("density must be provided")
    if not (density is not None):
        raise ValueError("density must be provided")
    dp_elevation = density * GRAVITY * elevation_change

    logger.debug(f"Elevation: Δh={elevation_change:.1f}m, ΔP={dp_elevation:.1f} Pa")
    return float(dp_elevation)


def _iterate_compressible_pressure(
    P1: float,
    P2_initial: float,
    coeff: float,
    resistance: float,
    max_iterations: int = 50,
    tolerance: float = 1.0,
) -> tuple[float, bool]:
    """Iteratively solve the isothermal compressible flow equation for P2.

    Uses fixed-point iteration on:
    P₂² = P₁² - coeff × (resistance + 2 × ln(P₁/P₂))

    Args:
        P1: Inlet pressure (Pa)
        P2_initial: Initial guess for outlet pressure (Pa)
        coeff: G² × Z × R × T / M coefficient
        resistance: f × L/D + sum(K) resistance term
        max_iterations: Maximum iteration count
        tolerance: Convergence tolerance (Pa)

    Returns:
        Tuple of (converged_P2, is_choked). If choked, P2 is meaningless.
    """
    if not (P1 is not None):
        raise ValueError("P1 must be provided")
    if not (P1 is not None):
        raise ValueError("P1 must be provided")
    P2 = P2_initial

    for iteration in range(max_iterations):
        P2_old = P2

        ln_term = 2.0 * math.log(P1 / P2) if P2 > 0 and P1 > P2 else 0.0
        rhs = coeff * (resistance + ln_term)
        P2_squared = P1**2 - rhs

        if P2_squared <= 0:
            logger.warning(
                "Compressible flow calculation indicates choked flow condition"
            )
            return P2, True

        P2 = math.sqrt(P2_squared)

        if abs(P2 - P2_old) < tolerance:
            logger.debug(f"Compressible flow converged in {iteration + 1} iterations")
            break

    return P2, False


def calculate_compressible_flow_correction(
    inlet_pressure: float,
    outlet_pressure: float,
    length: float,
    diameter: float,
    mass_flow_rate: float,
    temperature: float,
    molecular_weight: float,
    compressibility_factor: float,
    friction_factor: float,
    total_k_factor: float = 0.0,
) -> tuple[float, float]:
    """Calculate pressure drop accounting for compressibility effects.

    For high pressure drops (ΔP/P > 10%), gas density changes significantly
    along the pipe and requires correction.

    Uses isothermal compressible flow equation (derived from continuity and momentum):
    P₁² - P₂² = G² × (Z × R × T / M) × [f × L/D + ΣK + 2 × ln(P₁/P₂)]

    Args:
        inlet_pressure: Inlet pressure (Pa)
        outlet_pressure: Initial estimate of outlet pressure (Pa)
        length: Pipe length (m)
        diameter: Pipe diameter (m)
        mass_flow_rate: Mass flow rate (kg/s)
        temperature: Temperature (K)
        molecular_weight: Molecular weight (kg/kmol)
        compressibility_factor: Z-factor
        friction_factor: Darcy friction factor
        total_k_factor: Sum of fitting K-factors

    Returns:
        Tuple of (corrected_pressure_drop, corrected_outlet_pressure) in Pa

    Reference:
        Perry's Chemical Engineers' Handbook, 9th Ed, Section 6
        Crane TP-410: Flow of Compressible Fluids in Pipelines
    """
    # DbC preconditions
    if not (diameter > 0):
        raise ValueError(f"diameter must be positive, got {diameter}")
    if not (temperature > 0):
        raise ValueError(f"temperature must be positive (K), got {temperature}")
    if not (molecular_weight > 0):
        raise ValueError(f"molecular_weight must be positive, got {molecular_weight}")

    area = PI * (diameter**2) / 4.0
    G = mass_flow_rate / area
    resistance = friction_factor * (length / diameter) + total_k_factor

    coeff = (
        (G**2) * (compressibility_factor * R_UNIVERSAL * temperature) / molecular_weight
    )

    P2, is_choked = _iterate_compressible_pressure(
        inlet_pressure,
        outlet_pressure,
        coeff,
        resistance,
    )

    if is_choked:
        return inlet_pressure - outlet_pressure, outlet_pressure

    corrected_dp = inlet_pressure - P2

    pressure_ratio = P2 / inlet_pressure
    if pressure_ratio > 0:
        expansion_factor = math.sqrt(
            pressure_ratio * (1 - pressure_ratio**2) / (1 - pressure_ratio)
            if pressure_ratio < 1
            else 1.0
        )
    else:
        expansion_factor = 1.0

    logger.debug(
        f"Compressible flow correction: ΔP_incomp={inlet_pressure - outlet_pressure:.0f} Pa, "
        f"ΔP_comp={corrected_dp:.0f} Pa, Y={expansion_factor:.3f}"
    )

    return corrected_dp, P2


def calculate_expansion_factor(
    inlet_pressure: float,
    pressure_drop: float,
    friction_factor: float,
    length_over_diameter: float,
    gamma: float = 1.4,
) -> float:
    """Calculate gas expansion factor Y for compressible flow.

    The expansion factor Y accounts for gas expansion through restrictions
    and is used to correct incompressible flow equations for gas flow.

    Y = 1 - (0.41 + 0.35 × β⁴) × (ΔP / (κ × P₁))

    For pipe flow (β = 1):
    Y ≈ 1 - (ΔP / (3 × κ × P₁)) for ΔP/P₁ < 0.1
    Y = √[κ × (P₂/P₁)^(2/κ) × (1 - (P₂/P₁)^((κ-1)/κ)) / ((κ-1) × (1 - P₂/P₁))]

    Args:
        inlet_pressure: Inlet pressure (Pa)
        pressure_drop: Pressure drop (Pa)
        friction_factor: Darcy friction factor
        length_over_diameter: L/D ratio
        gamma: Heat capacity ratio (κ = Cp/Cv)

    Returns:
        Expansion factor Y (dimensionless, 0 < Y ≤ 1)

    Reference:
        Crane TP-410, Section 2-2: Compressible Flow
        ISO 5167: Measurement of fluid flow
    """
    if not (inlet_pressure is not None):
        raise ValueError("inlet_pressure must be provided")
    if not (inlet_pressure is not None):
        raise ValueError("inlet_pressure must be provided")
    if inlet_pressure <= 0 or pressure_drop < 0:
        return 1.0

    # Pressure ratio
    pressure_ratio = (inlet_pressure - pressure_drop) / inlet_pressure

    if pressure_ratio <= 0:
        # Choked flow condition
        return 0.0

    if pressure_ratio >= 0.99:
        # Nearly incompressible
        return 1.0

    # Calculate Y using the adiabatic expansion formula
    # Y = √[κ × r^(2/κ) × (1 - r^((κ-1)/κ)) / ((κ-1) × (1 - r))]
    # where r = P₂/P₁

    r = pressure_ratio
    k = gamma

    try:
        numerator = k * (r ** (2.0 / k)) * (1.0 - r ** ((k - 1.0) / k))
        denominator = (k - 1.0) * (1.0 - r)

        Y = 1.0 if denominator <= 0 else math.sqrt(numerator / denominator)
    except (ValueError, ZeroDivisionError):
        # Fallback to simplified formula
        Y = 1.0 - pressure_drop / (3.0 * gamma * inlet_pressure)

    # Bound Y to physical limits
    Y = max(0.0, min(Y, 1.0))

    return Y


# ============================================================================
# EROSIONAL VELOCITY
# ============================================================================


def calculate_erosional_velocity(
    density: float, service_type: str = "continuous"
) -> float:
    """Calculate erosional velocity limit using API RP 14E correlation.

    V_erosion = C / √ρ

    Args:
        density: Gas density (kg/m³)
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
    if not (density is not None):
        raise ValueError("density must be provided")
    if not (density is not None):
        raise ValueError("density must be provided")
    if service_type == "continuous":
        C = API_14E_C_CONTINUOUS
    elif service_type == "intermittent" or service_type == "non_corrosive":
        C = API_14E_C_INTERMITTENT
    else:
        C = API_14E_C_CONTINUOUS  # Conservative default

    # Convert C from ft/s to m/s units
    # API formula: V (ft/s) = C / √(ρ in lb/ft³)
    # For SI units: V (m/s) = C_si / √(ρ in kg/m³)

    # Conversion: C_si ≈ C × 0.0458 for density in kg/m³
    # C_si = C * 0.0458 / (3.281**0.5)  # Approximate conversion (unused)

    V_erosion = C / math.sqrt(
        density * KG_M3_TO_LB_FT3
    )  # Convert kg/m³ to lb/ft³ first
    V_erosion_si = V_erosion * FT_S_TO_M_S  # Convert ft/s to m/s

    logger.debug(f"Erosional velocity: {V_erosion_si:.2f} m/s (C={C})")
    return V_erosion_si


# ============================================================================
# MAIN CALCULATION ENGINE
# ============================================================================
