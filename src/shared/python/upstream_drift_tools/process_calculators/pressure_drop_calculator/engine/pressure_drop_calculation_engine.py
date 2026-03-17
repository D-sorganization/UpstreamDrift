#!/usr/bin/env python3
"""Advanced pressure drop calculation engine for combustion and gasification gases.

This module implements comprehensive pressure drop calculations using industry-standard
correlations with support for compressible flow corrections.

References:
    - Darcy-Weisbach equation for pipe friction
    - Colebrook-White equation for friction factor
    - Moody diagram relationships
    - Panhandle A and B equations for compressible flow
    - Weymouth equation
    - API RP 14E for erosional velocity
    - Crane TP-410 for fitting losses
    - Perry's Chemical Engineers' Handbook, 9th Edition
"""

import logging
import math

from ....utils.unit_constants import R_UNIVERSAL_KMOL, STANDARD_GRAVITY
from ...constants import (
    API_14E_C_CONTINUOUS,
    API_14E_C_INTERMITTENT,
    CHURCHILL_B_COEFF,
    COLEBROOK_ROUGHNESS_COEFF,
    FRICTION_FACTOR_DEFAULT_LAMINAR,
    FT_S_TO_M_S,
    HUNDRED_FEET_IN_METERS,
    KG_M3_TO_LB_FT3,
    LAMINAR_FRICTION_CONSTANT,
    METERS_TO_INCHES,
    RE_LAMINAR_UPPER,
    RE_TURBULENT_LOWER,
    SWAMEE_JAIN_COEFF,
)

# Local imports
from ..models.pressure_drop_data_models import (
    FlowProperties,
    GasComposition,
    PipeFitting,
    PressureDropInputs,
    PressureDropResults,
)
from ..utils.fitting_loss_coefficients import (
    calculate_two_k_factor,
    get_fitting_k_factor,
)
from ..utils.gas_properties import calculate_gas_properties

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================


GRAVITY = STANDARD_GRAVITY  # m/s²
R_UNIVERSAL = R_UNIVERSAL_KMOL  # J/(kmol·K)
PI = math.pi


# ============================================================================
# FRICTION FACTOR CORRELATIONS
# ============================================================================


def friction_factor_laminar(reynolds_number: float) -> float:
    """Calculate friction factor for laminar flow (Re < 2300).

    f = 64 / Re  (Hagen-Poiseuille equation)

    Args:
        reynolds_number: Reynolds number (must be positive)

    Returns:
        Darcy friction factor (always positive)

    Reference:
        Hagen, G. (1839), Poiseuille, J. (1840): Laminar flow in pipes
    """
    if reynolds_number <= 0:
        logger.error("Reynolds number must be positive")
        return FRICTION_FACTOR_DEFAULT_LAMINAR  # Default for Re ~ 1000

    result = LAMINAR_FRICTION_CONSTANT / reynolds_number
    assert result > 0, f"Friction factor must be positive, got {result}"
    return result


def friction_factor_colebrook(
    reynolds_number: float,
    relative_roughness: float,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
) -> float:
    """Calculate friction factor using Colebrook-White equation (implicit).

    Colebrook-White equation (turbulent flow, Re > 4000):
    1/√f = -2.0 × log10(ε/(3.7D) + 2.51/(Re×√f))

    Solved iteratively using Newton-Raphson method.

    Args:
        reynolds_number: Reynolds number
        relative_roughness: ε/D (roughness/diameter)
        max_iterations: Maximum iterations for convergence
        tolerance: Convergence tolerance

    Returns:
        Darcy friction factor

    Reference:
        Colebrook, C.F. (1939): "Turbulent Flow in Pipes, with Particular Reference
        to the Transition Region Between Smooth and Rough Pipe Laws"
        J. Inst. Civil Engineers, London, 11, 133-156

    Note:
        This is the most accurate correlation but requires iteration.
        The Moody diagram is a graphical representation of this equation.
    """
    assert reynolds_number is not None, "reynolds_number must be provided"
    assert reynolds_number is not None, "reynolds_number must be provided"
    if reynolds_number < RE_LAMINAR_UPPER:
        return friction_factor_laminar(reynolds_number)

    # Initial guess using Swamee-Jain as starting point
    f = friction_factor_swamee_jain(reynolds_number, relative_roughness)

    # Newton-Raphson iteration
    for i in range(max_iterations):
        f_old = f

        # Colebrook-White equation rearranged
        term1 = relative_roughness / COLEBROOK_ROUGHNESS_COEFF
        term2 = 2.51 / (reynolds_number * math.sqrt(f))
        f_new = 0.25 / (math.log10(term1 + term2) ** 2)

        # Check convergence
        if abs(f_new - f_old) < tolerance:
            logger.debug(f"Colebrook converged in {i + 1} iterations: f = {f_new:.6f}")
            return f_new

        f = f_new

    logger.warning(f"Colebrook did not converge in {max_iterations} iterations")
    return f


def friction_factor_swamee_jain(
    reynolds_number: float, relative_roughness: float
) -> float:
    """Calculate friction factor using Swamee-Jain explicit approximation.

    f = 0.25 / [log10(ε/(3.7D) + 5.74/Re^0.9)]²

    Accurate within 1% of Colebrook-White for:
    - 5000 < Re < 10^8
    - 10^-6 < ε/D < 10^-2

    Args:
        reynolds_number: Reynolds number
        relative_roughness: ε/D (roughness/diameter)

    Returns:
        Darcy friction factor

    Reference:
        Swamee, P.K., Jain, A.K. (1976): "Explicit Equations for Pipe-Flow Problems"
        J. Hydraulics Division, ASCE, 102(5), 657-664

    Note:
        Explicit formula, no iteration required. Excellent for computational efficiency.
    """
    assert reynolds_number is not None, "reynolds_number must be provided"
    assert reynolds_number is not None, "reynolds_number must be provided"
    if reynolds_number < RE_LAMINAR_UPPER:
        return friction_factor_laminar(reynolds_number)

    # Swamee-Jain equation
    term1 = relative_roughness / COLEBROOK_ROUGHNESS_COEFF
    term2 = SWAMEE_JAIN_COEFF / (reynolds_number**0.9)

    f = 0.25 / (math.log10(term1 + term2) ** 2)

    logger.debug(
        f"Swamee-Jain: Re={reynolds_number:.0f}, ε/D={relative_roughness:.6f}, f={f:.6f}"
    )
    return f


def friction_factor_churchill(
    reynolds_number: float, relative_roughness: float
) -> float:
    """Calculate friction factor using Churchill explicit correlation.

    Works for all Reynolds numbers (laminar, transitional, turbulent).

    f = 8[(8/Re)^12 + 1/(A + B)^1.5]^(1/12)

    where:
    A = [-2.457 ln((7/Re)^0.9 + 0.27(ε/D))]^16
    B = (37530/Re)^16

    Args:
        reynolds_number: Reynolds number
        relative_roughness: ε/D (roughness/diameter)

    Returns:
        Darcy friction factor

    Reference:
        Churchill, S.W. (1977): "Friction Factor Equation Spans All Fluid Flow Regimes"
        Chemical Engineering, 84(24), 91-92

    Note:
        Single equation valid for all flow regimes. Very useful for transitional flow.
    """
    assert reynolds_number is not None, "reynolds_number must be provided"
    assert reynolds_number is not None, "reynolds_number must be provided"
    Re = reynolds_number

    if Re < 1:
        return LAMINAR_FRICTION_CONSTANT  # Avoid division by zero

    # Churchill correlation
    term1 = (7.0 / Re) ** 0.9 + 0.27 * relative_roughness
    A = (-2.457 * math.log(term1)) ** 16

    B = (CHURCHILL_B_COEFF / Re) ** 16

    term2 = (8.0 / Re) ** 12
    term3 = 1.0 / ((A + B) ** 1.5)

    f = 8.0 * ((term2 + term3) ** (1.0 / 12.0))

    logger.debug(f"Churchill: Re={Re:.0f}, ε/D={relative_roughness:.6f}, f={f:.6f}")
    return float(f)


def friction_factor_haaland(reynolds_number: float, relative_roughness: float) -> float:
    """Calculate friction factor using Haaland explicit approximation.

    1/√f ≈ -1.8 × log10[(ε/D / 3.7)^1.11 + 6.9/Re]

    Simpler than Colebrook, accurate within 1.5%.

    Args:
        reynolds_number: Reynolds number
        relative_roughness: ε/D

    Returns:
        Darcy friction factor

    Reference:
        Haaland, S.E. (1983): "Simple and Explicit Formulas for Friction Factor"
        J. Fluids Engineering, 105(1), 89-90
    """
    assert reynolds_number is not None, "reynolds_number must be provided"
    assert reynolds_number is not None, "reynolds_number must be provided"
    if reynolds_number < RE_LAMINAR_UPPER:
        return friction_factor_laminar(reynolds_number)

    term1 = (relative_roughness / COLEBROOK_ROUGHNESS_COEFF) ** 1.11
    term2 = 6.9 / reynolds_number

    inv_sqrt_f = -1.8 * math.log10(term1 + term2)
    f = 1.0 / (inv_sqrt_f**2)

    return f


def select_friction_factor_method(
    method: str, reynolds_number: float, relative_roughness: float
) -> float:
    """Select and calculate friction factor using specified method.

    Args:
        method: Method name ('colebrook', 'swamee-jain', 'churchill', 'haaland')
        reynolds_number: Reynolds number
        relative_roughness: ε/D

    Returns:
        Darcy friction factor

    Raises:
        ValueError: If method is not recognized
    """
    method = method.lower()

    if method == "colebrook":
        return friction_factor_colebrook(reynolds_number, relative_roughness)
    if method == "swamee-jain" or method == "swamee_jain":
        return friction_factor_swamee_jain(reynolds_number, relative_roughness)
    if method == "churchill":
        return friction_factor_churchill(reynolds_number, relative_roughness)
    if method == "haaland":
        return friction_factor_haaland(reynolds_number, relative_roughness)
    available = ["colebrook", "swamee-jain", "churchill", "haaland"]
    raise ValueError(
        f"Unknown friction factor method '{method}'. Available: {available}"
    )


# ============================================================================
# FLOW PROPERTY CALCULATIONS
# ============================================================================


def calculate_flow_properties(inputs: PressureDropInputs) -> FlowProperties:
    """Calculate comprehensive flow properties for the gas mixture.

    Args:
        inputs: Pressure drop input parameters

    Returns:
        FlowProperties object with all calculated properties

    Raises:
        ValueError: If calculations fail
    """
    # DbC preconditions
    assert (
        inputs.pipe_diameter > 0
    ), f"Pipe diameter must be positive, got {inputs.pipe_diameter}"
    assert (
        inputs.mass_flow_rate > 0
    ), f"Mass flow rate must be positive, got {inputs.mass_flow_rate}"
    assert (
        inputs.inlet_temperature > 0
    ), f"Inlet temperature must be positive (K), got {inputs.inlet_temperature}"
    assert (
        inputs.inlet_pressure > 0
    ), f"Inlet pressure must be positive, got {inputs.inlet_pressure}"

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
    assert density > 0, f"Gas density must be positive, got {density}"
    assert viscosity > 0, f"Gas viscosity must be positive, got {viscosity}"
    assert speed_of_sound > 0, f"Speed of sound must be positive, got {speed_of_sound}"

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
    assert (
        flow_props.velocity > 0
    ), f"Flow velocity must be positive, got {flow_props.velocity}"
    assert (
        flow_props.reynolds_number > 0
    ), f"Reynolds number must be positive, got {flow_props.reynolds_number}"
    assert (
        0 <= flow_props.mach_number < 50
    ), f"Mach number out of physical range, got {flow_props.mach_number}"

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
    assert (
        friction_factor > 0
    ), f"friction_factor must be positive, got {friction_factor}"
    assert length > 0, f"length must be positive, got {length}"
    assert diameter > 0, f"diameter must be positive, got {diameter}"
    assert density > 0, f"density must be positive, got {density}"
    assert velocity > 0, f"velocity must be positive, got {velocity}"

    velocity_head = 0.5 * density * (velocity**2)
    dp_friction = friction_factor * (length / diameter) * velocity_head

    # DbC postcondition
    assert dp_friction >= 0, f"Pressure drop must be non-negative, got {dp_friction}"

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
    assert fittings is not None, "fittings must be provided"
    assert fittings is not None, "fittings must be provided"
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
    assert density is not None, "density must be provided"
    assert density is not None, "density must be provided"
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
    assert P1 is not None, "P1 must be provided"
    assert P1 is not None, "P1 must be provided"
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
    assert diameter > 0, f"diameter must be positive, got {diameter}"
    assert temperature > 0, f"temperature must be positive (K), got {temperature}"
    assert (
        molecular_weight > 0
    ), f"molecular_weight must be positive, got {molecular_weight}"

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
    assert inlet_pressure is not None, "inlet_pressure must be provided"
    assert inlet_pressure is not None, "inlet_pressure must be provided"
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
    assert density is not None, "density must be provided"
    assert density is not None, "density must be provided"
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


class PressureDropCalculationEngine:
    """Advanced pressure drop calculation engine."""

    def __init__(self) -> None:
        """Initialize the calculation engine."""
        logger.info("PressureDropCalculationEngine initialized")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_incompressible_components(
        inputs: PressureDropInputs,
        flow_props: FlowProperties,
        friction_factor: float,
    ) -> tuple[float, float, float, float]:
        """Compute the three incompressible ΔP terms.

        Returns:
            (dp_friction, dp_fittings, dp_elevation, total_k_factor)
        """
        assert inputs is not None, "inputs must be provided"
        assert inputs is not None, "inputs must be provided"
        dp_friction = calculate_frictional_pressure_drop(
            friction_factor,
            inputs.pipe_length,
            inputs.pipe_diameter,
            flow_props.density,
            flow_props.velocity,
        )

        diameter_inches = inputs.pipe_diameter * METERS_TO_INCHES
        dp_fittings = calculate_fitting_pressure_drop(
            inputs.fittings,
            flow_props.density,
            flow_props.velocity,
            flow_props.reynolds_number,
            diameter_inches,
        )

        dp_elevation = calculate_elevation_pressure_drop(
            flow_props.density, inputs.elevation_change
        )

        total_k_factor = sum(
            f.k_factor * f.quantity if f.k_factor > 0 else 0.0 for f in inputs.fittings
        )

        return dp_friction, dp_fittings, dp_elevation, total_k_factor

    @staticmethod
    def _apply_compressibility(
        inputs: PressureDropInputs,
        flow_props: FlowProperties,
        friction_factor: float,
        dp_incompressible: float,
        total_k_factor: float,
    ) -> tuple[float, float, float, list[str]]:
        """Decide whether to apply compressible-flow corrections.

        Returns:
            (total_dp, outlet_pressure, dp_acceleration, warnings)
        """
        assert inputs is not None, "inputs must be provided"
        assert inputs is not None, "inputs must be provided"
        warnings_list: list[str] = []
        pressure_ratio_initial = dp_incompressible / inputs.inlet_pressure

        if inputs.compressibility_correction and pressure_ratio_initial > 0.05:
            logger.info(
                f"Applying compressible flow correction "
                f"(ΔP/P = {pressure_ratio_initial * 100:.1f}%)"
            )
            total_dp, outlet_pressure = calculate_compressible_flow_correction(
                inlet_pressure=inputs.inlet_pressure,
                outlet_pressure=inputs.inlet_pressure - dp_incompressible,
                length=inputs.pipe_length,
                diameter=inputs.pipe_diameter,
                mass_flow_rate=inputs.mass_flow_rate,
                temperature=inputs.inlet_temperature,
                molecular_weight=flow_props.molecular_weight,
                compressibility_factor=flow_props.compressibility_factor,
                friction_factor=friction_factor,
                total_k_factor=total_k_factor,
            )
            dp_acceleration = max(total_dp - dp_incompressible, 0.0)

            if abs(total_dp - dp_incompressible) > 100:
                logger.info(
                    f"Compressibility effect: ΔP_incomp={dp_incompressible:.0f} Pa, "
                    f"ΔP_comp={total_dp:.0f} Pa "
                    f"(+{(total_dp / dp_incompressible - 1) * 100:.1f}%)"
                )
        else:
            dp_acceleration = 0.0
            total_dp = dp_incompressible
            outlet_pressure = inputs.inlet_pressure - total_dp

        # Negative outlet pressure → choked flow
        if outlet_pressure < 0:
            logger.error(
                f"Calculated negative outlet pressure: {outlet_pressure:.1f} Pa"
            )
            warnings_list.append(
                "Negative outlet pressure calculated - flow may be choked"
            )
            outlet_pressure = 0.0
            total_dp = inputs.inlet_pressure

        # Warn if correction disabled but needed
        pressure_ratio = total_dp / inputs.inlet_pressure
        if pressure_ratio > 0.1 and not inputs.compressibility_correction:
            warnings_list.append(
                f"High pressure drop ratio ({pressure_ratio * 100:.1f}%) - "
                "consider enabling compressibility_correction=True for better accuracy"
            )

        return total_dp, outlet_pressure, dp_acceleration, warnings_list

    @staticmethod
    def _build_results(
        *,
        inputs: PressureDropInputs,
        flow_props: FlowProperties,
        flow_regime: str,
        friction_factor: float,
        dp_friction: float,
        dp_fittings: float,
        dp_elevation: float,
        dp_acceleration: float,
        total_dp: float,
        outlet_pressure: float,
        warnings_list: list[str],
    ) -> PressureDropResults:
        """Construct the results object and perform final safety checks."""
        erosional_velocity = calculate_erosional_velocity(
            flow_props.density, "continuous"
        )
        erosion_ratio = flow_props.velocity / erosional_velocity

        if erosion_ratio > 0.5:
            warnings_list.append(
                f"Velocity is {erosion_ratio * 100:.0f}% of erosional limit"
            )
        if erosion_ratio > 1.0:
            warnings_list.append(
                "WARNING: Velocity exceeds erosional limit - risk of pipe erosion!"
            )

        velocity_pressure = 0.5 * flow_props.density * (flow_props.velocity**2)
        dp_per_100ft = (total_dp / inputs.pipe_length) * HUNDRED_FEET_IN_METERS

        results = PressureDropResults(
            total_pressure_drop=total_dp,
            outlet_pressure=outlet_pressure,
            friction_pressure_drop=dp_friction,
            fitting_pressure_drop=dp_fittings,
            elevation_pressure_drop=dp_elevation,
            acceleration_pressure_drop=dp_acceleration,
            friction_factor=friction_factor,
            flow_properties=flow_props,
            pressure_drop_per_100ft=dp_per_100ft,
            velocity_pressure=velocity_pressure,
            erosional_velocity=erosional_velocity,
            erosion_ratio=erosion_ratio,
            flow_regime=flow_regime,
            warnings=warnings_list,
        )

        # Log summary
        logger.info("=" * 80)
        logger.info("RESULTS SUMMARY")
        logger.info("=" * 80)
        logger.info(
            f"Total pressure drop: {total_dp / 1e5:.4f} bar ({total_dp:.1f} Pa)"
        )
        logger.info(
            f"  Friction: {dp_friction:.1f} Pa ({dp_friction / total_dp * 100:.1f}%)"
        )
        logger.info(
            f"  Fittings: {dp_fittings:.1f} Pa ({dp_fittings / total_dp * 100:.1f}%)"
        )
        logger.info(f"  Elevation: {dp_elevation:.1f} Pa")
        logger.info(f"Outlet pressure: {outlet_pressure / 1e5:.4f} bar")
        logger.info(f"Erosion ratio: {erosion_ratio * 100:.1f}%")
        logger.info("=" * 80)

        return results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate(self, inputs: PressureDropInputs) -> PressureDropResults:
        """Calculate comprehensive pressure drop analysis.

        Args:
            inputs: PressureDropInputs object with all parameters

        Returns:
            PressureDropResults object with complete analysis

        Raises:
            ValueError: If inputs are invalid
        """
        is_valid, error_msg = inputs.validate()
        if not is_valid:
            logger.error(f"Input validation failed: {error_msg}")
            raise ValueError(f"Invalid inputs: {error_msg}")

        logger.info("=" * 80)
        logger.info("PRESSURE DROP CALCULATION")
        logger.info("=" * 80)

        # Step 1: Flow properties & regime
        flow_props = calculate_flow_properties(inputs)
        flow_regime = classify_flow_regime(flow_props.reynolds_number)
        logger.info(
            f"Flow regime: {flow_regime} (Re = {flow_props.reynolds_number:.0f})"
        )

        # Step 2: Friction factor
        relative_roughness = inputs.pipe_roughness / inputs.pipe_diameter
        friction_factor = select_friction_factor_method(
            inputs.friction_method, flow_props.reynolds_number, relative_roughness
        )
        logger.info(
            f"Friction factor ({inputs.friction_method}): f = {friction_factor:.6f}"
        )

        # Step 3: Incompressible ΔP components
        dp_friction, dp_fittings, dp_elevation, total_k_factor = (
            self._compute_incompressible_components(inputs, flow_props, friction_factor)
        )
        dp_incompressible = dp_friction + dp_fittings + dp_elevation

        # Step 4: Compressibility correction (if applicable)
        total_dp, outlet_pressure, dp_acceleration, warnings_list = (
            self._apply_compressibility(
                inputs,
                flow_props,
                friction_factor,
                dp_incompressible,
                total_k_factor,
            )
        )

        # Step 5: Build result object
        return self._build_results(
            inputs=inputs,
            flow_props=flow_props,
            flow_regime=flow_regime,
            friction_factor=friction_factor,
            dp_friction=dp_friction,
            dp_fittings=dp_fittings,
            dp_elevation=dp_elevation,
            dp_acceleration=dp_acceleration,
            total_dp=total_dp,
            outlet_pressure=outlet_pressure,
            warnings_list=warnings_list,
        )


if __name__ == "__main__":
    # Demonstration
    logging.basicConfig(level=logging.INFO)

    logger.info("\n" + "=" * 80)
    logger.info("PRESSURE DROP CALCULATION ENGINE - EXAMPLE")
    logger.info("=" * 80)

    # Example: Syngas in 6" Schedule 40 pipe
    from ..models.pressure_drop_data_models import GasComposition, PipeFitting

    # Define gas composition (syngas)
    composition = GasComposition(
        components={
            "H2": 0.30,
            "CO": 0.40,
            "CO2": 0.15,
            "N2": 0.10,
            "CH4": 0.05,
        }
    )

    # Define fittings
    fittings = [
        PipeFitting("90_elbow_std", quantity=4, k_factor=30),
        PipeFitting("gate_valve_open", quantity=2, k_factor=8),
        PipeFitting("tee_through_run", quantity=1, k_factor=20),
    ]

    # Create inputs
    inputs = PressureDropInputs(
        pipe_diameter=0.15408,  # 6" Schedule 40 (154 mm ID)
        pipe_length=100.0,  # 100 m
        pipe_roughness=0.000045,  # Commercial steel (0.045 mm)
        elevation_change=0.0,  # Horizontal pipe
        mass_flow_rate=2.0,  # 2 kg/s
        inlet_pressure=25e5,  # 25 bar
        inlet_temperature=800,  # 800 K (527°C)
        gas_composition=composition,
        fittings=fittings,
        compressibility_correction=True,
        friction_method="colebrook",
    )

    # Calculate
    engine = PressureDropCalculationEngine()
    results = engine.calculate(inputs)

    # Display results
    logger.info("\n" + "-" * 80)
    logger.info("CALCULATION RESULTS")
    logger.info("-" * 80)
    for key, value in results.to_dict().items():
        if isinstance(value, float):
            logger.info(f"{key:40s}: {value:.6g}")
        else:
            logger.info(f"{key:40s}: {value}")

    if results.warnings:
        logger.warning("WARNINGS:")
        for warning in results.warnings:
            logger.warning(f"  {warning}")
