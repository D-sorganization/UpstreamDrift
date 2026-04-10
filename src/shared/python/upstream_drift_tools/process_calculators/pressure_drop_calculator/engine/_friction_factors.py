"""Friction factor correlations for pressure drop calculations.

Extracted from pressure_drop_calculation_engine.py.
Import via pressure_drop_calculation_engine module.

References:
    - Colebrook-White equation
    - Swamee-Jain approximation
    - Churchill correlation
    - Haaland correlation
"""

from __future__ import annotations

import logging
import math

from ....utils.unit_constants import R_UNIVERSAL_KMOL, STANDARD_GRAVITY
from ...constants import (
    CHURCHILL_B_COEFF,
    COLEBROOK_ROUGHNESS_COEFF,
    FRICTION_FACTOR_DEFAULT_LAMINAR,
    LAMINAR_FRICTION_CONSTANT,
    RE_LAMINAR_UPPER,
    SWAMEE_JAIN_COEFF,
)

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================


GRAVITY = STANDARD_GRAVITY  # m/s²
R_UNIVERSAL = R_UNIVERSAL_KMOL  # J/(kmol·K)
PI = math.pi


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
    if not (result > 0):
        raise ValueError(f"Friction factor must be positive, got {result}")
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
    if not (reynolds_number is not None):
        raise ValueError("reynolds_number must be provided")
    if not (reynolds_number is not None):
        raise ValueError("reynolds_number must be provided")
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
    if not (reynolds_number is not None):
        raise ValueError("reynolds_number must be provided")
    if not (reynolds_number is not None):
        raise ValueError("reynolds_number must be provided")
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
    if not (reynolds_number is not None):
        raise ValueError("reynolds_number must be provided")
    if not (reynolds_number is not None):
        raise ValueError("reynolds_number must be provided")
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
    if not (reynolds_number is not None):
        raise ValueError("reynolds_number must be provided")
    if not (reynolds_number is not None):
        raise ValueError("reynolds_number must be provided")
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
