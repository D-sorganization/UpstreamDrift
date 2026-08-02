"""Friction factor correlations for pipe flow calculations.

Provides Darcy friction factor implementations for laminar, transitional,
and turbulent flow regimes.

References:
    - Colebrook, C.F. (1939): J. Inst. Civil Engineers, London
    - Swamee, P.K., Jain, A.K. (1976): J. Hydraulics Division, ASCE
    - Churchill, S.W. (1977): Chemical Engineering, 84(24)
    - Haaland, S.E. (1983): J. Fluids Engineering, 105(1)
"""

import logging
import math

from ...constants import (
    CHURCHILL_B_COEFF,
    COLEBROOK_ROUGHNESS_COEFF,
    LAMINAR_FRICTION_CONSTANT,
    RE_LAMINAR_UPPER,
    SWAMEE_JAIN_COEFF,
)

__all__ = [
    "friction_factor_churchill",
    "friction_factor_colebrook",
    "friction_factor_haaland",
    "friction_factor_laminar",
    "friction_factor_swamee_jain",
    "select_friction_factor_method",
]

_logger = logging.getLogger(__name__)


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
        # Raise rather than silently returning a default (issue #3103 F6):
        # colebrook/swamee-jain delegate here for Re < 2300, so a negative Re
        # would otherwise yield 0.064 with no error and a wrong ΔP.
        raise ValueError(f"Reynolds number must be positive, got {reynolds_number}")

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
    1/sqrt(f) = -2.0 * log10(eps/(3.7D) + 2.51/(Re*sqrt(f)))

    Solved iteratively using Newton-Raphson method.

    Args:
        reynolds_number: Reynolds number
        relative_roughness: eps/D (roughness/diameter)
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
    if reynolds_number is None:
        raise ValueError("reynolds_number must be provided")
    if reynolds_number < RE_LAMINAR_UPPER:
        return friction_factor_laminar(reynolds_number)

    # Initial guess using Swamee-Jain as starting point
    f = friction_factor_swamee_jain(reynolds_number, relative_roughness)

    # Newton-Raphson iteration
    for i in range(max_iterations):
        f_old = f

        term1 = relative_roughness / COLEBROOK_ROUGHNESS_COEFF
        term2 = 2.51 / (reynolds_number * math.sqrt(f))
        f_new = 0.25 / (math.log10(term1 + term2) ** 2)

        if abs(f_new - f_old) < tolerance:
            _logger.debug(f"Colebrook converged in {i + 1} iterations: f = {f_new:.6f}")
            return f_new

        f = f_new

    _logger.warning(f"Colebrook did not converge in {max_iterations} iterations")
    return f


def friction_factor_swamee_jain(
    reynolds_number: float, relative_roughness: float
) -> float:
    """Calculate friction factor using Swamee-Jain explicit approximation.

    f = 0.25 / [log10(eps/(3.7D) + 5.74/Re^0.9)]^2

    Accurate within 1% of Colebrook-White for:
    - 5000 < Re < 10^8
    - 10^-6 < eps/D < 10^-2

    Args:
        reynolds_number: Reynolds number
        relative_roughness: eps/D (roughness/diameter)

    Returns:
        Darcy friction factor

    Reference:
        Swamee, P.K., Jain, A.K. (1976): "Explicit Equations for Pipe-Flow Problems"
        J. Hydraulics Division, ASCE, 102(5), 657-664

    Note:
        Explicit formula, no iteration required. Excellent for computational efficiency.
    """
    if reynolds_number is None:
        raise ValueError("reynolds_number must be provided")
    if reynolds_number < RE_LAMINAR_UPPER:
        return friction_factor_laminar(reynolds_number)

    term1 = relative_roughness / COLEBROOK_ROUGHNESS_COEFF
    term2 = SWAMEE_JAIN_COEFF / (reynolds_number**0.9)

    f = 0.25 / (math.log10(term1 + term2) ** 2)

    _logger.debug(
        f"Swamee-Jain: Re={reynolds_number:.0f}, "
        f"eps/D={relative_roughness:.6f}, f={f:.6f}"
    )
    return f


def friction_factor_churchill(
    reynolds_number: float, relative_roughness: float
) -> float:
    """Calculate friction factor using Churchill explicit correlation.

    Works for all Reynolds numbers (laminar, transitional, turbulent).

    f = 8[(8/Re)^12 + 1/(A + B)^1.5]^(1/12)

    where:
    A = [-2.457 ln((7/Re)^0.9 + 0.27(eps/D))]^16
    B = (37530/Re)^16

    Args:
        reynolds_number: Reynolds number
        relative_roughness: eps/D (roughness/diameter)

    Returns:
        Darcy friction factor

    Reference:
        Churchill, S.W. (1977): "Friction Factor Equation Spans All Fluid Flow Regimes"
        Chemical Engineering, 84(24), 91-92

    Note:
        Single equation valid for all flow regimes. Very useful for transitional flow.
    """
    if reynolds_number is None:
        raise ValueError("reynolds_number must be provided")
    Re = reynolds_number

    if Re < 1:
        return LAMINAR_FRICTION_CONSTANT

    term1 = (7.0 / Re) ** 0.9 + 0.27 * relative_roughness
    A = (-2.457 * math.log(term1)) ** 16

    B = (CHURCHILL_B_COEFF / Re) ** 16

    term2 = (8.0 / Re) ** 12
    term3 = 1.0 / ((A + B) ** 1.5)

    f = 8.0 * ((term2 + term3) ** (1.0 / 12.0))

    _logger.debug(f"Churchill: Re={Re:.0f}, eps/D={relative_roughness:.6f}, f={f:.6f}")
    return float(f)


def friction_factor_haaland(reynolds_number: float, relative_roughness: float) -> float:
    """Calculate friction factor using Haaland explicit approximation.

    1/sqrt(f) ~= -1.8 * log10[(eps/D / 3.7)^1.11 + 6.9/Re]

    Simpler than Colebrook, accurate within 1.5%.

    Args:
        reynolds_number: Reynolds number
        relative_roughness: eps/D

    Returns:
        Darcy friction factor

    Reference:
        Haaland, S.E. (1983): "Simple and Explicit Formulas for Friction Factor"
        J. Fluids Engineering, 105(1), 89-90
    """
    if reynolds_number is None:
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
        relative_roughness: eps/D

    Returns:
        Darcy friction factor

    Raises:
        ValueError: If method is not recognized
    """
    if method is None:
        raise ValueError("method must be provided")
    method = method.lower()

    if method == "colebrook":
        return friction_factor_colebrook(reynolds_number, relative_roughness)
    if method in ("swamee-jain", "swamee_jain"):
        return friction_factor_swamee_jain(reynolds_number, relative_roughness)
    if method == "churchill":
        return friction_factor_churchill(reynolds_number, relative_roughness)
    if method == "haaland":
        return friction_factor_haaland(reynolds_number, relative_roughness)
    available = ["colebrook", "swamee-jain", "churchill", "haaland"]
    raise ValueError(
        f"Unknown friction factor method '{method}'. Available: {available}"
    )
