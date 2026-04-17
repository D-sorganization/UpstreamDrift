"""Friction factor correlations for the pressure drop engine."""

from __future__ import annotations

import logging
import math

from ...constants import (
    CHURCHILL_B_COEFF,
    COLEBROOK_ROUGHNESS_COEFF,
    FRICTION_FACTOR_DEFAULT_LAMINAR,
    LAMINAR_FRICTION_CONSTANT,
    RE_LAMINAR_UPPER,
    SWAMEE_JAIN_COEFF,
)

logger = logging.getLogger(__name__)


def friction_factor_laminar(reynolds_number: float) -> float:
    """Calculate friction factor for laminar flow (Re < 2300)."""
    if reynolds_number <= 0:
        logger.error("Reynolds number must be positive")
        return FRICTION_FACTOR_DEFAULT_LAMINAR

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
    """Calculate friction factor using Colebrook-White equation (implicit)."""
    if reynolds_number is None:
        raise ValueError("reynolds_number must be provided")
    if reynolds_number < RE_LAMINAR_UPPER:
        return friction_factor_laminar(reynolds_number)

    f = friction_factor_swamee_jain(reynolds_number, relative_roughness)
    for iteration in range(max_iterations):
        f_old = f
        term1 = relative_roughness / COLEBROOK_ROUGHNESS_COEFF
        term2 = 2.51 / (reynolds_number * math.sqrt(f))
        f_new = 0.25 / (math.log10(term1 + term2) ** 2)
        if abs(f_new - f_old) < tolerance:
            logger.debug(
                "Colebrook converged in %s iterations: f = %.6f",
                iteration + 1,
                f_new,
            )
            return f_new
        f = f_new

    logger.warning("Colebrook did not converge in %s iterations", max_iterations)
    return f


def friction_factor_swamee_jain(
    reynolds_number: float, relative_roughness: float
) -> float:
    """Calculate friction factor using Swamee-Jain explicit approximation."""
    if reynolds_number is None:
        raise ValueError("reynolds_number must be provided")
    if reynolds_number < RE_LAMINAR_UPPER:
        return friction_factor_laminar(reynolds_number)

    term1 = relative_roughness / COLEBROOK_ROUGHNESS_COEFF
    term2 = SWAMEE_JAIN_COEFF / (reynolds_number**0.9)
    f = 0.25 / (math.log10(term1 + term2) ** 2)
    logger.debug(
        "Swamee-Jain: Re=%.0f, ε/D=%.6f, f=%.6f",
        reynolds_number,
        relative_roughness,
        f,
    )
    return f


def friction_factor_churchill(
    reynolds_number: float, relative_roughness: float
) -> float:
    """Calculate friction factor using Churchill explicit correlation."""
    if reynolds_number is None:
        raise ValueError("reynolds_number must be provided")
    if reynolds_number < 1:
        return LAMINAR_FRICTION_CONSTANT

    term1 = (7.0 / reynolds_number) ** 0.9 + 0.27 * relative_roughness
    a_term = (-2.457 * math.log(term1)) ** 16
    b_term = (CHURCHILL_B_COEFF / reynolds_number) ** 16
    term2 = (8.0 / reynolds_number) ** 12
    term3 = 1.0 / ((a_term + b_term) ** 1.5)
    f = 8.0 * ((term2 + term3) ** (1.0 / 12.0))
    logger.debug(
        "Churchill: Re=%.0f, ε/D=%.6f, f=%.6f",
        reynolds_number,
        relative_roughness,
        f,
    )
    return float(f)


def friction_factor_haaland(reynolds_number: float, relative_roughness: float) -> float:
    """Calculate friction factor using Haaland explicit approximation."""
    if reynolds_number is None:
        raise ValueError("reynolds_number must be provided")
    if reynolds_number < RE_LAMINAR_UPPER:
        return friction_factor_laminar(reynolds_number)

    term1 = (relative_roughness / COLEBROOK_ROUGHNESS_COEFF) ** 1.11
    term2 = 6.9 / reynolds_number
    inv_sqrt_f = -1.8 * math.log10(term1 + term2)
    return 1.0 / (inv_sqrt_f**2)


def select_friction_factor_method(
    method: str, reynolds_number: float, relative_roughness: float
) -> float:
    """Select and calculate friction factor using the specified method."""
    if method is None:
        raise ValueError("method must be provided")

    normalized = method.lower()
    if normalized == "colebrook":
        return friction_factor_colebrook(reynolds_number, relative_roughness)
    if normalized in {"swamee-jain", "swamee_jain"}:
        return friction_factor_swamee_jain(reynolds_number, relative_roughness)
    if normalized == "churchill":
        return friction_factor_churchill(reynolds_number, relative_roughness)
    if normalized == "haaland":
        return friction_factor_haaland(reynolds_number, relative_roughness)

    available = ["colebrook", "swamee-jain", "churchill", "haaland"]
    raise ValueError(
        f"Unknown friction factor method '{method}'. Available: {available}"
    )
