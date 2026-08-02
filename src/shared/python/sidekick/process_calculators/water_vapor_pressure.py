# TRACKED_TASK: see #2310 — architecture debt extraction schedule

"""Shared water-vapor-pressure correlation kernels.

Single source of truth for the Antoine, Buck, IAPWS-IF97 and Magnus
saturation-pressure formulas used across the process calculators and the
steam engine.  Previously each of ``syngas_water_calculator``,
``acid_gas_dewpoint_calculator``, ``calculators.thermo.steam_engine`` and the
``calc_backend`` syngas-water router re-implemented one or more of these
correlation bodies inline, so a numerical-stability or units fix in one copy
would silently fail to propagate to the others (issues #3675, #3677, #3678).

All functions take coefficients explicitly so callers can supply species- or
range-specific constants (e.g. the acid-gas Antoine fits) while sharing the
formula body and the overflow guard.

Conventions
-----------
- Antoine here uses the ``°C`` convention ``log10(P_mmHg) = A - B / (C + t_°C)``
  and returns pressure in **Pa**.  The inverse returns temperature in **°C**.
- Buck uses ``P_kPa = a * exp((b - t/d) * t / (c + t))`` and returns **Pa**.
- IAPWS-IF97 takes ``t_°C`` and returns **Pa**.
- Magnus uses ``P_hPa = A * exp(B * t / (t + C))`` and returns **Pa**.
"""

from __future__ import annotations

import math

from .constants import (
    CELSIUS_TO_KELVIN_OFFSET,
    IAPWS_COEFFICIENTS,
    IAPWS_CRITICAL_PRESSURE,
    IAPWS_CRITICAL_TEMP,
    IAPWS_TRIPLE_POINT_TEMP,
    MAGNUS_A,
    MAGNUS_B,
    MAGNUS_C,
    MMHG_TO_PA_CONV,
)

__all__ = [
    "PA_PER_HPA",
    "PA_PER_KPA",
    "antoine_pressure_pa",
    "antoine_temperature_c",
    "buck_pressure_pa",
    "iapws_pressure_pa",
    "magnus_pressure_pa",
    "safe_exp",
]

# Pressure unit factors used by the correlation bodies.
PA_PER_KPA: float = 1000.0
PA_PER_HPA: float = 100.0

# Maximum exponent for safe float64 exp() calls.  math.exp(709) is finite
# but math.exp(710) overflows.  We use 700 as a conservative upper bound.
_EXP_MAX_ARG: float = 700.0

# Natural log of 10, used to convert a base-10 Antoine exponent into the
# argument of math.exp so the clamped safe_exp guard can be applied.
_LN10: float = math.log(10.0)


def safe_exp(x: float) -> float:
    """Compute ``exp(x)`` with clamping to prevent float64 overflow.

    For ``x > _EXP_MAX_ARG`` the result is clamped to ``exp(_EXP_MAX_ARG)``
    (~1.01e+304).  For ``x < -_EXP_MAX_ARG`` the result is effectively 0.
    This avoids ``RuntimeWarning: overflow encountered in exp`` when extreme
    temperatures are pushed through the Buck / Magnus / IAPWS equations.

    Precondition:
        ``x`` must be a finite float (no NaN / inf).
    Postcondition:
        Return value is a finite, non-negative float.
    """
    if x is None:
        raise ValueError("x must be provided")
    clamped = max(-_EXP_MAX_ARG, min(x, _EXP_MAX_ARG))
    return float(math.exp(clamped))


def antoine_pressure_pa(a: float, b: float, c: float, temperature_c: float) -> float:
    """Forward Antoine vapor pressure in Pa (°C convention).

    Evaluates ``log10(P_mmHg) = a - b / (c + t_°C)`` and converts mmHg to Pa
    using the shared full-precision factor.  The base-10 exponent is routed
    through :func:`safe_exp` so extreme temperatures do not overflow.

    Args:
        a, b, c: Antoine coefficients in the mmHg / °C convention.
        temperature_c: Temperature in Celsius.

    Returns:
        Saturation (vapor) pressure in Pascal.
    """
    if temperature_c is None:
        raise ValueError("temperature_c must be provided")
    log10_p_mmhg = a - b / (c + temperature_c)
    p_mmhg = safe_exp(log10_p_mmhg * _LN10)
    return float(p_mmhg * MMHG_TO_PA_CONV)


def antoine_temperature_c(a: float, b: float, c: float, pressure_pa: float) -> float:
    """Inverse Antoine: dewpoint/saturation temperature in °C from pressure.

    Solves ``t_°C = b / (a - log10(P_mmHg)) - c`` for the °C-convention
    Antoine coefficients.

    Args:
        a, b, c: Antoine coefficients in the mmHg / °C convention.
        pressure_pa: Pressure in Pascal (must be > 0).

    Returns:
        Temperature in Celsius.

    Raises:
        ValueError: If ``pressure_pa`` is not positive or the inverse has a
            zero denominator.
    """
    if pressure_pa is None:
        raise ValueError("pressure_pa must be provided")
    if not (pressure_pa > 0):
        raise ValueError(f"pressure_pa must be > 0, got {pressure_pa}")
    p_mmhg = pressure_pa / MMHG_TO_PA_CONV
    if p_mmhg <= 0:
        raise ValueError(
            f"pressure in mmHg must be > 0, got {p_mmhg} (from {pressure_pa} Pa)"
        )
    denominator = a - math.log10(p_mmhg)
    if denominator == 0:
        raise ValueError(
            f"Antoine inverse has zero denominator for pressure_pa={pressure_pa}"
        )
    return float(b / denominator - c)


def buck_pressure_pa(
    a_kpa: float, b: float, c: float, d: float, temperature_c: float
) -> float:
    """Buck vapor pressure in Pa.

    Evaluates ``P_kPa = a * exp((b - t/d) * t / (c + t))`` and converts to Pa.
    This is the syngas-calculator coefficient order; callers that use the
    transposed steam-engine order (``(b - t/c) * t / (t + d)``) must swap their
    ``c`` and ``d`` arguments at the call site to preserve their legacy curve.

    Args:
        a_kpa: Buck pre-factor ``a`` in kPa.
        b, c, d: Buck coefficients (dimensionless / °C).
        temperature_c: Temperature in Celsius.

    Returns:
        Saturation (vapor) pressure in Pascal.
    """
    if temperature_c is None:
        raise ValueError("temperature_c must be provided")
    exponent = (b - temperature_c / d) * temperature_c / (c + temperature_c)
    p_kpa = a_kpa * safe_exp(exponent)
    return float(p_kpa * PA_PER_KPA)


def iapws_pressure_pa(temperature_c: float) -> float:
    """IAPWS-IF97 saturation pressure in Pa for liquid water.

    Args:
        temperature_c: Temperature in Celsius.

    Returns:
        Saturation pressure in Pascal.

    Raises:
        ValueError: If the temperature is outside the IAPWS-IF97 saturation
            range (triple point to critical point).
    """
    if temperature_c is None:
        raise ValueError("temperature_c must be provided")
    temperature_k = temperature_c + CELSIUS_TO_KELVIN_OFFSET
    tc = IAPWS_CRITICAL_TEMP
    pc = IAPWS_CRITICAL_PRESSURE
    if temperature_k < IAPWS_TRIPLE_POINT_TEMP or tc < temperature_k:
        raise ValueError("Temperature out of IAPWS-IF97 range")
    theta = 1 - temperature_k / tc
    a = IAPWS_COEFFICIENTS
    ln_p = (
        tc
        / temperature_k
        * (
            a[0] * theta
            + a[1] * theta**1.5
            + a[2] * theta**3
            + a[3] * theta**3.5
            + a[4] * theta**4
            + a[5] * theta**7.5
        )
    )
    return float(pc * safe_exp(ln_p))


def magnus_pressure_pa(temperature_c: float) -> float:
    """Magnus vapor pressure in Pa (very accurate for 0-100°C).

    Evaluates ``P_hPa = MAGNUS_A * exp(MAGNUS_B * t / (t + MAGNUS_C))``.

    Args:
        temperature_c: Temperature in Celsius (valid 0-100°C).

    Returns:
        Saturation (vapor) pressure in Pascal.

    Raises:
        ValueError: If the temperature is outside the 0-100°C validity range.
    """
    if temperature_c is None:
        raise ValueError("temperature_c must be provided")
    if temperature_c < 0 or temperature_c > 100:
        raise ValueError(
            f"Magnus equation valid for 0°C to 100°C, got {temperature_c}°C"
        )
    exponent = MAGNUS_B * temperature_c / (temperature_c + MAGNUS_C)
    p_hpa = MAGNUS_A * safe_exp(exponent)
    return float(p_hpa * PA_PER_HPA)
