from __future__ import annotations

import logging

import numpy as np

from ._constants import (
    ANTOINE_A,
    ANTOINE_B,
    ANTOINE_C_CELSIUS,
    BUCK_A,
    BUCK_B,
    BUCK_C,
    BUCK_D,
    DEFAULT_DEW_POINT_TEMPERATURE_CELSIUS,
    KPA_TO_PA_FACTOR,
    MBAR_TO_KPA_FACTOR,
    MMHG_TO_PASCAL_FACTOR,
    NEWTON_RAPHSON_DERIVATIVE_TOLERANCE,
    NEWTON_RAPHSON_MAX_ITERATIONS,
    NEWTON_RAPHSON_STEP_SIZE,
    NEWTON_RAPHSON_TOLERANCE,
)

logger = logging.getLogger(__name__)

try:
    from CoolProp.CoolProp import PropsSI

    _COOLPROP_AVAILABLE = True
except ImportError:
    _COOLPROP_AVAILABLE = False


def _antoine_equation(temperature_c: float) -> float:
    """Antoine equation for water vapor pressure (valid 1-100°C)"""
    if not (temperature_c is not None):
        raise ValueError("temperature_c must be provided")
    if not (temperature_c is not None):
        raise ValueError("temperature_c must be provided")
    log_p_mmhg = ANTOINE_A - ANTOINE_B / (ANTOINE_C_CELSIUS + temperature_c)
    p_mmhg = 10**log_p_mmhg
    return p_mmhg * MMHG_TO_PASCAL_FACTOR


def _buck_equation(temperature_c: float) -> float:
    """
    Buck equation for water vapor pressure (improved accuracy).
    """
    if not (temperature_c is not None):
        raise ValueError("temperature_c must be provided")
    if not (temperature_c is not None):
        raise ValueError("temperature_c must be provided")
    a_kpa = BUCK_A / MBAR_TO_KPA_FACTOR
    p_kpa = a_kpa * np.exp(
        (BUCK_B - temperature_c / BUCK_D) * temperature_c / (temperature_c + BUCK_C)
    )
    return float(p_kpa * KPA_TO_PA_FACTOR)


def _iapws_equation(temperature_c: float) -> float:
    """IAPWS-IF97 formulation for high-accuracy vapor pressure"""
    if not (temperature_c is not None):
        raise ValueError("temperature_c must be provided")
    if not (temperature_c is not None):
        raise ValueError("temperature_c must be provided")
    if _COOLPROP_AVAILABLE:
        try:
            temperature_k = temperature_c + 273.15
            return float(PropsSI("P", "T", temperature_k, "Q", 0, "Water"))
        except (ValueError, ZeroDivisionError, OverflowError, TypeError):
            pass
    return _buck_equation(temperature_c)


def calculate_water_vapor_pressure(temperature: float, method: str = "buck") -> float:
    """
    Calculate water vapor pressure using various correlations
    """
    try:
        if method == "antoine":
            return _antoine_equation(temperature)
        if method == "buck":
            return _buck_equation(temperature)
        if method == "iapws":
            return _iapws_equation(temperature)
        return _buck_equation(temperature)
    except (RuntimeError, ValueError, TypeError) as e:
        logger.exception("Water vapor pressure calculation failed: %s", e)
        return _antoine_equation(temperature)


def calculate_dew_point(partial_pressure_pa: float, total_pressure_pa: float) -> float:
    """
    Calculate dew point temperature from partial pressure
    """
    try:

        def objective_function(T: float) -> float:
            """Objective function for dew point calculation."""
            return calculate_water_vapor_pressure(T) - partial_pressure_pa

        T_guess = DEFAULT_DEW_POINT_TEMPERATURE_CELSIUS

        for _ in range(NEWTON_RAPHSON_MAX_ITERATIONS):
            f_val = objective_function(T_guess)
            if abs(f_val) < NEWTON_RAPHSON_TOLERANCE:
                break

            f_plus = objective_function(T_guess + NEWTON_RAPHSON_STEP_SIZE)
            f_minus = objective_function(T_guess - NEWTON_RAPHSON_STEP_SIZE)
            df_dT = (f_plus - f_minus) / (2 * NEWTON_RAPHSON_STEP_SIZE)

            if abs(df_dT) < NEWTON_RAPHSON_DERIVATIVE_TOLERANCE:
                break

            T_guess = T_guess - f_val / df_dT

            if T_guess < -50 or T_guess > 500:
                break

        return T_guess

    except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
        logger.exception("Dew point calculation failed: %s", e)
        return DEFAULT_DEW_POINT_TEMPERATURE_CELSIUS
