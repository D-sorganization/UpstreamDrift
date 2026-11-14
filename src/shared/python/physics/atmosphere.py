"""Atmospheric and aerodynamic property models for ball flight.

This module provides physically grounded helpers used by the ball flight
integrators:

- :func:`cd_dimpled_sphere`: Reynolds-number-dependent drag coefficient for
  dimpled spheres (golf balls), modelling the well-documented "drag crisis"
  transition.
- :func:`air_density`: ISA-standard atmosphere air density as a function of
  altitude, temperature, and (optionally) pressure.

These functions replace previously hardcoded ``rho = 1.225`` and the simple
piecewise drag model that did not capture the drag-crisis dip near
Re ~ 5e4 -- 1e5.

Implements two of the five sub-tasks tracked under issue #3504. The remaining
three (hydrodynamic lubrication, dimple geometry optimization, mud ball
physics) are still tracked under the same GitHub issue.

References
----------
Bearman, P. W., & Harvey, J. K. (1976). Golf ball aerodynamics.
    Aeronautical Quarterly, 27(2), 112-122.
Mehta, R. D. (1985). Aerodynamics of sports balls.
    Annual Review of Fluid Mechanics, 17, 151-189.
ISO 2533:1975. Standard Atmosphere.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# ISA standard atmosphere constants
# ---------------------------------------------------------------------------

#: Sea-level ISA standard temperature [K] (15 C; distinct from NIST 20 C lab ambient in physics_constants).
ISA_T0_K: float = 288.15

#: Sea-level standard pressure [Pa].
ISA_P0_PA: float = 101325.0

#: Sea-level standard air density [kg/m^3]; provided for reference and tests.
ISA_RHO0_KG_M3: float = 1.225

#: Tropospheric temperature lapse rate [K/m] (positive value -- temperature
#: decreases by this much per metre of altitude).
ISA_LAPSE_RATE_K_PER_M: float = 0.0065

#: Standard gravity [m/s^2].
ISA_GRAVITY_M_S2: float = 9.80665

#: Specific gas constant for dry air [J/(kg*K)].
DRY_AIR_R_SPECIFIC_J_KG_K: float = 287.05

#: Specific gas constant for water vapor [J/(kg*K)].
WATER_VAPOR_R_SPECIFIC_J_KG_K: float = 461.5

#: Validated altitude bounds (metres). Below the lower bound or above the
#: upper bound the ISA troposphere model is no longer reliable for golf
#: trajectories and we refuse to extrapolate silently.
MIN_VALID_ALTITUDE_M: float = -500.0
MAX_VALID_ALTITUDE_M: float = 9000.0

#: Validated temperature bounds (Celsius). Outside this range users almost
#: certainly passed a wrong value (e.g. Kelvin or Fahrenheit by mistake).
MIN_VALID_TEMPERATURE_C: float = -50.0
MAX_VALID_TEMPERATURE_C: float = 60.0


def _isa_pressure_and_temp_k(
    altitude_m: float,
    temperature_c: float,
    pressure_pa: float | None,
    *,
    temperature_is_local: bool,
) -> tuple[float, float]:
    """Validate inputs and return ``(pressure_pa, temperature_k)`` at altitude."""
    if not isinstance(altitude_m, (int, float)) or isinstance(altitude_m, bool):
        raise TypeError("altitude_m must be a real number")
    if not isinstance(temperature_c, (int, float)) or isinstance(temperature_c, bool):
        raise TypeError("temperature_c must be a real number")
    if pressure_pa is not None and (
        not isinstance(pressure_pa, (int, float)) or isinstance(pressure_pa, bool)
    ):
        raise TypeError("pressure_pa must be a real number or None")
    if not math.isfinite(altitude_m):
        raise ValueError("altitude_m must be finite")
    if not math.isfinite(temperature_c):
        raise ValueError("temperature_c must be finite")
    if altitude_m < MIN_VALID_ALTITUDE_M or altitude_m > MAX_VALID_ALTITUDE_M:
        raise ValueError(
            f"altitude_m={altitude_m} is outside the supported range "
            f"[{MIN_VALID_ALTITUDE_M}, {MAX_VALID_ALTITUDE_M}] metres"
        )
    if (
        temperature_c < MIN_VALID_TEMPERATURE_C
        or temperature_c > MAX_VALID_TEMPERATURE_C
    ):
        raise ValueError(
            f"temperature_c={temperature_c} is outside the supported range "
            f"[{MIN_VALID_TEMPERATURE_C}, {MAX_VALID_TEMPERATURE_C}] C"
        )
    p0 = ISA_P0_PA if pressure_pa is None else float(pressure_pa)
    if p0 <= 0:
        raise ValueError("pressure_pa must be positive when provided")

    lapse = ISA_LAPSE_RATE_K_PER_M
    exponent = ISA_GRAVITY_M_S2 / (DRY_AIR_R_SPECIFIC_J_KG_K * lapse)
    if temperature_is_local:
        t_at_alt = float(temperature_c) + 273.15
        t_isa = ISA_T0_K - lapse * float(altitude_m)
        if t_at_alt <= 0 or t_isa <= 0:
            raise ValueError("Computed temperature at altitude is non-positive")
        return p0 * (t_isa / ISA_T0_K) ** exponent, t_at_alt
    t0 = float(temperature_c) + 273.15
    t_at_alt = t0 - lapse * float(altitude_m)
    if t_at_alt <= 0:
        raise ValueError(
            "Computed temperature at altitude is non-positive; check inputs"
        )
    return p0 * (t_at_alt / t0) ** exponent, t_at_alt


def air_density(
    altitude_m: float,
    temperature_c: float = 15.0,
    pressure_pa: float | None = None,
    *,
    temperature_is_local: bool = False,
) -> float:
    """Return air density [kg/m^3] from the ISA-troposphere model.

    When ``temperature_is_local`` is ``False`` (default), ``temperature_c`` is
    the sea-level reference temperature ``T0`` and lapses with altitude; pass
    ``temperature_is_local=True`` to supply local ambient temperature at ``altitude_m``
    under standard ISA barometric pressure.
    """
    pressure_at_alt, t_at_alt = _isa_pressure_and_temp_k(
        altitude_m,
        temperature_c,
        pressure_pa,
        temperature_is_local=temperature_is_local,
    )
    return pressure_at_alt / (DRY_AIR_R_SPECIFIC_J_KG_K * t_at_alt)


def saturation_vapor_pressure_pa(temperature_c: float) -> float:
    """Return saturation vapor pressure over water [Pa]."""
    if not isinstance(temperature_c, (int, float)) or isinstance(temperature_c, bool):
        raise TypeError("temperature_c must be a real number")
    if not math.isfinite(temperature_c):
        raise ValueError("temperature_c must be finite")
    if (
        temperature_c < MIN_VALID_TEMPERATURE_C
        or temperature_c > MAX_VALID_TEMPERATURE_C
    ):
        raise ValueError(
            f"temperature_c={temperature_c} is outside the supported range "
            f"[{MIN_VALID_TEMPERATURE_C}, {MAX_VALID_TEMPERATURE_C}] C"
        )
    return 610.94 * math.exp(
        17.625 * float(temperature_c) / (float(temperature_c) + 243.04)
    )


def humid_air_density(
    altitude_m: float,
    temperature_c: float = 15.0,
    relative_humidity: float = 0.0,
    pressure_pa: float | None = None,
    *,
    temperature_is_local: bool = False,
) -> float:
    """Return moist-air density [kg/m^3] for ball-flight calculations.

    ``relative_humidity`` is a fraction in ``[0, 1]``. Humid air is less dense
    than dry air at the same pressure and temperature because water vapor has a
    larger specific gas constant than dry air.
    """
    if not isinstance(relative_humidity, (int, float)) or isinstance(
        relative_humidity, bool
    ):
        raise TypeError("relative_humidity must be a real number")
    if not math.isfinite(relative_humidity):
        raise ValueError("relative_humidity must be finite")
    rh = float(relative_humidity)
    if not 0.0 <= rh <= 1.0:
        raise ValueError("relative_humidity must be between 0 and 1")
    if rh == 0.0:
        return air_density(
            altitude_m=altitude_m,
            temperature_c=temperature_c,
            pressure_pa=pressure_pa,
            temperature_is_local=temperature_is_local,
        )

    pressure_at_alt, t_at_alt = _isa_pressure_and_temp_k(
        altitude_m,
        temperature_c,
        pressure_pa,
        temperature_is_local=temperature_is_local,
    )
    vapor_pressure = rh * saturation_vapor_pressure_pa(t_at_alt - 273.15)
    vapor_pressure = min(vapor_pressure, pressure_at_alt)
    dry_pressure = pressure_at_alt - vapor_pressure
    return dry_pressure / (DRY_AIR_R_SPECIFIC_J_KG_K * t_at_alt) + vapor_pressure / (
        WATER_VAPOR_R_SPECIFIC_J_KG_K * t_at_alt
    )


# ---------------------------------------------------------------------------
# Drag-crisis model for dimpled spheres
# ---------------------------------------------------------------------------

#: Minimum supported Reynolds number for the dimpled-sphere correlation.
MIN_VALID_REYNOLDS: float = 1.0e3

#: Maximum supported Reynolds number for the dimpled-sphere correlation.
MAX_VALID_REYNOLDS: float = 1.0e7

# Anchor points (Re, Cd) for the smoothed drag-crisis model. Values are
# representative of dimpled golf-ball measurements compiled by Bearman &
# Harvey (1976) and reviewed in Mehta (1985); they are not meant to be a
# manufacturer-specific calibration.
_CD_PRECRITICAL: float = 0.50  # subcritical / pre-crisis Cd for dimpled sphere
_RE_MIN_CD: float = 7.0e4  # log-Re centre of the drag-crisis transition
_CD_HIGH: float = 0.30  # high-Re asymptote Cd

# Width parameter for the tanh transition (in log10(Re) units).
_TRANSITION_WIDTH: float = 0.16


def _smoothstep_tanh(x: float, x0: float, width: float) -> float:
    """Return a smooth 0->1 transition centred at ``x0`` with given width.

    Uses ``0.5 * (1 + tanh((x - x0) / width))`` which has continuous
    derivatives of all orders -- important for ODE integrator stability.
    """
    if width <= 0:
        raise ValueError("width must be positive")
    return 0.5 * (1.0 + math.tanh((x - x0) / width))


def cd_dimpled_sphere(reynolds: float, base_cd: float = 0.21) -> float:
    """Return drag coefficient ``Cd`` for a dimpled sphere at given Reynolds.

    Models the three-region behaviour observed for dimpled golf balls
    (Bearman & Harvey 1976, Mehta 1985):

    1. **Subcritical** (Re below ~4e4): Cd is roughly constant near
       ``_CD_PRECRITICAL``.
    2. **Drag crisis** (Re ~ 4e4 -- 1e5): Cd drops sharply as the boundary
       layer transitions from laminar to turbulent and the wake narrows.
    3. **Trans/supercritical** (Re above ~1e5): Cd rises slowly toward an
       asymptote near ``_CD_HIGH``.

    The ``base_cd`` argument lets callers shift the post-crisis floor to
    account for ball-to-ball variation (the registry default is 0.21). Passing
    ``base_cd=0`` explicitly disables drag; negative values are rejected.

    A ``tanh``-based smooth blend is used so that ``Cd(Re)`` and its first
    derivative are continuous everywhere; this matters for the trajectory
    integrator's stability.

    Parameters
    ----------
    reynolds
        Reynolds number based on ball diameter and free-stream speed. Must
        lie within ``[MIN_VALID_REYNOLDS, MAX_VALID_REYNOLDS]``.
    base_cd
        Post-crisis floor Cd. Defaults to 0.21 (zero-spin polynomial baseline);
        ``DragModel`` and ``BallFlightSimulator`` pass ``GOLF_BALL_DRAG_COEFFICIENT``
        (0.25). Positive values are clamped into ``[0.18, 0.30]``, and 0 disables drag.

    Returns
    -------
    float
        Drag coefficient (dimensionless), either ``0`` or in ``[0.15, 0.55]``.

    Raises
    ------
    TypeError
        If arguments are not real numbers.
    ValueError
        If ``reynolds`` is non-positive or outside the supported range.

    Postconditions
    --------------
    Returned Cd is finite and lies in ``[0.15, 0.55]``.

    Notes
    -----
    Citations: Bearman & Harvey (1976), "Golf ball aerodynamics";
    Mehta (1985), "Aerodynamics of sports balls".
    """
    if not isinstance(reynolds, (int, float)) or isinstance(reynolds, bool):
        raise TypeError("reynolds must be a real number")
    if not isinstance(base_cd, (int, float)) or isinstance(base_cd, bool):
        raise TypeError("base_cd must be a real number")
    if math.isnan(reynolds) or math.isinf(reynolds) or reynolds <= 0:
        raise ValueError("reynolds must be a positive finite number")
    if reynolds < MIN_VALID_REYNOLDS or reynolds > MAX_VALID_REYNOLDS:
        raise ValueError(
            f"reynolds={reynolds} is outside the supported range "
            f"[{MIN_VALID_REYNOLDS:g}, {MAX_VALID_REYNOLDS:g}]"
        )

    if math.isnan(base_cd) or math.isinf(base_cd) or base_cd < 0:
        raise ValueError("base_cd must be a finite non-negative number")
    if base_cd == 0:
        return 0.0

    # Clamp the post-crisis floor into a physically plausible band so a
    # mistuned ``base_cd`` cannot drive the curve outside [0.15, 0.55].
    cd_floor = max(0.18, min(0.30, float(base_cd)))

    log_re = math.log10(reynolds)
    log_re_crisis = math.log10(_RE_MIN_CD)
    log_re_recovery = math.log10(7.0e5)

    # Crisis transition: high-drag plateau -> minimum Cd, centred near 7e4.
    crisis_blend = _smoothstep_tanh(log_re, log_re_crisis, _TRANSITION_WIDTH)
    cd_crisis = _CD_PRECRITICAL + (cd_floor - _CD_PRECRITICAL) * crisis_blend

    # Slow recovery: minimum Cd -> high-Re asymptote.
    recovery_blend = _smoothstep_tanh(log_re, log_re_recovery, 0.45)
    cd = cd_crisis + (_CD_HIGH - cd_floor) * recovery_blend

    # Defensive clamp: keep the return value strictly inside the documented
    # envelope so downstream integrators never see an unphysical value.
    return max(0.15, min(0.55, cd))


__all__ = [
    "MIN_VALID_ALTITUDE_M",
    "MAX_VALID_ALTITUDE_M",
    "MIN_VALID_TEMPERATURE_C",
    "MAX_VALID_TEMPERATURE_C",
    "MIN_VALID_REYNOLDS",
    "MAX_VALID_REYNOLDS",
    "ISA_T0_K",
    "ISA_P0_PA",
    "ISA_RHO0_KG_M3",
    "ISA_LAPSE_RATE_K_PER_M",
    "ISA_GRAVITY_M_S2",
    "DRY_AIR_R_SPECIFIC_J_KG_K",
    "WATER_VAPOR_R_SPECIFIC_J_KG_K",
    "air_density",
    "cd_dimpled_sphere",
    "humid_air_density",
    "saturation_vapor_pressure_pa",
]
