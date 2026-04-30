"""Mud-ball aerodynamic adjustment model.

Models the effect of mud accumulation on a golf ball's drag and lift
coefficients as well as its total mass.

References:
    - Achenbach, E. (1972). "Experiments on the flow past spheres at very
      high Reynolds numbers." J. Fluid Mech., 54(3), 565-575. Surface
      roughness raises the drag coefficient on spheres in the
      post-critical regime; mud coverage acts as a roughness perturbation.
    - Bearman, P. W., & Harvey, J. K. (1976). "Golf ball aerodynamics."
      Aeronautical Quarterly, 27(2), 112-122. Establishes the dimpled-
      sphere baseline (Cd ~0.21, Cl ~0.18) used as the un-fouled reference
      from which mud-fouling departures are computed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Module-level physical limits / tunables (also used by tests).
MAX_MUD_MASS_G: float = 30.0
MAX_CD_CLAMP: float = 0.55
MIN_CL_CLAMP: float = 0.0
CD_COVERAGE_GAIN: float = 0.6
CL_COVERAGE_LOSS: float = 0.7
DEFAULT_BALL_MASS_G: float = 45.93  # USGA tour-spec ball mass (grams)


@dataclass(frozen=True)
class MudBallAdjustment:
    """Adjusted aerodynamic coefficients and mass for a mud-coated ball.

    Attributes:
        cd_eff: Effective drag coefficient after mud fouling (clamped at
            ``MAX_CD_CLAMP``).
        cl_eff: Effective lift coefficient after mud fouling (clamped at
            ``MIN_CL_CLAMP``).
        mass_total_kg: Combined mass of the ball plus accumulated mud, in
            kilograms.
        cd_increase_factor: Ratio ``cd_eff / base_cd`` (>= 1).
        cl_decrease_factor: Ratio ``cl_eff / base_cl`` (in [0, 1]). Set to
            ``0.0`` if ``base_cl`` is zero.
    """

    cd_eff: float
    cl_eff: float
    mass_total_kg: float
    cd_increase_factor: float
    cl_decrease_factor: float


def _validate_finite_number(value: object, name: str) -> float:
    """Validate ``value`` is a real, finite number. Returns float(value)."""
    if value is None:
        raise TypeError(f"{name} must be a real number, got None")
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, got bool")
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    fval = float(value)
    if math.isnan(fval):
        raise ValueError(f"{name} must be finite, got NaN")
    if math.isinf(fval):
        raise ValueError(f"{name} must be finite, got infinity")
    return fval


def mud_ball_aero_adjustments(
    *,
    mud_mass_g: float,
    mud_coverage: float,
    base_cd: float = 0.21,
    base_cl: float = 0.18,
    ball_mass_g: float = DEFAULT_BALL_MASS_G,
) -> MudBallAdjustment:
    """Compute mud-fouled drag/lift coefficients and total ball mass.

    The effective drag rises with both surface coverage and the mass-fraction
    of mud relative to the ball, capturing the joint roughness/inertial
    contribution observed for fouled spheres (Achenbach, 1972). Lift falls
    nearly linearly with coverage as dimples are progressively filled
    (Bearman & Harvey, 1976).

    Args:
        mud_mass_g: Accumulated mud mass in grams. Must lie in
            ``[0, MAX_MUD_MASS_G]``.
        mud_coverage: Fraction of ball surface covered, in ``[0.0, 1.0]``.
        base_cd: Reference drag coefficient of the clean dimpled ball.
            Must be > 0 and finite.
        base_cl: Reference lift coefficient of the clean dimpled ball.
            Must be >= 0 and finite.
        ball_mass_g: Reference ball mass in grams. Must be > 0 and finite.

    Returns:
        :class:`MudBallAdjustment` with effective Cd, Cl, total mass and
        per-coefficient scaling factors.

    Raises:
        TypeError: If any argument is ``None`` or not a real number.
        ValueError: If any argument is non-finite or out of range.

    Postconditions:
        - ``0 <= cl_eff <= base_cl``
        - ``base_cd <= cd_eff <= MAX_CD_CLAMP``
        - ``mass_total_kg == (ball_mass_g + mud_mass_g) / 1000``
    """
    # --- Preconditions (DbC) ---
    mud_mass = _validate_finite_number(mud_mass_g, "mud_mass_g")
    coverage = _validate_finite_number(mud_coverage, "mud_coverage")
    cd0 = _validate_finite_number(base_cd, "base_cd")
    cl0 = _validate_finite_number(base_cl, "base_cl")
    ball_mass = _validate_finite_number(ball_mass_g, "ball_mass_g")

    if mud_mass < 0.0 or mud_mass > MAX_MUD_MASS_G:
        raise ValueError(
            f"mud_mass_g must be within [0, {MAX_MUD_MASS_G}] g, got {mud_mass}"
        )
    if coverage < 0.0 or coverage > 1.0:
        raise ValueError(f"mud_coverage must be within [0.0, 1.0], got {coverage}")
    if cd0 <= 0.0:
        raise ValueError(f"base_cd must be > 0, got {cd0}")
    if cl0 < 0.0:
        raise ValueError(f"base_cl must be >= 0, got {cl0}")
    if ball_mass <= 0.0:
        raise ValueError(f"ball_mass_g must be > 0, got {ball_mass}")

    # --- Model ---
    mass_factor = mud_mass / ball_mass
    cd_raw = cd0 * (1.0 + CD_COVERAGE_GAIN * coverage * (1.0 + mass_factor))
    cd_eff = min(cd_raw, MAX_CD_CLAMP)

    cl_raw = cl0 * (1.0 - CL_COVERAGE_LOSS * coverage)
    cl_eff = max(cl_raw, MIN_CL_CLAMP)

    mass_total_kg = (ball_mass + mud_mass) / 1000.0

    cd_increase_factor = cd_eff / cd0
    cl_decrease_factor = (cl_eff / cl0) if cl0 > 0.0 else 0.0

    return MudBallAdjustment(
        cd_eff=cd_eff,
        cl_eff=cl_eff,
        mass_total_kg=mass_total_kg,
        cd_increase_factor=cd_increase_factor,
        cl_decrease_factor=cl_decrease_factor,
    )


__all__ = [
    "MAX_CD_CLAMP",
    "MAX_MUD_MASS_G",
    "MIN_CL_CLAMP",
    "MudBallAdjustment",
    "mud_ball_aero_adjustments",
]
