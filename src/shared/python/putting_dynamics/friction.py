"""Advanced putting friction laws (epic #8345, P2).

Reused shared infrastructure (AGENTS.md section A discovery):

* DbC helpers from ``src.shared.python.contracts``.
* Gravity from ``src.shared.python.core.physics_constants``.
* The stimpmeter -> rolling-resistance conversion restates the
  derivation from Tools ``swing_sim.putting.roll`` (Tools branch
  ``feat/putting-vertical``, epic Tools#4125 H3).  The vendored Tools
  snapshot in ``vendor/ud-tools`` predates that branch, so the closed
  form is restated here with credit instead of imported; if the vendor
  pin is bumped past Tools#4125 this module should re-export the Tools
  functions instead (documented seam, no cycle: UD vendors Tools).

Laws
----
**Stimp -> rolling resistance** (restated from Tools
``swing_sim.putting.roll``): the USGA stimpmeter releases the ball at
``v_release ~= 1.83 m/s`` (36 in ramp, 20 deg release, V-groove
contact radius ~0.87 r).  A green that "stimps" ``S`` feet under a
constant rolling deceleration ``mu_r g`` satisfies::

    mu_r = v_release**2 / (2 g S)

Sanity: stimp 10 -> ``mu_r ~= 0.056``, inside the published 0.05-0.07
tournament band.

**Velocity-dependent rolling resistance**::

    mu_r(v) = mu_r0 * (1 + k_v * v)

Rationale (engineering model, documented rather than cited): turf
deformation depth and grass-blade drag both grow with ball speed, so
the effective rolling deceleration is not constant; a term linear in
``v`` is the smallest correction that (a) reduces exactly to the
constant-deceleration stimp model at ``k_v = 0`` (preserving the Tools
closed forms as an analytic limit, test-enforced) and (b) keeps the
ODE smooth for RK4.  Default ``k_v = 0``; no published value is
claimed for it.

**Static/kinetic transition at rest**: the ball rests when its speed
falls below ``v_stop`` *and* the in-plane gravity pull can be held by
the static *rolling* hold.  A resting ball needs no sliding to move —
it can re-start by rolling — so the relevant bound is the turf's
static rolling resistance (indentation holding torque), slightly above
the kinetic rolling band, not the sliding Coulomb coefficient.  On a
small slope of magnitude ``s = |grad h|`` the driving acceleration is
``g s`` and the maximum hold is ``mu_static g``, so the ball stays at
rest iff ``s <= mu_static`` — otherwise it re-starts rolling downhill
(both branches test-enforced).  This is also the self-consistency
condition of the roll ODE: on slopes steeper than the rolling
resistance a rolling ball cannot decelerate to rest at all.

**Grain anisotropy** (optional): bermuda-style grain drags the ball
less when rolling down-grain.  Modeled as a multiplicative factor::

    mu * (1 - g_strength * cos(theta_v - theta_grain))

with ``theta_grain`` the down-grain direction (the direction the laid
grass points).  Down-grain travel therefore gives the minimum factor
``1 - g_strength`` and directly opposing travel gives ``1 +
g_strength``.  Purely phenomenological; ``g_strength``
defaults to 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.shared.python.contracts import ensure, require, require_finite
from src.shared.python.core.physics_constants import GRAVITY_M_S2

__all__ = [
    "DEFAULT_SLIDING_MU",
    "DEFAULT_STATIC_MU",
    "STIMP_RELEASE_SPEED_MPS",
    "FrictionParams",
    "grain_factor",
    "is_static_hold",
    "rolling_mu",
    "rolling_mu_to_stimp",
    "sliding_mu",
    "stimp_to_rolling_mu",
]

#: Typical published grass sliding friction — the green-contact default
#: in ``rust_core/upstream-physics/src/contact.rs`` (friction = 0.4) and
#: the value used by Tools ``swing_sim.putting.roll``.
DEFAULT_SLIDING_MU = 0.40

#: Static rolling-hold default — slightly above the published 0.05-0.07
#: kinetic rolling band (classic static > kinetic ordering), so greens
#: hold putts on ordinary slopes but release them on steep ones.
DEFAULT_STATIC_MU = 0.08

#: Meters per foot (display conversions live in :mod:`.utils`; this is
#: the stimp definition, which is inherently in feet).
_FOOT_M = 0.3048

#: USGA stimpmeter release speed [m/s] (~6.0 ft/s).  Restated from the
#: Tools ``swing_sim.putting.roll`` derivation (see module docstring).
STIMP_RELEASE_SPEED_MPS = 1.83


@dataclass(frozen=True)
class FrictionParams:
    """Friction-law parameters for one green.

    Attributes:
        mu_roll0: Base rolling-resistance coefficient (stimp-derived
            via :func:`stimp_to_rolling_mu`).
        mu_slide: Kinetic sliding friction for the skid phase.
        mu_static: Static rolling-hold bound for the rest/restart
            check (module laws); must be at least the kinetic
            ``mu_roll0``.
        k_v_per_mps: Velocity coefficient of rolling resistance
            [s/m]; 0 recovers the constant-deceleration model.
        grain_strength: Grain anisotropy amplitude in [0, 0.9); 0
            disables grain.
        grain_direction_rad: Direction the grain points [rad], in the
            surface x-y frame.
        v_stop_mps: Speed below which the ball may come to rest.
    """

    mu_roll0: float
    mu_slide: float = DEFAULT_SLIDING_MU
    mu_static: float = DEFAULT_STATIC_MU
    k_v_per_mps: float = 0.0
    grain_strength: float = 0.0
    grain_direction_rad: float = 0.0
    v_stop_mps: float = 0.005

    def __post_init__(self) -> None:
        require_finite(self.mu_roll0, "mu_roll0")
        require(0.0 < self.mu_roll0 < 0.2, "mu_roll0 in (0, 0.2)", self.mu_roll0)
        require_finite(self.mu_slide, "mu_slide")
        require(0.0 < self.mu_slide <= 1.5, "mu_slide in (0, 1.5]", self.mu_slide)
        require_finite(self.mu_static, "mu_static")
        require(
            self.mu_roll0 <= self.mu_static <= 2.0,
            "mu_static must be >= mu_roll0 and <= 2",
            self.mu_static,
        )
        require_finite(self.k_v_per_mps, "k_v_per_mps")
        require(
            0.0 <= self.k_v_per_mps <= 2.0,
            "k_v must be in [0, 2] s/m",
            self.k_v_per_mps,
        )
        require_finite(self.grain_strength, "grain_strength")
        require(
            0.0 <= self.grain_strength < 0.9,
            "grain_strength in [0, 0.9)",
            self.grain_strength,
        )
        require_finite(self.grain_direction_rad, "grain_direction_rad")
        require(
            abs(self.grain_direction_rad) <= 2.0 * math.pi,
            "grain direction within +/- 2 pi",
            self.grain_direction_rad,
        )
        require_finite(self.v_stop_mps, "v_stop_mps")
        require(
            0.0 < self.v_stop_mps <= 0.1,
            "v_stop in (0, 0.1] m/s",
            self.v_stop_mps,
        )


def stimp_to_rolling_mu(stimp_ft: float) -> float:
    """Rolling-resistance coefficient for a stimp reading.

    ``mu_r = v_release**2 / (2 g S)`` — restated from Tools
    ``swing_sim.putting.roll`` with credit (see module docstring).

    Args:
        stimp_ft: Stimpmeter reading [feet]; greens span ~4-16.

    Returns:
        Dimensionless rolling deceleration coefficient.

    Raises:
        ValueError: If the stimp reading is out of range.
    """
    require_finite(stimp_ft, "stimp_ft")
    require(3.0 <= stimp_ft <= 16.0, "stimp must be in [3, 16] ft", stimp_ft)
    mu = STIMP_RELEASE_SPEED_MPS**2 / (2.0 * GRAVITY_M_S2 * stimp_ft * _FOOT_M)
    ensure(0.0 < mu < 0.2, "rolling mu must be physically small", mu)
    return mu


def rolling_mu_to_stimp(mu_r: float) -> float:
    """Inverse of :func:`stimp_to_rolling_mu` (round-trip exact).

    Args:
        mu_r: Rolling-resistance coefficient in (0, 0.2).

    Returns:
        Stimpmeter reading [feet].

    Raises:
        ValueError: If the coefficient is out of range.
    """
    require_finite(mu_r, "mu_r")
    require(0.0 < mu_r < 0.2, "mu_r must be in (0, 0.2)", mu_r)
    return STIMP_RELEASE_SPEED_MPS**2 / (2.0 * GRAVITY_M_S2 * mu_r * _FOOT_M)


def grain_factor(params: FrictionParams, travel_direction_rad: float) -> float:
    """Anisotropic grain multiplier ``1 - g cos(theta_v - theta_g)``.

    Args:
        params: Friction parameters (grain strength/direction).
        travel_direction_rad: Ball travel direction [rad].

    Returns:
        Multiplier in ``[1 - g, 1 + g]``; exactly 1 when grain is off.
    """
    if params.grain_strength == 0.0:
        return 1.0
    require_finite(travel_direction_rad, "travel_direction_rad")
    factor = 1.0 - params.grain_strength * math.cos(
        travel_direction_rad - params.grain_direction_rad
    )
    ensure(factor > 0.0, "grain factor must stay positive", factor)
    return factor


def rolling_mu(
    params: FrictionParams,
    speed_mps: float,
    spatial_multiplier: float = 1.0,
    travel_direction_rad: float = 0.0,
) -> float:
    """Effective rolling-resistance coefficient at a point.

    ``mu_r0 * mult * (1 + k_v v) * grain`` — see the module laws.

    Args:
        params: Friction parameters.
        speed_mps: Current ball speed [m/s], >= 0.
        spatial_multiplier: Local :class:`~.surfaces.FrictionField`
            rolling multiplier (> 0).
        travel_direction_rad: Ball travel direction [rad] (grain).

    Returns:
        Effective coefficient, > 0.

    Raises:
        ValueError: If inputs are out of range.
    """
    require_finite(speed_mps, "speed_mps")
    require(speed_mps >= 0.0, "speed must be non-negative", speed_mps)
    require_finite(spatial_multiplier, "spatial_multiplier")
    require(spatial_multiplier > 0.0, "multiplier must be > 0", spatial_multiplier)
    mu = (
        params.mu_roll0
        * spatial_multiplier
        * (1.0 + params.k_v_per_mps * speed_mps)
        * grain_factor(params, travel_direction_rad)
    )
    ensure(mu > 0.0, "effective rolling mu must be positive", mu)
    return mu


def sliding_mu(
    params: FrictionParams,
    spatial_multiplier: float = 1.0,
    travel_direction_rad: float = 0.0,
) -> float:
    """Effective kinetic sliding friction at a point.

    Speed-independent (Coulomb); scaled by the local field multiplier
    and the grain factor.

    Args:
        params: Friction parameters.
        spatial_multiplier: Local sliding multiplier (> 0).
        travel_direction_rad: Ball travel direction [rad] (grain).

    Returns:
        Effective coefficient, > 0.

    Raises:
        ValueError: If inputs are out of range.
    """
    require_finite(spatial_multiplier, "spatial_multiplier")
    require(spatial_multiplier > 0.0, "multiplier must be > 0", spatial_multiplier)
    mu = (
        params.mu_slide
        * spatial_multiplier
        * grain_factor(params, travel_direction_rad)
    )
    ensure(mu > 0.0, "effective sliding mu must be positive", mu)
    return mu


def is_static_hold(
    params: FrictionParams,
    slope_magnitude: float,
    spatial_multiplier: float = 1.0,
) -> bool:
    """Whether the static rolling hold keeps a resting ball in place.

    Small-angle static balance (module laws): the ball stays at rest
    iff ``|grad h| <= mu_static * mult``; on steeper slopes it
    re-starts rolling.

    Args:
        params: Friction parameters.
        slope_magnitude: ``|grad h|`` at the rest point (>= 0).
        spatial_multiplier: Local *rolling* multiplier (> 0) — the
            hold is a rolling-resistance bound, not a sliding one.

    Returns:
        True when the ball can remain at rest.

    Raises:
        ValueError: If inputs are out of range.
    """
    require_finite(slope_magnitude, "slope_magnitude")
    require(slope_magnitude >= 0.0, "slope must be >= 0", slope_magnitude)
    require_finite(spatial_multiplier, "spatial_multiplier")
    require(spatial_multiplier > 0.0, "multiplier must be > 0", spatial_multiplier)
    return slope_magnitude <= params.mu_static * spatial_multiplier
