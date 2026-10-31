"""Virtual USGA Stimpmeter (epic #14141, putting v4 step 4a).

Reused shared infrastructure (AGENTS.md section A discovery):

* DbC helpers from ``src.shared.python.contracts``.
* Gravity and ball constants from
  ``src.shared.python.core.physics_constants``.
* The ground roll itself is the existing :func:`~.solver.simulate_ball`
  mode machine and :func:`~.friction.stimp_to_rolling_mu` law — this
  module only adds the ramp-release geometry and the USGA two-direction
  measurement procedure on top of them (no duplicated roll physics).

USGA procedure (facts, external sources)
-----------------------------------------
The Stimpmeter is a 36 in aluminum bar with a V-groove track.  A ball
is placed in a notch :data:`STIMPMETER_NOTCH_DISTANCE_M` (30 in) from
the lower end; the bar is slowly raised until the ball starts rolling,
which happens at :data:`STIMPMETER_RELEASE_ANGLE_DEG` (~20 deg)
(Holmes, via Grober, "A Phenomenological Theory of Golf Putting",
arXiv:1106.1698, p.6; the measured release speed there is ``v_release
~= 6.0 ft/s = 1.83 m/s`` -- the same :data:`~.friction.STIMP_RELEASE_SPEED_MPS`
already used by :func:`~.friction.stimp_to_rolling_mu`).  The official
procedure (USGA "Stimpmeter Instructions", Hole Notes, Aug 2000) rolls
three balls in each of two opposite directions on a flat part of the
green, requires the three balls in each direction to stop within 8 in
of each other, and reports the stimp reading as the average of the
two directions' average rollouts -- this cancels any residual slope
under the test patch.

Ramp closed form vs. the measured release speed (documented finding)
----------------------------------------------------------------------
:func:`ramp_release_speed_mps` derives the release speed from first
principles: a solid sphere (moment of inertia ``I = (2/5) m R^2``)
rolling without slipping down a V-groove track loses gravitational PE
to *both* translational and rotational KE, so

::

    m g h = 1/2 m v^2 + 1/2 I omega^2,  omega = v / r_eff

with ``h = notch_distance * sin(angle)`` and ``r_eff`` the effective
rolling radius about the two groove contact lines.  For a V-groove
whose two contact lines are offset ``+-b`` ("groove_contact_half_width_m")
from the ball's vertical centerline, the contact points are at
perpendicular distance ``r_eff = sqrt(R^2 - b^2)`` from the ball's
spin axis (a rail that bites deeper into the ball -- larger ``b`` --
narrows ``r_eff`` and raises the effective rotational inertia).  Solving::

    v = sqrt(2 g h / (1 + (2/5) (R / r_eff)^2))

At ``b = 0`` (contact at the ball's bottom, the no-groove / flat-track
limit -- the only value this module adopts *without* an independent
citation for the actual USGA groove width) this gives ``v ~= 1.911
m/s`` for the USGA geometry, about **0.08 m/s above** the measured
1.83 m/s -- outside the ~+-0.05 m/s band a reasonable measurement
tolerance would allow.  A nonzero groove width lowers the closed-form
speed further (a narrower effective contact radius raises the
rotational-inertia penalty) and numerically a half-width around ``0.49
R`` closes the gap almost exactly -- but that number was reached by
solving for the half-width that reproduces 1.83 m/s, not from an
independently sourced groove dimension, so it is *not* adopted as a
default here (a USPTO Stimpmeter-adjacent patent search surfaced a
145 deg included-angle, ~0.5 in contact-separation description in one
secondary source and a conflicting 60 deg / 1.75 in description in
another, neither corroborated well enough to cite with confidence).
``groove_contact_half_width_m`` is therefore exposed as a parameter so
a caller with a sourced value can supply it; the default stays at the
uncited-but-honest ``0.0`` and the gap to the measured 1.83 m/s is
accepted as a documented, open finding rather than tuned away.  See
``tests/unit/putting_dynamics/test_stimpmeter.py`` for the strict
xfail that pins this finding.
"""

from __future__ import annotations

import math

from src.shared.python.contracts import ensure, require, require_finite
from src.shared.python.core.physics_constants import GOLF_BALL_RADIUS_M, GRAVITY_M_S2

from .friction import STIMP_RELEASE_SPEED_MPS
from .solver import simulate_ball
from .surfaces import SurfaceSpec
from .types import BallState
from .utils import m_to_feet

__all__ = [
    "STIMPMETER_NOTCH_DISTANCE_M",
    "STIMPMETER_RELEASE_ANGLE_DEG",
    "measure_stimp_ft",
    "ramp_release_speed_mps",
    "stimpmeter_rollout_ft",
]

#: Release-notch distance from the bar's lower (exit) end [m] (30 in;
#: Holmes via Grober 2011, arXiv:1106.1698 p.6).
STIMPMETER_NOTCH_DISTANCE_M = 30.0 * 0.0254

#: Bar angle at which the ball starts rolling [deg] (~20 deg; Holmes
#: via Grober 2011, arXiv:1106.1698 p.6).
STIMPMETER_RELEASE_ANGLE_DEG = 20.0

#: Solid-sphere rotational-inertia factor, ``I = (2/5) m R^2``.
_SPHERE_INERTIA_FACTOR = 2.0 / 5.0


def ramp_release_speed_mps(
    notch_distance_m: float = STIMPMETER_NOTCH_DISTANCE_M,
    angle_deg: float = STIMPMETER_RELEASE_ANGLE_DEG,
    groove_contact_half_width_m: float = 0.0,
    ball_radius_m: float = GOLF_BALL_RADIUS_M,
) -> float:
    """Closed-form rolling-without-slipping release speed off the ramp.

    See the module docstring for the energy-balance derivation and the
    documented gap between this closed form (at the uncited default
    ``groove_contact_half_width_m=0``) and the measured USGA release
    speed.

    Args:
        notch_distance_m: Distance from the release notch to the exit
            end of the bar [m], > 0.
        angle_deg: Bar angle at release [deg], in (0, 90).
        groove_contact_half_width_m: Half the lateral separation of
            the V-groove's two contact lines [m]; 0 is the flat-track
            limit (no independently sourced default available, see
            module docstring).  Must be in ``[0, ball_radius_m)``.
        ball_radius_m: Ball radius [m], > 0.

    Returns:
        Release speed at the bottom of the ramp [m/s], > 0.

    Raises:
        ValueError: If any argument is out of its documented range.
    """
    require_finite(notch_distance_m, "notch_distance_m")
    require(notch_distance_m > 0.0, "notch_distance_m must be > 0", notch_distance_m)
    require_finite(angle_deg, "angle_deg")
    require(0.0 < angle_deg < 90.0, "angle_deg must be in (0, 90)", angle_deg)
    require_finite(ball_radius_m, "ball_radius_m")
    require(ball_radius_m > 0.0, "ball_radius_m must be > 0", ball_radius_m)
    require_finite(groove_contact_half_width_m, "groove_contact_half_width_m")
    require(
        0.0 <= groove_contact_half_width_m < ball_radius_m,
        "groove_contact_half_width_m must be in [0, ball_radius_m)",
        groove_contact_half_width_m,
    )

    height_m = notch_distance_m * math.sin(math.radians(angle_deg))
    r_eff_m = math.sqrt(ball_radius_m**2 - groove_contact_half_width_m**2)
    inertia_term = _SPHERE_INERTIA_FACTOR * (ball_radius_m / r_eff_m) ** 2
    speed_mps = math.sqrt(2.0 * GRAVITY_M_S2 * height_m / (1.0 + inertia_term))
    ensure(
        math.isfinite(speed_mps) and speed_mps > 0.0,
        "release speed must be finite and positive",
        speed_mps,
    )
    return speed_mps


def stimpmeter_rollout_ft(
    surface: SurfaceSpec,
    release_speed_mps: float = STIMP_RELEASE_SPEED_MPS,
    heading_rad: float = 0.0,
) -> float:
    """Roll a ball from a pure-rolling release and report the rollout.

    Starts the ball already in pure rolling (``spin = v / R``, the
    state the ball is in once it leaves the ramp) at the origin and
    integrates it to rest on ``surface`` via :func:`~.solver.simulate_ball`
    (no hole; the Stimpmeter measures rollout, not capture).

    Args:
        surface: Green description to roll the calibration ball on.
        release_speed_mps: Speed leaving the ramp [m/s], > 0.
        heading_rad: Direction of travel [rad], 0 along +x.

    Returns:
        Straight-line distance from release to rest [ft], >= 0.

    Raises:
        ValueError: If ``release_speed_mps`` is not finite and > 0, or
            ``heading_rad`` is not finite.
    """
    require_finite(release_speed_mps, "release_speed_mps")
    require(
        release_speed_mps > 0.0, "release_speed_mps must be > 0", release_speed_mps
    )
    require_finite(heading_rad, "heading_rad")

    vx_mps = release_speed_mps * math.cos(heading_rad)
    vy_mps = release_speed_mps * math.sin(heading_rad)
    ball = BallState(
        x_m=0.0,
        y_m=0.0,
        vx_mps=vx_mps,
        vy_mps=vy_mps,
        spin_rad_s=release_speed_mps / GOLF_BALL_RADIUS_M,
    )
    result = simulate_ball(ball, surface)
    distance_m = math.hypot(result.rest_x_m, result.rest_y_m)
    rollout_ft = m_to_feet(distance_m)
    ensure(
        math.isfinite(rollout_ft) and rollout_ft >= 0.0,
        "rollout must be finite and non-negative",
        rollout_ft,
    )
    return rollout_ft


def measure_stimp_ft(
    surface: SurfaceSpec,
    release_speed_mps: float = STIMP_RELEASE_SPEED_MPS,
    heading_rad: float = 0.0,
) -> float:
    """USGA two-direction stimp reading (module docstring, procedure).

    The USGA procedure averages three balls rolled in each of two
    opposite directions; in this deterministic model three balls
    released identically in the same direction travel identically, so
    "three balls within 8 in" is vacuous here and only the
    two-direction average is computed.

    Args:
        surface: Green description to calibrate.
        release_speed_mps: Speed leaving the ramp [m/s], > 0.
        heading_rad: One measurement direction [rad]; the opposite
            direction is ``heading_rad + pi``.

    Returns:
        The two-direction average stimp reading [ft], > 0.

    Raises:
        ValueError: If ``release_speed_mps`` is not finite and > 0, or
            ``heading_rad`` is not finite.
    """
    require_finite(heading_rad, "heading_rad")
    forward_ft = stimpmeter_rollout_ft(surface, release_speed_mps, heading_rad)
    backward_ft = stimpmeter_rollout_ft(
        surface, release_speed_mps, heading_rad + math.pi
    )
    stimp_ft = 0.5 * (forward_ft + backward_ft)
    ensure(
        math.isfinite(stimp_ft) and stimp_ft > 0.0,
        "stimp reading must be finite and positive",
        stimp_ft,
    )
    return stimp_ft
