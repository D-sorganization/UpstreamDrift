"""Mode-machine putt integration over heterogeneous greens (#8345).

Reused shared infrastructure (AGENTS.md section A discovery):

* Gravity and ball constants from
  ``src.shared.python.core.physics_constants``.
* DbC helpers from ``src.shared.python.contracts``.
* The skid/roll ODE shape, the fixed 2 ms RK4 discipline (mode held
  constant within a step, transitions applied between steps), and the
  hole-capture bound restate the derivations in Tools
  ``swing_sim.putting`` (``roll.py`` / ``green.py``, branch
  ``feat/putting-vertical``, epic Tools#4125 H3).  The vendored Tools
  snapshot in ``vendor/ud-tools`` predates that branch, so the
  formulas are restated here with credit rather than imported (UD
  vendors Tools; no cycle).  Capture bound (Tools derivation, after
  Holmes, "Putting: How a golf ball and hole interact", Am. J. Phys.
  59 (1991)): free fall must drop the ball half a diameter within a
  travel budget of one full hole diameter::

      v_capture = 2 R * sqrt(g / (2 r)) ~= 1.64 m/s

Mode machine
------------
``AIRBORNE -> (bounce ...) -> SLIDE -> ROLL -> REST`` with two extra
edges: SLIDE starts only while the contact point slips
(``omega r < |v|``), and REST re-opens to ROLL when the local slope
exceeds the static-friction bound (steep-slope restart, see
:func:`~.friction.is_static_hold`).

Ground modes integrate ``(x, y, vx, vy, s)`` — ``s = omega r`` is the
contact-surface speed along travel (restated Tools simplification: the
spin axis stays perpendicular to the velocity; exact for straight
putts, first-order accurate for real break angles) — with classic RK4
at a fixed ``dt = 2 ms`` (restated from Tools ``green.py``, where the
same step pins a TS parity mirror; kept identical here so future
parity is bit-comparable).  In-plane gravity is ``-g grad h`` from the
height field each evaluation (small-slope), friction comes from the
:mod:`.friction` laws with the local :class:`~.surfaces.FrictionField`
multipliers.

The airborne phase (post-impact rise) is ballistic (no drag at putt
speeds, documented) with symplectic-Euler height updates; each ground
contact reflects the vertical speed with the green restitution
``e = 0.78`` (``rust_core/upstream-physics/src/contact.rs``) and hands
off to the ground modes once the remaining hop apex is below a
millimetre.  Spin is frozen during flight and bounce (documented
simplification: no bounce friction impulse).
"""

from __future__ import annotations

import math

from src.shared.python.contracts import ensure, require, require_finite
from src.shared.python.core.physics_constants import (
    GOLF_BALL_RADIUS_M,
    GRAVITY_M_S2,
)

from .collision import GROUND_RESTITUTION, strike
from .friction import is_static_hold, rolling_mu, sliding_mu
from .surfaces import SurfaceSpec
from .types import (
    BallState,
    CollisionReport,
    Mode,
    PuttResult,
    PutterState,
    TrajectorySample,
)

__all__ = [
    "DT_S",
    "HOLE_RADIUS_M",
    "capture_speed_mps",
    "simulate_ball",
    "simulate_strike",
]

#: USGA hole radius [m] (4.25 in diameter).
HOLE_RADIUS_M = 0.054

#: Fixed RK4 step [s] — restated from Tools ``green.py`` (parity-pinned
#: there); kept identical for future cross-repo parity.
DT_S = 0.002

#: Integration time cap [s].
_MAX_TIME_S = 60.0

#: Hop apex below this height hands off to the ground modes [m].
_SETTLE_APEX_M = 0.001

#: Gravity as a plain float for the inner loop.
_G = float(GRAVITY_M_S2)

#: Ball radius as a plain float for the inner loop.
_R_BALL = float(GOLF_BALL_RADIUS_M)

_State = tuple[float, float, float, float, float]


def capture_speed_mps() -> float:
    """Full-chord capture bound ``2 R sqrt(g / (2 r))`` (module credit).

    Returns:
        The dead-centre capture bound [m/s], ~1.64.
    """
    return 2.0 * HOLE_RADIUS_M * math.sqrt(_G / (2.0 * _R_BALL))


def _derivative(state: _State, sliding: bool, surface: SurfaceSpec) -> _State:
    """Ground-mode ODE right-hand side (module docstring)."""
    x, y, vx, vy, surface_speed = state
    grad = surface.height.gradient(x, y)
    gx, gy = -_G * grad[0], -_G * grad[1]
    speed = math.hypot(vx, vy)
    if speed <= 0.0:
        return (0.0, 0.0, gx, gy, 0.0)
    heading = math.atan2(vy, vx)
    roll_mult, slide_mult = surface.friction_field.multipliers(x, y)
    if sliding:
        mu = sliding_mu(surface.friction, slide_mult, heading)
        slip = speed - surface_speed
        slip_sign = 1.0 if slip >= 0.0 else -1.0
        ds = slip_sign * 2.5 * mu * _G
    else:
        mu = rolling_mu(surface.friction, speed, roll_mult, heading)
        slip_sign = 1.0
        ds = 0.0
    ax = -slip_sign * mu * _G * vx / speed + gx
    ay = -slip_sign * mu * _G * vy / speed + gy
    return (vx, vy, ax, ay, ds)


def _rk4_step(state: _State, sliding: bool, surface: SurfaceSpec) -> _State:
    """One classic RK4 step with the mode held constant (Tools shape)."""
    k1 = _derivative(state, sliding, surface)
    mid1 = tuple(s + 0.5 * DT_S * k for s, k in zip(state, k1, strict=True))
    k2 = _derivative(mid1, sliding, surface)  # type: ignore[arg-type]
    mid2 = tuple(s + 0.5 * DT_S * k for s, k in zip(state, k2, strict=True))
    k3 = _derivative(mid2, sliding, surface)  # type: ignore[arg-type]
    end = tuple(s + DT_S * k for s, k in zip(state, k3, strict=True))
    k4 = _derivative(end, sliding, surface)  # type: ignore[arg-type]
    return tuple(  # type: ignore[return-value]
        s + (DT_S / 6.0) * (a + 2.0 * b + 2.0 * c + d)
        for s, a, b, c, d in zip(state, k1, k2, k3, k4, strict=True)
    )


class _Recorder:
    """Sample/bookkeeping accumulator for one putt."""

    def __init__(self, hole_x_m: float | None, hole_y_m: float | None) -> None:
        self.samples: list[TrajectorySample] = []
        self.distance = 0.0
        self.skid_distance = 0.0
        self.hole = (
            (hole_x_m, hole_y_m)
            if hole_x_m is not None and hole_y_m is not None
            else None
        )
        self.speed_at_hole: float | None = None
        self.holed = False

    def record(
        self,
        t: float,
        x: float,
        y: float,
        height: float,
        speed: float,
        spin: float,
        mode: Mode,
    ) -> None:
        self.samples.append(
            TrajectorySample(
                t_s=t,
                x_m=x,
                y_m=y,
                height_m=height,
                speed_mps=speed,
                spin_rad_s=spin,
                mode=mode,
            )
        )

    def check_hole(self, x: float, y: float, speed: float, on_ground: bool) -> bool:
        """Track hole crossing; True when captured."""
        if self.hole is None or not on_ground:
            return False
        if math.hypot(x - self.hole[0], y - self.hole[1]) > HOLE_RADIUS_M:
            return False
        if self.speed_at_hole is None:
            self.speed_at_hole = speed
        if speed <= capture_speed_mps():
            self.holed = True
            return True
        return False


def _fly(
    ball: BallState, surface: SurfaceSpec, rec: _Recorder, t: float
) -> tuple[_State, float, float]:
    """Airborne rise/bounce phase; returns ground state, spin, time."""
    x, y = ball.x_m, ball.y_m
    vx, vy, vz = ball.vx_mps, ball.vy_mps, ball.vz_mps
    z = surface.height.elevation(x, y) + ball.height_m
    while t < _MAX_TIME_S:
        vz -= _G * DT_S
        x += vx * DT_S
        y += vy * DT_S
        z += vz * DT_S
        t += DT_S
        rec.distance += math.hypot(vx * DT_S, vy * DT_S)
        floor = surface.height.elevation(x, y)
        height = max(z - floor, 0.0)
        rec.record(t, x, y, height, math.hypot(vx, vy), ball.spin_rad_s, Mode.AIRBORNE)
        if z <= floor and vz < 0.0:
            vz = -GROUND_RESTITUTION * vz
            z = floor
            if vz**2 / (2.0 * _G) < _SETTLE_APEX_M:
                break
    return (x, y, vx, vy, ball.spin_rad_s * _R_BALL), ball.spin_rad_s, t


def _restart_state(x: float, y: float, surface: SurfaceSpec) -> _State | None:
    """Steep-slope restart: tiny downhill velocity, or None to rest."""
    grad = surface.height.gradient(x, y)
    slope = math.hypot(grad[0], grad[1])
    roll_mult, _slide_mult = surface.friction_field.multipliers(x, y)
    if is_static_hold(surface.friction, slope, roll_mult):
        return None
    v0 = surface.friction.v_stop_mps
    return (x, y, -v0 * grad[0] / slope, -v0 * grad[1] / slope, v0)


def simulate_ball(
    ball: BallState,
    surface: SurfaceSpec,
    hole_x_m: float | None = None,
    hole_y_m: float | None = None,
    collision: CollisionReport | None = None,
) -> PuttResult:
    """Integrate a ball state over a surface to rest or capture.

    Args:
        ball: Initial ball state (may be airborne or at rest).
        surface: Green description.
        hole_x_m: Hole center x [m]; hole disabled when None.
        hole_y_m: Hole center y [m]; hole disabled when None.
        collision: Optional impact report to attach to the result.

    Returns:
        The integrated :class:`~.types.PuttResult`.

    Raises:
        ValueError: If the hole position is half-specified.
    """
    require(
        (hole_x_m is None) == (hole_y_m is None),
        "hole position needs both coordinates",
        (hole_x_m, hole_y_m),
    )
    if hole_x_m is not None:
        require_finite(hole_x_m, "hole_x_m")
        require_finite(hole_y_m, "hole_y_m")
    rec = _Recorder(hole_x_m, hole_y_m)
    t = 0.0
    spin = ball.spin_rad_s
    airborne = ball.height_m > 0.0 or ball.vz_mps > 0.0
    initial_mode = (
        Mode.AIRBORNE
        if airborne
        else (
            Mode.SLIDE
            if not math.isclose(
                spin * _R_BALL, ball.speed_mps, rel_tol=0.0, abs_tol=1e-12
            )
            else Mode.ROLL
        )
    )
    rec.record(
        0.0, ball.x_m, ball.y_m, ball.height_m, ball.speed_mps, spin, initial_mode
    )
    if airborne:
        state, spin, t = _fly(ball, surface, rec, t)
    else:
        state = (ball.x_m, ball.y_m, ball.vx_mps, ball.vy_mps, spin * _R_BALL)

    sliding = not math.isclose(
        state[4], math.hypot(state[2], state[3]), rel_tol=0.0, abs_tol=1e-12
    )
    while t < _MAX_TIME_S:
        speed = math.hypot(state[2], state[3])
        if (
            speed <= surface.friction.v_stop_mps
            and abs(state[4]) <= surface.friction.v_stop_mps
        ):
            restarted = _restart_state(state[0], state[1], surface)
            if restarted is None:
                break
            state, sliding = restarted, False
        prev = state
        previous_slip = math.hypot(prev[2], prev[3]) - prev[4]
        state = _rk4_step(state, sliding, surface)
        t += DT_S
        step = math.hypot(state[0] - prev[0], state[1] - prev[1])
        rec.distance += step
        speed = math.hypot(state[2], state[3])
        if sliding:
            rec.skid_distance += step
            current_slip = speed - state[4]
            crossed_zero = previous_slip * current_slip <= 0.0
            within_one_step = (
                abs(current_slip) <= abs(current_slip - previous_slip) + 1e-12
            )
            if crossed_zero or within_one_step:
                sliding = False
        if not sliding:
            # Pin the contact-surface speed to pure roll.
            state = (state[0], state[1], state[2], state[3], speed)
        spin = state[4] / _R_BALL
        mode = Mode.SLIDE if sliding else Mode.ROLL
        rec.record(t, state[0], state[1], 0.0, speed, spin, mode)
        if rec.check_hole(state[0], state[1], speed, on_ground=True):
            break

    last = rec.samples[-1]
    if not rec.holed and last.speed_mps <= surface.friction.v_stop_mps:
        rec.record(last.t_s, last.x_m, last.y_m, 0.0, 0.0, spin, Mode.REST)
    ensure(rec.distance >= 0.0, "distance must be non-negative", rec.distance)
    ensure(
        rec.skid_distance <= rec.distance + 1e-9,
        "skid cannot exceed the total path",
        (rec.skid_distance, rec.distance),
    )
    return PuttResult(
        samples=tuple(rec.samples),
        collision=collision,
        holed=rec.holed,
        rest_x_m=rec.samples[-1].x_m,
        rest_y_m=rec.samples[-1].y_m,
        total_distance_m=rec.distance,
        time_s=rec.samples[-1].t_s,
        skid_distance_m=rec.skid_distance,
        speed_at_hole_mps=rec.speed_at_hole,
    )


def simulate_strike(
    putter: PutterState,
    surface: SurfaceSpec,
    impact_toe_m: float = 0.0,
    hole_x_m: float | None = None,
    hole_y_m: float | None = None,
) -> PuttResult:
    """Strike the ball with a putter and integrate the full putt.

    Chains :func:`~.collision.strike` (impulse + hosel twist), the
    airborne rise/settle phase, and the surface mode machine.

    Args:
        putter: Putter head at impact.
        surface: Green description.
        impact_toe_m: Strike offset toe-ward of the head CG [m].
        hole_x_m: Hole center x [m]; hole disabled when None.
        hole_y_m: Hole center y [m]; hole disabled when None.

    Returns:
        The integrated :class:`~.types.PuttResult` with its
        :class:`~.types.CollisionReport` attached.
    """
    report = strike(putter, impact_toe_m=impact_toe_m)
    ball = BallState(
        x_m=0.0,
        y_m=0.0,
        height_m=0.0,
        vx_mps=report.horizontal_speed_mps,
        vy_mps=0.0,
        vz_mps=max(report.vertical_speed_mps, 0.0),
        spin_rad_s=report.spin_rad_s,
    )
    return simulate_ball(
        ball,
        surface,
        hole_x_m=hole_x_m,
        hole_y_m=hole_y_m,
        collision=report,
    )
