"""Vectorised ground-roll integrator for many putts at once (#14171).

Integrates N balls together with the same mode machine, fixed RK4 step,
roll-onset projection and static-hold restart as
:func:`putting_dynamics.solver.simulate_ball`, which stays the reference
(parity is test-enforced). Constants and laws are imported from
``putting_dynamics``; the one restatement is the array form of
``rolling_mu`` / ``sliding_mu`` in :func:`_effective_mu`, pinned to the
scalar laws by a test.

Capture has two rules:

* ``"point"``: the solver's rule (centre inside the cup at the end of a
  step, speed at most ``capture_speed_mps()``), kept for parity.
* ``"effective_radius"`` (default): per step, the closest approach of the
  step's segment to the cup centre must lie within the shrinking hole
  ``R_eff(v) = R sqrt(1 - (v / v_cap)^2)`` (Holmes 1991; Penner 2002, as
  Tools ``swing_sim.putting.capture``). ``v_cap`` is the
  ``putting_dynamics`` bound so both rules agree at a dead-centre pass.
  It also catches a fast ball that skips over the cup between steps.

Balls that pass within the cup radius but are not captured are recorded
as lip-outs. Balls start on the ground (no airborne hop). Finished balls
are dropped from the working arrays, so cost falls as putts finish.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.shared.python.contracts import require
from src.shared.python.core.physics_constants import GRAVITY_M_S2
from src.shared.python.putting_dynamics.solver import (
    _MAX_TIME_S,
    DT_S,
    HOLE_RADIUS_M,
    capture_speed_mps,
)
from src.shared.python.putting_dynamics.surfaces import SurfaceSpec

Capture = Literal["point", "effective_radius"]

_G = float(GRAVITY_M_S2)
_SLIP_EPS = 1e-12  # same tolerances as simulate_ball's slip tests
_SLIDE_SPIN_GAIN = 2.5  # 5/2 from I = (2/5) m r^2, as in the solver ODE
_PLANE_TOL = 1e-12  # relative spread under which a height field is a plane
_CHUNK = 4096  # balls per integration chunk (fits the working set in cache)


@dataclass(frozen=True)
class BatchRollResult:
    """Per-ball outcome arrays (SI units)."""

    rest_x_m: np.ndarray
    rest_y_m: np.ndarray
    holed: np.ndarray
    lipped_out: np.ndarray  # passed within the cup radius, not captured
    timed_out: np.ndarray
    speed_at_hole_mps: np.ndarray  # NaN where the ball never reached the cup


def effective_hole_radius_m(speed_mps: np.ndarray) -> np.ndarray:
    """Array ``R_eff(v)``: the full cup at rest, zero at ``capture_speed_mps()``."""
    ratio = np.minimum(np.asarray(speed_mps) / capture_speed_mps(), 1.0)
    return HOLE_RADIUS_M * np.sqrt(1.0 - ratio * ratio)


def _cell(x, y, origin, spacing, shape):  # type: ignore[no-untyped-def]
    """Vectorised clamped bilinear cell index and fractions (solver rule)."""
    fx = (x - origin[0]) / spacing
    fy = (y - origin[1]) / spacing
    ix = np.clip(np.floor(fx), 0, shape[1] - 2).astype(np.intp)
    iy = np.clip(np.floor(fy), 0, shape[0] - 2).astype(np.intp)
    return ix, iy, np.clip(fx - ix, 0.0, 1.0), np.clip(fy - iy, 0.0, 1.0)


def _plane_gradient(h: np.ndarray, spacing: float) -> tuple[float, float] | None:
    """Constant ``(dh/dx, dh/dy)`` when every bilinear patch is the same plane."""
    dx, dy = np.diff(h, axis=1) / spacing, np.diff(h, axis=0) / spacing
    scale = max(1.0, float(np.abs(dx).max()), float(np.abs(dy).max()))
    if np.ptp(dx) > _PLANE_TOL * scale or np.ptp(dy) > _PLANE_TOL * scale:
        return None
    return float(dx.mean()), float(dy.mean())


class _Field:
    """Array versions of the surface lookups used by the ODE."""

    def __init__(self, surface: SurfaceSpec) -> None:
        self.h = surface.height
        self.f = surface.friction_field
        self.plane = _plane_gradient(self.h.heights_m, self.h.spacing_m)
        roll, slide = self.f.roll_multiplier, self.f.slide_multiplier
        self.uniform = bool(np.all(roll == 1.0) and np.all(slide == 1.0))

    def gradient(self, x: np.ndarray, y: np.ndarray):  # type: ignore[no-untyped-def]
        """Exact bilinear-patch gradient, as ``HeightField.gradient``."""
        if self.plane is not None:
            return self.plane
        h, sp = self.h.heights_m, self.h.spacing_m
        ix, iy, tx, ty = _cell(x, y, self.h.origin_m, sp, h.shape)
        h00, h10 = h[iy, ix], h[iy, ix + 1]
        h01, h11 = h[iy + 1, ix], h[iy + 1, ix + 1]
        dhdx = ((h10 - h00) * (1 - ty) + (h11 - h01) * ty) / sp
        dhdy = ((h01 - h00) * (1 - tx) + (h11 - h10) * tx) / sp
        return dhdx, dhdy

    def multipliers(self, x: np.ndarray, y: np.ndarray):  # type: ignore[no-untyped-def]
        """Bilinear (roll, slide) multipliers, as ``FrictionField.multipliers``."""
        if self.uniform:
            return 1.0, 1.0
        ix, iy, tx, ty = _cell(
            x, y, self.f.origin_m, self.f.spacing_m, self.f.roll_multiplier.shape
        )
        w00, w10 = (1 - tx) * (1 - ty), tx * (1 - ty)
        w01, w11 = (1 - tx) * ty, tx * ty
        return tuple(
            g[iy, ix] * w00
            + g[iy, ix + 1] * w10
            + g[iy + 1, ix] * w01
            + g[iy + 1, ix + 1] * w11
            for g in (self.f.roll_multiplier, self.f.slide_multiplier)
        )


def _effective_mu(surface, speed, vx, vy, sliding, mults):  # type: ignore[no-untyped-def]
    """Array form of ``friction.rolling_mu`` / ``friction.sliding_mu``.

    ``sliding`` may be ``None`` when no ball is sliding.
    """
    p = surface.friction
    grain = 1.0
    if p.grain_strength != 0.0:
        cg, sg = np.cos(p.grain_direction_rad), np.sin(p.grain_direction_rad)
        moving = speed > 0.0
        inv = np.divide(1.0, speed, out=np.zeros_like(speed), where=moving)
        cos_rel = np.where(moving, (vx * cg + vy * sg) * inv, cg)
        grain = 1.0 - p.grain_strength * cos_rel
    roll = p.mu_roll0 * mults[0] * (1.0 + p.k_v_per_mps * speed) * grain
    if sliding is None:
        return roll
    return np.where(sliding, p.mu_slide * mults[1] * grain, roll)


def _derivative(state, sliding, surface, field):  # type: ignore[no-untyped-def]
    """Ground-mode ODE, row-for-row the solver's ``_derivative``."""
    x, y, vx, vy, s = state
    gx, gy = field.gradient(x, y)
    speed = np.hypot(vx, vy)
    moving = speed > 0.0
    inv = np.divide(1.0, speed, out=np.zeros_like(speed), where=moving)
    mu = _effective_mu(surface, speed, vx, vy, sliding, field.multipliers(x, y))
    out = np.empty_like(state)
    out[0], out[1], out[4] = vx, vy, 0.0
    drag = mu * _G * inv
    if sliding is not None:
        sign = np.where(sliding & (speed - s < 0.0), -1.0, 1.0)
        drag = sign * drag
        out[4] = np.where(sliding & moving, sign * _SLIDE_SPIN_GAIN * mu * _G, 0.0)
    out[2] = -drag * vx - _G * gx
    out[3] = -drag * vy - _G * gy
    return out


def _rk4(state, sliding, surface, field):  # type: ignore[no-untyped-def]
    slide = sliding if sliding.any() else None
    k1 = _derivative(state, slide, surface, field)
    k2 = _derivative(state + 0.5 * DT_S * k1, slide, surface, field)
    k3 = _derivative(state + 0.5 * DT_S * k2, slide, surface, field)
    k4 = _derivative(state + DT_S * k3, slide, surface, field)
    return state + (DT_S / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _settle(state, sliding, surface, field):  # type: ignore[no-untyped-def]
    """Rest check before a step; returns the mask of balls held at rest.

    Restarts the others on a steep slope with a tiny downhill roll, as the
    solver's ``_restart_state``.
    """
    v_stop = surface.friction.v_stop_mps
    speed = np.hypot(state[2], state[3])
    near = (speed <= v_stop) & (np.abs(state[4]) <= v_stop)
    if not near.any():
        return near
    gx, gy = field.gradient(state[0], state[1])
    slope = np.hypot(gx, gy)
    limit = surface.friction.mu_static * field.multipliers(state[0], state[1])[0]
    hold = near & (slope <= limit)
    restart = near & ~hold
    if restart.any():
        safe = np.where(restart, slope, 1.0)
        state[2] = np.where(restart, -v_stop * gx / safe, state[2])
        state[3] = np.where(restart, -v_stop * gy / safe, state[3])
        state[4] = np.where(restart, v_stop, state[4])
        sliding &= ~restart
    return hold


def _roll_onset(state, sliding, prev_slip):  # type: ignore[no-untyped-def]
    """Slip-crossing test and roll projection, as ``_roll_onset_state``."""
    speed = np.hypot(state[2], state[3])
    if not sliding.any():
        state[4] = speed
        return speed
    slip = speed - state[4]
    onset = sliding & (
        (prev_slip * slip <= 0.0)
        | (np.abs(slip) <= np.abs(slip - prev_slip) + _SLIP_EPS)
    )
    v_roll = (speed + 0.4 * state[4]) / 1.4
    ratio = np.divide(v_roll, speed, out=np.zeros_like(speed), where=speed > 0.0)
    scale = np.where(onset, ratio, 1.0)
    state[2] *= scale
    state[3] *= scale
    sliding &= ~onset
    speed = np.where(onset, v_roll, speed)
    state[4] = np.where(sliding, state[4], speed)
    return speed


def _approach(p0, p1, v0, v1, hole, rule):  # type: ignore[no-untyped-def]
    """Distance to the cup centre and speed there, for this step."""
    if rule == "point":
        return np.hypot(p1[0] - hole[0], p1[1] - hole[1]), v1
    d = p1 - p0
    w = np.array(hole, dtype=float)[:, None] - p0
    dd = d[0] * d[0] + d[1] * d[1]
    along = np.divide(
        w[0] * d[0] + w[1] * d[1], dd, out=np.zeros_like(dd), where=dd > 0
    )
    t = np.clip(along, 0.0, 1.0)
    return np.hypot(w[0] - t * d[0], w[1] - t * d[1]), v0 + t * (v1 - v0)


def roll_batch(
    start: np.ndarray,
    surface: SurfaceSpec,
    hole_xy_m: tuple[float, float],
    capture: Capture = "effective_radius",
) -> BatchRollResult:
    """Roll every ball in ``start`` to rest, capture or the time cap.

    Balls are integrated in chunks of ``_CHUNK`` so the working arrays stay
    in cache; results do not depend on the chunking.

    Args:
        start: Shape ``(5, N)`` rows ``x, y, vx, vy, s`` [m, m/s], where
            ``s = omega * r`` is the contact-surface speed (``s == |v|`` for
            a ball already rolling).
        surface: Green description shared by every ball.
        hole_xy_m: Hole centre [m].
        capture: ``"effective_radius"`` or the solver's ``"point"`` rule.

    Returns:
        Per-ball :class:`BatchRollResult`.
    """
    start = np.asarray(start, dtype=float)
    require(start.ndim == 2 and start.shape[0] == 5, "start must be (5, N)")
    require(bool(np.all(np.isfinite(start))), "start must be finite")
    require(capture in ("point", "effective_radius"), "unknown capture", capture)
    field = _Field(surface)
    parts = [
        _roll_chunk(start[:, i : i + _CHUNK], surface, field, hole_xy_m, capture)
        for i in range(0, start.shape[1], _CHUNK)
    ]
    return BatchRollResult(*(np.concatenate(col) for col in zip(*parts, strict=True)))


def _roll_chunk(start, surface, field, hole_xy_m, capture):  # type: ignore[no-untyped-def]
    """Integrate one chunk; returns the :class:`BatchRollResult` columns."""
    n = start.shape[1]
    out, ids, state = np.empty((2, n)), np.arange(n), start.copy()
    holed, near_any = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    at_hole = np.full(n, np.nan)
    speed = np.hypot(state[2], state[3])
    sliding = ~np.isclose(state[4], speed, rtol=0.0, atol=_SLIP_EPS)
    v_cap, t = capture_speed_mps(), 0.0
    while t < _MAX_TIME_S and ids.size:
        rest = _settle(state, sliding, surface, field)
        if rest.any():
            out[:, ids[rest]] = state[:2, rest]
            state, sliding, ids = state[:, ~rest], sliding[~rest], ids[~rest]
        p0, v0 = state[:2].copy(), np.hypot(state[2], state[3])
        prev_slip = v0 - state[4]
        state = _rk4(state, sliding, surface, field)
        t += DT_S
        speed = _roll_onset(state, sliding, prev_slip)
        dist, v_at = _approach(p0, state[:2], v0, speed, hole_xy_m, capture)
        near = dist <= HOLE_RADIUS_M
        if not near.any():
            continue
        first = near & np.isnan(at_hole[ids])
        at_hole[ids[first]] = v_at[first]
        near_any[ids[near]] = True
        if capture == "point":
            caught = near & (v_at <= v_cap)
        else:
            caught = near & (v_at < v_cap) & (dist <= effective_hole_radius_m(v_at))
        if caught.any():
            holed[ids[caught]] = True
            out[:, ids[caught]] = state[:2, caught]
            state, sliding, ids = state[:, ~caught], sliding[~caught], ids[~caught]
    out[:, ids] = state[:2]
    timed_out = np.zeros(n, dtype=bool)
    timed_out[ids] = True
    lipped = near_any & ~holed
    return out[0], out[1], holed, lipped, timed_out, at_hole
