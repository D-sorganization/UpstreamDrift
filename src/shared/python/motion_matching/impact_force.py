"""Ball-impact external force: the one hook every dynamics replay uses (GCV-20).

The tracked swings used to pass through the ball with nothing pushing back,
so the computed-torque controller had to decelerate the head itself and the
12 Hz reference spread that deceleration ~25 ms before impact (#11767). Here
the ball's collision impulse (:mod:`impact_parameters.ball_impact`) is
applied to the clubhead as a short constant world force at the face centre:

1. At the start of the contact window the plan is *latched* from the
   simulated state: face normal ``n = R a`` (``a`` the face axis in the club
   frame), face-centre velocity ``J v`` and the plant's effective mass along
   ``n``, ``m = 1 / (n . J (a(tau + J^T n) - a(tau)))`` (one extra
   acceleration solve, exact because the acceleration is affine in force).
2. Over ``[t_start, t_start + duration]`` the generalized force ``J(q)^T F``
   with the constant ``F = impulse_on_club / duration`` is added to the plant
   acceleration. Integrator steps are split at the window edges so the
   delivered impulse is exact.
3. The latched force is recorded (``to_record``) so open-loop replays in any
   engine apply the identical force (``from_record``, ``shifted`` for a
   segment clock).

The point Jacobian is a central finite difference of the frame pose in the
generalized coordinates; it equals the velocity Jacobian because the
full-body coordinates have ``nq == nv`` (Euler-angle root).

The reference-rate helpers at the end split the controller's velocity and
acceleration tables at impact, so the feedforward does not differentiate
across the velocity step the ball causes (no torque spike).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.impact_parameters.ball_impact import (
    BALL_MASS_KG,
    CONTACT_DURATION_S,
    COR_LIMIT,
    BallCollision,
    BallSpec,
    FaceContact,
    collision_impulse,
    validate_window,
)

Array = NDArray[np.float64]
PoseFn = Callable[[Array], Mapping[str, Array]]
AccelFn = Callable[[Array, Array, Array], Array]
ExternalFn = Callable[[Array], Array]
StepFn = Callable[[float, float, Array, Array, ExternalFn | None], tuple[Any, ...]]

_FD_STEP = 1e-6
_EDGE_TOL_S = 1e-12


def _finite3(value: Any, name: str, *, nonzero: bool = False) -> Array:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.shape != (3,) or not np.isfinite(arr).all():
        raise ValueError(f"{name} must be a finite 3-vector")
    if nonzero and float(np.linalg.norm(arr)) < 1e-12:
        raise ValueError(f"{name} must be nonzero")
    return arr


@dataclass(frozen=True)
class ImpactForce:
    """Planned (and, once latched, determined) ball force on the clubhead."""

    frame: str
    point_in_frame: Array
    axis_in_frame: Array
    t_start_s: float
    duration_s: float = CONTACT_DURATION_S
    swing_span_s: tuple[float, float] = (0.0, np.inf)
    ball_mass_kg: float = BALL_MASS_KG
    cor: float = COR_LIMIT
    ball_centre_m: Array | None = None
    force_world: Array | None = None
    collision: BallCollision | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.frame, str) or not self.frame:
            raise ValueError("frame must name the clubhead frame")
        object.__setattr__(
            self, "point_in_frame", _finite3(self.point_in_frame, "point_in_frame")
        )
        axis = _finite3(self.axis_in_frame, "axis_in_frame", nonzero=True)
        object.__setattr__(self, "axis_in_frame", axis / np.linalg.norm(axis))
        for name in ("duration_s", "ball_mass_kg"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if not 0.0 < float(self.cor) <= 1.0:
            raise ValueError(f"cor must lie in (0, 1], got {self.cor!r}")
        validate_window(float(self.t_start_s), self.duration_s, self.swing_span_s)
        if self.ball_centre_m is not None:
            object.__setattr__(
                self, "ball_centre_m", _finite3(self.ball_centre_m, "ball_centre_m")
            )
        if self.force_world is not None:
            object.__setattr__(
                self, "force_world", _finite3(self.force_world, "force_world")
            )

    @classmethod
    def for_spec(
        cls,
        spec: Mapping[str, Any],
        *,
        t_start_s: float,
        swing_span_s: tuple[float, float],
        ball_centre_m: Any = None,
    ) -> ImpactForce:
        """Plan at the spec's rendered face centre along its face normal."""
        from src.shared.python.motion_matching import club_face_target as cft

        return cls(
            frame=cft.FACE_FRAME,
            point_in_frame=cft.face_centre_in_frame(spec),
            axis_in_frame=cft.face_axis_in_frame(spec),
            t_start_s=t_start_s,
            swing_span_s=swing_span_s,
            ball_centre_m=ball_centre_m,
        )

    @property
    def is_latched(self) -> bool:
        return self.force_world is not None

    @property
    def t_end_s(self) -> float:
        return float(self.t_start_s) + float(self.duration_s)

    def with_force(self, force_world: Any) -> ImpactForce:
        """Copy with an explicit world force (replays of a recorded impact)."""
        return replace(self, force_world=_finite3(force_world, "force_world"))

    def unlatched(self) -> ImpactForce:
        """Copy that re-latches from the next plant it runs in."""
        return replace(self, force_world=None, collision=None)

    def shifted(self, offset_s: float) -> ImpactForce:
        """Copy on a clock shifted by ``offset_s`` (segment replays)."""
        span = (self.swing_span_s[0] + offset_s, self.swing_span_s[1] + offset_s)
        hit = self.collision
        if hit is not None:
            hit = replace(hit, time_s=hit.time_s + offset_s)
        return replace(
            self, t_start_s=self.t_start_s + offset_s, swing_span_s=span, collision=hit
        )

    # -- kinematics -------------------------------------------------------
    def _point_and_normal(self, q: Array, poses: PoseFn) -> tuple[Array, Array]:
        pose = np.asarray(poses(q)[self.frame], dtype=float)
        rotation = pose[:3, :3]
        return rotation @ self.point_in_frame + pose[
            :3, 3
        ], rotation @ self.axis_in_frame

    def point_jacobian(self, q: Array, poses: PoseFn) -> Array:
        """(3, nv) world Jacobian of the application point (central differences)."""
        q = np.asarray(q, dtype=float)
        columns = []
        for k in range(q.size):
            dq = np.zeros_like(q)
            dq[k] = _FD_STEP
            plus = self._point_and_normal(q + dq, poses)[0]
            minus = self._point_and_normal(q - dq, poses)[0]
            columns.append((plus - minus) / (2.0 * _FD_STEP))
        return np.column_stack(columns)

    def generalized_force(self, q: Array, poses: PoseFn) -> Array:
        """``J(q)^T F`` of the latched world force."""
        if self.force_world is None:
            raise ValueError("impact force is not latched yet")
        return self.point_jacobian(q, poses).T @ self.force_world

    def latch(self, q: Array, v: Array, poses: PoseFn, accel: AccelFn) -> ImpactForce:
        """Fix the force from the state at the start of contact.

        ``accel(q, v, f)`` must return the plant acceleration with the extra
        generalized force ``f`` (no control effort), so the difference of two
        calls isolates ``M^-1`` (including contact and grip constraints).
        """
        q = np.asarray(q, dtype=float)
        v = np.asarray(v, dtype=float)
        point, normal = self._point_and_normal(q, poses)
        jac = self.point_jacobian(q, poses)
        base = np.asarray(accel(q, v, np.zeros(q.size)), dtype=float)
        probe = np.asarray(accel(q, v, jac.T @ normal), dtype=float)
        compliance = float(normal @ (jac @ (probe - base)))
        if not np.isfinite(compliance) or compliance <= 0.0:
            raise ValueError(
                f"face has no positive mobility along its normal ({compliance!r})"
            )
        hit = collision_impulse(
            FaceContact(
                face_normal=normal,
                face_velocity_mps=jac @ v,
                application_point_m=point,
                effective_mass_kg=1.0 / compliance,
            ),
            time_s=float(self.t_start_s),
            swing_span_s=self.swing_span_s,
            ball=BallSpec(
                mass_kg=self.ball_mass_kg,
                cor=self.cor,
                duration_s=self.duration_s,
                centre_m=self.ball_centre_m,
            ),
        )
        return replace(self, force_world=hit.mean_force_on_club_n, collision=hit)

    # -- time stepping ----------------------------------------------------
    def sub_intervals(self, t: float, dt: float) -> list[tuple[float, float, bool]]:
        """Split ``[t, t + dt]`` at the window edges; flag the forced parts."""
        end = t + dt
        cuts = [t]
        for edge in (float(self.t_start_s), self.t_end_s):
            if t + _EDGE_TOL_S < edge < end - _EDGE_TOL_S:
                cuts.append(edge)
        cuts.append(end)
        return [
            (a, b, self.t_start_s <= 0.5 * (a + b) <= self.t_end_s)
            for a, b in zip(cuts[:-1], cuts[1:], strict=True)
        ]

    def to_record(self) -> dict[str, Any]:
        """JSON-serialisable plan plus the latched force and collision."""

        def vec(x: Array | None) -> list[float] | None:
            return None if x is None else [float(c) for c in x]

        return {
            "frame": self.frame,
            "point_in_frame": vec(self.point_in_frame),
            "axis_in_frame": vec(self.axis_in_frame),
            "t_start_s": float(self.t_start_s),
            "duration_s": float(self.duration_s),
            "swing_span_s": [float(x) for x in self.swing_span_s],
            "ball_mass_kg": float(self.ball_mass_kg),
            "cor": float(self.cor),
            "ball_centre_m": vec(self.ball_centre_m),
            "force_world_n": vec(self.force_world),
            "collision": None if self.collision is None else self.collision.to_record(),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> ImpactForce:
        """Rebuild a plan (and its latched force) from :meth:`to_record`."""
        force = record.get("force_world_n")
        span = record["swing_span_s"]
        return cls(
            frame=str(record["frame"]),
            point_in_frame=np.asarray(record["point_in_frame"], dtype=float),
            axis_in_frame=np.asarray(record["axis_in_frame"], dtype=float),
            t_start_s=float(record["t_start_s"]),
            duration_s=float(record["duration_s"]),
            swing_span_s=(float(span[0]), float(span[1])),
            ball_mass_kg=float(record["ball_mass_kg"]),
            cor=float(record["cor"]),
            ball_centre_m=(
                None
                if record.get("ball_centre_m") is None
                else np.asarray(record["ball_centre_m"], dtype=float)
            ),
            force_world=None if force is None else np.asarray(force, dtype=float),
        )


def step_through_impact(
    impact: ImpactForce | None,
    t: float,
    dt: float,
    q: Array,
    v: Array,
    step: StepFn,
    *,
    poses: PoseFn,
    accel: AccelFn,
) -> tuple[Array, Array, ImpactForce | None, Any]:
    """Advance ``[t, t + dt]`` with ``step``, applying the ball force inside it.

    ``step(t_a, h, q, v, external)`` advances by ``h`` and returns
    ``(q, v, extra)``; ``external`` maps ``q`` to the generalized ball force
    (``None`` outside the window). The step is split at the window edges, the
    plan is latched at the window start, and the first ``extra`` is returned.
    Returns ``(q, v, impact, extra)`` with the (possibly newly latched) plan.
    """
    if impact is None:
        q, v, extra = step(t, dt, q, v, None)
        return q, v, None, extra
    parts = impact.sub_intervals(t, dt)
    if len(parts) == 1 and not parts[0][2]:
        q, v, extra = step(t, dt, q, v, None)  # bit-identical outside the window
        return q, v, impact, extra
    first: Any = None
    for index, (a, b, active) in enumerate(parts):
        external: ExternalFn | None = None
        if active:
            if not impact.is_latched:
                impact = impact.latch(q, v, poses, accel)
            plan = impact

            def external(q_k: Array, plan: ImpactForce = plan) -> Array:
                return plan.generalized_force(q_k, poses)

        q, v, extra = step(a, b - a, q, v, external)
        if index == 0:
            first = extra
    return q, v, impact, first


# -- reference rates split at impact -----------------------------------------
def reference_rates(
    times: Array, reference: Array, split_time_s: float | None
) -> tuple[Array, Array, int | None]:
    """Velocity and acceleration tables of ``reference`` (rows on ``times``).

    With ``split_time_s`` inside ``(times[0], times[-1])`` the samples at or
    before it and those after it are differentiated separately, so the
    velocity step at impact is not smeared into an acceleration spike.
    Returns ``(velocity, acceleration, last_pre_impact_row)``.
    """
    t = np.asarray(times, dtype=float)
    ref = np.asarray(reference, dtype=float)
    if t.size < 2:
        zero = np.zeros_like(ref)
        return zero, zero.copy(), None
    if split_time_s is None:
        vel = np.gradient(ref, t, axis=0)
        return vel, np.gradient(vel, t, axis=0), None
    last = int(np.searchsorted(t, float(split_time_s), side="right")) - 1
    if not 1 <= last < t.size - 2:
        raise ValueError(
            f"split_time_s {split_time_s!r} must leave two samples on each side"
        )
    parts = [(t[: last + 1], ref[: last + 1]), (t[last + 1 :], ref[last + 1 :])]
    vels = [np.gradient(r, tt, axis=0) for tt, r in parts]
    accs = [
        np.gradient(vv, tt, axis=0) for vv, (tt, _) in zip(vels, parts, strict=True)
    ]
    return np.vstack(vels), np.vstack(accs), last


def sample_rate_table(
    times: Array, table: Array, t: float, split: tuple[float, int] | None
) -> Array:
    """Linear sample of a rate table, clamped to the side of impact ``t`` is on.

    ``split`` is ``(split_time_s, last_pre_impact_row)`` or ``None``.
    """
    if split is None:
        lo, hi = 0, times.size
    elif t <= split[0]:
        lo, hi = 0, split[1] + 1
    else:
        lo, hi = split[1] + 1, times.size
    seg_t, seg = times[lo:hi], table[lo:hi]
    return np.array([np.interp(t, seg_t, seg[:, k]) for k in range(seg.shape[1])])
