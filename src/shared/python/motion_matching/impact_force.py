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

Contact timing. A replay lags its reference by a few milliseconds (about
14 cm of head travel at impact), so contact at the *capture's* impact time
would strike the head well short of the ball. With ``trigger == "passage"``
the plan is armed at ``t_start_s`` and the contact starts when the simulated
face centre reaches its closest approach to ``passage_point_m`` (the address
face centre: the ball's contact point, the same definition as
:func:`model_appearance.club_face.ball_passage`) within
``PASSAGE_GATE_M``. ``t_start_s`` then becomes that event time, sub-step
accurate from the face-centre velocity at the step start.

The opt-in ``surface`` trigger uses the finite rendered face triangles and
actual unforced step callback to bracket sphere contact. The legacy ``passage``
trigger remains a diagnostic centre-proximity event; it does not establish
finite-face contact.

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
import math
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
from src.shared.python.model_appearance.ball import BALL_RADIUS_M
from src.shared.python.motion_matching.contact_event import (
    ContactRoot,
    find_contact_event,
)

Array = NDArray[np.float64]
PoseFn = Callable[[Array], Mapping[str, Array]]
AccelFn = Callable[[Array, Array, Array], Array]
ExternalFn = Callable[[Array], Array]
StepFn = Callable[[float, float, Array, Array, ExternalFn | None], tuple[Any, ...]]

_FD_STEP = 1e-6
_EDGE_TOL_S = 1e-12
#: Arm the passage trigger this long before the capture's impact time.
PASSAGE_ARM_LEAD_S = 0.020
#: The face centre must pass within this distance of the ball point (the
#: ``ball_passage`` tolerance, ``IMPACT_BALL_RADIUS_M``).
PASSAGE_GATE_M = 0.15


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
    trigger: str = "time"
    passage_point_m: Array | None = None
    scheduled: bool = False
    face_triangles_in_frame: Array | None = field(default=None, compare=False)
    armed_at_s: float | None = None
    contact_event: ContactRoot | None = field(default=None, compare=False)
    contact_miss_penalty_m: float | None = None
    contact_miss_penalty_logsum: float | None = None
    contact_miss_penalty_weight_s: float = 0.0
    contact_point_in_frame: Array | None = None
    # A candidate bracket is sampled evidence, never first-contact certification.
    contact_search_status: str | None = None
    contact_search_has_prior_unresolved_intervals: bool = False

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
        if self.trigger not in ("time", "passage", "surface"):
            raise ValueError(
                f"trigger must be 'time', 'passage', or 'surface', not {self.trigger}"
            )
        if not isinstance(self.contact_search_has_prior_unresolved_intervals, bool):
            raise ValueError("prior unresolved contact status must be boolean")
        if self.passage_point_m is not None:
            object.__setattr__(
                self,
                "passage_point_m",
                _finite3(self.passage_point_m, "passage_point_m"),
            )
        if self.trigger == "passage" and self.passage_point_m is None:
            raise ValueError("a passage trigger needs passage_point_m")
        if self.face_triangles_in_frame is not None:
            triangles = np.asarray(self.face_triangles_in_frame, dtype=float)
            if (
                triangles.ndim != 3
                or triangles.shape[1:] != (3, 3)
                or len(triangles) == 0
                or not np.isfinite(triangles).all()
            ):
                raise ValueError(
                    "face_triangles_in_frame must be finite (F, 3, 3) geometry"
                )
            object.__setattr__(self, "face_triangles_in_frame", triangles.copy())
        if self.trigger == "surface":
            if self.ball_centre_m is None or self.face_triangles_in_frame is None:
                raise ValueError("surface contact needs ball centre and face triangles")
            armed = float(
                self.t_start_s if self.armed_at_s is None else self.armed_at_s
            )
            if (
                not np.isfinite(armed)
                or armed < self.swing_span_s[0]
                or armed > self.t_start_s
            ):
                raise ValueError(
                    "armed_at_s must be inside the swing and at or before contact"
                )
            object.__setattr__(self, "armed_at_s", armed)
            valid_search_statuses = {
                "unsearched",
                "sampled_bracket_candidate",
                "unresolved_no_accepted_candidate",
                "sampled_bracket_outside_window",
            }
            search_status = self.contact_search_status
            if search_status is None:
                search_status = (
                    "sampled_bracket_candidate"
                    if self.contact_event is not None
                    else "unsearched"
                )
                object.__setattr__(self, "contact_search_status", search_status)
            if search_status not in valid_search_statuses:
                raise ValueError("surface contact search status is invalid")
            if self.scheduled != (search_status == "sampled_bracket_candidate"):
                raise ValueError(
                    "surface contact status disagrees with scheduled state"
                )
            if self.scheduled != (self.contact_event is not None):
                raise ValueError(
                    "scheduled surface contact needs exactly one contact_event"
                )
            if self.force_world is not None and (
                self.contact_event is None or self.contact_point_in_frame is None
            ):
                raise ValueError(
                    "surface replay force requires an event and local point"
                )
            if self.contact_event is not None:
                event = self.contact_event
                point = np.asarray(event.contact_point_m, dtype=float)
                normal = np.asarray(event.contact_normal, dtype=float)
                if (
                    event.q.ndim != 1
                    or event.qd.ndim != 1
                    or not np.isfinite(event.q).all()
                    or not np.isfinite(event.qd).all()
                    or point.shape != (3,)
                    or normal.shape != (3,)
                    or not np.isfinite(point).all()
                    or not np.isfinite(normal).all()
                    or not np.isfinite(event.time_s)
                    or not np.isfinite(event.approach_speed_mps)
                    or event.approach_speed_mps <= 0.0
                    or event.triangle_index < 0
                    or event.triangle_index >= len(self.face_triangles_in_frame)
                    or abs(float(np.linalg.norm(event.contact_normal)) - 1.0) > 1e-8
                    or abs(event.time_s - float(self.t_start_s)) > 1e-12
                ):
                    raise ValueError("contact_event fields are invalid")
                validate_window(event.time_s, self.duration_s, self.swing_span_s)
        else:
            if self.contact_event is not None:
                raise ValueError("contact_event is only valid for a surface trigger")
            if self.contact_search_status is not None:
                raise ValueError(
                    "contact search status is only valid for a surface trigger"
                )
            if self.contact_search_has_prior_unresolved_intervals:
                raise ValueError(
                    "prior unresolved contact status is only valid for a surface trigger"
                )
        if self.contact_miss_penalty_m is not None and (
            not np.isfinite(self.contact_miss_penalty_m)
            or self.contact_miss_penalty_m < 0.0
        ):
            raise ValueError("contact_miss_penalty_m must be finite and nonnegative")
        if not np.isfinite(self.contact_miss_penalty_weight_s) or (
            self.contact_miss_penalty_weight_s < 0.0
        ):
            raise ValueError(
                "contact_miss_penalty_weight_s must be finite and nonnegative"
            )
        if self.contact_miss_penalty_logsum is not None and not np.isfinite(
            self.contact_miss_penalty_logsum
        ):
            raise ValueError("contact_miss_penalty_logsum must be finite")
        if self.contact_point_in_frame is not None:
            object.__setattr__(
                self,
                "contact_point_in_frame",
                _finite3(self.contact_point_in_frame, "contact_point_in_frame"),
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

    @classmethod
    def for_spec_surface_contact(
        cls,
        spec: Mapping[str, Any],
        *,
        t_start_s: float,
        swing_span_s: tuple[float, float],
        ball_centre_m: Any,
    ) -> ImpactForce:
        """Opt-in spec-backed finite-face contact plan."""
        from src.shared.python.motion_matching import club_face_target as cft

        plan = cls.for_spec(
            spec,
            t_start_s=t_start_s,
            swing_span_s=swing_span_s,
            ball_centre_m=ball_centre_m,
        )
        return replace(
            plan,
            face_triangles_in_frame=cft.face_triangles_in_frame(spec),
        ).with_surface_contact()

    @property
    def is_latched(self) -> bool:
        return self.force_world is not None

    @property
    def is_scheduled(self) -> bool:
        """Contact start known (fixed time or a trigger event has fired)."""
        return self.trigger == "time" or self.scheduled

    def with_passage(self, point_m: Any) -> ImpactForce:
        """Arm the legacy centre-proximity diagnostic at ``point_m``.

        This trigger does not verify the finite clubface or sphere contact.
        """
        return replace(
            self,
            trigger="passage",
            passage_point_m=point_m,
            scheduled=False,
            contact_event=None,
            contact_search_status=None,
            contact_search_has_prior_unresolved_intervals=False,
        )

    def with_surface_contact(self) -> ImpactForce:
        """Opt into finite face-triangle contact, failing closed without geometry."""
        if self.is_latched:
            raise ValueError("cannot change a latched impact trigger")
        if self.ball_centre_m is None or self.face_triangles_in_frame is None:
            raise ValueError("surface contact needs ball centre and face triangles")
        return replace(
            self,
            trigger="surface",
            scheduled=False,
            armed_at_s=float(self.t_start_s),
            contact_event=None,
            contact_miss_penalty_m=None,
            contact_miss_penalty_logsum=None,
            contact_miss_penalty_weight_s=0.0,
            contact_point_in_frame=None,
            contact_search_status="unsearched",
            contact_search_has_prior_unresolved_intervals=False,
        )

    def at_address(self, q0: Array, poses: PoseFn) -> ImpactForce:
        """Passage plan whose ball is at this run's address face (``q0``).

        The passage point is the address face centre and the ball centre the
        shared :func:`model_appearance.ball.ball_position_at_address`, centred
        at the face-centre height (the ``capture_impact`` convention).
        """
        from src.shared.python.model_appearance.ball import (
            BALL_RADIUS_M,
            ball_position_at_address,
        )

        point, normal = self._point_and_normal(np.asarray(q0, dtype=float), poses)
        ball = ball_position_at_address(
            point, normal, ground_height_m=float(point[2]) - BALL_RADIUS_M
        )
        return replace(self.with_passage(point), ball_centre_m=ball)

    def rearmed(self, lead_s: float = PASSAGE_ARM_LEAD_S) -> ImpactForce:
        """Unlatched copy for a new closed-loop run in another plant.

        A passage plan is re-armed ``lead_s`` before its recorded contact so
        the new run finds its own contact; a fixed-time plan keeps its time.
        """
        start = float(self.t_start_s)
        if self.trigger in ("passage", "surface"):
            start = max(start - lead_s, float(self.swing_span_s[0]))
        return replace(
            self,
            force_world=None,
            collision=None,
            scheduled=False,
            t_start_s=start,
            armed_at_s=start if self.trigger == "surface" else None,
            contact_event=None,
            contact_miss_penalty_m=None,
            contact_miss_penalty_logsum=None,
            contact_miss_penalty_weight_s=0.0,
            contact_point_in_frame=None,
            contact_search_status="unsearched" if self.trigger == "surface" else None,
            contact_search_has_prior_unresolved_intervals=False,
        )

    def schedule(
        self,
        t: float,
        dt: float,
        q: Array,
        v: Array,
        poses: PoseFn,
        *,
        advance: Callable[[float], tuple[Array, Array]] | None = None,
    ) -> ImpactForce:
        """Fire the configured event if it occurs within the actual step."""
        if self.trigger == "surface":
            return self._schedule_surface(t, dt, q, v, poses, advance)
        if self.is_scheduled or t + dt < float(self.t_start_s):
            return self
        q = np.asarray(q, dtype=float)
        v = np.asarray(v, dtype=float)
        point, _ = self._point_and_normal(q, poses)
        ahead = self._point_and_normal(q + _FD_STEP * v, poses)[0]
        behind = self._point_and_normal(q - _FD_STEP * v, poses)[0]
        velocity = (ahead - behind) / (2.0 * _FD_STEP)
        offset = point - self.passage_point_m  # type: ignore[operator]
        speed2 = float(velocity @ velocity)
        if float(np.linalg.norm(offset)) > PASSAGE_GATE_M or speed2 <= 0.0:
            return self
        lead = -float(offset @ velocity) / speed2
        if lead >= dt:
            return self
        return replace(self, t_start_s=t + max(lead, 0.0), scheduled=True)

    def _schedule_surface(
        self,
        t: float,
        dt: float,
        q: Array,
        v: Array,
        poses: PoseFn,
        advance: Callable[[float], tuple[Array, Array]] | None,
    ) -> ImpactForce:
        """Bracket finite-face contact using the integrator's actual advance."""
        if self.scheduled or self.is_latched:
            return self
        if advance is None:
            raise ValueError("surface contact search needs an integrator advance")
        if self.ball_centre_m is None:
            raise ValueError("surface contact needs a ball centre")
        arm = float(self.armed_at_s if self.armed_at_s is not None else self.t_start_s)
        offset = max(0.0, arm - t)
        search_duration = dt - offset
        if search_duration <= 0.0:
            return self
        if offset <= _EDGE_TOL_S:
            q_arm, v_arm = np.asarray(q, dtype=float), np.asarray(v, dtype=float)
        else:
            q_arm, v_arm = (np.asarray(x, dtype=float) for x in advance(offset))

        def armed_advance(seconds: float) -> tuple[Array, Array]:
            return advance(offset + seconds)

        def triangles_world(q_state: Array) -> Array:
            pose = np.asarray(poses(q_state)[self.frame], dtype=float)
            triangles = self.face_triangles_in_frame
            if triangles is None:
                raise ValueError("surface contact lost its face triangles")
            return triangles @ pose[:3, :3].T + pose[:3, 3]

        result = find_contact_event(
            lambda seconds: (
                (q_arm, v_arm) if seconds == 0.0 else armed_advance(seconds)
            ),
            triangles_world,
            self.ball_centre_m,
            BALL_RADIUS_M,
            duration_s=search_duration,
        )
        penalty = result.closest_approach_penalty_m
        weight_s = search_duration
        logsum = math.log(weight_s) - penalty / BALL_RADIUS_M
        if self.contact_miss_penalty_logsum is not None:
            logsum = float(np.logaddexp(self.contact_miss_penalty_logsum, logsum))
        total_weight = self.contact_miss_penalty_weight_s + weight_s
        aggregate = -BALL_RADIUS_M * (logsum - math.log(total_weight))
        accumulated = replace(
            self,
            contact_miss_penalty_m=aggregate,
            contact_miss_penalty_logsum=logsum,
            contact_miss_penalty_weight_s=total_weight,
            contact_search_status=(
                "unresolved_no_accepted_candidate"
                if result.event is not None
                else result.coverage_status
            ),
            contact_search_has_prior_unresolved_intervals=(
                self.contact_search_has_prior_unresolved_intervals
                or result.event is None
            ),
        )
        if result.event is None:
            return accumulated
        event = replace(result.event, time_s=t + offset + result.event.time_s)
        try:
            validate_window(event.time_s, self.duration_s, self.swing_span_s)
        except ValueError:
            return replace(
                accumulated,
                contact_search_status="sampled_bracket_outside_window",
                contact_search_has_prior_unresolved_intervals=True,
            )
        return replace(
            accumulated,
            t_start_s=event.time_s,
            scheduled=True,
            contact_event=event,
            contact_search_status="sampled_bracket_candidate",
        )

    @property
    def t_end_s(self) -> float:
        return float(self.t_start_s) + float(self.duration_s)

    def with_force(self, force_world: Any) -> ImpactForce:
        """Copy with an explicit world force (replays of a recorded impact)."""
        return replace(self, force_world=_finite3(force_world, "force_world"))

    def shifted(self, offset_s: float) -> ImpactForce:
        """Copy on a clock shifted by ``offset_s`` (segment replays)."""
        span = (self.swing_span_s[0] + offset_s, self.swing_span_s[1] + offset_s)
        hit = self.collision
        if hit is not None:
            hit = replace(hit, time_s=hit.time_s + offset_s)
        event = self.contact_event
        if event is not None:
            event = replace(event, time_s=event.time_s + offset_s)
        return replace(
            self,
            t_start_s=self.t_start_s + offset_s,
            swing_span_s=span,
            collision=hit,
            armed_at_s=(
                None if self.armed_at_s is None else self.armed_at_s + offset_s
            ),
            contact_event=event,
        )

    # -- kinematics -------------------------------------------------------
    def _point_and_normal(
        self, q: Array, poses: PoseFn, point_in_frame: Array | None = None
    ) -> tuple[Array, Array]:
        pose = np.asarray(poses(q)[self.frame], dtype=float)
        rotation = pose[:3, :3]
        point = self.point_in_frame if point_in_frame is None else point_in_frame
        return rotation @ point + pose[:3, 3], rotation @ self.axis_in_frame

    def point_jacobian(
        self, q: Array, poses: PoseFn, point_in_frame: Array | None = None
    ) -> Array:
        """(3, nv) world Jacobian of the application point (central differences)."""
        q = np.asarray(q, dtype=float)
        columns = []
        for k in range(q.size):
            dq = np.zeros_like(q)
            dq[k] = _FD_STEP
            plus = self._point_and_normal(q + dq, poses, point_in_frame)[0]
            minus = self._point_and_normal(q - dq, poses, point_in_frame)[0]
            columns.append((plus - minus) / (2.0 * _FD_STEP))
        return np.column_stack(columns)

    def generalized_force(self, q: Array, poses: PoseFn) -> Array:
        """``J(q)^T F`` of the latched world force."""
        if self.force_world is None:
            raise ValueError("impact force is not latched yet")
        point = self.contact_point_in_frame
        return self.point_jacobian(q, poses, point).T @ self.force_world

    def latch(self, q: Array, v: Array, poses: PoseFn, accel: AccelFn) -> ImpactForce:
        """Fix the force from the state at the start of contact.

        ``accel(q, v, f)`` must return the plant acceleration with the extra
        generalized force ``f`` (no control effort), so the difference of two
        calls isolates ``M^-1`` (including contact and grip constraints).
        """
        q = np.asarray(q, dtype=float)
        v = np.asarray(v, dtype=float)
        point_in_frame = self.point_in_frame
        if self.trigger == "surface":
            if self.contact_event is None:
                raise ValueError("surface impact cannot latch without a contact root")
            pose = np.asarray(poses(q)[self.frame], dtype=float)
            point_in_frame = pose[:3, :3].T @ (
                self.contact_event.contact_point_m - pose[:3, 3]
            )
            point = self.contact_event.contact_point_m
            normal = self.contact_event.contact_normal
        else:
            point, normal = self._point_and_normal(q, poses)
        jac = self.point_jacobian(q, poses, point_in_frame)
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
        return replace(
            self,
            force_world=hit.mean_force_on_club_n,
            collision=hit,
            contact_point_in_frame=(
                point_in_frame if self.trigger == "surface" else None
            ),
        )

    # -- time stepping ----------------------------------------------------
    def sub_intervals(self, t: float, dt: float) -> list[tuple[float, float, bool]]:
        """Split ``[t, t + dt]`` at the window edges; flag the forced parts."""
        end = t + dt
        cuts = [t]
        edges = [float(self.t_start_s), self.t_end_s]
        if self.trigger == "surface" and self.armed_at_s is not None:
            edges.append(float(self.armed_at_s))
        if self.trigger == "surface" and not self.scheduled:
            edges = [float(self.armed_at_s)] if self.armed_at_s is not None else []
        interior = sorted(
            edge for edge in edges if t + _EDGE_TOL_S < edge < end - _EDGE_TOL_S
        )
        for edge in interior:
            if edge - cuts[-1] > _EDGE_TOL_S:
                cuts.append(edge)
        cuts.append(end)
        return [
            (
                a,
                b,
                self.is_scheduled and self.t_start_s <= 0.5 * (a + b) <= self.t_end_s,
            )
            for a, b in zip(cuts[:-1], cuts[1:], strict=True)
        ]

    def to_record(self) -> dict[str, Any]:
        """JSON-serialisable plan plus the latched force and collision."""

        def vec(x: Array | None) -> list[float] | None:
            return None if x is None else [float(c) for c in x]

        record = {
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
            "trigger": self.trigger,
            "passage_point_m": vec(self.passage_point_m),
            "scheduled": bool(self.scheduled),
            "collision": None if self.collision is None else self.collision.to_record(),
        }
        if self.trigger == "surface":
            if self.face_triangles_in_frame is None or self.armed_at_s is None:
                raise ValueError("surface contact requires geometry and arming time")
            record.update(
                {
                    "face_triangles_in_frame": self.face_triangles_in_frame.tolist(),
                    "armed_at_s": float(self.armed_at_s),
                    "contact_miss_penalty_m": self.contact_miss_penalty_m,
                    "contact_miss_penalty_logsum": self.contact_miss_penalty_logsum,
                    "contact_miss_penalty_weight_s": self.contact_miss_penalty_weight_s,
                    "contact_search_status": self.contact_search_status,
                    "contact_search_has_prior_unresolved_intervals": (
                        self.contact_search_has_prior_unresolved_intervals
                    ),
                    "contact_point_in_frame": vec(self.contact_point_in_frame),
                    "contact_event": (
                        None
                        if self.contact_event is None
                        else {
                            "time_s": self.contact_event.time_s,
                            "q": self.contact_event.q.tolist(),
                            "qd": self.contact_event.qd.tolist(),
                            "contact_point_m": self.contact_event.contact_point_m.tolist(),
                            "contact_normal": self.contact_event.contact_normal.tolist(),
                            "approach_speed_mps": self.contact_event.approach_speed_mps,
                            "triangle_index": self.contact_event.triangle_index,
                        }
                    ),
                }
            )
        return record

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> ImpactForce:
        """Rebuild a plan (and its latched force) from :meth:`to_record`."""
        force = record.get("force_world_n")
        span = record["swing_span_s"]
        raw_event = record.get("contact_event")
        event = None
        if raw_event is not None:
            if not isinstance(raw_event, Mapping):
                raise ValueError("contact_event must be an object")
            event = ContactRoot(
                time_s=float(raw_event["time_s"]),
                q=np.asarray(raw_event["q"], dtype=float),
                qd=np.asarray(raw_event["qd"], dtype=float),
                contact_point_m=_finite3(
                    raw_event["contact_point_m"], "contact_event.contact_point_m"
                ),
                contact_normal=_finite3(
                    raw_event["contact_normal"], "contact_event.contact_normal"
                ),
                approach_speed_mps=float(raw_event["approach_speed_mps"]),
                triangle_index=int(raw_event["triangle_index"]),
            )
            if (
                not np.isfinite(event.time_s)
                or event.q.ndim != 1
                or event.qd.ndim != 1
                or not np.isfinite(event.q).all()
                or not np.isfinite(event.qd).all()
                or not np.isfinite(event.approach_speed_mps)
                or event.approach_speed_mps <= 0.0
                or abs(float(np.linalg.norm(event.contact_normal)) - 1.0) > 1e-8
                or event.triangle_index < 0
            ):
                raise ValueError("contact_event fields are invalid")
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
            trigger=str(record.get("trigger", "time")),
            passage_point_m=(
                None
                if record.get("passage_point_m") is None
                else np.asarray(record["passage_point_m"], dtype=float)
            ),
            scheduled=bool(record.get("scheduled", False)),
            face_triangles_in_frame=(
                None
                if record.get("face_triangles_in_frame") is None
                else np.asarray(record["face_triangles_in_frame"], dtype=float)
            ),
            armed_at_s=(
                None
                if record.get("armed_at_s") is None
                else float(record["armed_at_s"])
            ),
            contact_event=event,
            contact_miss_penalty_m=(
                None
                if record.get("contact_miss_penalty_m") is None
                else float(record["contact_miss_penalty_m"])
            ),
            contact_miss_penalty_logsum=(
                None
                if record.get("contact_miss_penalty_logsum") is None
                else float(record["contact_miss_penalty_logsum"])
            ),
            contact_miss_penalty_weight_s=float(
                record.get("contact_miss_penalty_weight_s", 0.0)
            ),
            contact_point_in_frame=(
                None
                if record.get("contact_point_in_frame") is None
                else np.asarray(record["contact_point_in_frame"], dtype=float)
            ),
            contact_search_status=(
                None
                if record.get("contact_search_status") is None
                else str(record["contact_search_status"])
            ),
            contact_search_has_prior_unresolved_intervals=record.get(
                "contact_search_has_prior_unresolved_intervals", False
            ),
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
    trial_step: StepFn | None = None,
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
    search_plan = impact
    probe = step if trial_step is None else trial_step

    def advance_unforced(seconds: float) -> tuple[Array, Array]:
        arm = search_plan.armed_at_s if search_plan.trigger == "surface" else None
        if arm is not None and t < arm < t + seconds:
            arm_duration = arm - t
            q_arm, v_arm, _ = probe(t, arm_duration, q, v, None)
            q_end, v_end, _ = probe(arm, seconds - arm_duration, q_arm, v_arm, None)
        else:
            q_end, v_end, _ = probe(t, seconds, q, v, None)
        return np.asarray(q_end, dtype=float), np.asarray(v_end, dtype=float)

    impact = impact.schedule(
        t,
        dt,
        q,
        v,
        poses,
        advance=advance_unforced if impact.trigger == "surface" else None,
    )
    if not impact.is_scheduled:
        parts = impact.sub_intervals(t, dt)
        if len(parts) == 1:
            q, v, extra = step(t, dt, q, v, None)
            return q, v, impact, extra
        first_unforced = None
        for index, (a, b, _) in enumerate(parts):
            q, v, extra = step(a, b - a, q, v, None)
            if index == 0:
                first_unforced = extra
        return q, v, impact, first_unforced
    parts = impact.sub_intervals(t, dt)
    if len(parts) == 1 and not parts[0][2]:
        q, v, extra = step(t, dt, q, v, None)  # bit-identical outside the window
        return q, v, impact, extra
    first: Any = None
    for index, (a, b, active) in enumerate(parts):
        external: ExternalFn | None = None
        if active:
            if (
                not impact.is_latched
                and impact.trigger == "surface"
                and impact.contact_event is not None
            ):
                event = impact.contact_event
                if abs(a - event.time_s) <= _EDGE_TOL_S:
                    if not (
                        np.allclose(q, event.q, rtol=1e-9, atol=1e-10)
                        and np.allclose(v, event.qd, rtol=1e-9, atol=1e-10)
                    ):
                        raise RuntimeError(
                            "surface contact root state differs from step replay"
                        )
                    q, v = event.q.copy(), event.qd.copy()
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
