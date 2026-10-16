"""Event-triggered impact outcome map (NSX-83, epic #12593 V2/V3).

The single place that turns a club state at the physical impact event into
launch-monitor outcomes and their Jacobians ``dy/dx``:

    y = (face angle, club path, attack angle, dynamic loft, spin loft, CHS).

* **Event, not a fixed time.**  The event is the face centre reaching the ball
  plane, ``g(x) = n . (r_face - p_ball) = 0``.  The reported Jacobian is the
  event-section one, ``J_event = J_y Pi`` with ``Pi`` from
  :mod:`event_section` (the timing/saltation correction
  ``dt* = -(g_x dx)/(g_x f + g_t)``).  The fixed-time Jacobian ``J_y`` is kept
  only as ``jacobian_fixed_time`` and is a **diagnostic**; peak-speed-proxy
  impact times are never used here.
* **One extractor.**  Values come from
  :func:`impact_parameters.extract_impact_parameters`; the engine-free
  Jacobian is a central finite difference on that same routine.  An engine
  adapter may supply ``kinematic_jacobian`` (analytic ``dv_face/dx`` and
  ``dn/dx``) and the chain rule to the outcomes is then closed form.
* **Fail closed.**  Face rows are ``UNAVAILABLE`` with the propagated reason
  when axial club rotation is unobserved; path, AoA and CHS stay available.

State coordinates (``RigidClubState``): ``x = (dp, dtheta, dv, dw)`` with
``p`` the club-body origin, ``R <- Exp(dtheta) R`` (world-frame left
perturbation), ``v`` the origin velocity and ``w`` the world-frame angular
velocity.  Near-grazing contact is flagged ``UNRELIABLE_LINEAR_ESTIMATE``:
Monte Carlo is required there.  Software correctness only; no scientific
qualification is claimed.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.shared.python.impact_parameters import TargetFrame
from src.shared.python.impact_parameters.adapters.club_face import (
    NATIVE_CLUB_FACE,
    ClubFaceSpec,
    rigid_body_series,
)
from src.shared.python.impact_parameters.clubhead_series import ClubheadSeries
from src.shared.python.impact_parameters.extract import (
    ImpactParameters,
    extract_impact_parameters,
)

from .event_section import (
    GRAZING_TRANSVERSALITY,
    event_projection,
    event_time_shift_row,
    transversality,
)
from .evidence import (
    AllocationDependence,
    InputProvenance,
    ModelProvenance,
    QuantityRecord,
)

Array = NDArray[np.float64]

OUTCOME_NAMES = (
    "face_angle_deg",
    "club_path_deg",
    "attack_angle_deg",
    "dynamic_loft_deg",
    "spin_loft_deg",
    "clubhead_speed_mps",
)
FACE_OUTCOMES = ("face_angle_deg", "dynamic_loft_deg", "spin_loft_deg")
_UNITS = {name: ("m/s" if name.endswith("_mps") else "deg") for name in OUTCOME_NAMES}
_WRAPPED = ("face_angle_deg", "club_path_deg")
STATE_DIM = 12
DEFAULT_STEP = 1e-4
EVENT_TOLERANCE_M = 1e-6
FIXED_TIME_LABEL = (
    "DIAGNOSTIC ONLY: fixed-time sensitivity omits the event-time shift; "
    "use the event-section Jacobian"
)
MODEL = ModelProvenance("impact_outcome_map", "1")
Flow = Callable[["RigidClubState"], Array]


def _skew(vec: Array) -> Array:
    return np.array(
        [[0.0, -vec[2], vec[1]], [vec[2], 0.0, -vec[0]], [-vec[1], vec[0], 0.0]]
    )


def exp_so3(rotvec: ArrayLike) -> Array:
    """Rodrigues rotation matrix for a rotation vector."""
    w = np.asarray(rotvec, dtype=np.float64)
    angle = float(np.linalg.norm(w))
    k = _skew(w)
    if angle < 1e-8:
        return np.eye(3) + k + 0.5 * (k @ k)
    return (
        np.eye(3)
        + (math.sin(angle) / angle) * k
        + ((1.0 - math.cos(angle)) / angle**2) * (k @ k)
    )


def _vec3(value: ArrayLike, name: str) -> Array:
    arr = np.array(value, dtype=np.float64, copy=True)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a finite 3-vector, got shape {arr.shape}")
    arr.flags.writeable = False
    return arr


@dataclass(frozen=True)
class RigidClubState:
    """Club-body rigid state at one instant (world frame).

    ``face_unobservable_reason`` set means the axial club rotation (and so the
    face normal) is unobserved: face outcomes are then unavailable.

    Raises:
        ValueError: On non-finite or mis-shaped inputs or a non-rotation.
    """

    origin_m: Array
    rotation: Array
    velocity_mps: Array
    angular_velocity_rps: Array
    face_spec: ClubFaceSpec = NATIVE_CLUB_FACE
    face_unobservable_reason: str | None = None

    def __post_init__(self) -> None:
        for name in ("origin_m", "velocity_mps", "angular_velocity_rps"):
            object.__setattr__(self, name, _vec3(getattr(self, name), name))
        rot = np.array(self.rotation, dtype=np.float64, copy=True)
        if rot.shape != (3, 3) or not np.all(np.isfinite(rot)):
            raise ValueError("rotation must be a finite 3x3 matrix")
        if np.max(np.abs(rot.T @ rot - np.eye(3))) > 1e-9 or (
            abs(np.linalg.det(rot) - 1.0) > 1e-9
        ):
            raise ValueError("rotation must be a proper rotation matrix")
        rot.flags.writeable = False
        object.__setattr__(self, "rotation", rot)

    @property
    def face_offset_m(self) -> Array:
        """World-frame vector from the body origin to the face centre."""
        return self.rotation @ np.asarray(self.face_spec.face_center_body_m)

    def retract(self, dx: ArrayLike) -> RigidClubState:
        """State moved by ``dx = (dp, dtheta, dv, dw)`` (12-vector)."""
        d = np.asarray(dx, dtype=np.float64)
        if d.shape != (STATE_DIM,) or not np.all(np.isfinite(d)):
            raise ValueError(f"dx must be a finite vector of length {STATE_DIM}")
        return dataclasses.replace(
            self,
            origin_m=self.origin_m + d[0:3],
            rotation=exp_so3(d[3:6]) @ self.rotation,
            velocity_mps=self.velocity_mps + d[6:9],
            angular_velocity_rps=self.angular_velocity_rps + d[9:12],
        )

    def series(self) -> ClubheadSeries:
        """Two-sample series holding this one instant (impact index 1).

        The state is duplicated so the shared extractor runs unchanged; the
        outcomes read only the last sample (the event instant).
        """
        two = np.stack([self.origin_m, self.origin_m])
        series = rigid_body_series(
            [0.0, 1.0],
            two,
            np.stack([self.rotation, self.rotation]),
            np.stack([self.velocity_mps, self.velocity_mps]),
            np.stack([self.angular_velocity_rps, self.angular_velocity_rps]),
            self.face_spec,
        )
        if self.face_unobservable_reason is None:
            return series
        return dataclasses.replace(
            series,
            face_normal=None,
            toe_axis=None,
            grip_axis=None,
            face_unobservable_reason=self.face_unobservable_reason,
        )


def constant_twist_flow(state: RigidClubState) -> Array:
    """Flow ``f = (v, w, 0, 0)`` of a constant-twist rigid body (declared model)."""
    return np.concatenate([state.velocity_mps, state.angular_velocity_rps, np.zeros(6)])


def advance_constant_twist(state: RigidClubState, dt: float) -> RigidClubState:
    """Exact constant-twist state ``dt`` later (matches :func:`constant_twist_flow`)."""
    return dataclasses.replace(
        state,
        origin_m=state.origin_m + state.velocity_mps * dt,
        rotation=exp_so3(state.angular_velocity_rps * dt) @ state.rotation,
    )


@dataclass(frozen=True)
class BallPlaneEvent:
    """Event ``g = normal . (r_face - point) = 0``: face centre at the ball plane."""

    point_m: tuple[float, float, float]
    normal: tuple[float, float, float]

    def __post_init__(self) -> None:
        _vec3(self.point_m, "point_m")
        norm = float(np.linalg.norm(_vec3(self.normal, "normal")))
        if norm < 1e-9:
            raise ValueError("normal must be nonzero")
        unit = tuple(float(c) / norm for c in self.normal)
        object.__setattr__(self, "normal", unit)

    def value(self, state: RigidClubState) -> float:
        """Signed face-centre distance from the plane (metres)."""
        face = state.origin_m + state.face_offset_m
        return float(np.asarray(self.normal) @ (face - np.asarray(self.point_m)))

    def gradient(self, state: RigidClubState) -> Array:
        """``g_x`` in state coordinates: ``(n, r x n, 0, 0)``."""
        n = np.asarray(self.normal)
        return np.concatenate([n, np.cross(state.face_offset_m, n), np.zeros(6)])


def _outcome_vector(
    state: RigidClubState, frame: TargetFrame
) -> tuple[Array, dict[str, str]]:
    """Outcome values via the shared extractor (NaN where unavailable)."""
    params: ImpactParameters = extract_impact_parameters(
        state.series(), frame, impact_index=1, use_tools=False
    )
    out = np.array(
        [
            np.nan if getattr(params, n) is None else getattr(params, n)
            for n in OUTCOME_NAMES
        ]
    )
    missing = {
        n: params.unavailable[n] for n in OUTCOME_NAMES if n in params.unavailable
    }
    return out, missing


def wrapped_difference(plus: Array, minus: Array, wrap: Array) -> Array:
    """``plus - minus`` with entries flagged in ``wrap`` mapped to ``[-180, 180)``."""
    diff = plus - minus
    return np.where(wrap, (diff + 180.0) % 360.0 - 180.0, diff)


def central_difference_jacobian(
    fn: Callable[[Array], Array], dim: int, step: float, wrap: Array
) -> Array:
    """Richardson-extrapolated central-difference Jacobian of ``fn`` at 0.

    Uses ``(4 D(h/2) - D(h)) / 3`` with ``D`` the central difference, which
    cancels the ``O(h^2)`` term (needed where the outcome is a stiff
    ``acos``).  This is the one FD routine for every engine-free Jacobian.

    Raises:
        ValueError: If ``step`` is not positive, or an output that is finite at
            the nominal point becomes undefined within the step.
    """
    if not step > 0.0:
        raise ValueError(f"step must be positive, got {step}")
    base = fn(np.zeros(dim))
    jac = np.full((base.size, dim), np.nan)
    for i in range(dim):
        e = np.zeros(dim)
        cols = []
        for h in (step, 0.5 * step):
            e[i] = h
            cols.append(wrapped_difference(fn(e), fn(-e), wrap) / (2.0 * h))
        col = (4.0 * cols[1] - cols[0]) / 3.0
        bad = np.isfinite(base) & ~np.isfinite(col)
        if np.any(bad):
            raise ValueError(
                f"output {np.flatnonzero(bad).tolist()} undefined within step {step}"
            )
        jac[:, i] = col
    jac[~np.isfinite(base)] = np.nan
    return jac


@dataclass(frozen=True)
class KinematicJacobian:
    """Analytic ``dv_face/dx`` and ``dn/dx`` (each 3 x 12) at a state."""

    dv: Array
    dn: Array


def rigid_kinematic_jacobian(state: RigidClubState) -> KinematicJacobian:
    """Closed-form kinematic Jacobian of the rigid club state (engine-free).

    ``v_f = v + w x r``, ``n = R n0``; with ``r = R c``:
    ``dv_f = dv - [w]x[r]x dtheta - [r]x dw`` and ``dn = -[n]x dtheta``.
    """
    r = state.face_offset_m
    n = state.rotation @ state.face_spec.axes()[0]
    dv = np.zeros((3, STATE_DIM))
    dv[:, 3:6] = -_skew(state.angular_velocity_rps) @ _skew(r)
    dv[:, 6:9] = np.eye(3)
    dv[:, 9:12] = -_skew(r)
    dn = np.zeros((3, STATE_DIM))
    dn[:, 3:6] = -_skew(n)
    return KinematicJacobian(dv=dv, dn=dn)


def _heading_grad(frame: TargetFrame, vec: Array) -> Array:
    x, y, _ = frame.components(vec)
    s = frame.lateral_sign
    return (
        math.degrees(1.0)
        * (x * (-s) * frame.y_t - (-s * y) * frame.x_t)
        / (x * x + y * y)
    )


def _elevation_grad(frame: TargetFrame, vec: Array) -> Array:
    x, y, z = frame.components(vec)
    h = math.hypot(x, y)
    dh = (x * frame.x_t + y * frame.y_t) / h
    return math.degrees(1.0) * (h * frame.z_t - z * dh) / (h * h + z * z)


def _spin_loft_grads(v: Array, n: Array) -> tuple[Array, Array]:
    nv, nn = float(np.linalg.norm(v)), float(np.linalg.norm(n))
    cosine = float(np.clip(v @ n / (nv * nn), -1.0, 1.0))
    scale = -math.degrees(1.0) / math.sqrt(max(1.0 - cosine * cosine, 1e-300))
    return (
        scale * (n / nn - cosine * v / nv) / nv,
        scale * (v / nv - cosine * n / nn) / nn,
    )


def analytic_outcome_jacobian(
    kin: KinematicJacobian, v: Array, n: Array, frame: TargetFrame
) -> Array:
    """Chain rule from ``(dv_f, dn)`` to the six outcomes (6 x n, closed form)."""
    dv_spin, dn_spin = _spin_loft_grads(v, n)
    rows = (
        _heading_grad(frame, n) @ kin.dn,
        _heading_grad(frame, v) @ kin.dv,
        _elevation_grad(frame, v) @ kin.dv,
        _elevation_grad(frame, n) @ kin.dn,
        dv_spin @ kin.dv + dn_spin @ kin.dn,
        (v / np.linalg.norm(v)) @ kin.dv,
    )
    return np.stack(rows)


@dataclass(frozen=True)
class ImpactOutcomeMap:
    """Outcomes and Jacobians at the impact event.

    ``jacobian`` (6 x n) is the **event-section** one; ``jacobian_fixed_time``
    is a labelled diagnostic.  Unavailable outcomes have NaN value and row and
    an entry in ``unavailable`` (and an ``UNAVAILABLE`` record).
    """

    names: tuple[str, ...]
    values: Array
    jacobian: Array
    jacobian_fixed_time: Array
    projection: Array
    event_time_shift_row: Array
    outcome_rates: Array
    records: dict[str, QuantityRecord]
    unavailable: dict[str, str]
    jacobian_method: str
    transversality: float
    reliability: str
    reliability_note: str
    section: str = "event"
    fixed_time_label: str = FIXED_TIME_LABEL


def _records(
    values: Array, unavailable: dict[str, str], source: str
) -> dict[str, QuantityRecord]:
    inputs = InputProvenance(kind="rigid_club_state", source=source)
    conv = ("event section: face centre at ball plane",)
    out: dict[str, QuantityRecord] = {}
    for name, value in zip(OUTCOME_NAMES, values, strict=True):
        reason = unavailable.get(name)
        out[name] = QuantityRecord(
            values=np.array([np.nan if reason else value]),
            units=_UNITS[name],
            allocation_dependence=(
                AllocationDependence.UNAVAILABLE
                if reason
                else AllocationDependence.INVARIANT
            ),
            model_provenance=MODEL,
            input_provenance=inputs,
            conventions=conv,
            reason=reason,
        )
    return out


def _reliability(trans: float) -> tuple[str, str]:
    if trans < GRAZING_TRANSVERSALITY:
        return (
            "UNRELIABLE_LINEAR_ESTIMATE",
            f"near-grazing contact (transversality {trans:.3g} < "
            f"{GRAZING_TRANSVERSALITY}); the event-time shift is not linear, "
            "Monte Carlo is required",
        )
    return "LINEAR_OK", ""


def impact_outcome_map(
    state: RigidClubState,
    event: BallPlaneEvent,
    frame: TargetFrame,
    *,
    flow: Flow,
    kinematic_jacobian: Callable[[RigidClubState], KinematicJacobian] | None = None,
    step: float = DEFAULT_STEP,
    check_on_event: bool = True,
) -> ImpactOutcomeMap:
    """Outcome map at the impact event with event-section Jacobians.

    Args:
        state: Club state at the event (``event.value(state) ~ 0``).
        event: Event surface (face centre reaches the ball plane).
        frame: Target frame for the outcome definitions.
        flow: ``state -> f`` (12-vector) flow just before the event.
        kinematic_jacobian: Optional analytic hook (engine frame Jacobian);
            ``None`` selects central FD on the shared extractor.
        step: FD step (state units).
        check_on_event: Require ``|g| <= 1e-6 m`` (disable only to evaluate
            outcomes at an off-event state for diagnostics).

    Postconditions: ``jacobian @ f == 0`` (a pure time shift along the flow
    changes no event outcome); rows of unavailable outcomes are NaN.

    Raises:
        TypeError: If ``state``/``event``/``frame`` have the wrong type.
        ValueError: If the state is off the event, the event is not
            transversal, or ``step`` is not positive.
    """
    if not isinstance(state, RigidClubState):
        raise TypeError("state must be a RigidClubState")
    if not isinstance(event, BallPlaneEvent):
        raise TypeError("event must be a BallPlaneEvent (no fixed-time default)")
    if not isinstance(frame, TargetFrame):
        raise TypeError("frame must be a TargetFrame")
    if not step > 0.0:
        raise ValueError(f"step must be positive, got {step}")
    if check_on_event and abs(event.value(state)) > EVENT_TOLERANCE_M:
        raise ValueError(
            f"state is not on the event (|g| = {abs(event.value(state)):.3g} m); "
            "advance it to the event first"
        )
    values, unavailable = _outcome_vector(state, frame)
    wrap = np.array([n in _WRAPPED for n in OUTCOME_NAMES])
    if kinematic_jacobian is None:
        fixed = central_difference_jacobian(
            lambda dx: _outcome_vector(state.retract(dx), frame)[0],
            STATE_DIM,
            step,
            wrap,
        )
        method = "central_fd"
    else:
        series = state.series()
        n_vec = np.zeros(3) if series.face_normal is None else series.face_normal[1]
        fixed = analytic_outcome_jacobian(
            kinematic_jacobian(state), series.velocity_mps[1], n_vec, frame
        )
        fixed[~np.isfinite(values)] = np.nan
        method = "analytic"
    f = np.asarray(flow(state), dtype=np.float64)
    grad_g = event.gradient(state)
    pi = event_projection(grad_g, f)
    trans = transversality(grad_g, f)
    reliability, note = _reliability(trans)
    return ImpactOutcomeMap(
        names=OUTCOME_NAMES,
        values=values,
        jacobian=fixed @ pi,
        jacobian_fixed_time=fixed,
        projection=pi,
        event_time_shift_row=event_time_shift_row(grad_g, f),
        outcome_rates=fixed @ f,
        records=_records(values, unavailable, "rigid_club_state@event"),
        unavailable=unavailable,
        jacobian_method=method,
        transversality=trans,
        reliability=reliability,
        reliability_note=note,
    )
