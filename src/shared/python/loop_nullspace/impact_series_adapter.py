"""Clubhead series to rigid club state, and event location (NSX-83).

:func:`impact_outcome_map` takes an engine-free :class:`RigidClubState`.  This
module is the one bridge from the engine-agnostic
:class:`~impact_parameters.ClubheadSeries` (what every engine adapter already
emits through ``rigid_body_series``) to that state, so the outcome map is fed
by the same samples the shared extractor reads and no engine re-derives them.

* **Rotation** is rebuilt from the series face axes and the club-face spec
  (``R a_k = w_k`` for each body axis ``a_k`` and world axis ``w_k``) and
  projected onto SO(3).
* **Angular velocity** is not stored in a series; it is the central difference
  ``log(R_{i+1} R_{i-1}^T) / (t_{i+1} - t_{i-1})`` (one-sided at the ends).  It
  is exact for a constant-twist rotation and second-order accurate otherwise.
* **Unobserved axial rotation** (``face_normal is None``): the state keeps the
  recorded reason, takes the face centre as the body origin and the face-centre
  velocity as the origin velocity, and carries no angular velocity.  Face
  outcomes are then ``UNAVAILABLE``; path, attack angle and speed are
  unaffected.
* :func:`advance_to_event` moves a state along the declared constant-twist flow
  to the instant the face centre reaches the ball plane (Newton on the exact
  flow), so the outcome map is evaluated on the physical event, not on a
  sample time.

Software correctness only; no scientific qualification is claimed.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.shared.python.impact_parameters.adapters.club_face import (
    NATIVE_CLUB_FACE,
    ClubFaceSpec,
)
from src.shared.python.impact_parameters.clubhead_series import ClubheadSeries

from .impact_outcome_map import (
    BallPlaneEvent,
    RigidClubState,
    advance_constant_twist,
    constant_twist_flow,
)

Array = NDArray[np.float64]
Axes = tuple[Array, Array, Array]

ORTHONORMAL_TOL = 1e-6
HALF_TURN_MARGIN_RAD = 1e-3
DEFAULT_HORIZON_S = 0.05
EVENT_TOL_M = 1e-12
PAST_TOL_S = 1e-9
MAX_NEWTON_ITERATIONS = 30
_SMALL_ANGLE = 1e-9


def log_so3(rot: Array) -> Array:
    """Rotation vector of a proper rotation matrix.

    Raises:
        ValueError: If the rotation is within ``HALF_TURN_MARGIN_RAD`` of a half
            turn, where the rotation vector is not unique.
    """
    vee = 0.5 * np.array(
        [rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]]
    )
    sine = float(np.linalg.norm(vee))
    angle = math.atan2(sine, 0.5 * (float(np.trace(rot)) - 1.0))
    if angle > math.pi - HALF_TURN_MARGIN_RAD:
        raise ValueError(
            "rotation between samples is a half turn; the angular velocity "
            "is ambiguous (sampling too coarse)"
        )
    return vee if sine < _SMALL_ANGLE else (angle / sine) * vee


def _rotation_at(axes: Axes, index: int, spec: ClubFaceSpec) -> Array:
    """Body-to-world rotation from the world face axes at one sample."""
    world = np.stack([axis[index] for axis in axes])
    world = world / np.linalg.norm(world, axis=1, keepdims=True)
    if np.max(np.abs(world @ world.T - np.eye(3))) > ORTHONORMAL_TOL:
        raise ValueError(
            f"face axes at sample {index} are not orthonormal within "
            f"{ORTHONORMAL_TOL:g}; the club rotation is undefined"
        )
    left, _, right = np.linalg.svd(world.T @ spec.axes())
    rot = left @ right
    if np.linalg.det(rot) < 0.0:
        raise ValueError(f"face axes at sample {index} form a left-handed frame")
    return rot


def _check_inputs(series: object, index: object, spec: object) -> int:
    if not isinstance(series, ClubheadSeries):
        raise TypeError("series must be a ClubheadSeries")
    if not isinstance(spec, ClubFaceSpec):
        raise TypeError("spec must be a ClubFaceSpec")
    if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
        raise TypeError("index must be an integer")
    if not 0 <= int(index) < len(series):
        raise ValueError(f"index {index} out of range [0, {len(series)})")
    return int(index)


def club_state_from_series(
    series: ClubheadSeries, index: int, spec: ClubFaceSpec = NATIVE_CLUB_FACE
) -> RigidClubState:
    """Rigid club state at sample ``index`` of a clubhead series.

    Preconditions: ``series`` carries all three face axes, or none with a
    ``face_unobservable_reason``; observed axes are orthonormal within
    ``ORTHONORMAL_TOL`` at the sample and its neighbours.
    Postconditions: ``rigid_body_series`` applied to the returned state
    reproduces the sample's face centre, velocity and axes.

    Raises:
        TypeError: On wrong argument types.
        ValueError: On a bad index, a partly observed or non-orthonormal axis
            set, or a half turn between neighbouring samples.
    """
    idx = _check_inputs(series, index, spec)
    if series.face_normal is None:
        return RigidClubState(
            origin_m=series.face_center_m[idx],
            rotation=np.eye(3),
            velocity_mps=series.velocity_mps[idx],
            angular_velocity_rps=np.zeros(3),
            face_spec=dataclasses.replace(spec, face_center_body_m=(0.0, 0.0, 0.0)),
            face_unobservable_reason=series.face_unobservable_reason,
        )
    if series.toe_axis is None or series.grip_axis is None:
        raise ValueError(
            "toe_axis and grip_axis are required with face_normal; a partly "
            "observed rotation is not guessed"
        )
    lo, hi = max(idx - 1, 0), min(idx + 1, len(series) - 1)
    axes = (series.face_normal, series.toe_axis, series.grip_axis)
    rot = _rotation_at(axes, idx, spec)
    relative = _rotation_at(axes, hi, spec) @ _rotation_at(axes, lo, spec).T
    omega = log_so3(relative) / float(series.times_s[hi] - series.times_s[lo])
    offset = rot @ np.asarray(spec.face_center_body_m)
    return RigidClubState(
        origin_m=series.face_center_m[idx] - offset,
        rotation=rot,
        velocity_mps=series.velocity_mps[idx] - np.cross(omega, offset),
        angular_velocity_rps=omega,
        face_spec=spec,
    )


@dataclass(frozen=True)
class EventCrossing:
    """State at the event and the time ``time_s`` it took to reach it."""

    state: RigidClubState
    time_s: float


def _newton_time(state: RigidClubState, event: BallPlaneEvent) -> float | None:
    """Root of ``g(advance(state, t)) = 0`` by Newton, or ``None`` if stuck."""
    t = 0.0
    for _ in range(MAX_NEWTON_ITERATIONS):
        moved = advance_constant_twist(state, t)
        value = event.value(moved)
        if abs(value) <= EVENT_TOL_M:
            return t
        rate = float(event.gradient(moved) @ constant_twist_flow(moved))
        if abs(rate) < _SMALL_ANGLE:
            return None
        t -= value / rate
    return None


def advance_to_event(
    state: RigidClubState,
    event: BallPlaneEvent,
    *,
    max_horizon_s: float = DEFAULT_HORIZON_S,
) -> EventCrossing:
    """Advance ``state`` along the constant-twist flow to ``event``.

    Postconditions: ``|event.value(result.state)| <= 1e-12 m`` and
    ``0 <= result.time_s <= max_horizon_s`` (up to ``1e-9 s`` of rounding).

    Raises:
        TypeError: On wrong argument types.
        ValueError: If ``max_horizon_s`` is not positive, or the face does not
            reach the plane within the horizon (receding, tangent, or too far).
    """
    if not isinstance(state, RigidClubState):
        raise TypeError("state must be a RigidClubState")
    if not isinstance(event, BallPlaneEvent):
        raise TypeError("event must be a BallPlaneEvent")
    if not max_horizon_s > 0.0:
        raise ValueError(f"max_horizon_s must be positive, got {max_horizon_s}")
    time_s = _newton_time(state, event)
    if time_s is None or not -PAST_TOL_S <= time_s <= max_horizon_s:
        raise ValueError(
            f"the face centre does not reach the ball plane within {max_horizon_s:g} s"
        )
    time_s = max(time_s, 0.0)
    return EventCrossing(advance_constant_twist(state, time_s), time_s)
