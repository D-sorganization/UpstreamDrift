"""Head gaze: eye point, gaze error, release schedule, stability metrics, neck IK.

OSV-3 (#11729). All functions are pure and vectorised over frames. Z is up,
lengths are metres, angles radians unless a name says ``_deg``/``_mm``.

Definitions (binding, from the issue):

* Eye point ``e = head_R @ eye_offset + head_t``. The offset is the
  anthropometric midpoint between the eyes in the head body frame (origin at
  the neck joint, +x anterior, +y left, +z up; see ``EYE_OFFSET_HEAD_M``).
* Gaze direction ``g = head_R @ GAZE_AXIS_HEAD`` and gaze error
  ``theta_gaze = angle(g, ball - e)``.
* Schedule: the target is the ball until ``t_impact + t_hold`` and then blends
  to the target-line direction at eye height (``e + d * x_t``; ``d`` cancels
  in the direction) with a minimum-jerk blend of duration ``t_rel``.
* The ball at address comes from
  :func:`src.shared.python.model_appearance.ball.ball_position_at_address`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from src.shared.python.motion_matching.range_of_motion import UPPER_RANGES_DEG

Array = NDArray[np.float64]

# de Leva (1996) male head: vertex to cervical joint, 0.2429 m of stature
# fraction; matches ``anthropometry.head.length_m`` in the full-body documents.
HEAD_LENGTH_M = 0.2429
# Vertex to the eye midpoint (~0.115 m, adult male survey value; an assumption
# recorded in the modelling reference, not a fitted number) puts the eyes at
# 0.2429 - 0.115 = 0.128 m above the neck joint; the eyes sit ~0.09 m anterior
# of the head frame axis (HeadFront forehead marker is at x = 0.10 m).
EYE_OFFSET_HEAD_M: tuple[float, float, float] = (0.09, 0.0, 0.128)
GAZE_AXIS_HEAD: tuple[float, float, float] = (1.0, 0.0, 0.0)
DEFAULT_T_HOLD_S = 0.03
DEFAULT_T_RELEASE_S = 0.35
# Neck joint primitives are Rx, Ry, Rz in that order (parent to child).
NECK_COORDINATES: tuple[str, str, str] = ("NeckInputX", "NeckInputY", "NeckInputZ")
_EULER = "XYZ"  # intrinsic, matching the Rx . Ry . Rz primitive chain


def _unit(v: Array, name: str) -> Array:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    if not np.all(n > 1e-12):
        raise ValueError(f"{name} must be nonzero")
    return v / n


def _rotations(head_r: Sequence | Array) -> Array:
    r = np.asarray(head_r, dtype=float)
    if r.ndim not in (2, 3) or r.shape[-2:] != (3, 3):
        raise ValueError("head rotation must be (3, 3) or (frames, 3, 3)")
    if not np.isfinite(r).all():
        raise ValueError("head rotation must be finite")
    return r


def eye_point(
    head_r: Sequence | Array,
    head_t: Sequence | Array,
    eye_offset: Sequence[float] = EYE_OFFSET_HEAD_M,
) -> Array:
    """World eye point ``head_R @ eye_offset + head_t`` ((3,) or (frames, 3))."""
    r = _rotations(head_r)
    t = np.asarray(head_t, dtype=float)
    off = np.asarray(eye_offset, dtype=float)
    if off.shape != (3,) or t.shape[-1:] != (3,):
        raise ValueError("eye offset and head translation must be 3-vectors")
    return np.einsum("...ij,j->...i", r, off) + t


def gaze_direction(
    head_r: Sequence | Array, axis: Sequence[float] = GAZE_AXIS_HEAD
) -> Array:
    """World gaze direction ``head_R @ axis`` (unit)."""
    a = _unit(np.asarray(axis, dtype=float), "gaze axis")
    return np.einsum("...ij,j->...i", _rotations(head_r), a)


def gaze_error(
    head_r: Sequence | Array,
    eye: Sequence | Array,
    ball: Sequence | Array,
    axis: Sequence[float] = GAZE_AXIS_HEAD,
) -> Array:
    """Angle (rad) between the gaze direction and the line of sight to the ball.

    Precondition: the eye point is not on the ball. Postcondition: in [0, pi].
    """
    g = gaze_direction(head_r, axis)
    sight = np.asarray(ball, dtype=float) - np.asarray(eye, dtype=float)
    sight = _unit(sight, "line of sight (eye coincides with ball)")
    cosine = np.clip(np.sum(g * sight, axis=-1), -1.0, 1.0)
    return np.arccos(cosine)


def min_jerk(s: Sequence[float] | Array) -> Array:
    """Minimum-jerk position profile ``10 s^3 - 15 s^4 + 6 s^5`` clamped to [0, 1].

    Velocity and acceleration vanish at both ends (C2 junction with the holds).
    """
    x = np.clip(np.asarray(s, dtype=float), 0.0, 1.0)
    return x**3 * (10.0 - 15.0 * x + 6.0 * x**2)


def gaze_target_direction(
    t: Sequence[float] | Array,
    eye: Sequence[float] | Array,
    ball: Sequence[float] | Array,
    target_dir: Sequence[float] | Array,
    *,
    t_impact: float,
    t_hold: float = DEFAULT_T_HOLD_S,
    t_rel: float = DEFAULT_T_RELEASE_S,
) -> Array:
    """Unit gaze target direction per time sample following the gaze schedule.

    The direction is ``ball - eye`` until ``t_impact + t_hold``, then a
    great-circle (slerp) blend with a minimum-jerk angle fraction over
    ``t_rel`` to ``target_dir`` (the target-line direction at eye height), then
    ``target_dir``. ``eye`` is (3,) or (frames, 3).

    Preconditions: ``t_hold >= 0``, ``t_rel > 0``, directions nonzero and not
    antiparallel (the blend plane would be undefined).
    """
    if t_hold < 0 or t_rel <= 0:
        raise ValueError("t_hold must be >= 0 and t_rel > 0")
    times = np.asarray(t, dtype=float)
    eyes = np.broadcast_to(np.asarray(eye, dtype=float), times.shape + (3,))
    start = _unit(np.asarray(ball, dtype=float) - eyes, "line of sight")
    end = np.broadcast_to(
        _unit(np.asarray(target_dir, dtype=float), "target direction"), start.shape
    )
    cosine = np.clip(np.sum(start * end, axis=-1), -1.0, 1.0)
    omega = np.arccos(cosine)
    if np.any(omega > np.pi - 1e-6):
        raise ValueError("ball and target directions are antiparallel")
    frac = min_jerk((times - (t_impact + t_hold)) / t_rel)
    sin_omega = np.sin(omega)
    safe = np.where(sin_omega < 1e-9, 1.0, sin_omega)
    a = np.where(sin_omega < 1e-9, 1.0 - frac, np.sin((1.0 - frac) * omega) / safe)
    b = np.where(sin_omega < 1e-9, frac, np.sin(frac * omega) / safe)
    out = a[..., None] * start + b[..., None] * end
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def impact_index(
    time_s: Sequence[float] | Array,
    clubhead_m: Sequence | Array,
    window_s: float = 0.05,
) -> int:
    """Impact frame: lowest clubhead point within ``window_s`` of peak speed.

    Peak speed comes from ``detect_impact_index`` (loaders/_align.py); the
    height minimum near it refines the frame. Never a fixed time.
    """
    from src.shared.python.motion_matching.loaders._align import detect_impact_index

    t = np.asarray(time_s, dtype=float)
    head = np.asarray(clubhead_m, dtype=float)
    if head.ndim != 2 or head.shape[1] != 3 or head.shape[0] != t.shape[0]:
        raise ValueError("clubhead must be (frames, 3) matching time")
    if window_s < 0:
        raise ValueError("window_s must be nonnegative")
    peak = int(detect_impact_index(t, head))
    near = np.flatnonzero(np.abs(t - t[peak]) <= window_s)
    return int(near[np.argmin(head[near, 2])])


@dataclass(frozen=True)
class GazeMetrics:
    """Head stability over an address-to-impact window (issue definitions)."""

    eye_translation_range_mm: tuple[float, float, float]
    head_yaw_range_deg: float
    head_pitch_range_deg: float
    head_roll_range_deg: float
    theta_gaze_max_deg: float
    theta_gaze_rms_deg: float
    frames: int

    def as_dict(self) -> dict[str, float | int | list[float]]:
        return {
            "eye_translation_range_mm": list(self.eye_translation_range_mm),
            "head_yaw_range_deg": self.head_yaw_range_deg,
            "head_pitch_range_deg": self.head_pitch_range_deg,
            "head_roll_range_deg": self.head_roll_range_deg,
            "theta_gaze_max_deg": self.theta_gaze_max_deg,
            "theta_gaze_rms_deg": self.theta_gaze_rms_deg,
            "frames": self.frames,
        }


def head_stability_metrics(
    head_r: Sequence | Array,
    head_t: Sequence | Array,
    ball: Sequence[float] | Array,
    first: int,
    last: int,
    *,
    eye_offset: Sequence[float] = EYE_OFFSET_HEAD_M,
    axis: Sequence[float] = GAZE_AXIS_HEAD,
) -> GazeMetrics:
    """Metrics over frames ``first..last`` inclusive (address to impact)."""
    r = _rotations(head_r)
    t = np.asarray(head_t, dtype=float)
    if r.ndim != 3 or t.shape != (r.shape[0], 3):
        raise ValueError("head_r (frames, 3, 3) and head_t (frames, 3) must agree")
    if not 0 <= first <= last < r.shape[0]:
        raise ValueError("first/last must satisfy 0 <= first <= last < frames")
    rs, ts = r[first : last + 1], t[first : last + 1]
    eyes = eye_point(rs, ts, eye_offset)
    translation = (eyes.max(axis=0) - eyes.min(axis=0)) * 1e3
    # ZYX Euler (yaw about world Z, pitch, roll), unwrapped so that a range is
    # a sweep rather than a branch jump.
    angles = np.degrees(np.unwrap(Rotation.from_matrix(rs).as_euler("ZYX"), axis=0))
    spans = angles.max(axis=0) - angles.min(axis=0)
    theta = np.degrees(gaze_error(rs, eyes, ball, axis))
    return GazeMetrics(
        eye_translation_range_mm=(
            float(translation[0]),
            float(translation[1]),
            float(translation[2]),
        ),
        head_yaw_range_deg=float(spans[0]),
        head_pitch_range_deg=float(spans[1]),
        head_roll_range_deg=float(spans[2]),
        theta_gaze_max_deg=float(theta.max()),
        theta_gaze_rms_deg=float(np.sqrt(np.mean(theta**2))),
        frames=int(last - first + 1),
    )


@dataclass(frozen=True)
class NeckIKResult:
    """Neck angles (X, Y, Z, rad) for a gaze direction, with ROM clamping flags."""

    angles_rad: Array
    residual_deg: float
    clamped: NDArray[np.bool_]


def neck_ik(
    parent_r: Sequence | Array,
    direction: Sequence[float] | Array,
    *,
    axis: Sequence[float] = GAZE_AXIS_HEAD,
    ranges_deg: dict[str, tuple[float, float]] | None = None,
    q0: Sequence[float] | None = None,
    pose_prior: float = 1e-3,
) -> NeckIKResult:
    """Neck angles that point the head forward axis along ``direction``.

    Solves ``parent_R . Rx(q0) Ry(q1) Rz(q2) . axis = direction`` as a bounded
    least squares inside the neck ROM (``range_of_motion.UPPER_RANGES_DEG``).
    The axis alone leaves the roll about it free, so a small prior toward
    ``q0`` (default zero) picks the minimal-motion solution. Out-of-range
    requests are clamped, never extrapolated: ``clamped`` flags coordinates
    resting on a bound and ``residual_deg`` reports the remaining aim error.
    """
    parent = np.asarray(parent_r, dtype=float)
    if parent.shape != (3, 3) or not np.isfinite(parent).all():
        raise ValueError("parent rotation must be a finite (3, 3) matrix")
    want = _unit(np.asarray(direction, dtype=float), "direction")
    local_axis = _unit(np.asarray(axis, dtype=float), "gaze axis")
    table = UPPER_RANGES_DEG if ranges_deg is None else ranges_deg
    lo = np.radians([table[n][0] for n in NECK_COORDINATES])
    hi = np.radians([table[n][1] for n in NECK_COORDINATES])
    start = np.zeros(3) if q0 is None else np.asarray(q0, dtype=float)
    if start.shape != (3,):
        raise ValueError("q0 must be a 3-vector")
    start = np.clip(start, lo, hi)

    def residual(q: Array) -> Array:
        head = parent @ Rotation.from_euler(_EULER, q).as_matrix()
        return np.concatenate([head @ local_axis - want, pose_prior * (q - start)])

    best = None
    for seed in (start, 0.5 * (lo + hi), lo * 0.5, hi * 0.5):
        sol = least_squares(residual, np.clip(seed, lo, hi), bounds=(lo, hi))
        aim = np.linalg.norm(residual(sol.x)[:3])
        if best is None or aim + 1e-9 < best[0]:
            best = (aim, sol.x)
    assert best is not None
    q = np.asarray(best[1], dtype=float)
    head_axis = parent @ Rotation.from_euler(_EULER, q).as_matrix() @ local_axis
    err = float(np.degrees(np.arccos(np.clip(head_axis @ want, -1.0, 1.0))))
    tol = 1e-6
    clamped = (q <= lo + tol) | (q >= hi - tol)
    return NeckIKResult(angles_rad=q, residual_deg=err, clamped=clamped)
