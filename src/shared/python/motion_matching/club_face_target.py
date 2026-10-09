"""Club-face orientation targets for the shared marker IK (OSV-10, #11759).

The clubhead marker triad is only about 12 cm across, so a 2-3 cm marker
error is a 10-20 degree error in the club's roll about the shaft. Marker
distances alone therefore leave the face free to drift open through the
release while the marker RMS barely changes. This module turns the capture
head triad into an explicit face-orientation residual:

* ``observe_frame_rotations`` fits, per capture frame, the rotation of the
  marker frame (the spec ``Clubhead`` frame the triad offsets are expressed
  in) that best maps the calibrated triad offsets onto the captured triad
  (Kabsch; no translation enters the rotation).
* ``face_axis_in_frame`` is the rendered face normal (the rolled spec
  ``Clubface Vector`` of :mod:`model_appearance.club_assembly`) expressed in
  that marker frame.
* ``face_axis_targets`` combines the two into the per-frame ``axis_targets``
  understood by every ``BaseFullBodyIK`` provider: the residual is
  ``sqrt(w) * (R_model(q) a - R_capture a)`` with ``a`` the face axis, a
  dimensionless chord (about the angle in radians for small errors).

Everything is engine independent: the targets feed the shared
``solve_trajectory``, so MuJoCo, Drake and Pinocchio fits, and every engine
that replays the fitted trajectory, inherit the same face.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from src.shared.python.body_part_viz.fitters._kabsch import kabsch_rotation
from src.shared.python.motion_matching.tour_capture_contract import MARKER_SEGMENTS

Array = NDArray[np.float64]
AxisTarget = tuple[tuple[float, float, float], tuple[float, float, float], float]

#: Spec frame the clubhead markers are attached to.
FACE_FRAME = "Clubhead"
#: Clubhead marker triad (the three head markers of the tour capture contract).
HEAD_TRIAD_LABELS: tuple[str, ...] = MARKER_SEGMENTS["club"][:3]
#: Weight of the face-orientation residual relative to unit-weight marker
#: residuals in metres (cost ``w |R a - n|^2``). A 5 degree face error (chord
#: 0.087) then costs as much as an 8.7 cm error on each of the three head
#: markers. Sweep on the reference stage (driver / 7-iron, OSV-10, #11759):
#: w = 0, 0.3, 1, 3, 10 gave face-separation RMS 8.6/7.9, 3.0, 1.7, 0.8/0.6,
#: 0.3/0.2 deg and full-capture IK marker RMS 32.8/31.7, 32.5, 33.7, 33.6/31.6,
#: 33.6/31.5 mm; w = 10 doubled the driver closure error (8.5 mm), so 3.
FACE_ORIENTATION_WEIGHT = 3.0
#: Minimum spread (m) of the triad offsets; a degenerate triad has no roll.
MIN_TRIAD_SPREAD_M = 0.01


def _finite_vector(value: Sequence[float] | Array, name: str) -> Array:
    vec = np.asarray(value, dtype=float)
    if vec.shape != (3,) or not np.isfinite(vec).all():
        raise ValueError(f"{name} must be a finite 3-vector")
    return vec


def validate_face_weight(weight: float) -> float:
    """Return ``weight`` as a float; raise for a negative or non-finite value."""
    if isinstance(weight, bool) or not isinstance(weight, (int, float)):
        raise TypeError("face weight must be a real number")
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError(f"face weight must be finite and >= 0, got {weight}")
    return float(weight)


def _frame_placement(spec: Mapping[str, Any], frame: str) -> tuple[str, Array]:
    for entry in spec.get("frames", ()):
        if entry.get("name") == frame:
            placement = np.asarray(entry["placement"], dtype=float)
            if placement.shape != (4, 4) or not np.isfinite(placement).all():
                raise ValueError(f"frame {frame} needs a finite 4x4 placement")
            return str(entry["body"]), placement
    raise ValueError(f"spec has no frame named {frame}")


def face_axis_in_frame(spec: Mapping[str, Any], frame: str = FACE_FRAME) -> Array:
    """Unit rendered face normal expressed in the spec ``frame``.

    The face normal is the assembly ``Clubface Vector`` (club-body frame,
    including loft and the shared address roll). Raises ``ValueError`` when
    the spec has no club or ``frame`` is not on the club body.
    Postcondition: a unit 3-vector.
    """
    from src.shared.python.model_appearance import club_assembly as ca

    assembly = ca.assembly_from_spec(spec)
    club_body = ca.club_body_name(spec)
    if assembly is None or club_body is None:
        raise ValueError("spec has no club body")
    body, placement = _frame_placement(spec, frame)
    if body != club_body:
        raise ValueError(f"frame {frame} is on {body}, not on the club body")
    axis = placement[:3, :3].T @ ca.clubface_vector(assembly)
    return np.asarray(axis / np.linalg.norm(axis), dtype=float)


def face_centre_in_frame(spec: Mapping[str, Any], frame: str = FACE_FRAME) -> Array:
    """Rendered face-centre point expressed in the spec ``frame`` (metres)."""
    from src.shared.python.model_appearance import club_assembly as ca

    assembly = ca.assembly_from_spec(spec)
    if assembly is None:
        raise ValueError("spec has no club body")
    _, placement = _frame_placement(spec, frame)
    face_axis_in_frame(spec, frame)  # same club-body checks
    centre = ca.clubface_centre(assembly) - placement[:3, 3]
    return np.asarray(placement[:3, :3].T @ centre, dtype=float)


def observe_frame_rotations(
    points: Array,
    valid: NDArray[np.bool_],
    labels: Sequence[str],
    offsets: Mapping[str, Sequence[float] | Array],
) -> Array:
    """Per-frame world rotation of the marker frame seen by the capture triad.

    ``points`` is ``(frames, markers, 3)`` in the native world, ``valid`` the
    matching ``(frames, markers)`` mask and ``offsets`` the triad offsets in
    the marker frame. Frames where any triad marker is missing or non-finite
    are NaN. Raises ``ValueError`` for mismatched shapes, unknown labels,
    fewer than three markers or a degenerate (collinear) triad.
    """
    pts = np.asarray(points, dtype=float)
    mask = np.asarray(valid, dtype=bool)
    names = list(labels)
    if pts.ndim != 3 or pts.shape[2] != 3 or mask.shape != pts.shape[:2]:
        raise ValueError("points must be (frames, markers, 3) with a matching mask")
    if len(names) != pts.shape[1]:
        raise ValueError("labels must name every marker column")
    if len(offsets) < 3:
        raise ValueError("at least three triad markers are required")
    missing = [label for label in offsets if label not in names]
    if missing:
        raise ValueError(f"triad markers not in the capture: {missing}")
    local = np.array([_finite_vector(v, f"offset {k}") for k, v in offsets.items()])
    centred = local - local.mean(axis=0)
    if np.linalg.svd(centred, compute_uv=False)[1] < MIN_TRIAD_SPREAD_M:
        raise ValueError("triad offsets are degenerate (collinear or coincident)")
    cols = [names.index(label) for label in offsets]
    triad = pts[:, cols, :]
    seen = mask[:, cols].all(axis=1) & np.isfinite(triad).all(axis=(1, 2))
    out = np.full((len(pts), 3, 3), np.nan)
    for k in np.flatnonzero(seen):
        world = triad[k] - triad[k].mean(axis=0)
        out[k] = kabsch_rotation(centred, world)
    return out


def observe_capture_face(
    points: Array,
    valid: NDArray[np.bool_],
    labels: Sequence[str],
    attachments: Mapping[str, tuple[str, Sequence[float]]],
    spec: Mapping[str, Any],
    frame: str = FACE_FRAME,
) -> tuple[Array, Array]:
    """World face normal and face-centre point the capture triad implies.

    Returns ``(normals, centres)``, both ``(frames, 3)``, NaN where the triad
    is not observed. The rigid pose of ``frame`` comes from the calibrated
    triad offsets (Kabsch rotation, centroid translation); the face axis and
    centre are the rendered ones (:func:`face_axis_in_frame`,
    :func:`face_centre_in_frame`). Raises ``ValueError`` when the triad is not
    attached to ``frame``.
    """
    offsets = triad_offsets(attachments, frame)
    if len(offsets) < len(HEAD_TRIAD_LABELS):
        raise ValueError(f"the head triad is not attached to {frame}")
    rotations = observe_frame_rotations(points, valid, labels, offsets)
    names = list(labels)
    cols = [names.index(label) for label in offsets]
    local_mean = np.mean(list(offsets.values()), axis=0)
    world_mean = np.asarray(points, dtype=float)[:, cols, :].mean(axis=1)
    axis = face_axis_in_frame(spec, frame)
    centre = face_centre_in_frame(spec, frame) - local_mean
    normals = np.einsum("nij,j->ni", rotations, axis)
    centres = np.einsum("nij,j->ni", rotations, centre) + world_mean
    centres[~np.isfinite(normals).all(axis=1)] = np.nan
    return normals, centres


def triad_offsets(
    attachments: Mapping[str, tuple[str, Sequence[float]]],
    frame: str = FACE_FRAME,
    triad: Sequence[str] = HEAD_TRIAD_LABELS,
) -> dict[str, Array]:
    """Offsets of the triad markers attached to ``frame`` (empty when absent)."""
    out: dict[str, Array] = {}
    for label in triad:
        if label in attachments and attachments[label][0] == frame:
            out[label] = _finite_vector(attachments[label][1], f"offset {label}")
    return out


def face_axis_targets(
    points: Array,
    valid: NDArray[np.bool_],
    labels: Sequence[str],
    attachments: Mapping[str, tuple[str, Sequence[float]]],
    spec: Mapping[str, Any],
    *,
    weight: float = FACE_ORIENTATION_WEIGHT,
    frame: str = FACE_FRAME,
) -> list[dict[str, AxisTarget] | None]:
    """Per-frame face ``axis_targets`` for ``BaseFullBodyIK.solve_trajectory``.

    Each observed frame gets ``{frame: (a, R_capture a, weight)}`` where ``a``
    is :func:`face_axis_in_frame`; unobserved frames get ``None``. A zero
    weight, or a capture/spec without the full head triad on ``frame``,
    returns ``None`` for every frame (the legacy marker-only fit).
    """
    w = validate_face_weight(weight)
    frames = int(np.asarray(points).shape[0])
    offsets = triad_offsets(attachments, frame)
    if w == 0.0 or len(offsets) < len(HEAD_TRIAD_LABELS):
        return [None] * frames
    axis = face_axis_in_frame(spec, frame)
    rotations = observe_frame_rotations(points, valid, labels, offsets)
    a = (float(axis[0]), float(axis[1]), float(axis[2]))
    out: list[dict[str, AxisTarget] | None] = []
    for rot in rotations:
        if not np.isfinite(rot).all():
            out.append(None)
            continue
        d = rot @ axis
        out.append({frame: (a, (float(d[0]), float(d[1]), float(d[2])), w)})
    return out


def merge_axis_targets(
    *per_frame: Sequence[Mapping[str, Any] | None] | None,
) -> list[dict[str, Any] | None] | None:
    """Frame-wise union of several per-frame axis-target lists.

    ``None`` lists are skipped; the lists must have equal length. A frame
    whose union is empty is ``None``. Raises ``ValueError`` when two lists
    target the same frame name on the same capture frame.
    """
    lists = [list(item) for item in per_frame if item is not None]
    if not lists:
        return None
    if len({len(item) for item in lists}) != 1:
        raise ValueError("axis-target lists must have one entry per capture frame")
    merged: list[dict[str, Any] | None] = []
    for entries in zip(*lists, strict=True):
        frame: dict[str, Any] = {}
        for entry in entries:
            for name, target in (entry or {}).items():
                if name in frame:
                    raise ValueError(f"two axis targets for frame {name}")
                frame[name] = target
        merged.append(frame or None)
    return merged


def face_separation_deg(model_axis: Array, capture_axis: Array) -> Array:
    """Per-frame angle (degrees) between model and capture world face normals.

    Both arrays are ``(frames, 3)``; NaN rows give NaN.
    """
    m = np.asarray(model_axis, dtype=float)
    c = np.asarray(capture_axis, dtype=float)
    if m.shape != c.shape or m.ndim != 2 or m.shape[1] != 3:
        raise ValueError("model and capture axes must both be (frames, 3)")
    cos = np.einsum("ij,ij->i", m, c) / (
        np.linalg.norm(m, axis=1) * np.linalg.norm(c, axis=1)
    )
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def model_face_normals(
    kin: Any, q: Array, spec: Mapping[str, Any], frame: str = FACE_FRAME
) -> Array:
    """World rendered face normal per row of ``q`` from the IK provider's FK.

    ``kin`` must expose ``body_poses(q, [frame])`` (MuJoCo and Drake marker
    kinematics do). Raises ``ValueError`` for a non-2-D ``q``.
    """
    rows = np.asarray(q, dtype=float)
    if rows.ndim != 2:
        raise ValueError("q must be (frames, coordinates)")
    axis = face_axis_in_frame(spec, frame)
    return np.array([kin.body_poses(row, [frame])[frame][0] @ axis for row in rows])


def model_face_centres(
    kin: Any, q: Array, spec: Mapping[str, Any], frame: str = FACE_FRAME
) -> Array:
    """World rendered face centre per row of ``q`` from the IK provider's FK.

    Same contract as :func:`model_face_normals` (GCV-20 clubhead speed).
    """
    rows = np.asarray(q, dtype=float)
    if rows.ndim != 2:
        raise ValueError("q must be (frames, coordinates)")
    centre = face_centre_in_frame(spec, frame)
    out = []
    for row in rows:
        rot, pos = kin.body_poses(row, [frame])[frame]
        out.append(rot @ centre + pos)
    return np.array(out)


def face_fit_summary(model_normals: Array, capture_normals: Array) -> dict[str, Any]:
    """Separation statistics (degrees) over the frames the capture observes.

    Unobserved frames are reported as a count, never as a zero error.
    """
    sep = face_separation_deg(model_normals, capture_normals)
    seen = np.isfinite(sep)
    if not seen.any():
        return {"frames_observed": 0, "reason": "head triad never observed"}
    return {
        "frames_observed": int(seen.sum()),
        "frames_unobserved": int((~seen).sum()),
        "rms_deg": float(np.sqrt(np.mean(sep[seen] ** 2))),
        "p95_deg": float(np.percentile(sep[seen], 95)),
        "max_deg": float(sep[seen].max()),
    }


def fill_unobserved(
    time: Sequence[float] | np.ndarray, centres: np.ndarray
) -> np.ndarray:
    """Copy of ``centres`` with unobserved (NaN) rows interpolated in time.

    Capture head-triad gaps would otherwise break the impact search. Raises
    ``ValueError`` for mismatched shapes or fewer than two observed rows.
    """
    t = np.asarray(time, dtype=float)
    c = np.asarray(centres, dtype=float)
    if c.ndim != 2 or c.shape[1] != 3 or t.shape != (len(c),):
        raise ValueError("time must be (n,) and centres (n, 3)")
    seen = np.isfinite(c).all(axis=1)
    if seen.sum() < 2:
        raise ValueError("need at least two observed face centres to interpolate")
    return np.column_stack([np.interp(t, t[seen], c[seen, j]) for j in range(3)])


class CaptureImpact(NamedTuple):
    """Capture ball passage on the capture clock (GCV-20, #11767).

    ``frame_index`` is the last pre-contact frame (``times[frame_index] <=
    time_s <= times[frame_index + 1]``); named to avoid shadowing
    ``tuple.index``. ``ball_centre_m`` is the shared ball
    (:func:`model_appearance.ball.ball_position_at_address`) at the address
    face, centred at the address face-centre height, or ``None`` when no
    face normal was observed.
    """

    time_s: float
    frame_index: int
    ball_centre_m: Array | None


def capture_impact(
    times: Sequence[float] | Array,
    points: Array,
    valid: NDArray[np.bool_],
    labels: Sequence[str],
    attachments: Mapping[str, tuple[str, Sequence[float]]],
    spec: Mapping[str, Any],
) -> CaptureImpact:
    """Sub-sample capture impact and the ball it strikes.

    The capture face centre (head triad through the calibrated
    ``attachments``, gaps interpolated) passes the ball at the sub-sample
    :func:`model_appearance.club_face.ball_passage`. Raises ``ValueError``
    when ``times`` does not match the frames, the triad cannot be observed,
    or the face centre never returns to the ball.
    """
    from src.shared.python.model_appearance.ball import (
        BALL_RADIUS_M,
        ball_position_at_address,
    )
    from src.shared.python.model_appearance.club_face import ball_passage

    normals, centres = observe_capture_face(points, valid, labels, attachments, spec)
    t = np.asarray(times, dtype=float)
    if t.shape != (len(centres),):
        raise ValueError("times must hold one entry per capture frame")
    filled = fill_unobserved(t, centres)
    t_impact, k, _ = ball_passage(t, filled)
    seen = np.isfinite(normals).all(axis=1) & (np.linalg.norm(normals, axis=1) > 0)
    ball = None
    if seen.any():
        normal = normals[int(np.argmax(seen))]
        ball = ball_position_at_address(
            filled[0], normal, ground_height_m=float(filled[0, 2]) - BALL_RADIUS_M
        )
    return CaptureImpact(t_impact, int(k), ball)


def capture_impact_index(
    times: Sequence[float] | Array,
    points: Array,
    valid: NDArray[np.bool_],
    labels: Sequence[str],
    attachments: Mapping[str, tuple[str, Sequence[float]]],
    spec: Mapping[str, Any],
) -> int:
    """Last pre-contact capture frame (:func:`capture_impact` ``.frame_index``)."""
    return capture_impact(times, points, valid, labels, attachments, spec).frame_index
