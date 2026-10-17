"""Bench press on a bench contact body (LIFT-5, #11745).

The pulls and the squat stand on the foot spheres.  A bench presser lies on
his back, and the same-input spec's root is unactuated, so without a support
under the trunk the body falls.  This module adds that support and poses the
lifter on it:

* **Bench contact body.** Four back spheres mirror OSL-2's OpenSim bench
  (``msk_lift_model._BACK_SPHERES``, #13040): one at each posterior shoulder
  and posterior iliac site, radius 0.04 m at the generic 1.70 m stature,
  scaled with the lifter, touching the skin site from the anterior side.  On
  this body those sites are the spec's ``BackLeft``/``BackRight`` (upper
  trunk) and ``WaistLBack``/``WaistRBack`` (pelvis) markers.  The bench top
  is a raised contact support (``contact.supports``, shared law,
  :func:`~src.shared.python.motion_matching.contact_law.support_planes`) at
  the lowest back-sphere point of the posed hold, as OSL-2 places its
  ``bench_top`` half-space; only the back spheres press on it.
* **Supine hold.** The standing pose's trunk is turned onto its back so the
  four back spheres lie level; the legs take the standard's bench start
  (hip extension and knee flexion of the ``bench_press`` via-pose) with the
  feet flat on the floor.  The bench height follows from that pose.
* **Press.** The Movement Optimizer's bench model is an arm chain on a fixed
  shoulder: its bar offset from the shoulder (along the body and vertical)
  is scaled by the model's straight-up reach over the optimizer's lockout
  reach and applied to the model's shoulder midpoint.  Everything but the
  arm chain is held at the hold.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from src.shared.python.lifting.back_bar import HandBarSetup
from src.shared.python.lifting.bundles import (
    ARM_CHAIN_COORDINATES,
    BACK_BAR_CLOSURE_TOLERANCE_M,
    BACK_BAR_REACH_TOLERANCE_M,
    HAND_BAR_LIMITATIONS,
    LANDMARK_ATTACHMENTS,
    STANCE_SPHERES,
    TOE_COORDINATES,
    FullBodyLift,
    LiftIKOptions,
    _arm_hang,
    _attachments,
    _axis_targets_for,
    _engine_kinematics,
    _stand,
    _Standing,
    _track,
    _TrackSetup,
)
from src.shared.python.lifting.lift_spec import lift_via_pose
from src.shared.python.lifting.sagittal_retarget import (
    LANDMARK_LABELS,
    SagittalFrame,
    SagittalSolution,
    rotate_sagittal,
)

Array = NDArray[np.float64]

BENCH_LIFT = "bench_press"
#: Name of the raised contact support the back spheres press on.
BENCH_SUPPORT = "bench_top"
#: OSL-2's back-sphere radius (``msk_lift_model._BACK_SPHERES``) and the
#: generic stature it is given at (``msk_lift_model.GENERIC_STATURE_M``).
BACK_SPHERE_RADIUS_M = 0.04
GENERIC_STATURE_M = 1.70
#: Back sphere name and the spec marker at its skin site (OSL-2's RPSH,
#: LPSH, RPSI and LPSI).
BACK_SPHERES: tuple[tuple[str, str], ...] = (
    ("back_RPSH", "BackRight"),
    ("back_LPSH", "BackLeft"),
    ("back_RPSI", "WaistRBack"),
    ("back_LPSI", "WaistLBack"),
)
#: Root rotation (``Rx`` then ``Rz`` of the spec's ``Rx Ry Rz`` root) that
#: lies the upright body on its back, head toward ``+y`` and face up.  Lying
#: along ``x`` instead needs ``HipInputY = -90 deg``, the gimbal lock of the
#: root's Euler chain, where its ``Rx`` and ``Rz`` axes coincide.
_SUPINE_ROOT: tuple[tuple[str, float], ...] = (
    ("HipInputX", -np.pi / 2),
    ("HipInputZ", -np.pi / 2),
)
#: Turn about the vertical from the standing pose's sagittal plane (feet
#: toward ``+x``) to the bench's (feet toward ``-y``).
_BENCH_YAW_RAD = -np.pi / 2
#: Marker weight of the back-sphere centres in the hold fit: the spheres
#: must lie level on the bench even where the trunk landmarks give way.
_SPHERE_WEIGHT = 10.0
BENCH_LIMITATIONS = (
    "Bench press (#11745): the lifter lies on a bench contact body, four back "
    "spheres (OSL-2 geometry: posterior shoulder and posterior iliac sites) "
    "pressing on a raised support at the hold's lowest back-sphere point; the "
    "feet stand on the floor. Only the arm chain moves: the trunk, legs, "
    "scapulae and head are held at the hold. The bar centre is tracked to the "
    "model's shoulder midpoint plus the optimizer's arm-chain bar offset, "
    "scaled by the model's straight-up reach; the optimizer's chest position "
    "is its own (the bar is not stopped by the chest, which has no contact). "
    "The bar mass enters only the planar optimizer, not the full-body "
    "dynamics (LIFT-4, #11744). "
)


def _placement(frame: Mapping[str, Any]) -> tuple[Array, Array]:
    matrix = np.asarray(frame["placement"], dtype=float)
    return matrix[:3, :3], matrix[:3, 3]


def back_sphere_documents(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The four back spheres of a full-body document, in its body frames.

    Each centre is its marker's station moved one radius along the marker
    frame's anterior ``+x`` (the sphere touches the skin site from inside);
    the radius is :data:`BACK_SPHERE_RADIUS_M` scaled by the subject's
    stature over :data:`GENERIC_STATURE_M`.

    Raises:
        ValueError: When the document lacks a back marker, its frame or a
            positive subject stature.
    """
    stature = float(document.get("subject", {}).get("stature_m", float("nan")))
    if not (np.isfinite(stature) and stature > 0.0):
        raise ValueError("document needs a positive subject stature_m")
    radius = BACK_SPHERE_RADIUS_M * stature / GENERIC_STATURE_M
    frames = {frame["name"]: frame for frame in document["frames"]}
    spheres = []
    for name, marker in BACK_SPHERES:
        attachment = document["marker_attachments"].get(marker)
        if attachment is None or attachment["body"] not in frames:
            raise ValueError(f"document has no back marker {marker} on a frame")
        frame = frames[attachment["body"]]
        rotation, origin = _placement(frame)
        station = rotation @ np.asarray(attachment["offset_m"], dtype=float) + origin
        centre = station + radius * rotation[:, 0]
        spheres.append(
            {
                "name": name,
                "body": frame["body"],
                "position_m": [float(x) for x in centre],
                "radius_m": radius,
            }
        )
    return spheres


def bench_spec(spec_bytes: bytes, height_m: float) -> bytes:
    """``spec_bytes`` with the back spheres and a bench top at ``height_m``.

    Postconditions: the foot spheres and everything outside ``contact`` are
    unchanged; only the back spheres name the ``bench_top`` support.

    Raises:
        TypeError: If *spec_bytes* is not bytes.
        ValueError: For a non-finite height or a spec that already has back
            spheres.
    """
    if not isinstance(spec_bytes, bytes | bytearray):
        raise TypeError("spec_bytes must be the raw full-body spec bytes")
    if not np.isfinite(height_m):
        raise ValueError(f"bench height must be finite: {height_m!r}")
    document = json.loads(bytes(spec_bytes))
    spheres = back_sphere_documents(document)
    contact = dict(document["contact"])
    names = {sphere["name"] for sphere in contact["spheres"]}
    if names & {sphere["name"] for sphere in spheres}:
        raise ValueError("spec already carries the back spheres")
    contact["spheres"] = list(contact["spheres"]) + spheres
    contact["supports"] = [
        {
            "name": BENCH_SUPPORT,
            "height_m": float(height_m),
            "spheres": [sphere["name"] for sphere in spheres],
            "provenance": "OSL-2 (#13040) bench_top geometry; #11745",
        }
    ]
    document["contact"] = contact
    return json.dumps(document).encode("utf-8")


@dataclass(frozen=True)
class BenchHold:
    """The supine hold on the bench and its bench (provenance in ``record``)."""

    standing: _Standing
    q: Array
    points: Mapping[str, Array]
    height_m: float
    record: Mapping[str, Any]


def _sphere_attachments(document: Mapping[str, Any]) -> dict[str, Any]:
    return {
        sphere["name"]: (sphere["body"], tuple(sphere["position_m"]))
        for sphere in back_sphere_documents(document)
    }


def _supine_trunk(
    standing: _Standing, centres: Mapping[str, Array]
) -> tuple[float, Array]:
    """Rotation (rad) laying the back spheres level, and its pivot.

    The line from the pelvis spheres' midpoint to the shoulder spheres'
    midpoint turns horizontal, the head toward ``-forward``.
    """
    pelvis = 0.5 * (centres["back_RPSI"] + centres["back_LPSI"])
    shoulders = 0.5 * (centres["back_RPSH"] + centres["back_LPSH"])
    line = shoulders - pelvis
    forward, up = np.asarray(standing.frame.forward), np.asarray(standing.frame.up)
    angle = float(np.arctan2(line @ up, line @ forward)) - np.pi
    return angle, pelvis


def _leg_targets(
    standing: _Standing, hips: Mapping[str, Array], frame: SagittalFrame
) -> dict[str, Array]:
    """Knees and ankles of the standard's bench start below the supine hips.

    The thigh leaves the hip toward the feet (``+forward``) extended
    downward by the standard's hip extension; the shank folds by its knee
    flexion.  The model's own segment lengths and lateral offsets are kept.
    """
    via = lift_via_pose(BENCH_LIFT, "start").coordinates
    points = standing.points
    targets: dict[str, Array] = {}
    for side in "lr":
        extension = -float(via[f"hip_{side}_flex"])
        flexion = -float(via[f"knee_{side}_flex"])
        thigh_dir = rotate_sagittal(np.asarray(frame.forward), extension, frame)
        shank_dir = rotate_sagittal(thigh_dir, flexion, frame)
        thigh = float(np.linalg.norm(points[f"knee_{side}"] - points[f"hip_{side}"]))
        shank = float(np.linalg.norm(points[f"ankle_{side}"] - points[f"knee_{side}"]))
        targets[f"knee_{side}"] = hips[f"hip_{side}"] + thigh * thigh_dir
        targets[f"ankle_{side}"] = targets[f"knee_{side}"] + shank * shank_dir
    return targets


def _hold_targets(
    standing: _Standing, centres: Mapping[str, Array]
) -> dict[str, Array]:
    """Supine targets: trunk landmarks, back-sphere centres, legs (world)."""
    angle, pivot = _supine_trunk(standing, centres)
    frame = standing.frame
    trunk = {
        label: pivot + rotate_sagittal(point - pivot, angle, frame)
        for label, point in (
            *((k, standing.points[k]) for k in ("hip_l", "hip_r")),
            *((k, standing.points[k]) for k in ("shoulder_l", "shoulder_r")),
            *centres.items(),
        )
    }
    targets = trunk | _leg_targets(standing, trunk, frame)
    up = np.asarray(frame.up)
    ankle_height = np.mean([standing.points[f"ankle_{s}"] @ up for s in "lr"])
    lift = ankle_height - np.mean([targets[f"ankle_{s}"] @ up for s in "lr"])
    yaw = Rotation.from_rotvec(_BENCH_YAW_RAD * up).as_matrix()
    return {
        label: pivot + yaw @ (point - pivot) + lift * up
        for label, point in targets.items()
    }


def _supine_seed(standing: _Standing) -> Array:
    """The standing pose lying on its back with the standard's leg angles."""
    via = lift_via_pose(BENCH_LIFT, "start").coordinates
    q = standing.q.copy()
    order = standing.order
    for name, value in _SUPINE_ROOT:
        q[order.index(name)] = value
    for side in "lr":
        q[order.index(f"hip_flexion_{side}")] = float(via[f"hip_{side}_flex"])
        q[order.index(f"knee_angle_{side}")] = float(via[f"knee_{side}_flex"])
    return q


def solve_bench_hold(
    spec_bytes: bytes, seed_q: Array, *, options: LiftIKOptions | None = None
) -> BenchHold:
    """The supine hold on the bench and the bench height under it.

    Solves the standing pose (:func:`~.bundles.lift_to_full_body`'s), turns
    its trunk onto the back spheres and its legs to the standard's bench
    start, and fits the full body to those targets with the feet flat on the
    floor; the bench top is the lowest back-sphere point of the fit.

    Postcondition: ``height_m`` equals the lowest back-sphere point at ``q``;
    ``record["sphere_clearance_m"]`` gives each back sphere's gap above it.
    """
    standing = _stand(spec_bytes, seed_q, options)
    document = json.loads(bytes(spec_bytes))
    spheres = _sphere_attachments(document)
    kin = _engine_kinematics(spec_bytes, {**LANDMARK_ATTACHMENTS, **spheres})
    labels = (*LANDMARK_LABELS, *spheres)
    at_standing = dict(zip(labels, kin.marker_positions(standing.q), strict=True))
    targets = _hold_targets(standing, {k: at_standing[k] for k in spheres})
    rows = np.array([targets.get(k, np.zeros(3)) for k in labels])
    fit = kin.solve_pose(
        rows,
        np.array([k in targets for k in labels], dtype=bool),
        _supine_seed(standing),
        locked=standing.locked,
        flat_feet=STANCE_SPHERES,
        iterations=standing.options.standing_iterations,
        ground=standing.ground,
        prior_weight=standing.options.prior_weight,
        posture_prior=standing.posture,
        marker_weights=dict.fromkeys(spheres, _SPHERE_WEIGHT),
        **standing.limits(),
    )
    q = np.asarray(fit.q, dtype=float)
    reached = dict(zip(labels, kin.marker_positions(q), strict=True))
    up = np.asarray(standing.frame.up)
    radius = float(back_sphere_documents(document)[0]["radius_m"])
    bottoms = {k: float(reached[k] @ up) - radius for k in spheres}
    height = min(bottoms.values())
    record = {
        "support": BENCH_SUPPORT,
        "height_m": height,
        "sphere_radius_m": radius,
        "spheres": {k: {"marker": m} for k, m in BACK_SPHERES},
        "sphere_clearance_m": {k: v - height for k, v in bottoms.items()},
        "fit_residual_m": {k: float(fit.per_marker_m[k]) for k in targets},
        "geometry_source": "OSL-2 msk_lift_model (#13040): bench_top at the "
        "lowest back-sphere point of the supine hold",
        "leg_pose_source": "biomech parity standard bench_press start via-pose",
    }
    return BenchHold(
        standing=standing,
        q=q,
        points={k: reached[k] for k in LANDMARK_LABELS},
        height_m=height,
        record=record,
    )


def bench_frame(hold: BenchHold) -> SagittalFrame:
    """Press plane: ``forward`` along the body toward the feet, ``up`` vertical."""
    points = hold.points
    hips = 0.5 * (points["hip_l"] + points["hip_r"])
    shoulders = 0.5 * (points["shoulder_l"] + points["shoulder_r"])
    up = np.asarray(hold.standing.frame.up)
    toward_feet = hips - shoulders
    toward_feet -= (toward_feet @ up) * up
    toward_feet /= np.linalg.norm(toward_feet)
    fx, fy, fz = (float(x) for x in toward_feet)
    ux, uy, uz = (float(x) for x in up)
    return SagittalFrame(forward=(fx, fy, fz), up=(ux, uy, uz))


def bench_bar_targets(
    solution: SagittalSolution,
    shoulder_mid: Array,
    frame: SagittalFrame,
    scale: float,
) -> Array:
    """Bar targets (N, 3): the optimizer's arm-chain bar offset, scaled.

    The optimizer's bench bar is the arm tip relative to its fixed shoulder
    (``forward`` along the body toward the feet, ``up`` vertical).

    Postcondition: row ``k`` is ``shoulder_mid + scale * bar_m[k]`` in the
    press plane.
    """
    if not (np.isfinite(scale) and scale > 0.0):
        raise ValueError(f"scale must be positive and finite: {scale!r}")
    mid = np.asarray(shoulder_mid, dtype=float)
    forward, up = np.asarray(frame.forward), np.asarray(frame.up)
    bar = scale * solution.bar_m
    return np.asarray(mid + bar[:, :1] * forward + bar[:, 1:] * up, dtype=float)


@dataclass(frozen=True)
class BenchLift:
    """The bench press full-body lift and the bench it lies on."""

    lift: FullBodyLift
    bench: Mapping[str, Any]

    @property
    def limitations(self) -> str:
        """Limitations recorded with the bundle (hand-held bar plus bench)."""
        return BENCH_LIMITATIONS + HAND_BAR_LIMITATIONS


def _press_setup(
    spec_bytes: bytes, hold: BenchHold
) -> tuple[HandBarSetup, Any, dict[str, Any]]:
    """Bench-and-grip spec, its kinematics and the bar axis target."""
    setup = HandBarSetup.from_standard(bytes(spec_bytes), BENCH_LIFT)
    setup = replace(setup, spec_bytes=bench_spec(setup.spec_bytes, hold.height_m))
    kin = _engine_kinematics(setup.spec_bytes, _attachments(setup))
    axes = _axis_targets_for(setup.bar_axis, bench_frame(hold))
    return setup, kin, axes


def lift_bench_to_full_body(
    spec_bytes: bytes,
    solution: SagittalSolution,
    seed_q: Array,
    *,
    options: LiftIKOptions | None = None,
) -> BenchLift:
    """Full-body bench press on the bench contact body (#11745).

    Postconditions: ``lift.back_bar.spec_bytes`` is the derived spec (grip
    width and bench); every coordinate outside the arm chain is constant.

    Raises:
        ValueError: For a solution without a positive lockout bar offset.
        RuntimeError: If the arms cannot hold the bar at the first sample.
    """
    if not isinstance(spec_bytes, bytes | bytearray):
        raise TypeError("spec_bytes must be the raw full-body spec bytes")
    lockout = float(np.linalg.norm(solution.bar_m[0]))
    if not (np.isfinite(lockout) and lockout > 0.0):
        raise ValueError("solution needs a positive bar offset at its first sample")
    hold = solve_bench_hold(bytes(spec_bytes), seed_q, options=options)
    setup, kin, axes = _press_setup(bytes(spec_bytes), hold)
    frame = bench_frame(hold)
    standing = replace(hold.standing, q=hold.q, frame=frame, points=dict(hold.points))
    up = np.asarray(frame.up)
    reach = _arm_hang(kin, standing, hold.q, axes, direction=up)
    setup = replace(setup, arm_hang_m=reach, bar_offset_scale=reach / lockout)
    held = {
        n: float(hold.q[standing.order.index(n)])
        for n in standing.order
        if n not in ARM_CHAIN_COORDINATES and n not in TOE_COORDINATES
    }
    shoulders = 0.5 * (hold.points["shoulder_l"] + hold.points["shoulder_r"])
    targets = np.repeat(
        np.array([hold.points[k] for k in LANDMARK_LABELS])[None],
        solution.time_s.size,
        axis=0,
    )
    bar_index = LANDMARK_LABELS.index("bar")
    targets[:, bar_index] = bench_bar_targets(
        solution, shoulders, frame, setup.bar_offset_scale
    )
    first = _first_hold(kin, standing, targets[0], held, axes)
    lift = _track(
        standing,
        solution,
        _TrackSetup(
            kin=kin,
            points=hold.points,
            q_start=first,
            held=held,
            axes=axes,
            back_bar=setup,
            targets=targets,
        ),
    )
    return BenchLift(lift=lift, bench=dict(hold.record))


def _first_hold(
    kin: Any,
    standing: _Standing,
    targets: Array,
    held: Mapping[str, float],
    axes: Mapping[str, Any],
) -> Array:
    """Arm chain holding the bar at the first target, the grip closed.

    Raises:
        RuntimeError: If the bar centre or the grip closure misses its
            tolerance.
    """
    order = standing.order
    toes = {n: float(standing.q[order.index(n)]) for n in TOE_COORDINATES}
    fit = kin.solve_pose(
        targets,
        np.ones(len(LANDMARK_LABELS), dtype=bool),
        standing.q,
        locked=dict(held) | toes,
        flat_feet=STANCE_SPHERES,
        iterations=standing.options.standing_iterations,
        ground=standing.ground,
        prior_weight=standing.options.prior_weight,
        **standing.limits(),
        axis_targets=axes,
    )
    reach = fit.per_marker_m["bar"]
    if reach > BACK_BAR_REACH_TOLERANCE_M or (
        fit.closure_error_m > BACK_BAR_CLOSURE_TOLERANCE_M
    ):
        raise RuntimeError(
            f"arms cannot hold the bench bar at the start: bar {reach:.4f} m "
            f"off, grip closure {fit.closure_error_m:.2e} m"
        )
    return np.asarray(fit.q, dtype=float)
