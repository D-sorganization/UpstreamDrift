"""OpenSim inverse kinematics and inverse dynamics of the barbell lifts (OSL-3, #13041).

Builds on the OSL-2 lift models (:mod:`.msk_lift_model`):

* :func:`bundle_segment_scales` -- measured segment scaling: thigh, shank and
  trunk factors from the bundle's landmark distances over the generic
  Rajagopal joint-centre markers (``XHJC``, ``XKJC``, ``XAJC``, ``XSJC``),
  and (OSL-12, #13048) the pelvis scaled mediolaterally by the bundle's hip
  width over the generic one (0.1851 m / 0.1545 m on the OSL-14 lift body,
  #13057), uniformly otherwise.
* :func:`pelvis_landmark_targets` -- virtual ASIS/PSIS targets: the model's
  own ``RASI``/``LASI``/``RPSI``/``LPSI`` markers carried into the bundle by
  its Rajagopal pelvis frame (``bundle_targets.bundle_pelvis_rotation``) and
  hip-centre midpoint, with the same pelvis scaling.  They are tracked only
  when the bundle's own hip flexion lies inside the model's range.  The
  golf-body LIFT-5 squat's did not (159-171 deg against 120 deg); the OSL-14
  lift body (#13057) keeps its hips inside the cited range, so they are
  tracked.
* :func:`run_inverse_kinematics` -- OpenSim's ``InverseKinematicsSolver``
  (the engine of ``InverseKinematicsTool``) tracks the bundle landmarks.
  Coordinate ranges are clamped (OpenSim's IK default) except the arms, whose
  clamping makes the two-hand bar weld infeasible; every coordinate that
  leaves the model's declared range is reported.
* :func:`run_inverse_dynamics` -- OpenSim's ``InverseDynamicsSolver`` with
  the bundle's external loads applied as body forces: the GCV-1 foot wrench
  on each calcaneus and the GCV-7 bar wrench on each hand (the barbell body is
  made massless, so the bar is carried only through that measured wrench).
  Coordinate derivatives come from a quintic GCV spline with zero error
  variance -- what ``InverseDynamicsTool`` does without a low-pass filter.
  Model forces (muscles, actuators, contact) are removed; gravity stays.
  The six pelvis generalized forces are the residuals.
* :func:`static_hold_grf` -- the contact-derived ground reaction of a hold:
  ``pelvis_ty`` and the three pelvis rotations are solved where OpenSim's own
  inverse dynamics (contacts on, every joint at the hold pose, at rest) has
  zero vertical and rotational pelvis residual -- the hold rocks rigidly on
  its contacts into equilibrium; the per-sphere contact forces are then read
  from the ``SmoothSphereHalfSpaceForce`` record values and reduced by GCV-1,
  and every sphere's penetration is checked against the declared limit.

Thresholds are declared here, before any measurement, and never loosened:

* IK marker error: RMS <= 2 cm and max <= 4 cm (OpenSim IK best practice).
* IK coordinate error: the convention-free sagittal segment angles (shank,
  thigh, trunk; :func:`segment_angles`) RMS <= 2 deg, the strict end of
  OpenSim's 2-5 deg guidance.
* Residuals (Hicks et al. 2015, J Biomech Eng 137:020905): max residual force
  <= 5 % of the peak external (ground) force; max residual moment <= 1 % of
  the centre-of-mass height times that force.

OpenSim is imported lazily; callers check availability first.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq, least_squares, root
from scipy.spatial.transform import Rotation

from src.engines.physics_engines.opensim.python.msk_lift_foot import (
    bundle_foot_scale,
)
from src.engines.physics_engines.opensim.python.msk_lift_model import (
    BAR_BODY,
    BENCH_TOP,
    CONTACT_PREFIX,
    GENERIC_STATURE_M,
    PENETRATION_LIMIT_M,
    LiftModelBuild,
)
from src.shared.python.biomechanics.ground_reaction import (
    ContactSet,
    analyze_ground_reaction,
)
from src.shared.python.engine_core.opensim_runtime import import_opensim
from src.shared.python.lifting.bundle_targets import IK_LANDMARKS, LiftTargets
from src.shared.python.lifting.pack_audit.frames import OPENSIM_Y_UP, to_canonical
from src.shared.python.lifting.sagittal_retarget import (
    LANDMARK_LABELS,
    SEGMENTS,
    SagittalFrame,
    segment_angles,
)

Array = NDArray[np.float64]

__all__ = [
    "ARM_TASK_WEIGHT",
    "BAR_WEIGHT",
    "IK_COORDINATE_RMS_THRESHOLD_RAD",
    "IK_MARKER_MAX_THRESHOLD_M",
    "IK_MARKER_RMS_THRESHOLD_M",
    "RESIDUAL_FORCE_FRACTION",
    "RESIDUAL_MOMENT_FRACTION",
    "RESIDUAL_COORDINATES",
    "PELVIS_LANDMARKS",
    "IKResult",
    "IDResult",
    "PelvisLandmarks",
    "bundle_segment_scales",
    "pelvis_landmark_targets",
    "ik_receipt",
    "id_receipt",
    "run_inverse_dynamics",
    "run_inverse_kinematics",
    "static_hold_grf",
]

IK_MARKER_RMS_THRESHOLD_M = 0.02
IK_MARKER_MAX_THRESHOLD_M = 0.04
IK_COORDINATE_RMS_THRESHOLD_RAD = math.radians(2.0)
RESIDUAL_FORCE_FRACTION = 0.05
RESIDUAL_MOMENT_FRACTION = 0.01
RESIDUAL_COORDINATES: tuple[str, ...] = (
    "pelvis_tilt",
    "pelvis_list",
    "pelvis_rotation",
    "pelvis_tx",
    "pelvis_ty",
    "pelvis_tz",
)
#: Marker weights, fixed before measuring (OSL-3): the lower-limb centres and
#: feet are anatomically shared with the bundle (Rajagopal lower limbs); the
#: shoulders and bar sit on a different (golf) upper body, so they weigh less.
LOWER_LIMB_WEIGHT = 10.0
UPPER_BODY_WEIGHT = 1.0
_UPPER_BODY = ("shoulder_l", "shoulder_r", "bar")
#: Bar weight: the same as the shoulders.  OSL-12 first dropped it to 0.1
#: because the hands, held at the OSL-2 back-rack hold behind the shoulders,
#: stayed 0.18-0.28 m from the bundle's bar, which its hands hold in front of
#: and above the shoulders; at weight 1 the bar then dragged the trunk.  With
#: the arms seeded on the bundle's grips (``msk_lift_arm_reach``)
#: the bar is tracked at full weight again (#13048).
BAR_WEIGHT = UPPER_BODY_WEIGHT
_MARKER_PREFIX = "ik_"
#: Landmark -> OpenSim body whose origin it is (feet: calcaneus stations).
_BODY_OF = {
    "ankle_l": "talus_l",
    "ankle_r": "talus_r",
    "knee_l": "tibia_l",
    "knee_r": "tibia_r",
    "hip_l": "femur_l",
    "hip_r": "femur_r",
    "shoulder_l": "humerus_l",
    "shoulder_r": "humerus_r",
    "bar": BAR_BODY,
}
_IK_ACCURACY = 1e-8
#: Virtual pelvis landmark -> the generic model's own pelvis marker.
PELVIS_LANDMARKS: Mapping[str, str] = {
    "asis_l": "LASI",
    "asis_r": "RASI",
    "psis_l": "LPSI",
    "psis_r": "RPSI",
}
#: Pelvis landmark weight when tracked: same as the hip centres it joins.
PELVIS_WEIGHT = LOWER_LIMB_WEIGHT
#: Arm coordinates: held by weak coordinate tasks rather than clamped, since
#: clamping them together with those tasks leaves OpenSim's assembler unable
#: to close the two-hand bar weld (Ipopt failure at the first squat frame).
_UNCLAMPED_PREFIXES = ("arm_", "elbow_", "pro_sup", "wrist_")
#: Arm coordinate-task weight (marker weights are 1 and 10).
ARM_TASK_WEIGHT = 0.01
#: Stated limitation of the landmark set (recorded in every IK receipt).
_PELVIS_OBSERVABILITY = (
    "pelvis orientation is weakly observed in the bundle itself: LIFT-5 "
    "tracked only the hip centres on the pelvis, so the bundle's pelvis tilt "
    "is whatever its own IK left; with coordinate ranges clamped, OpenSim "
    "resolves the hip/pelvis/lumbar trade-off inside the Rajagopal ranges"
)
_GCV_DEGREE = 5
_OPENSIM_SAGITTAL = SagittalFrame(forward=(1.0, 0.0, 0.0), up=(0.0, 1.0, 0.0))
_FOOT_BODY = {"left": "calcn_l", "right": "calcn_r"}
_HAND_BODY = {"L": "hand_l", "R": "hand_r"}
_STATIC_XTOL_M = 1e-12
_STATIC_BRACKET_M = 0.2


def _vec3(v: Any) -> Array:
    return np.array([float(v.get(i)) for i in range(3)], dtype=float)


# --------------------------------------------------------------------------
# Segment scaling
# --------------------------------------------------------------------------


def _generic_marker_points(
    base_model_path: Path, frame: str = "ground"
) -> dict[str, Array]:
    """Joint-centre and pelvis markers of the generic model at its default pose.

    ``frame`` is ``ground`` or a body name the locations are expressed in.
    """
    osim = import_opensim()
    model = osim.Model(str(base_model_path))
    state = model.initSystem()
    markers = model.getMarkerSet()
    names = [f"{s}{j}JC" for s in "RL" for j in "HKAS"] + list(
        PELVIS_LANDMARKS.values()
    )
    into = model.getGround() if frame == "ground" else model.getBodySet().get(frame)
    out = {}
    for name in names:
        marker = markers.get(name)
        out[name] = _vec3(
            marker.getParentFrame().findStationLocationInAnotherFrame(
                state, marker.get_location(), into
            )
        )
    return out


def _hip_widths(targets: LiftTargets, generic: Mapping[str, Array]) -> tuple:
    """(bundle, generic) hip-centre separations, metres (bundle frame 0)."""
    bundle = targets.landmarks(0)
    return (
        float(np.linalg.norm(bundle["hip_r"] - bundle["hip_l"])),
        float(np.linalg.norm(generic["RHJC"] - generic["LHJC"])),
    )


def _pelvis_scale(
    targets: LiftTargets, generic: Mapping[str, Array]
) -> tuple[float, float, float]:
    """Pelvis (x, y, z) factors: stature-uniform, hip width mediolaterally."""
    bundle_width, generic_width = _hip_widths(targets, generic)
    uniform = targets.height_m / GENERIC_STATURE_M
    return (uniform, uniform, bundle_width / generic_width)


def bundle_segment_scales(
    targets: LiftTargets, base_model_path: str | Path
) -> dict[str, float | tuple[float, float, float]]:
    """Per-body scale factors fitting the generic body to the bundle (frame 0).

    Thigh (``femur``, ``patella``): hip-to-knee distance; shank (``tibia``):
    knee-to-ankle distance; foot (``talus``, ``calcn``, ``toes``): the
    bundle's heel-to-forefoot distance over the same two calcaneus stations'
    separation (:func:`.msk_lift_foot.bundle_foot_scale`; OSL-18 #13157: the bundle foot is the
    generic size, 0.150 m, while the shank ratio is 1.10); ``torso``: the
    triple (trunk, trunk, shoulder width) -- hip-centre midpoint to
    shoulder-centre midpoint along x and y, bundle shoulder-centre separation
    over the generic ``RSJC``-``LSJC`` one along the torso ``z`` (OSL-14: the
    lift body's shoulders are 0.406 m apart against the generic 0.340 m);
    ``pelvis``: the triple
    (stature, stature, bundle hip width / generic hip width) -- the bundle's
    hip centres are 0.1851 m apart (OSL-14 lift body) along its pelvis ``z``
    against the generic 0.1545 m, while nothing in the bundle measures pelvis
    depth or height.
    Arms keep the uniform height scale (the bundle's golf arms are not
    Rajagopal's, so their landmarks do not measure the model's segments).

    Postcondition: every factor is positive and finite.
    """
    generic = _generic_marker_points(Path(base_model_path))
    bundle = targets.landmarks(0)
    factors: dict[str, float] = {}
    for side, prefix in (("r", "R"), ("l", "L")):
        thigh = np.linalg.norm(bundle[f"hip_{side}"] - bundle[f"knee_{side}"])
        shank = np.linalg.norm(bundle[f"knee_{side}"] - bundle[f"ankle_{side}"])
        g_thigh = np.linalg.norm(generic[f"{prefix}HJC"] - generic[f"{prefix}KJC"])
        g_shank = np.linalg.norm(generic[f"{prefix}KJC"] - generic[f"{prefix}AJC"])
        for body in (f"femur_{side}", f"patella_{side}"):
            factors[body] = float(thigh / g_thigh)
        factors[f"tibia_{side}"] = float(shank / g_shank)
        foot = bundle_foot_scale(targets, side)
        for body in (f"talus_{side}", f"calcn_{side}", f"toes_{side}"):
            factors[body] = foot
    mid = {
        key: 0.5 * (pts[f"{key}_l"] + pts[f"{key}_r"])
        for key, pts in (("hip", bundle), ("shoulder", bundle))
    }
    g_mid_hip = 0.5 * (generic["RHJC"] + generic["LHJC"])
    g_mid_shoulder = 0.5 * (generic["RSJC"] + generic["LSJC"])
    trunk = float(
        np.linalg.norm(mid["shoulder"] - mid["hip"])
        / np.linalg.norm(g_mid_shoulder - g_mid_hip)
    )
    shoulders = float(
        np.linalg.norm(bundle["shoulder_r"] - bundle["shoulder_l"])
        / np.linalg.norm(generic["RSJC"] - generic["LSJC"])
    )
    pelvis = _pelvis_scale(
        targets, _generic_marker_points(Path(base_model_path), "pelvis")
    )
    scales: dict[str, float | tuple[float, float, float]] = dict(factors)
    scales["torso"] = (trunk, trunk, shoulders)
    scales["pelvis"] = pelvis
    flat = [*factors.values(), trunk, shoulders, *pelvis]
    if not all(math.isfinite(f) and f > 0.0 for f in flat):
        raise ValueError(f"degenerate segment scales {scales}")
    return scales


# --------------------------------------------------------------------------
# Inverse kinematics
# --------------------------------------------------------------------------


#: Tolerance of the pelvis gate: none.  OSL-14 briefly allowed the shared
#: 0.5 deg range-flag tolerance because the diverged LIFT-4 re-track's hip
#: overshot the 120 deg bound by 0.012 deg; after the #13072 controller fix
#: the loaded re-track overshoots by 0.86 deg, outside either way, so the gate
#: is strict again (a bound a source needs to cross is not observed).
PELVIS_GATE_TOLERANCE_RAD = 0.0


def hip_flexion_inside(source: tuple[float, float], low: float, high: float) -> bool:
    """Whether a bundle hip-flexion span ``(min, max)`` lies in ``[low, high]``.

    Within :data:`PELVIS_GATE_TOLERANCE_RAD` (zero) of a bound counts as inside.
    """
    tol = PELVIS_GATE_TOLERANCE_RAD
    return low - tol <= source[0] and source[1] <= high + tol


@dataclass(frozen=True)
class PelvisLandmarks:
    """Virtual ASIS/PSIS targets in the OpenSim lift frame and their provenance.

    ``positions_m`` maps each :data:`PELVIS_LANDMARKS` label to (N, 3) target
    points; ``scale`` is the pelvis (x, y, z) factor triple the model uses;
    ``tracked`` says whether IK follows them, ``reason`` why (not).
    """

    positions_m: Mapping[str, Array]
    scale: tuple[float, float, float]
    hip_width_bundle_m: float
    hip_width_generic_m: float
    source_hip_flexion_rad: tuple[float, float]
    model_hip_flexion_range_rad: tuple[float, float]
    bundle_pelvis_tilt_rad: tuple[float, float]
    tracked: bool
    reason: str

    def record(self) -> dict[str, Any]:
        return {
            "source": "the generic RajagopalLaiUhlrich2023.osim pelvis markers "
            + "/".join(PELVIS_LANDMARKS.values()),
            "mapping": "bundle hip-centre midpoint + bundle Rajagopal pelvis "
            "frame (bundle_targets.bundle_pelvis_rotation) applied to the "
            "generic marker offsets from the generic hip-centre midpoint, "
            "scaled by the pelvis factors",
            "hip_width_bundle_m": self.hip_width_bundle_m,
            "hip_width_generic_m": self.hip_width_generic_m,
            "pelvis_scale_xyz": list(self.scale),
            "source_hip_flexion_rad": list(self.source_hip_flexion_rad),
            "model_hip_flexion_range_rad": list(self.model_hip_flexion_range_rad),
            "bundle_pelvis_tilt_rad": list(self.bundle_pelvis_tilt_rad),
            "tracked": self.tracked,
            "reason": self.reason,
        }


def _span(values: Array) -> tuple[float, float]:
    return float(np.min(values)), float(np.max(values))


def pelvis_landmark_targets(
    build: LiftModelBuild, targets: LiftTargets, base_model_path: str | Path
) -> PelvisLandmarks:
    """Virtual ASIS/PSIS targets mapped from the bundle's pelvis.

    Each generic pelvis marker's offset from the generic hip-centre midpoint
    (pelvis frame) is scaled by the pelvis factors of
    :func:`bundle_segment_scales` and placed with the bundle's Rajagopal
    pelvis frame at the bundle's hip-centre midpoint, so a model scaled with
    those factors whose pelvis coincides with the bundle's carries its own
    markers exactly onto the targets.  They are tracked only when the
    bundle's own hip flexion lies inside the model's ``hip_flexion`` range: a
    source pelvis that needs out-of-range hips is not a physical observation.

    Postcondition: every position array is (N, 3) and finite.
    """
    generic = _generic_marker_points(Path(base_model_path), "pelvis")
    scale = _pelvis_scale(targets, generic)
    bundle_width, generic_width = _hip_widths(targets, generic)
    mid_generic = 0.5 * (generic["RHJC"] + generic["LHJC"])
    hips = [IK_LANDMARKS.index(x) for x in ("hip_l", "hip_r")]
    mid_bundle = targets.positions_m[:, hips].mean(axis=1)
    positions = {
        label: mid_bundle
        + np.einsum(
            "nij,j->ni",
            targets.pelvis_rotation,
            np.asarray(scale) * (generic[marker] - mid_generic),
        )
        for label, marker in PELVIS_LANDMARKS.items()
    }
    if not all(np.isfinite(p).all() for p in positions.values()):
        raise ValueError("non-finite pelvis landmark targets")
    flexion = np.concatenate(
        [targets.source_coordinates[f"hip_flexion_{s}"] for s in "lr"]
    )
    coordinate = build.model.getCoordinateSet().get("hip_flexion_r")
    low, high = float(coordinate.getRangeMin()), float(coordinate.getRangeMax())
    source = _span(flexion)
    tracked = hip_flexion_inside(source, low, high)
    tilt = [Rotation.from_matrix(r).as_euler("ZXY")[0] for r in targets.pelvis_rotation]
    reason = (
        "the bundle's hip flexion lies inside the model range"
        if tracked
        else f"the bundle's own hip flexion spans {math.degrees(source[0]):.1f}"
        f"-{math.degrees(source[1]):.1f} deg, outside the model's "
        f"{math.degrees(low):.0f}-{math.degrees(high):.0f} deg range: its "
        "pelvis tilt was never observed (LIFT-5 tracked only the hip centres), "
        "so its pelvis landmarks are reported, not tracked"
    )
    return PelvisLandmarks(
        positions_m=positions,
        scale=scale,
        hip_width_bundle_m=bundle_width,
        hip_width_generic_m=generic_width,
        source_hip_flexion_rad=source,
        model_hip_flexion_range_rad=(low, high),
        bundle_pelvis_tilt_rad=_span(np.asarray(tilt)),
        tracked=tracked,
        reason=reason,
    )


@dataclass(frozen=True)
class IKResult:
    """IK solution per target state (coordinate-set order, radians/metres).

    ``markers`` are the tracked labels (:data:`IK_LANDMARKS`, then the pelvis
    landmarks when tracked); ``pelvis_points_m`` (N, 4, 3) are the model's
    own pelvis markers in :data:`PELVIS_LANDMARKS` order, tracked or not.
    """

    coordinates: tuple[str, ...]
    q: Array
    markers: tuple[str, ...]
    marker_error_m: Array
    model_points_m: Array
    range_violations: Mapping[str, dict[str, float | int]]
    pelvis_points_m: Array


def _station(build: LiftModelBuild, label: str, targets: LiftTargets) -> tuple:
    """(body, local station) of a landmark on the scaled OpenSim model."""
    if label in _BODY_OF:
        return _BODY_OF[label], (0.0, 0.0, 0.0)
    body, station = targets.foot_stations_m[label]
    factors = build.receipt["segment_scale_factors"]
    scale = float(factors.get(body, build.receipt["scale_factor"]))
    return body, tuple(scale * x for x in station)


def _add_markers(
    build: LiftModelBuild, targets: LiftTargets, pelvis: PelvisLandmarks
) -> tuple[str, ...]:
    """Add one ``ik_<label>`` marker per tracked label; return the labels."""
    osim = import_opensim()
    model = build.model
    for label in IK_LANDMARKS:
        body, station = _station(build, label, targets)
        frame = model.getBodySet().get(body)
        model.addMarker(osim.Marker(_MARKER_PREFIX + label, frame, osim.Vec3(*station)))
    labels = IK_LANDMARKS
    if pelvis.tracked:
        markers = model.getMarkerSet()
        frame = model.getBodySet().get("pelvis")
        for label, marker in PELVIS_LANDMARKS.items():
            location = markers.get(marker).get_location()
            model.addMarker(osim.Marker(_MARKER_PREFIX + label, frame, location))
        labels = IK_LANDMARKS + tuple(PELVIS_LANDMARKS)
    model.finalizeConnections()
    return labels


def _marker_weight(label: str) -> float:
    if label in PELVIS_LANDMARKS:
        return PELVIS_WEIGHT
    if label == "bar":
        return BAR_WEIGHT
    return UPPER_BODY_WEIGHT if label in _UPPER_BODY else LOWER_LIMB_WEIGHT


def _markers_reference(
    targets: LiftTargets,
    pelvis: PelvisLandmarks,
    labels: tuple[str, ...],
    bar_weight: float,
) -> Any:
    osim = import_opensim()
    names = osim.StdVectorString()
    for label in labels:
        names.append(_MARKER_PREFIX + label)
    points = {x: targets.positions_m[:, IK_LANDMARKS.index(x)] for x in IK_LANDMARKS}
    points |= dict(pelvis.positions_m)
    table = osim.TimeSeriesTableVec3()
    table.setColumnLabels(names)
    for k, t in enumerate(targets.time_s):
        vecs = [osim.Vec3(*map(float, points[x][k])) for x in labels]
        table.appendRow(float(t), osim.RowVectorVec3(vecs))
    table.addTableMetaDataString("Units", "m")
    weights = osim.SetMarkerWeights()
    for label in labels:
        weight = bar_weight if label == "bar" else _marker_weight(label)
        weights.cloneAndAppend(osim.MarkerWeight(_MARKER_PREFIX + label, weight))
    return osim.MarkersReference(table, weights)


def _range_violations(model: Any, names: tuple[str, ...], q: Array) -> dict:
    out: dict[str, dict[str, float | int]] = {}
    coords = model.getCoordinateSet()
    for j, name in enumerate(names):
        c = coords.get(name)
        low, high = float(c.getRangeMin()), float(c.getRangeMax())
        outside = (q[:, j] < low - 1e-9) | (q[:, j] > high + 1e-9)
        if outside.any():
            out[name] = {
                "frames": int(outside.sum()),
                "range_min_rad": low,
                "range_max_rad": high,
                "min_rad": float(q[:, j].min()),
                "max_rad": float(q[:, j].max()),
            }
    return out


def _arm_coordinate_tasks(model: Any) -> Any:
    """Weak coordinate tasks holding every arm coordinate at its hold value.

    Only the shoulders and the bar observe the arms, so elbow, forearm and
    wrist angles form a null space the solver otherwise wanders through
    (OSL-12 measured elbow hyperextension to -146 deg).  The task weight
    :data:`ARM_TASK_WEIGHT` is small against the marker weights.
    """
    osim = import_opensim()
    tasks = osim.SimTKArrayCoordinateReference()
    for coordinate in model.getCoordinateSet():
        name = coordinate.getName()
        if name.startswith(_UNCLAMPED_PREFIXES):
            task = osim.CoordinateReference(
                name, osim.Constant(coordinate.getDefaultValue())
            )
            task.setWeight(ARM_TASK_WEIGHT)
            tasks.push_back(task)
    return tasks


def run_inverse_kinematics(
    build: LiftModelBuild,
    targets: LiftTargets,
    pelvis: PelvisLandmarks,
    bar_weight: float = BAR_WEIGHT,
) -> IKResult:
    """Track the bundle landmarks with OpenSim's ``InverseKinematicsSolver``.

    Mutates ``build.model``: adds one ``ik_<label>`` marker per tracked
    landmark and clamps every coordinate to its range except the arms
    (``_UNCLAMPED_PREFIXES``), which weak coordinate tasks hold near the
    hold pose instead (:func:`_arm_coordinate_tasks`).  ``bar_weight``
    overrides :data:`BAR_WEIGHT` (the OSL-12 seed pass uses 0 so the bar
    does not pull a body it cannot yet reach).

    Raises:
        ValueError: when ``bar_weight`` is negative or not finite.

    Postconditions: ``q`` is (N, coordinates); ``marker_error_m`` (N, M) and
    ``model_points_m`` (N, M, 3) follow ``markers``.
    """
    if not (math.isfinite(bar_weight) and bar_weight >= 0.0):
        raise ValueError(f"bar_weight must be finite and >= 0, got {bar_weight}")
    osim = import_opensim()
    model = build.model
    labels = _add_markers(build, targets, pelvis)
    for coordinate in model.updCoordinateSet():
        coordinate.set_clamped(not coordinate.getName().startswith(_UNCLAMPED_PREFIXES))
    state = model.initSystem()
    solver = osim.InverseKinematicsSolver(
        model,
        _markers_reference(targets, pelvis, labels, bar_weight),
        _arm_coordinate_tasks(model),
        math.inf,
    )
    solver.setAccuracy(_IK_ACCURACY)
    names = tuple(c.getName() for c in model.getCoordinateSet())
    markers = model.getMarkerSet()
    q_rows, errors, points, pelvis_points = [], [], [], []
    for k, t in enumerate(targets.time_s):
        state.setTime(float(t))
        if k == 0:
            solver.assemble(state)
        else:
            solver.track(state)
        coords = model.getCoordinateSet()
        q_rows.append([coords.get(n).getValue(state) for n in names])
        errors.append(
            [solver.computeCurrentMarkerError(_MARKER_PREFIX + x) for x in labels]
        )
        model.realizePosition(state)
        points.append(
            [
                _vec3(markers.get(_MARKER_PREFIX + x).getLocationInGround(state))
                for x in labels
            ]
        )
        pelvis_points.append(
            [
                _vec3(markers.get(m).getLocationInGround(state))
                for m in PELVIS_LANDMARKS.values()
            ]
        )
    q = np.asarray(q_rows, dtype=float)
    return IKResult(
        coordinates=names,
        q=q,
        markers=labels,
        marker_error_m=np.asarray(errors, dtype=float),
        model_points_m=np.asarray(points, dtype=float),
        range_violations=_range_violations(model, names, q),
        pelvis_points_m=np.asarray(pelvis_points, dtype=float),
    )


def _segment_angles(points: Array) -> Array:
    by_label = dict(zip(IK_LANDMARKS, points[: len(IK_LANDMARKS)], strict=True))
    return segment_angles({x: by_label[x] for x in LANDMARK_LABELS}, _OPENSIM_SAGITTAL)


def _pelvis_receipt(pelvis: PelvisLandmarks, ik: IKResult) -> dict[str, Any]:
    """Pelvis landmark provenance and the model's deviation from the targets."""
    deviation = np.stack(
        [
            np.linalg.norm(ik.pelvis_points_m[:, j] - pelvis.positions_m[label], axis=1)
            for j, label in enumerate(PELVIS_LANDMARKS)
        ],
        axis=1,
    )
    column = ik.coordinates.index("pelvis_tilt")
    return {
        **pelvis.record(),
        "model_pelvis_tilt_rad": _span(ik.q[:, column]),
        "model_deviation_rms_m": dict(
            zip(
                PELVIS_LANDMARKS,
                np.sqrt(np.mean(deviation**2, axis=0)).tolist(),
                strict=True,
            )
        ),
        "model_deviation_max_m": dict(
            zip(PELVIS_LANDMARKS, deviation.max(axis=0).tolist(), strict=True)
        ),
    }


def ik_receipt(
    targets: LiftTargets, ik: IKResult, pelvis: PelvisLandmarks
) -> dict[str, Any]:
    """Marker and convention-free coordinate errors against the bundle."""
    err = ik.marker_error_m
    angle_error = np.array(
        [
            _segment_angles(model) - _segment_angles(bundle)
            for model, bundle in zip(
                ik.model_points_m, targets.positions_m, strict=True
            )
        ]
    )
    rms = float(np.sqrt(np.mean(err**2)))
    peak = float(err.max())
    return {
        "solver": "OpenSim InverseKinematicsSolver (engine of InverseKinematicsTool)",
        "accuracy": _IK_ACCURACY,
        "markers": list(ik.markers),
        "weights": {x: _marker_weight(x) for x in ik.markers},
        "frames": int(err.shape[0]),
        "marker_error": {
            "rms_m": rms,
            "max_m": peak,
            "per_marker_rms_m": dict(
                zip(ik.markers, np.sqrt(np.mean(err**2, axis=0)).tolist(), strict=True)
            ),
            "per_marker_max_m": dict(
                zip(ik.markers, err.max(axis=0).tolist(), strict=True)
            ),
            "threshold_rms_m": IK_MARKER_RMS_THRESHOLD_M,
            "threshold_max_m": IK_MARKER_MAX_THRESHOLD_M,
            "within_best_practice": bool(
                rms <= IK_MARKER_RMS_THRESHOLD_M and peak <= IK_MARKER_MAX_THRESHOLD_M
            ),
        },
        "segment_angle_error": {
            "definition": "sagittal shank, thigh and trunk angles from the "
            "landmarks (sagittal_retarget.segment_angles), model minus bundle",
            "rms_rad": float(np.sqrt(np.mean(angle_error**2))),
            "max_abs_rad": float(np.abs(angle_error).max()),
            "per_segment_rms_rad": dict(
                zip(
                    SEGMENTS,
                    np.sqrt(np.mean(angle_error**2, axis=0)).tolist(),
                    strict=True,
                )
            ),
            "threshold_rms_rad": IK_COORDINATE_RMS_THRESHOLD_RAD,
        },
        "clamping": "on (OpenSim IK default) except the arm coordinates "
        + "/".join(_UNCLAMPED_PREFIXES)
        + f", which coordinate tasks (weight {ARM_TASK_WEIGHT}) hold at the "
        "hold pose; coordinates outside the model range are listed",
        "range_violations": dict(ik.range_violations),
        "observability": _PELVIS_OBSERVABILITY,
        "pelvis_landmarks": _pelvis_receipt(pelvis, ik),
    }


# --------------------------------------------------------------------------
# Inverse dynamics
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class IDResult:
    """Generalized forces per target state (coordinate-set order)."""

    coordinates: tuple[str, ...]
    tau: Array
    com_height_m: float


def _bare_model(model: Any, keep_contact: bool) -> Any:
    """Unlocked copy of ``model`` without force elements (contacts optional).

    Inverse dynamics ignores constraints, so a locked coordinate only blocks
    setting its value; it is unlocked and keeps its default value.
    """
    osim = import_opensim()
    bare = osim.Model(model)
    for coordinate in bare.updCoordinateSet():
        coordinate.setDefaultLocked(False)
    forces = bare.updForceSet()
    for i in reversed(range(forces.getSize())):
        is_contact = forces.get(i).getName().startswith(CONTACT_PREFIX)
        if not (keep_contact and is_contact):
            forces.remove(i)
    return bare


def _u_index(model: Any, state: Any) -> dict[str, int]:
    """Mobility (``u``) index of each coordinate."""
    osim = import_opensim()
    index = {}
    for coordinate in model.getCoordinateSet():
        probe = osim.State(state)
        for other in model.getCoordinateSet():
            other.setSpeedValue(probe, 0.0)
        coordinate.setSpeedValue(probe, 1.0)
        u = probe.getU()
        hits = [i for i in range(u.size()) if u.get(i) == 1.0]
        if len(hits) != 1:
            raise RuntimeError(f"coordinate {coordinate.getName()} has no unique u")
        index[coordinate.getName()] = hits[0]
    return index


def _splines(ik: IKResult, time_s: Array) -> Any:
    osim = import_opensim()
    labels = osim.StdVectorString()
    for name in ik.coordinates:
        labels.append(name)
    table = osim.TimeSeriesTable()
    table.setColumnLabels(labels)
    for t, row in zip(time_s, ik.q, strict=True):
        table.appendRow(float(t), osim.RowVector(list(map(float, row))))
    return osim.GCVSplineSet(table, osim.StdVectorString(), _GCV_DEGREE, 0.0)


def _derivatives(splines: Any, n: int, t: float) -> tuple[Array, Array, Array]:
    osim = import_opensim()
    x = osim.Vector(1, float(t))
    first, second = osim.StdVectorInt(), osim.StdVectorInt()
    first.append(0)
    second.append(0)
    second.append(0)
    values = np.empty((3, n))
    for j in range(n):
        f = splines.get(j)
        values[0, j] = f.calcValue(x)
        values[1, j] = f.calcDerivative(first, x)
        values[2, j] = f.calcDerivative(second, x)
    return values[0], values[1], values[2]


def _body_wrench(
    model: Any,
    state: Any,
    body: str,
    force: Array,
    moment_about_point: Array,
    point: Array,
) -> tuple[int, Any]:
    """Spatial force on ``body`` (ground axes, moment about the body origin)."""
    osim = import_opensim()
    frame = model.getBodySet().get(body)
    origin = _vec3(frame.getPositionInGround(state))
    moment = moment_about_point + np.cross(point - origin, force)
    return int(frame.getMobilizedBodyIndex()), osim.SpatialVec(
        osim.Vec3(*map(float, moment)), osim.Vec3(*map(float, force))
    )


def _external_body_forces(model: Any, state: Any, targets: LiftTargets, k: int) -> Any:
    osim = import_opensim()
    count = model.getBodySet().getSize() + 1  # plus ground
    forces = osim.VectorOfSpatialVec(count, osim.SpatialVec(osim.Vec3(0), osim.Vec3(0)))
    zero: Array = np.zeros(3)
    wrenches = [
        (
            _FOOT_BODY[foot],
            targets.grf_force_n[foot][k],
            targets.grf_moment_nm[foot][k],
            zero,
        )
        for foot in _FOOT_BODY
    ]
    for side, body in _HAND_BODY.items():
        hand = model.getBodySet().get(body)
        grip = _vec3(hand.findStationLocationInGround(state, hand.getMassCenter()))
        wrenches.append(
            (body, targets.hand_force_n[side][k], targets.hand_torque_nm[side][k], grip)
        )
    for body, force, moment, point in wrenches:
        index, spatial = _body_wrench(model, state, body, force, moment, point)
        forces.set(index, spatial)
    return forces


def _massless_bar(model: Any) -> None:
    osim = import_opensim()
    bar = model.updBodySet().get(BAR_BODY)
    bar.setMass(0.0)
    bar.setInertia(osim.Inertia(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))


def run_inverse_dynamics(
    build: LiftModelBuild, targets: LiftTargets, ik: IKResult
) -> IDResult:
    """OpenSim inverse dynamics of the IK motion with the bundle's loads.

    ``tau = M qdd + C - G - J^T F_ext``: the model part comes from
    ``InverseDynamicsSolver.solve(state, udot)`` (gravity is the only model
    force left) and the external part from the solver's applied-body-force
    overload, which is linear in the body forces.

    Postcondition: ``tau`` is (N, coordinates) in ``ik.coordinates`` order.
    """
    osim = import_opensim()
    model = _bare_model(build.model, keep_contact=False)
    _massless_bar(model)
    state = model.initSystem()
    solver = osim.InverseDynamicsSolver(model)
    u_index = _u_index(model, state)
    nu = state.getNU()
    splines = _splines(ik, targets.time_s)
    coords = model.getCoordinateSet()
    names = ik.coordinates
    zero_u = osim.Vector(nu, 0.0)
    no_forces = osim.VectorOfSpatialVec(
        model.getBodySet().getSize() + 1, osim.SpatialVec(osim.Vec3(0), osim.Vec3(0))
    )
    rows, com = [], []
    for k, t in enumerate(targets.time_s):
        q, u, udot_values = _derivatives(splines, len(names), float(t))
        for j, name in enumerate(names):
            coords.get(name).setValue(state, float(q[j]), False)
            coords.get(name).setSpeedValue(state, float(u[j]))
        model.realizeVelocity(state)
        udot = osim.Vector(nu, 0.0)
        for j, name in enumerate(names):
            udot.set(u_index[name], float(udot_values[j]))
        inertial = solver.solve(state, udot)
        loaded = solver.solve(
            state, zero_u, zero_u, _external_body_forces(model, state, targets, k)
        )
        bias = solver.solve(state, zero_u, zero_u, no_forces)
        tau = [
            inertial.get(u_index[n]) + loaded.get(u_index[n]) - bias.get(u_index[n])
            for n in names
        ]
        rows.append(tau)
        com.append(float(_vec3(model.calcMassCenterPosition(state))[1]))
    return IDResult(
        coordinates=names,
        tau=np.asarray(rows, dtype=float),
        com_height_m=float(np.mean(com)),
    )


def id_receipt(targets: LiftTargets, inverse: IDResult) -> dict[str, Any]:
    """Residuals (with the Hicks 2015 gate) and joint-moment summary."""
    column = {n: j for j, n in enumerate(inverse.coordinates)}
    residuals: dict[str, dict[str, Any]] = {}
    peaks: dict[str, float] = {}
    for name in RESIDUAL_COORDINATES:
        series = inverse.tau[:, column[name]]
        peaks[name] = float(np.abs(series).max())
        residuals[name] = {
            "unit": "N" if name.endswith(("_tx", "_ty", "_tz")) else "N m",
            "max_abs": float(np.abs(series).max()),
            "rms": float(np.sqrt(np.mean(series**2))),
        }
    grf = targets.grf_force_n["left"] + targets.grf_force_n["right"]
    peak_force = float(np.linalg.norm(grf, axis=1).max())
    max_force = RESIDUAL_FORCE_FRACTION * peak_force
    max_moment = RESIDUAL_MOMENT_FRACTION * inverse.com_height_m * peak_force
    forces = [peaks[n] for n in RESIDUAL_COORDINATES[3:]]
    moments = [peaks[n] for n in RESIDUAL_COORDINATES[:3]]
    joints = [n for n in inverse.coordinates if n not in RESIDUAL_COORDINATES]
    return {
        "solver": "OpenSim InverseDynamicsSolver; quintic GCV spline derivatives, "
        "no low-pass filter (InverseDynamicsTool default)",
        "external_loads": {
            "feet": "GCV-1 foot wrench of the bundle's shared-law contact "
            "(kinetics_parity.bundle_kinetics), applied to calcn_l/calcn_r",
            "hands": "GCV-7 per-hand bushing wrench of the bundle bar "
            "(BushingSample.grip_analysis), negated onto hand_l/hand_r at the "
            "model's grip point; the barbell body is massless in ID",
            "model_forces": "removed (muscles, actuators, contact); gravity kept",
        },
        "residuals": residuals,
        "residual_gate": {
            "reference": "Hicks et al. 2015, J Biomech Eng 137:020905",
            "peak_external_force_n": peak_force,
            "com_height_m": inverse.com_height_m,
            "max_force_n": max_force,
            "max_moment_nm": max_moment,
            "forces_within": bool(max(forces) <= max_force),
            "moments_within": bool(max(moments) <= max_moment),
        },
        "joint_moments_peak_abs": {
            n: float(np.abs(inverse.tau[:, column[n]]).max()) for n in joints
        },
        "all_finite": bool(np.isfinite(inverse.tau).all()),
    }


# --------------------------------------------------------------------------
# Static hold
# --------------------------------------------------------------------------


def _contact_spheres(model: Any) -> dict[str, Any]:
    osim = import_opensim()
    spheres = {}
    for geometry in model.getContactGeometrySet():
        sphere = osim.ContactSphere.safeDownCast(geometry)
        if sphere is not None:
            spheres[CONTACT_PREFIX + sphere.getName()] = sphere
    return spheres


def _support_of(sphere: Any) -> str:
    body = sphere.getFrame().findBaseFrame().getName()
    return {"calcn_l": "left", "calcn_r": "right"}.get(body, "bench")


def _contact_sets(model: Any, state: Any) -> dict[str, tuple[list, list]]:
    """Per support: canonical (z-up) sphere forces and lowest sphere points."""
    model.realizeDynamics(state)
    spheres = _contact_spheres(model)
    sets: dict[str, tuple[list, list]] = {}
    for force in model.getForceSet():
        if force.getName() not in spheres:
            continue
        sphere = spheres[force.getName()]
        record = force.getRecordValues(state)
        on_sphere = np.array([record.get(i) for i in range(3)])
        centre = _vec3(
            sphere.getFrame().findStationLocationInGround(state, sphere.get_location())
        )
        lowest = centre - np.array([0.0, sphere.getRadius(), 0.0])
        forces, points = sets.setdefault(_support_of(sphere), ([], []))
        forces.append(to_canonical(OPENSIM_Y_UP, on_sphere.tolist()))
        points.append(to_canonical(OPENSIM_Y_UP, lowest.tolist()))
    return sets


def _penetrations(model: Any, state: Any) -> dict[str, float]:
    """Depth (m, positive = into the surface) of every contact sphere."""
    osim = import_opensim()
    heights = {}
    for geometry in model.getContactGeometrySet():
        plane = osim.ContactHalfSpace.safeDownCast(geometry)
        if plane is not None:
            heights[plane.getName()] = float(plane.get_location().get(1))
    model.realizePosition(state)
    depths = {}
    for name, sphere in _contact_spheres(model).items():
        surface = BENCH_TOP if _support_of(sphere) == "bench" else "floor"
        centre = _vec3(
            sphere.getFrame().findStationLocationInGround(state, sphere.get_location())
        )
        depths[name] = heights[surface] - (centre[1] - float(sphere.getRadius()))
    return depths


#: Pelvis coordinates solved for the hold equilibrium (height, then rotations).
_HOLD_UNKNOWNS = ("pelvis_ty", "pelvis_tilt", "pelvis_list", "pelvis_rotation")
#: Coordinates moved by the standing ankle lean (sign applied to the lean).
_LEAN_COORDINATES = {"ankle_angle_l": 1.0, "ankle_angle_r": 1.0, "pelvis_tilt": -1.0}
#: Equilibrium solve tolerance on the residual vector (N and N m).
_HOLD_RESIDUAL_TOL = 1e-6
#: Largest settling rotation accepted as the same hold (rad, 15 deg).
_SETTLE_BOUND_RAD = math.radians(15.0)


def _equilibrium(
    model: Any, state: Any, solver: Any, u_index: Mapping, *, lean: bool
) -> Array:
    """Settle the hold on its contacts until its pelvis residual vanishes.

    The unknowns are the pelvis height, a fore-aft unknown and the pelvis
    list and rotation.  For a standing hold (``lean``) the fore-aft unknown
    is an extra ankle lean (both ankles ``+d``, pelvis tilt ``-d``), so the
    feet stay flat while the body above them moves; a supine hold rocks
    rigidly on its pelvis tilt instead.  The height is bracketed first
    (Brent, as OSL-3); then all four are solved with Powell's hybrid method,
    which keeps a symmetric hold symmetric (the frictionless-at-rest
    contacts leave pelvis yaw nearly neutral).  A root whose rotations
    leave ``_SETTLE_BOUND_RAD`` is a different posture, so the solve then
    falls back to least squares with the rotations bounded.

    Returns ``[height, fore-aft, list, rotation]`` as offsets from the
    starting values (the height absolute).

    Raises:
        ValueError: when no height in the bracket balances the weight or the
            joint solve leaves a residual above ``_HOLD_RESIDUAL_TOL``.
    """
    osim = import_opensim()
    zero = osim.Vector(state.getNU(), 0.0)
    coords = model.getCoordinateSet()
    fore_aft = _LEAN_COORDINATES if lean else {"pelvis_tilt": 1.0}
    base = {
        n: float(coords.get(n).getValue(state))
        for n in (*_HOLD_UNKNOWNS, *_LEAN_COORDINATES)
    }

    def residuals(x: Array) -> Array:
        coords.get("pelvis_ty").setValue(state, float(x[0]), False)
        for name, sign in fore_aft.items():
            coords.get(name).setValue(state, base[name] + sign * float(x[1]), False)
        for name, value in zip(_HOLD_UNKNOWNS[2:], x[2:], strict=True):
            coords.get(name).setValue(state, base[name] + float(value), False)
        tau = solver.solve(state, zero)
        return np.array([tau.get(u_index[n]) for n in _HOLD_UNKNOWNS])

    def vertical(height: float) -> float:
        return float(residuals(np.array([height, 0.0, 0.0, 0.0]))[0])

    top = base["pelvis_ty"]
    low = top - _STATIC_BRACKET_M
    if vertical(top) * vertical(low) > 0.0:
        raise ValueError("no pelvis height in the bracket balances the hold")
    height = float(brentq(vertical, low, top, xtol=_STATIC_XTOL_M))
    guess = np.array([height, 0.0, 0.0, 0.0])
    bound = np.r_[_STATIC_BRACKET_M, np.full(3, _SETTLE_BOUND_RAD)]
    x = root(residuals, guess, method="hybr", options={"xtol": 1e-14}).x
    if np.any(np.abs(x - guess) > bound):
        x = least_squares(
            residuals,
            guess,
            bounds=(guess - bound, guess + bound),
            xtol=1e-15,
            ftol=1e-15,
            gtol=1e-15,
        ).x
    final = residuals(x)
    if not np.all(np.abs(final) <= _HOLD_RESIDUAL_TOL):
        raise ValueError(f"the hold does not reach equilibrium: residual {final}")
    if np.any(np.abs(x[1:]) > _SETTLE_BOUND_RAD):
        raise ValueError(f"the hold settles {np.degrees(x[1:])} deg from its pose")
    return np.asarray(x, dtype=float)


def static_hold_grf(build: LiftModelBuild) -> dict[str, Any]:
    """Contact-derived ground reaction of the lift's hold pose at equilibrium.

    Postconditions: ``total_grf_n`` (canonical z-up) is the sum of every
    contact force; ``relative_error`` compares its vertical component with
    ``(skeleton mass + bar mass) * g`` from the build receipt; the pelvis
    residuals left at the equilibrium are ``balance_residual`` (zero within
    ``_HOLD_RESIDUAL_TOL``); ``settling_rad`` is how far the hold moved to
    get there (``fore_aft`` is the extra ankle lean of a standing hold, the
    pelvis-tilt rocking of a supine one); ``penetration`` lists every
    sphere's depth against :data:`PENETRATION_LIMIT_M`.

    Raises:
        ValueError: when the hold has no equilibrium on its contacts.
    """
    osim = import_opensim()
    model = _bare_model(build.model, keep_contact=True)
    state = model.initSystem()
    solver = osim.InverseDynamicsSolver(model)
    u_index = _u_index(model, state)
    balance = build.receipt["hold_pose"].get("balance")
    solution = _equilibrium(model, state, solver, u_index, lean=balance is not None)
    tau = solver.solve(state, osim.Vector(state.getNU(), 0.0))
    gravity = float(np.linalg.norm(_vec3(model.getGravity())))
    breakdowns = {}
    for support, (forces, points) in _contact_sets(model, state).items():
        height = 0.0 if support != "bench" else float(build.receipt["bench_height_m"])
        contact = ContactSet(np.array(forces), np.array(points))
        breakdowns[support] = analyze_ground_reaction(
            {support: contact}, np.zeros(3), ground_height_m=height
        ).net
    total = np.sum([r.force_n for r in breakdowns.values()], axis=0)
    expected = (
        build.receipt["skeleton_mass_kg"] + build.receipt["bar_mass_kg"]
    ) * gravity
    depths = _penetrations(model, state)
    deepest = max(depths.values())
    com = to_canonical(
        OPENSIM_Y_UP, _vec3(model.calcMassCenterPosition(state)).tolist()
    )
    return {
        "method": "pelvis height, a fore-aft unknown (standing: extra ankle "
        "lean with the feet flat; supine: pelvis-tilt rocking) and the pelvis "
        "list and rotation solved where OpenSim inverse dynamics (contacts on, "
        "at rest) has zero vertical and rotational pelvis residual; contact "
        "forces from SmoothSphereHalfSpaceForce record values, reduced by GCV-1",
        "pelvis_ty_m": float(solution[0]),
        "settling_rad": {
            "fore_aft": float(solution[1]),
            "pelvis_list": float(solution[2]),
            "pelvis_rotation": float(solution[3]),
        },
        "gravity_m_s2": gravity,
        "expected_weight_n": float(expected),
        "total_grf_n": total.tolist(),
        "relative_error": float((total[2] - expected) / expected),
        "support_vertical_n": {k: float(r.force_n[2]) for k, r in breakdowns.items()},
        "support_cop_m": {
            k: None if r.cop_m is None else r.cop_m.tolist()
            for k, r in breakdowns.items()
        },
        "com_m": com.tolist(),
        "penetration": {
            "per_sphere_m": depths,
            "max_m": deepest,
            "limit_m": PENETRATION_LIMIT_M,
            "within_limit": bool(deepest <= PENETRATION_LIMIT_M),
        },
        "balance": balance,
        "balance_residual": {
            n: float(tau.get(u_index[n]))
            for n in RESIDUAL_COORDINATES
            if n != "pelvis_ty"
        },
    }
