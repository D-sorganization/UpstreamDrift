"""OpenSim inverse kinematics and inverse dynamics of the barbell lifts (OSL-3, #13041).

Builds on the OSL-2 lift models (:mod:`.msk_lift_model`):

* :func:`bundle_segment_scales` -- measured segment scaling: thigh, shank and
  trunk factors from the bundle's landmark distances over the generic
  Rajagopal joint-centre markers (``XHJC``, ``XKJC``, ``XAJC``, ``XSJC``).
* :func:`run_inverse_kinematics` -- OpenSim's ``InverseKinematicsSolver``
  (the engine of ``InverseKinematicsTool``) tracks the bundle landmarks.
  Coordinate range clamping is switched off so the solver follows the bundle;
  every coordinate that leaves the model's declared range is reported.
* :func:`run_inverse_dynamics` -- OpenSim's ``InverseDynamicsSolver`` with
  the bundle's external loads applied as body forces: the GCV-1 foot wrench
  on each calcaneus and the GCV-7 bar wrench on each hand (the barbell body is
  made massless, so the bar is carried only through that measured wrench).
  Coordinate derivatives come from a quintic GCV spline with zero error
  variance -- what ``InverseDynamicsTool`` does without a low-pass filter.
  Model forces (muscles, actuators, contact) are removed; gravity stays.
  The six pelvis generalized forces are the residuals.
* :func:`static_hold_grf` -- the contact-derived ground reaction of a hold:
  ``pelvis_ty`` is solved where OpenSim's own inverse dynamics (contacts on,
  every other coordinate at the hold pose, at rest) has zero vertical
  residual; the per-sphere contact forces are then read from the
  ``SmoothSphereHalfSpaceForce`` record values and reduced by GCV-1.

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
from scipy.optimize import brentq

from src.engines.physics_engines.opensim.python.msk_lift_model import (
    BAR_BODY,
    CONTACT_PREFIX,
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
    "IK_COORDINATE_RMS_THRESHOLD_RAD",
    "IK_MARKER_MAX_THRESHOLD_M",
    "IK_MARKER_RMS_THRESHOLD_M",
    "RESIDUAL_FORCE_FRACTION",
    "RESIDUAL_MOMENT_FRACTION",
    "RESIDUAL_COORDINATES",
    "IKResult",
    "IDResult",
    "bundle_segment_scales",
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
#: Marker weights, fixed before measuring: the lower-limb joint centres and
#: feet are anatomically shared with the bundle (Rajagopal lower limbs); the
#: shoulders and bar sit on a different (golf) upper body, so they weigh less.
LOWER_LIMB_WEIGHT = 10.0
UPPER_BODY_WEIGHT = 1.0
_UPPER_BODY = ("shoulder_l", "shoulder_r", "bar")
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
#: Stated limitation of the landmark set (recorded in every IK receipt).
_PELVIS_OBSERVABILITY = (
    "pelvis orientation is weakly observed: the only pelvis-borne landmarks "
    "are the two hip centres, so pelvis tilt trades off against hip flexion "
    "and lumbar extension; the bundle spec's pelvis frame does not map onto "
    "the Rajagopal pelvis markers by a verified transform, so no virtual "
    "pelvis markers are added"
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


def _generic_marker_points(base_model_path: Path) -> dict[str, Array]:
    osim = import_opensim()
    model = osim.Model(str(base_model_path))
    state = model.initSystem()
    markers = model.getMarkerSet()
    names = [f"{s}{j}JC" for s in "RL" for j in "HKAS"]
    return {n: _vec3(markers.get(n).getLocationInGround(state)) for n in names}


def bundle_segment_scales(
    targets: LiftTargets, base_model_path: str | Path
) -> dict[str, float]:
    """Per-body scale factors fitting the generic body to the bundle (frame 0).

    Thigh (``femur``, ``patella``): hip-to-knee distance; shank (``tibia``,
    ``talus``, ``calcn``, ``toes``): knee-to-ankle distance; ``torso``:
    hip-centre midpoint to shoulder-centre midpoint.  Pelvis and arms keep the
    uniform height scale (the bundle's golf pelvis and arms are not
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
        for body in (f"tibia_{side}", f"talus_{side}", f"calcn_{side}", f"toes_{side}"):
            factors[body] = float(shank / g_shank)
    mid = {
        key: 0.5 * (pts[f"{key}_l"] + pts[f"{key}_r"])
        for key, pts in (("hip", bundle), ("shoulder", bundle))
    }
    g_mid_hip = 0.5 * (generic["RHJC"] + generic["LHJC"])
    g_mid_shoulder = 0.5 * (generic["RSJC"] + generic["LSJC"])
    trunk = np.linalg.norm(mid["shoulder"] - mid["hip"])
    factors["torso"] = float(trunk / np.linalg.norm(g_mid_shoulder - g_mid_hip))
    if not all(math.isfinite(f) and f > 0.0 for f in factors.values()):
        raise ValueError(f"degenerate segment scales {factors}")
    return factors


# --------------------------------------------------------------------------
# Inverse kinematics
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class IKResult:
    """IK solution per target state (coordinate-set order, radians/metres)."""

    coordinates: tuple[str, ...]
    q: Array
    markers: tuple[str, ...]
    marker_error_m: Array
    model_points_m: Array
    range_violations: Mapping[str, dict[str, float | int]]


def _station(build: LiftModelBuild, label: str, targets: LiftTargets) -> tuple:
    """(body, local station) of a landmark on the scaled OpenSim model."""
    if label in _BODY_OF:
        return _BODY_OF[label], (0.0, 0.0, 0.0)
    body, station = targets.foot_stations_m[label]
    factors = build.receipt["segment_scale_factors"]
    scale = float(factors.get(body, build.receipt["scale_factor"]))
    return body, tuple(scale * x for x in station)


def _add_markers(build: LiftModelBuild, targets: LiftTargets) -> None:
    osim = import_opensim()
    model = build.model
    for label in IK_LANDMARKS:
        body, station = _station(build, label, targets)
        frame = model.getBodySet().get(body)
        model.addMarker(osim.Marker(_MARKER_PREFIX + label, frame, osim.Vec3(*station)))
    model.finalizeConnections()


def _markers_reference(targets: LiftTargets) -> Any:
    osim = import_opensim()
    labels = osim.StdVectorString()
    for label in IK_LANDMARKS:
        labels.append(_MARKER_PREFIX + label)
    table = osim.TimeSeriesTableVec3()
    table.setColumnLabels(labels)
    for t, row in zip(targets.time_s, targets.positions_m, strict=True):
        vecs = [osim.Vec3(*map(float, p)) for p in row]
        table.appendRow(float(t), osim.RowVectorVec3(vecs))
    table.addTableMetaDataString("Units", "m")
    weights = osim.SetMarkerWeights()
    for label in IK_LANDMARKS:
        weight = UPPER_BODY_WEIGHT if label in _UPPER_BODY else LOWER_LIMB_WEIGHT
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


def run_inverse_kinematics(build: LiftModelBuild, targets: LiftTargets) -> IKResult:
    """Track the bundle landmarks with OpenSim's ``InverseKinematicsSolver``.

    Mutates ``build.model``: adds one ``ik_<landmark>`` marker per landmark
    and leaves every coordinate unclamped.

    Postconditions: ``q`` is (N, coordinates); ``marker_error_m`` (N, M) and
    ``model_points_m`` (N, M, 3) follow :data:`IK_LANDMARKS`.
    """
    osim = import_opensim()
    model = build.model
    _add_markers(build, targets)
    for coordinate in model.updCoordinateSet():
        coordinate.set_clamped(False)
    state = model.initSystem()
    solver = osim.InverseKinematicsSolver(
        model,
        _markers_reference(targets),
        osim.SimTKArrayCoordinateReference(),
        math.inf,
    )
    solver.setAccuracy(_IK_ACCURACY)
    names = tuple(c.getName() for c in model.getCoordinateSet())
    markers = model.getMarkerSet()
    q_rows, errors, points = [], [], []
    for k, t in enumerate(targets.time_s):
        state.setTime(float(t))
        if k == 0:
            solver.assemble(state)
        else:
            solver.track(state)
        coords = model.getCoordinateSet()
        q_rows.append([coords.get(n).getValue(state) for n in names])
        errors.append(
            [solver.computeCurrentMarkerError(_MARKER_PREFIX + x) for x in IK_LANDMARKS]
        )
        model.realizePosition(state)
        points.append(
            [
                _vec3(markers.get(_MARKER_PREFIX + x).getLocationInGround(state))
                for x in IK_LANDMARKS
            ]
        )
    q = np.asarray(q_rows, dtype=float)
    return IKResult(
        coordinates=names,
        q=q,
        markers=IK_LANDMARKS,
        marker_error_m=np.asarray(errors, dtype=float),
        model_points_m=np.asarray(points, dtype=float),
        range_violations=_range_violations(model, names, q),
    )


def _segment_angles(points: Array) -> Array:
    by_label = dict(zip(IK_LANDMARKS, points, strict=True))
    return segment_angles({x: by_label[x] for x in LANDMARK_LABELS}, _OPENSIM_SAGITTAL)


def ik_receipt(targets: LiftTargets, ik: IKResult) -> dict[str, Any]:
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
        "weights": {
            x: UPPER_BODY_WEIGHT if x in _UPPER_BODY else LOWER_LIMB_WEIGHT
            for x in ik.markers
        },
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
        "clamping": "off; coordinates outside the model range are listed",
        "range_violations": dict(ik.range_violations),
        "observability": _PELVIS_OBSERVABILITY,
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
    zero = np.zeros(3)
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


def static_hold_grf(build: LiftModelBuild) -> dict[str, Any]:
    """Contact-derived ground reaction of the lift's hold pose at equilibrium.

    Postconditions: ``total_grf_n`` (canonical z-up) is the sum of every
    contact force; ``relative_error`` compares its vertical component with
    ``(skeleton mass + bar mass) * g`` from the build receipt; the remaining
    pelvis residuals at the equilibrium are reported as ``balance_residual``.

    Raises:
        ValueError: when no vertical position in the bracket balances.
    """
    osim = import_opensim()
    model = _bare_model(build.model, keep_contact=True)
    state = model.initSystem()
    solver = osim.InverseDynamicsSolver(model)
    u_index = _u_index(model, state)
    zero = osim.Vector(state.getNU(), 0.0)
    ty = model.getCoordinateSet().get("pelvis_ty")
    start = float(ty.getValue(state))

    def residual(value: float) -> float:
        ty.setValue(state, value, False)
        return float(solver.solve(state, zero).get(u_index["pelvis_ty"]))

    low = start - _STATIC_BRACKET_M
    if residual(start) * residual(low) > 0.0:
        raise ValueError("no pelvis height in the bracket balances the hold")
    root = float(brentq(residual, low, start, xtol=_STATIC_XTOL_M))
    ty.setValue(state, root, False)
    tau = solver.solve(state, zero)
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
    return {
        "method": "pelvis_ty solved where OpenSim inverse dynamics (contacts on, "
        "at rest, hold pose) has zero vertical residual; contact forces from "
        "SmoothSphereHalfSpaceForce record values, reduced by GCV-1",
        "pelvis_ty_m": root,
        "penetration_drop_m": start - root,
        "gravity_m_s2": gravity,
        "expected_weight_n": float(expected),
        "total_grf_n": total.tolist(),
        "relative_error": float((total[2] - expected) / expected),
        "support_vertical_n": {k: float(r.force_n[2]) for k, r in breakdowns.items()},
        "support_cop_m": {
            k: None if r.cop_m is None else r.cop_m.tolist()
            for k, r in breakdowns.items()
        },
        "balance_residual": {
            n: float(tau.get(u_index[n]))
            for n in RESIDUAL_COORDINATES
            if n != "pelvis_ty"
        },
    }
