"""Musculoskeletal OpenSim barbell-lift models (OSL-2, #13040; LIFT-6, #11746).

Composes, in the real OpenSim runtime, one full-body musculoskeletal model per
barbell lift (``squat``, ``deadlift``, ``bench_press``, ``snatch``,
``clean_and_jerk``):

* **Body.** The muscle-bearing Rajagopal-Lai-Uhlrich 2023 model (80 Millard
  lower-limb muscles, upper-body ``CoordinateActuator`` stand-ins), resolved at
  run time from the ``opensim-org/opensim-models`` submodule (or
  ``$UPSTREAMDRIFT_MSK_BASE_MODEL``) by
  :func:`musculoskeletal_swing.resolve_base_model`; nothing is copied into the
  repository (licence audit: ``src/engines/physics_engines/opensim/models/
  README.md``).  It is scaled uniformly by ``height_m / GENERIC_STATURE_M``
  and to ``body_mass_kg`` with ``Model.scale`` (mass distribution preserved;
  muscle fibre and tendon lengths scale with the bodies).
* **Barbell.** One ``barbell`` body of the spec's mass with the shared
  :class:`~src.shared.python.lifting.barbell_bushing.Barbell` inertia (bar
  long axis = body ``x``), held by the LIFT-4 hand interface of the OpenSim
  pack: ``WeldJoint hand_l_to_barbell`` (left) plus ``WeldConstraint
  hand_r_to_barbell`` (right).  The grip points are the hands' centres of
  mass (the palm-centre attach point of the parity standard), and the grip
  width is solved to the standard's ``width_m`` scaled by
  ``height_m / standard_height_m``.
* **Contact.** ``SmoothSphereHalfSpaceForce`` between a floor half-space and
  the three MS-20 sphere stations per foot (heel, forefoot, toe; the layout of
  ``motion_matching.execution.spec_builder``).  The bench press adds a
  bench-top half-space paired only with four back spheres placed at the
  model's posterior shoulder (``RPSH``/``LPSH``) and posterior iliac
  (``RPSI``/``LPSI``) markers.  The stiffness is OpenSim_Models PR #476's
  ``CONTACT_STIFFNESS`` (1e7 Pa, which seats lifter plus bar 4.3 mm deep
  there), not MS-20's 1e5 Pa (a 0.119 m sink under 200 kg, OSL-12 #13048);
  dissipation and friction stay MS-20's.

Hold pose (assumption, documented): one static pose per lift that the
Rajagopal joint ranges admit -- squat: standing with the bar on the upper
back; deadlift: lockout; bench press: supine, arms extended (unrack);
snatch and clean and jerk: hang (standing, arms straight, each lift's grip).
Standing holds are balanced (OSL-12, #13048): the body leans at the ankles,
feet flat, until the centre of mass of body plus bar is over the centre of
the foot spheres (:func:`_balance_lean`).
The Rajagopal shoulder (``arm_flex`` <= 90 deg, ``arm_add`` >= -120 deg) and
elbow (<= 150 deg) cannot reach a straight-arm overhead lockout or a front
rack at the clean grip width, so neither is claimed.  No shoulder, back or chest contact with the bar is modelled: in the
hold the hands carry the whole bar.  The standard's via-poses are not
retargeted here; inverse kinematics and dynamics are OSL-3/OSL-4.

OpenSim is imported lazily; callers check availability first.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation

from src.engines.physics_engines.opensim.python.msk_lift_foot import (
    apply_tibial_torsion,
)
from src.engines.physics_engines.opensim.python.musculoskeletal_swing import (
    resolve_base_model,
)
from src.shared.python.engine_core.opensim_runtime import import_opensim
from src.shared.python.lifting.barbell_bushing import Barbell
from src.shared.python.lifting.lift_spec import lift_grip, standard_height_m
from src.shared.python.lifting.pack_audit.frames import OPENSIM_Y_UP, to_canonical
from src.shared.python.lifting.pack_audit.names import LIFTS
from src.shared.python.motion_matching.execution import spec_builder as contact_spec

__all__ = [
    "ARM_COORDINATE_PREFIXES",
    "BAR_BODY",
    "BODY_SOURCE",
    "GENERIC_STATURE_M",
    "HOLD_POSES",
    "LEFT_WELD_JOINT",
    "LIFT_CONTACT_STIFFNESS_PA",
    "PENETRATION_LIMIT_M",
    "RIGHT_WELD_CONSTRAINT",
    "HoldPose",
    "LiftModelBuild",
    "build_lift_model",
    "native_static_hold",
]

BODY_SOURCE = "RajagopalLaiUhlrich2023"
#: Stature of the generic Rajagopal et al. (2016) adult (about 170 cm, 75 kg).
#: Uniform height scaling is an approximation; OSL-3 passes measured segment
#: factors (``segment_scales``) when it fits a same-input bundle.
GENERIC_STATURE_M = 1.70
BAR_BODY = "barbell"
LEFT_WELD_JOINT = "hand_l_to_barbell"
RIGHT_WELD_CONSTRAINT = "hand_r_to_barbell"
FLOOR = "floor"
BENCH_TOP = "bench_top"
CONTACT_PREFIX = "contact_"
SIDES = ("r", "l")
RIGHT_HAND_BODY = "hand_r"
#: Back spheres (bench press): marker name, body, radius in metres (generic).
_BACK_SPHERES = (
    ("RPSH", "torso", 0.04),
    ("LPSH", "torso", 0.04),
    ("RPSI", "pelvis", 0.04),
    ("LPSI", "pelvis", 0.04),
)
BENCH_EXTERNAL_LOAD_SOURCE = (
    "shared-law back-sphere contact wrench of the bundle (VectorPlant), "
    "applied to torso (back_RPSH, back_LPSH) and pelvis (back_RPSI, back_LPSI) "
    "at the contact points"
)


def external_load_sources(targets: Any) -> dict[str, str]:
    """External wrench provenance, one source for ID and joint reactions."""
    sources = {
        "feet": "GCV-1 foot wrench of the bundle's shared-law contact "
        "(kinetics_parity.bundle_kinetics), applied to calcn_l/calcn_r",
        "hands": "GCV-7 per-hand bushing wrench of the bundle bar "
        "(BushingSample.grip_analysis), negated onto hand_l/hand_r at the "
        "model's grip point; the barbell body is massless in ID",
    }
    if targets.bench_force_n:
        sources["bench"] = BENCH_EXTERNAL_LOAD_SOURCE
    return sources


#: Contact stiffness (Pa) of every sphere: OpenSim_Models PR #476
#: (``shared/contact/policy.py`` ``CONTACT_STIFFNESS``), which seats an 80 kg
#: lifter plus a 120 kg bar 4.3 mm deep on eight 0.02 m spheres.
LIFT_CONTACT_STIFFNESS_PA = 1.0e7
#: Largest accepted sphere penetration at a balanced hold: the 1-8 mm stance
#: band OpenSim_Models PR #476 states for its seated feet.  Declared before
#: measuring; never loosened.
PENETRATION_LIMIT_M = 0.008
#: Bracket (deg) of the ankle lean that balances a standing hold.
_LEAN_BRACKET_DEG = 25.0
_LEAN_XTOL_RAD = 1e-10
#: Coordinates a caller may set through ``arm_coordinates`` (the arm chain).
ARM_COORDINATE_PREFIXES = ("arm_", "elbow_", "pro_sup", "wrist_")
#: Bracket (deg) searched either side of the nominal ``arm_add`` for the grip.
_ADD_BRACKET_DEG = 30.0
_WIDTH_XTOL_M = 1e-6


@dataclass(frozen=True)
class HoldPose:
    """Static hold pose of one lift (Rajagopal coordinate names, degrees).

    ``arms`` is ``(arm_flex, arm_add, arm_rot, elbow_flex)`` applied to both
    sides; ``arm_add`` is the nominal value the grip-width solve starts from.
    ``supine`` lies the body on its back (``pelvis_tilt`` 90 deg) with the
    knees flexed 90 deg so the feet reach the floor.
    """

    name: str
    description: str
    arms: tuple[float, float, float, float]
    supine: bool = False


HOLD_POSES: Mapping[str, HoldPose] = {
    "squat": HoldPose(
        "back_rack_standing",
        "standing, hands behind the shoulders at the upper-back bar position",
        (90.0, -44.0, -90.0, 145.0),
    ),
    "deadlift": HoldPose(
        "lockout", "standing lockout, arms hanging", (5.0, -3.0, 0.0, 0.0)
    ),
    "bench_press": HoldPose(
        "unrack",
        "supine, arms extended over the shoulders",
        (88.0, -17.0, 0.0, 5.0),
        supine=True,
    ),
    "snatch": HoldPose(
        "hang", "standing hang, arms straight, wide grip", (5.0, -26.0, 0.0, 0.0)
    ),
    "clean_and_jerk": HoldPose(
        "hang", "standing hang, arms straight, clean grip", (5.0, -8.0, 0.0, 0.0)
    ),
}


@dataclass
class LiftModelBuild:
    """A composed lift model and its receipt (counts, source, pose, grip)."""

    model: Any
    receipt: dict[str, Any] = field(default_factory=dict)


def _vec3(v: Any) -> np.ndarray:
    return np.array([float(v.get(i)) for i in range(3)], dtype=float)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _arm_names(side: str) -> tuple[str, str, str, str]:
    return (
        f"arm_flex_{side}",
        f"arm_add_{side}",
        f"arm_rot_{side}",
        f"elbow_flex_{side}",
    )


def _pose_coordinates(pose: HoldPose, arm_add_deg: float) -> dict[str, float]:
    """Hold-pose coordinate defaults in radians (every other coordinate 0)."""
    flex, _, rot, elbow = pose.arms
    values: dict[str, float] = {}
    for side in SIDES:
        names = _arm_names(side)
        for name, deg in zip(names, (flex, arm_add_deg, rot, elbow), strict=True):
            values[name] = math.radians(deg)
        if pose.supine:
            values[f"knee_angle_{side}"] = math.radians(90.0)
    if pose.supine:
        values["pelvis_tilt"] = math.radians(90.0)
    return values


def _apply_defaults(model: Any, values: Mapping[str, float]) -> None:
    coords = model.updCoordinateSet()
    for name, value in values.items():
        coords.get(name).setDefaultValue(float(value))


def _hand_point(model: Any, state: Any, side: str) -> np.ndarray:
    """Ground position (OpenSim frame) of a hand's centre of mass."""
    hand = model.getBodySet().get(f"hand_{side}")
    return _vec3(hand.findStationLocationInGround(state, hand.getMassCenter()))


def _grip_width(model: Any, state: Any) -> float:
    model.realizePosition(state)
    return float(
        np.linalg.norm(_hand_point(model, state, "r") - _hand_point(model, state, "l"))
    )


def _solve_arm_add(model: Any, pose: HoldPose, target_m: float) -> float:
    """``arm_add`` (deg, both sides) giving the target hand-centre grip width.

    Raises:
        ValueError: if the bracket around the nominal value holds no root.
    """
    state = model.initSystem()
    coords = model.updCoordinateSet()

    def residual(add_deg: float) -> float:
        for side in SIDES:
            coords.get(f"arm_add_{side}").setValue(state, math.radians(add_deg), False)
        return _grip_width(model, state) - target_m

    nominal = pose.arms[1]
    low = max(nominal - _ADD_BRACKET_DEG, -120.0)
    high = min(nominal + _ADD_BRACKET_DEG, 90.0)
    if residual(low) * residual(high) > 0.0:
        raise ValueError(
            f"{pose.name}: no arm_add in [{low}, {high}] deg gives the "
            f"{target_m:.3f} m grip width"
        )
    return float(brentq(residual, low, high, xtol=_WIDTH_XTOL_M))


SegmentScale = float | tuple[float, float, float]


def _axis_factors(body: str, factor: SegmentScale) -> tuple[float, float, float]:
    """Per-axis (x, y, z) factors of one segment scale.

    Raises:
        ValueError: when a factor is not positive and finite or not 1 or 3 long.
    """
    axes = (factor,) * 3 if isinstance(factor, (int, float)) else tuple(factor)
    if len(axes) != 3 or not all(math.isfinite(f) and f > 0.0 for f in axes):
        raise ValueError(f"segment scale of {body!r} must be positive, got {factor}")
    return (float(axes[0]), float(axes[1]), float(axes[2]))


def _scale_body(
    model: Any,
    scale: float,
    body_mass_kg: float,
    segment_scales: Mapping[str, SegmentScale] | None = None,
) -> dict[str, tuple[float, float, float]]:
    """Scale every body by ``scale`` except those given in ``segment_scales``.

    A segment scale is one factor or an ``(x, y, z)`` triple in the body
    frame (OSL-12 scales the pelvis mediolaterally to the bundle hip width).

    Raises:
        ValueError: on an unknown body or a non-positive segment factor.
    """
    osim = import_opensim()
    state = model.initSystem()
    factors = {body.getName(): (scale,) * 3 for body in model.getBodySet()}
    for body, factor in (segment_scales or {}).items():
        if body not in factors:
            raise ValueError(f"segment_scales names unknown body {body!r}")
        factors[body] = _axis_factors(body, factor)
    scale_set = osim.ScaleSet()
    for body, axes in factors.items():
        entry = osim.Scale()
        entry.setSegmentName(body)
        entry.setScaleFactors(osim.Vec3(*axes))
        entry.setApply(True)
        scale_set.cloneAndAppend(entry)
    model.scale(state, scale_set, True, float(body_mass_kg))
    return factors


def _foot_spheres(scale: float) -> list[tuple[str, str, np.ndarray, float]]:
    """MS-20 calibrated foot spheres (name, body, station, radius), scaled."""
    return [
        (f"{kind}_{side}", f"calcn_{side}", np.asarray(pos, float) * scale, r * scale)
        for side in SIDES
        for kind, pos, r in contact_spec.SPHERES
    ]


def _back_spheres(model: Any, scale: float) -> list[tuple[str, str, np.ndarray, float]]:
    """Back spheres touching the skin markers from the anterior side."""
    markers = model.getMarkerSet()
    spheres = []
    for marker, body, radius in _BACK_SPHERES:
        station = _vec3(markers.get(marker).get_location())
        r = radius * scale
        spheres.append((f"back_{marker}", body, station + np.array([r, 0.0, 0.0]), r))
    return spheres


def _lowest_point(
    model: Any, state: Any, spheres: list[tuple[str, str, np.ndarray, float]]
) -> float:
    model.realizePosition(state)
    bodies = model.getBodySet()
    osim = import_opensim()
    return min(
        float(
            bodies.get(body)
            .findStationLocationInGround(state, osim.Vec3(*station))
            .get(1)
        )
        - radius
        for _, body, station, radius in spheres
    )


def _add_contact(
    model: Any,
    half_space: Any,
    spheres: list[tuple[str, str, np.ndarray, float]],
) -> list[str]:
    osim = import_opensim()
    names = []
    for name, body, station, radius in spheres:
        sphere = osim.ContactSphere(
            float(radius), osim.Vec3(*station), model.getBodySet().get(body), name
        )
        model.addContactGeometry(sphere)
        force = osim.SmoothSphereHalfSpaceForce(
            CONTACT_PREFIX + name, sphere, half_space
        )
        force.set_stiffness(LIFT_CONTACT_STIFFNESS_PA)
        force.set_dissipation(contact_spec.CONTACT_DISSIPATION_S_M)
        force.set_static_friction(contact_spec.CONTACT_STATIC_FRICTION)
        force.set_dynamic_friction(contact_spec.CONTACT_DYNAMIC_FRICTION)
        force.set_viscous_friction(contact_spec.CONTACT_VISCOUS_FRICTION)
        force.set_transition_velocity(contact_spec.CONTACT_TRANSITION_VELOCITY_M_S)
        model.addForce(force)
        names.append(force.getName())
    return names


def _ground_points(
    model: Any, state: Any, spheres: list[tuple[str, str, np.ndarray, float]]
) -> np.ndarray:
    """Lowest point (ground frame) of each sphere, in ``spheres`` order."""
    osim = import_opensim()
    model.realizePosition(state)
    bodies = model.getBodySet()
    points = []
    for _, body, station, radius in spheres:
        centre = _vec3(
            bodies.get(body).findStationLocationInGround(state, osim.Vec3(*station))
        )
        points.append(centre - np.array([0.0, radius, 0.0]))
    return np.asarray(points)


def _system_com(model: Any, state: Any, bar_mass_kg: float) -> np.ndarray:
    """Centre of mass of body plus bar, the bar at the hand-centre midpoint.

    Called before the barbell body exists, so the bar is added by hand.
    """
    model.realizePosition(state)
    body_mass = float(model.getTotalMass(state))
    body_com = _vec3(model.calcMassCenterPosition(state))
    bar = 0.5 * (_hand_point(model, state, "l") + _hand_point(model, state, "r"))
    return (body_mass * body_com + bar_mass_kg * bar) / (body_mass + bar_mass_kg)


def _balance_lean(
    model: Any,
    coordinates: dict[str, float],
    feet: list[tuple[str, str, np.ndarray, float]],
    bar_mass_kg: float,
) -> dict[str, float]:
    """Lean a standing hold at the ankles until it balances over the feet.

    Both ``ankle_angle`` coordinates change by ``lean`` and ``pelvis_tilt`` by
    ``-lean``, which rotates everything above the ankles while the feet stay
    flat (the oblique Rajagopal ankle axis leaves a sub-degree foot roll).
    ``lean`` is solved so the centre of mass of body plus bar lies over the
    mean of the foot spheres' lowest points (fore-aft).  Mutates
    ``coordinates`` and the model defaults.

    Postcondition: the returned ``com_offset_m`` is below 1e-6 m.

    Raises:
        ValueError: when no lean within +/-25 deg balances the hold.
    """
    state = model.initSystem()
    coords = model.updCoordinateSet()
    base = {
        name: coordinates.get(name, 0.0)
        for name in ("ankle_angle_l", "ankle_angle_r", "pelvis_tilt")
    }

    def offset(lean: float) -> float:
        for name, value in base.items():
            sign = -1.0 if name == "pelvis_tilt" else 1.0
            coords.get(name).setValue(state, value + sign * lean, False)
        com = _system_com(model, state, bar_mass_kg)
        return float(com[0] - _ground_points(model, state, feet)[:, 0].mean())

    bound = math.radians(_LEAN_BRACKET_DEG)
    if offset(-bound) * offset(bound) > 0.0:
        raise ValueError("no ankle lean within +/-25 deg balances the hold")
    lean = float(brentq(offset, -bound, bound, xtol=_LEAN_XTOL_RAD))
    residual = offset(lean)
    for name, value in base.items():
        sign = -1.0 if name == "pelvis_tilt" else 1.0
        coordinates[name] = value + sign * lean
    _apply_defaults(model, {name: coordinates[name] for name in base})
    return {"ankle_lean_rad": lean, "com_offset_m": residual}


def _half_space(model: Any, name: str, height_m: float) -> Any:
    """A ground half-space whose outward normal is +Y at ``height_m``."""
    osim = import_opensim()
    plane = osim.ContactHalfSpace(
        osim.Vec3(0.0, float(height_m), 0.0),
        osim.Vec3(0.0, 0.0, -math.pi / 2.0),
        model.getGround(),
        name,
    )
    model.addContactGeometry(plane)
    return plane


def _body_rotation(model: Any, state: Any, body: str) -> np.ndarray:
    """Columns: the body's (or ``ground``'s) axes expressed in ground."""
    osim = import_opensim()
    frame = model.getGround() if body == "ground" else model.getBodySet().get(body)
    return np.column_stack(
        [
            _vec3(frame.expressVectorInGround(state, osim.Vec3(*axis)))
            for axis in np.eye(3)
        ]
    )


def _bar_rotation(grip_l: np.ndarray, grip_r: np.ndarray) -> np.ndarray:
    """Bar frame in ground: ``x`` left-to-right along the bar, ``y`` up-most."""
    x_axis = (grip_r - grip_l) / np.linalg.norm(grip_r - grip_l)
    up = np.array([0.0, 1.0, 0.0])
    y_axis = up - float(up @ x_axis) * x_axis
    y_axis /= np.linalg.norm(y_axis)
    return np.column_stack([x_axis, y_axis, np.cross(x_axis, y_axis)])


def _xyz(rotation: np.ndarray) -> Any:
    osim = import_opensim()
    return osim.Vec3(*Rotation.from_matrix(rotation).as_euler("XYZ"))


@dataclass(frozen=True)
class _BarPlacement:
    """Bar pose at the hold, measured before any component is added."""

    half_width_m: float
    in_hand_l: np.ndarray
    in_hand_r: np.ndarray


def _bar_placement(model: Any, state: Any) -> _BarPlacement:
    """Grip half-width and the bar orientation in each hand frame."""
    model.realizePosition(state)
    grip_l = _hand_point(model, state, "l")
    grip_r = _hand_point(model, state, "r")
    bar_r = _bar_rotation(grip_l, grip_r)
    return _BarPlacement(
        half_width_m=0.5 * float(np.linalg.norm(grip_r - grip_l)),
        in_hand_l=_body_rotation(model, state, "hand_l").T @ bar_r,
        in_hand_r=_body_rotation(model, state, "hand_r").T @ bar_r,
    )


def _attach_barbell(model: Any, placement: _BarPlacement, bar_mass_kg: float) -> None:
    """Add the barbell: ``WeldJoint`` to the left hand, ``WeldConstraint`` right."""
    osim = import_opensim()
    shared_bar = Barbell(load_kg=float(bar_mass_kg))
    axial = shared_bar.axial_inertia_kg_m2
    transverse = shared_bar.transverse_inertia_kg_m2
    bar = osim.Body(
        BAR_BODY,
        float(bar_mass_kg),
        osim.Vec3(0.0),
        osim.Inertia(axial, transverse, transverse, 0.0, 0.0, 0.0),
    )
    bodies = model.getBodySet()
    hand_l, hand_r = bodies.get("hand_l"), bodies.get("hand_r")
    half = placement.half_width_m
    weld = osim.WeldJoint(
        LEFT_WELD_JOINT,
        hand_l,
        hand_l.getMassCenter(),
        _xyz(placement.in_hand_l),
        bar,
        osim.Vec3(-half, 0.0, 0.0),
        osim.Vec3(0.0),
    )

    # Visual geometry for barbell shaft and plates (OSL-8, #13045).
    # Dimensions cited from src.shared.python.lifting.barbell_bushing.Barbell:
    # shaft_length_m=2.20, shaft_radius_m=0.014, plate_radius_m=0.225,
    # plate_offset_m=0.70. Simbody cylinders lie along local Y; rotate 90 deg
    # about Z to align with the barbell local X axis connecting the hands.
    rot = osim.Rotation(math.pi / 2.0, osim.Vec3(0.0, 0.0, 1.0))
    t_shaft = osim.Transform(rot, osim.Vec3(0.0, 0.0, 0.0))
    pof_shaft = osim.PhysicalOffsetFrame(f"{BAR_BODY}_shaft_frame", bar, t_shaft)
    shaft_geom = osim.Cylinder(
        float(shared_bar.shaft_radius_m), float(shared_bar.shaft_length_m / 2.0)
    )
    shaft_geom.setColor(osim.Vec3(0.7, 0.7, 0.7))
    pof_shaft.attachGeometry(shaft_geom)
    bar.addComponent(pof_shaft)

    for side, sign in (("l", -1.0), ("r", 1.0)):
        t_plate = osim.Transform(
            rot, osim.Vec3(sign * float(shared_bar.plate_offset_m), 0.0, 0.0)
        )
        pof_plate = osim.PhysicalOffsetFrame(
            f"{BAR_BODY}_plate_{side}_frame", bar, t_plate
        )
        plate_geom = osim.Cylinder(float(shared_bar.plate_radius_m), 0.025)
        plate_geom.setColor(osim.Vec3(0.2, 0.2, 0.2))
        plate_geom.setOpacity(0.6)
        pof_plate.attachGeometry(plate_geom)
        bar.addComponent(pof_plate)

    model.addBody(bar)
    model.addJoint(weld)
    closure = osim.WeldConstraint(
        RIGHT_WELD_CONSTRAINT,
        hand_r,
        hand_r.getMassCenter(),
        _xyz(placement.in_hand_r),
        bar,
        osim.Vec3(half, 0.0, 0.0),
        osim.Vec3(0.0),
    )
    model.addConstraint(closure)


def _require_inputs(
    lift: str, body_mass_kg: float, height_m: float, bar_mass_kg: float
) -> None:
    if lift not in LIFTS:
        raise ValueError(f"lift must be one of {list(LIFTS)}, got {lift!r}")
    for name, value in (
        ("body_mass_kg", body_mass_kg),
        ("height_m", height_m),
        ("bar_mass_kg", bar_mass_kg),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite, got {value}")


def _require_arm_coordinates(arm_coordinates: Mapping[str, float] | None) -> None:
    """Given arm coordinates must name the arm chain and be finite (radians)."""
    for name, value in (arm_coordinates or {}).items():
        if not name.startswith(ARM_COORDINATE_PREFIXES):
            raise ValueError(
                f"arm_coordinates may only set the arm chain "
                f"{ARM_COORDINATE_PREFIXES}, got {name!r}"
            )
        if not math.isfinite(value):
            raise ValueError(f"arm coordinate {name!r} must be finite, got {value}")


def build_lift_model(
    lift: str,
    *,
    body_mass_kg: float,
    height_m: float,
    bar_mass_kg: float,
    base_model_path: str | Path | None = None,
    segment_scales: Mapping[str, SegmentScale] | None = None,
    arm_coordinates: Mapping[str, float] | None = None,
    tibial_torsion: Mapping[str, Any] | None = None,
) -> LiftModelBuild:
    """Compose the musculoskeletal OpenSim model of one barbell lift.

    ``segment_scales`` overrides the uniform ``height_m`` scale per body (the
    measured segment scaling of OSL-3; a triple scales the body's x, y, z
    axes separately); every other body keeps the uniform factor.
    ``arm_coordinates`` (radians, arm chain only) replaces the hold pose's
    arms; the grip width is then whatever those arms give, not solved
    (OSL-12 seeds the arms on a bundle's grips).  A standing hold is
    balanced by an ankle lean (:func:`_balance_lean`).  ``tibial_torsion`` (a
    bundle spec's ``tibial_torsion_deg`` record) turns each foot out about its
    shank as the bundle's foot is turned (OSL-18, #13157;
    :func:`.msk_lift_foot.apply_tibial_torsion`); ``None`` keeps the
    generic Rajagopal foot.

    Postconditions: the returned model passes ``initSystem``; its skeleton
    mass equals ``body_mass_kg``; without ``arm_coordinates`` the hand-centre
    grip width equals the standard's ``width_m`` scaled to ``height_m``; the
    receipt records the body source path and SHA-256, the scale factor, the
    hold pose (and its ``arm_source``) and the contact layout.

    Raises:
        ValueError: on an unknown lift, a non-positive input, a bar lighter
            than the shared 20 kg shaft, an unreachable grip width, or an
            ``arm_coordinates`` entry outside the arm chain or not finite, or
            a tibial-torsion offset outside the cited adult range.
        FileNotFoundError: when the muscle-bearing body is not available.
    """
    _require_inputs(lift, body_mass_kg, height_m, bar_mass_kg)
    _require_arm_coordinates(arm_coordinates)
    osim = import_opensim()
    base_path = resolve_base_model(base_model_path)
    Barbell(load_kg=float(bar_mass_kg))  # precondition: at least the shaft mass
    model = osim.Model(str(base_path))
    model.setName(f"{BODY_SOURCE}_{lift}")
    torsion = apply_tibial_torsion(model, tibial_torsion)
    scale = float(height_m) / GENERIC_STATURE_M
    factors = _scale_body(model, scale, body_mass_kg, segment_scales)
    skeleton_mass = float(sum(body.getMass() for body in model.getBodySet()))
    posed = _pose_for_hold(model, lift, height_m, bar_mass_kg, scale, arm_coordinates)
    contacts, bench_height, grip_width = _add_contacts_and_bar(
        model, posed, scale, bar_mass_kg
    )
    receipt = {
        "body_source": _body_source(base_path),
        "generic_stature_m": GENERIC_STATURE_M,
        "scale_factor": scale,
        "segment_scale_factors": {
            body: axes[0] if len(set(axes)) == 1 else list(axes)
            for body, axes in factors.items()
            if axes != (scale,) * 3
        },
        "skeleton_mass_kg": skeleton_mass,
        "bar_mass_kg": float(bar_mass_kg),
        "tibial_torsion": torsion,
        "hold_pose": posed.hold_record(),
        "grip_width_m": grip_width,
        "grip_width_target_m": posed.target_width_m,
        "bench_height_m": bench_height,
        "contact_forces": contacts,
        "contact_parameters": _contact_parameters(),
    }
    return LiftModelBuild(model=model, receipt=receipt)


@dataclass(frozen=True)
class _PosedHold:
    """A lift model posed in its hold, standing on (or lying above) the floor."""

    pose: HoldPose
    coordinates: dict[str, float]
    feet: list[tuple[str, str, np.ndarray, float]]
    balance: dict[str, Any] | None
    target_width_m: float | None
    state: Any

    def hold_record(self) -> dict[str, Any]:
        """The receipt's ``hold_pose`` entry."""
        return {
            "name": self.pose.name,
            "description": self.pose.description,
            "supine": self.pose.supine,
            "arm_source": "hold_pose" if self.target_width_m is not None else "given",
            "coordinates_rad": self.coordinates,
            "balance": self.balance,
        }


def _pose_for_hold(
    model: Any,
    lift: str,
    height_m: float,
    bar_mass_kg: float,
    scale: float,
    arm_coordinates: Mapping[str, float] | None = None,
) -> _PosedHold:
    """Set the hold pose's defaults, reach the grip width, balance, lift to floor.

    Given ``arm_coordinates`` replace the pose's arms and skip the width solve.
    """
    pose = HOLD_POSES[lift]
    target_width: float | None = None
    if arm_coordinates is None:
        target_width = lift_grip(lift).width_m * float(height_m) / standard_height_m()
        _apply_defaults(model, _pose_coordinates(pose, pose.arms[1]))
        arm_add = _solve_arm_add(model, pose, target_width)
        coordinates = _pose_coordinates(pose, arm_add)
    else:
        coordinates = _pose_coordinates(pose, pose.arms[1])
        coordinates.update({k: float(v) for k, v in arm_coordinates.items()})
    _apply_defaults(model, coordinates)

    feet = _foot_spheres(scale)
    balance = (
        None if pose.supine else _balance_lean(model, coordinates, feet, bar_mass_kg)
    )
    state = model.initSystem()
    lift_y = -_lowest_point(model, state, feet)
    coordinates["pelvis_ty"] = (
        model.getCoordinateSet().get("pelvis_ty").getDefaultValue() + lift_y
    )
    _apply_defaults(model, {"pelvis_ty": coordinates["pelvis_ty"]})
    state = model.initSystem()
    return _PosedHold(pose, coordinates, feet, balance, target_width, state)


def _add_contacts_and_bar(
    model: Any, posed: _PosedHold, scale: float, bar_mass_kg: float
) -> tuple[list[str], float | None, float]:
    """Floor (and bench) contacts plus the welded bar; returns the grip width."""
    # Measure everything posed first: adding a component drops the System.
    back = _back_spheres(model, scale) if posed.pose.supine else []
    bench_height = _lowest_point(model, posed.state, back) if back else None
    placement = _bar_placement(model, posed.state)

    contacts = _add_contact(model, _half_space(model, FLOOR, 0.0), posed.feet)
    if bench_height is not None:
        bench = _half_space(model, BENCH_TOP, bench_height)
        contacts += _add_contact(model, bench, back)
    _attach_barbell(model, placement, bar_mass_kg)
    model.finalizeConnections()
    model.initSystem()
    return contacts, bench_height, 2.0 * placement.half_width_m


def _body_source(base_path: Path) -> dict[str, str]:
    return {
        "name": BODY_SOURCE,
        "path": str(base_path),
        "sha256": _sha256(base_path),
        "licence": "opensim-org/opensim-models submodule, referenced not vendored; "
        "audit in src/engines/physics_engines/opensim/models/README.md",
    }


def _contact_parameters() -> dict[str, Any]:
    return {
        "stiffness_pa": LIFT_CONTACT_STIFFNESS_PA,
        "stiffness_source": "OpenSim_Models PR #476 CONTACT_STIFFNESS",
        "dissipation_friction_source": "MS-20 (#10335) receipt "
        + contact_spec.CONTACT_RECEIPT_SHA256,
        "penetration_limit_m": PENETRATION_LIMIT_M,
    }


def _ancestors(model: Any, body: str) -> list[str]:
    """``body`` and its ancestors up to ground, nearest first."""
    parent_of = {}
    for joint in model.getJointSet():
        child = joint.getChildFrame().findBaseFrame().getName()
        parent_of[child] = joint.getParentFrame().findBaseFrame().getName()
    chain = [body]
    while chain[-1] in parent_of:
        chain.append(parent_of[chain[-1]])
    return chain


def _common_ancestor(model: Any, body_a: str, body_b: str) -> str:
    """Nearest body that is an ancestor of both (``ground`` at worst)."""
    others = set(_ancestors(model, body_b))
    return next(name for name in _ancestors(model, body_a) if name in others)


def native_static_hold(model: Any, left_joint: Any, right_weld: Any) -> dict[str, Any]:
    """Hand-bar reactions OpenSim computes for a static hold at the default pose.

    Every coordinate is locked in the state, so the system is at rest; after
    ``realizeAcceleration`` the left ``WeldJoint`` reaction on the bar and the
    right ``WeldConstraint`` force on the bar are read back.  Their split is
    not physical (a rigid two-hand hold is statically indeterminate and the
    solver picks one) -- only their sum is checked against the bar weight.
    Simbody's ``calcConstraintForces`` returns the negated applied force, so
    the bar-side entry is negated.  Vectors are in the canonical z-up frame.
    """
    osim = import_opensim()
    state = model.initSystem()
    for coordinate in model.getCoordinateSet():
        coordinate.setLocked(state, True)
    model.realizeAcceleration(state)
    left = _vec3(left_joint.calcReactionOnChildExpressedInGround(state).get(1))
    body_forces = osim.VectorOfSpatialVec()
    right_weld.calcConstraintForces(state, body_forces, osim.Vector())
    # Simbody expresses the constraint forces in the constrained bodies'
    # common ancestor (here the torso), not in ground.
    ancestor = _common_ancestor(model, RIGHT_HAND_BODY, BAR_BODY)
    in_ancestor = -_vec3(body_forces.get(1).get(1))
    right = _body_rotation(model, state, ancestor) @ in_ancestor
    left_c = to_canonical(OPENSIM_Y_UP, left.tolist())
    right_c = to_canonical(OPENSIM_Y_UP, right.tolist())
    bar = model.getBodySet().get(BAR_BODY)
    weight = float(bar.getMass()) * float(np.linalg.norm(_vec3(model.getGravity())))
    total = float(left_c[2] + right_c[2])
    return {
        "hand_force_n": {"L": left_c.tolist(), "R": right_c.tolist()},
        "sum_vertical_n": total,
        "bar_weight_n": weight,
        "relative_error": abs(total - weight) / weight,
        "method": "OpenSim realizeAcceleration with every coordinate locked: "
        "WeldJoint reaction on the bar plus WeldConstraint force on the bar",
    }
