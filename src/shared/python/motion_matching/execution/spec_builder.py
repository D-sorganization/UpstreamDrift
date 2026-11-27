"""Build the anthropometric full-body candidate document (AN-1, #10099, #10520).

Upper body from ``anthropometric_geometry.build_upper_body`` (subject stature
and mass), Rajagopal legs through ``leg_extension`` with fixed pelvis alignment
(OpenSim pelvis axes onto the pelvis frame, hip centres at subject hip half width),
toe spheres, calibrated contact law, and anatomical seed offsets.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from defusedxml import ElementTree as ET

from src.shared.python.motion_matching.arm_mirror import mirror_document_arm
from src.shared.python.motion_matching.anthropometric_geometry import (
    GIRDLE_MASS_ADDED,
    GIRDLE_MASS_RULES,
    GRIP_ROTATION_DEG,
    LEG_VISUAL_RADIUS_M,
    build_upper_body,
    pelvis_alignment_for,
)
from src.shared.python.motion_matching.anthropometry import (
    de_leva_table_dict,
    de_leva_table_sha256,
)
from src.shared.python.motion_matching.club_models import (
    CLUBS,
    SOLE_POINT_SOURCE,
    ClubSpec,
    checked_sole_point_m,
    recorded_club,
)
from src.shared.python.motion_matching.club_shaft_chain import (
    DEFAULT_SHAFT_MODEL,
    SHAFT_MODELS,
    add_shaft_arguments,
    apply_shaft_chain,
    build_shaft_chain,
    checked_shaft_options,
    shaft_options,
)
from src.shared.python.motion_matching.execution.assets import (
    get_candidate_geometry_spec,
    get_club_markers,
    get_native_geometry_spec,
    get_opensim_model,
)
from src.shared.python.motion_matching.execution.marker_seed_overrides import (
    apply_marker_seed_overrides,
)
from src.shared.python.motion_matching.execution.subject_preset import (
    load_subject_preset,
)
from src.shared.python.motion_matching.execution.club_markers import (
    CLUB_MARKER_BODY,
    Offset,
    load_club_markers,
)
from src.shared.python.motion_matching.execution.osim_leg_lengths import (
    positive_length_m,
    scale_leg_lengths,
)
from src.shared.python.motion_matching.execution.grip_selection import (
    add_grip_arguments,
    overrides_from_args,
)
from src.shared.python.motion_matching.execution.spec_revisions import (
    with_current_golfer_model,
    with_grip_revision,
    write_grip_sidecar,
)
from src.shared.python.motion_matching.hip_calibration import hip_is_mirrored
from src.shared.python.motion_matching.full_body_spec import (
    BodySpec,
    ContactSpec,
    ContactSphere,
    JointSpec,
    LowerLimbExtension,
    MarkerAttachment,
    canonical_sha256,
    derive_full_body_spec,
    save_full_body_spec,
)
from src.shared.python.motion_matching.range_of_motion import (
    HUMAN_RANGES_DEG,
    UPPER_RANGES_DEG,
    as_document,
)
from src.shared.python.motion_matching.knee_axial_rotation import (
    DEFAULT_KNEE_AXIAL_RANGE_DEG,
    KNEE_AXIAL_KEY,
    apply_knee_axial_rotation,
)
from src.shared.python.motion_matching.tibial_torsion import (
    DEFAULT_TIBIAL_TORSION_OFFSET_DEG,
    apply_tibial_torsion,
)
from src.shared.python.motion_matching.tour_capture_contract import (
    MARKER_SEGMENTS,
    tracked_labels,
)

# Calibrated contact parameters from MS-20 (#10335) identifiability sweep
CONTACT_RECEIPT_SHA256 = (
    "9bbe5fc4e2fef05d0160af03f1214cc2768068fe0c6f3c71e9cae11c4a7e9556"
)
CONTACT_STIFFNESS_N_M = 1.0e5
CONTACT_DISSIPATION_S_M = 0.5
CONTACT_STATIC_FRICTION = 0.6
CONTACT_DYNAMIC_FRICTION = 0.4
CONTACT_VISCOUS_FRICTION = 0.01
CONTACT_TRANSITION_VELOCITY_M_S = 0.01
SPHERES = (
    ("heel", (0.01, -0.005, 0.0), 0.035),
    ("forefoot", (0.16, -0.005, 0.0), 0.03),
    ("toe", (0.23, -0.010, 0.0), 0.025),
)

LEG_BODIES = ("femur", "tibia", "talus", "calcn", "toes")
LEG_MARKER_BODIES = {
    "KneeOut": "femur",
    "AnkleOut": "tibia",
    "ToeIn": "calcn",
    "ToeOut": "calcn",
}

HIP_PERMUTATION = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
KNEE_PERMUTATION = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
_LIFT_UPPER_RANGES_DEG: dict[str, tuple[float, float]] = {
    **UPPER_RANGES_DEG,
    "LScapInputX": (-10.0, 30.0),
    "RScapInputX": (-10.0, 30.0),
    "LScapInputY": (-40.0, 40.0),
    "RScapInputY": (-40.0, 40.0),
    "LWInputX": (-40.0, 25.0),
    "RWInputX": (-40.0, 25.0),
    "LWInputY": (-70.0, 70.0),
    "RWInputY": (-70.0, 70.0),
}


def body_fixed_xyz(angles: np.ndarray) -> np.ndarray:
    """SimTK BodyFixed XYZ Euler rotation: R = Rx(a) Ry(b) Rz(c)."""
    a, b, c = angles

    def rx(t: float) -> np.ndarray:
        return np.array(
            [[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]]
        )

    def ry(t: float) -> np.ndarray:
        return np.array(
            [[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]]
        )

    def rz(t: float) -> np.ndarray:
        return np.array(
            [[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]]
        )

    return rx(a) @ ry(b) @ rz(c)


def transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = rotation
    t[:3, 3] = translation
    return t


def _offset_frames(model: Any) -> dict[str, tuple[str, Any]]:
    """Map body-owned frame paths (``/bodyset/<body>/<frame>``) to (body, frame)."""
    frames = {}
    for body in model.findall("BodySet/objects/Body"):
        name = body.get("name")
        for frame in body.findall("components/PhysicalOffsetFrame"):
            frames[f"/bodyset/{name}/{frame.get('name')}"] = (name, frame)
    return frames


def _joint_frame(
    joint: Any, socket: str, owned: dict[str, tuple[str, Any]]
) -> tuple[str, Any]:
    """Resolve a joint socket to (body name, PhysicalOffsetFrame element).

    Accepts both layouts OpenSim 4 writes: a frame declared inside the joint's
    ``<frames>`` (its ``socket_parent`` names the body) and a frame owned by a
    body and referenced by absolute path.
    """
    path = joint.findtext(socket)
    for frame in joint.findall("frames/PhysicalOffsetFrame"):
        if frame.get("name") == path:
            return frame.findtext("socket_parent").rsplit("/", 1)[1], frame
    if path in owned:
        return owned[path]
    raise ValueError(f"Joint {joint.get('name')!r}: cannot resolve {socket}={path!r}")


def _vector(frame: Any, tag: str) -> np.ndarray:
    return np.array(frame.findtext(tag).split(), dtype=float)


def read_osim(osim: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    model = ET.parse(str(osim)).getroot().find("Model")
    if model is None:
        raise ValueError(f"No Model tag found in OpenSim file {osim}")
    bodies = {}
    for body in model.findall("BodySet/objects/Body"):
        bodies[body.get("name")] = {
            "mass": float(body.findtext("mass")),  # type: ignore[arg-type]
            "com": np.array(body.findtext("mass_center").split(), dtype=float),  # type: ignore[union-attr]
            "inertia": np.array(body.findtext("inertia").split(), dtype=float),  # type: ignore[union-attr]
        }
    owned = _offset_frames(model)
    joints = {}
    for joint in model.findall("JointSet/objects/*"):
        parent_body, parent = _joint_frame(joint, "socket_parent_frame", owned)
        child_body, child = _joint_frame(joint, "socket_child_frame", owned)
        joints[joint.get("name")] = {
            "parent_body": parent_body,
            "child_body": child_body,
            "parent_translation": _vector(parent, "translation"),
            "parent_orientation": _vector(parent, "orientation"),
            "child_translation": _vector(child, "translation"),
            "child_orientation": _vector(child, "orientation"),
        }
    return bodies, joints


#: Conjugating a hip's base and follower frames by ``Rx(pi)`` turns
#: ``Rx(a) Ry(b) Rz(c)`` into ``Rx(a) Ry(-b) Rz(-c)``: OpenSim's mirrored side,
#: unchanged at the zero pose.
HIP_MIRROR = transform(np.diag([1.0, -1.0, -1.0]), np.zeros(3))

#: Neutral name for the same ``Rx(pi)`` conjugation (also used for the knee).
AXIS_MIRROR = HIP_MIRROR


def hip_axis_signs(osim: Path) -> dict[str, float]:
    """Per side, the OpenSim coefficient on hip adduction and rotation (+1 or -1).

    DbC Postcondition: flexion has coefficient +1 and adduction and rotation
    share one sign on each side; otherwise ``ValueError`` (the spec's three
    primitives cannot express it).
    """
    model = ET.parse(str(osim)).getroot().find("Model")
    if model is None:
        raise ValueError(f"No Model tag found in OpenSim file {osim}")
    signs: dict[str, float] = {}
    for side in ("r", "l"):
        joint = model.find(f"JointSet/objects/CustomJoint[@name='hip_{side}']")
        if joint is None:
            raise ValueError(f"{osim} has no CustomJoint hip_{side}")
        coefficient = {}
        for axis in joint.findall("SpatialTransform/TransformAxis"):
            name = axis.findtext("coordinates", "").strip()
            values = axis.findtext("LinearFunction/coefficients", "1 0").split()
            coefficient[name] = float(values[0])
        flexion = coefficient.get(f"hip_flexion_{side}")
        pair = {
            coefficient.get(f"hip_adduction_{side}"),
            coefficient.get(f"hip_rotation_{side}"),
        }
        sign = pair.pop() if len(pair) == 1 else None
        if flexion != 1.0 or sign is None or sign not in (1.0, -1.0):
            raise ValueError(f"Unsupported hip_{side} coefficients: {coefficient}")
        signs[side] = sign
    return signs


def mirror_document_hip(document: dict[str, Any], side: str) -> bool:
    """Conjugate ``hip_<side>`` in a spec document by :data:`HIP_MIRROR`, in place.

    Idempotent: returns ``False`` (no change) when the hip is already mirrored.
    """
    joint = next((j for j in document["joints"] if j["name"] == f"hip_{side}"), None)
    if joint is None:
        raise ValueError(f"Document has no hip_{side} joint")
    if hip_is_mirrored(joint):
        return False
    for key in ("parent_to_base", "child_to_follower"):
        joint[key] = (np.asarray(joint[key], dtype=float) @ HIP_MIRROR).tolist()
    return True


def knee_follows_convention(joint: Mapping[str, Any]) -> bool:
    """True when a knee hinge axis points to the subject's right (+z).

    The spec's knee follows the gait2392 convention: the base ``z`` column
    (the ``Rz`` axis in the femur frame) has a positive ``z`` component on both
    sides, so a positive angle is extension and flexion is negative.
    """
    base = np.asarray(joint["parent_to_base"], dtype=float)
    return bool(base[2, 2] > 0.0)


def mirror_document_knee(document: dict[str, Any], side: str) -> bool:
    """Conjugate ``knee_<side>`` in a spec document by :data:`AXIS_MIRROR`, in place.

    Idempotent: returns ``False`` (no change) when the knee already follows the
    convention. Postcondition: ``knee_follows_convention`` holds for the joint
    and the zero pose is unchanged.

    Raises:
        ValueError: ``side`` is not ``"r"``/``"l"`` or the joint is missing.
    """
    if side not in ("r", "l"):
        raise ValueError(f"Unknown side for knee_{side}: expected 'r' or 'l'")
    joint = next((j for j in document["joints"] if j["name"] == f"knee_{side}"), None)
    if joint is None:
        raise ValueError(f"Document has no knee_{side} joint")
    if knee_follows_convention(joint):
        return False
    for key in ("parent_to_base", "child_to_follower"):
        joint[key] = (np.asarray(joint[key], dtype=float) @ AXIS_MIRROR).tolist()
    return True


def _frame(entry: dict[str, Any], which: str) -> np.ndarray:
    return transform(
        body_fixed_xyz(entry[f"{which}_orientation"]), entry[f"{which}_translation"]
    )


def leg_extension(
    bodies: dict[str, Any],
    joints: dict[str, Any],
    pelvis_body: str,
    hip_frame_in_pelvis: np.ndarray,
    mirrored_hips: frozenset[str] = frozenset(),
) -> tuple[LowerLimbExtension, list[str]]:
    """Rajagopal legs as spec bodies and joints.

    ``mirrored_hips`` lists the sides ("r"/"l") whose OpenSim hip adduction and
    rotation coefficients are -1 (see :func:`hip_axis_signs`); their hip frames
    are conjugated by :data:`HIP_MIRROR` so the spec keeps OpenSim's signs.
    """
    if not mirrored_hips <= {"r", "l"}:
        raise ValueError(f"mirrored_hips must be a subset of r/l: {mirrored_hips}")
    specs: list[BodySpec] = []
    edges: list[JointSpec] = []
    notes = [
        "patella bodies and patellofemoral coupler constraints dropped",
        "walker knee reduced to a hinge about its primary axis; coupled spline "
        "translations ignored",
        "OpenSim body inertia interpreted as about the mass centre in the body frame",
    ]
    for side in ("r", "l"):
        for name in LEG_BODIES:
            b = bodies[f"{name}_{side}"]
            specs.append(
                BodySpec(
                    f"{name}_{side}", b["mass"], tuple(b["com"]), tuple(b["inertia"])
                )
            )
        hip = joints[f"hip_{side}"]
        axes = transform(HIP_PERMUTATION, np.zeros(3))
        if side in mirrored_hips:
            axes = axes @ HIP_MIRROR
        edges.append(
            JointSpec(
                f"hip_{side}",
                pelvis_body,
                f"femur_{side}",
                (hip_frame_in_pelvis @ _frame(hip, "parent") @ axes).tolist(),
                (_frame(hip, "child") @ axes).tolist(),
                ("Rx", "Ry", "Rz"),
                (
                    f"hip_flexion_{side}",
                    f"hip_adduction_{side}",
                    f"hip_rotation_{side}",
                ),
            )
        )
        knee = joints[f"walker_knee_{side}"]
        knee_axes = transform(KNEE_PERMUTATION, np.zeros(3))
        # Geometric rule, independent of the OpenSim axis sign: the hinge axis
        # (base z column) must point to +z on both sides (flexion negative).
        if (_frame(knee, "parent") @ knee_axes)[2, 2] <= 0.0:
            knee_axes = knee_axes @ AXIS_MIRROR
        edges.append(
            JointSpec(
                f"knee_{side}",
                f"femur_{side}",
                f"tibia_{side}",
                (_frame(knee, "parent") @ knee_axes).tolist(),
                (_frame(knee, "child") @ knee_axes).tolist(),
                ("Rz",),
                (f"knee_angle_{side}",),
            )
        )
        for joint_name, parent, child, coordinate in (
            (f"ankle_{side}", f"tibia_{side}", f"talus_{side}", f"ankle_angle_{side}"),
            (
                f"subtalar_{side}",
                f"talus_{side}",
                f"calcn_{side}",
                f"subtalar_angle_{side}",
            ),
            (f"mtp_{side}", f"calcn_{side}", f"toes_{side}", f"mtp_angle_{side}"),
        ):
            pin = joints[joint_name]
            edges.append(
                JointSpec(
                    joint_name,
                    parent,
                    child,
                    _frame(pin, "parent").tolist(),
                    _frame(pin, "child").tolist(),
                    ("Rz",),
                    (coordinate,),
                )
            )
    return (
        LowerLimbExtension(
            specs,
            edges,
            provenance="Rajagopal et al. 2016 via golf_humanoid.osim (OpenSense variant); "
            + "; ".join(notes),
        ),
        notes,
    )


def marker_seeds(
    hub_height_m: float,
    clav_m: float,
    upper_arm_m: float,
    forearm_m: float,
    native: dict[str, Any],
    club_offsets: Mapping[str, Sequence[float]] | None = None,
) -> dict[str, MarkerAttachment]:
    """Anatomical seed offsets in the new frames (x forward, y left, z up).

    The ``Clubhead`` cluster comes from ``native`` (the driver-fitted
    candidate) unless ``club_offsets`` gives the club's own cluster (#12210);
    its labels must equal the native cluster's (``ValueError`` otherwise).
    """
    club = {
        label: MarkerAttachment(native_body, tuple(offset))
        for label, native_body, offset in (
            (lbl, native["marker_bodies"][i], native["marker_offsets_m"][i])
            for i, lbl in enumerate(native["marker_labels"])
        )
        if native_body == CLUB_MARKER_BODY
    }
    if club_offsets is not None:
        if set(club_offsets) != set(club):
            raise ValueError(
                f"club marker labels {sorted(club_offsets)} must equal the "
                f"native {CLUB_MARKER_BODY} cluster {sorted(club)}"
            )
        club = {
            label: MarkerAttachment(
                CLUB_MARKER_BODY,
                (
                    float(club_offsets[label][0]),
                    float(club_offsets[label][1]),
                    float(club_offsets[label][2]),
                ),
            )
            for label in club
        }
    seeds: dict[str, MarkerAttachment] = {
        "WaistLeft": MarkerAttachment("Hip", (0.02, 0.15, 0.08)),
        "WaistRight": MarkerAttachment("Hip", (0.02, -0.15, 0.08)),
        "WaistLBack": MarkerAttachment("Hip", (-0.10, 0.07, 0.10)),
        "WaistRBack": MarkerAttachment("Hip", (-0.10, -0.07, 0.10)),
        "BackTop": MarkerAttachment("Spine", (-0.10, 0.0, hub_height_m + 0.02)),
        "BackLeft": MarkerAttachment("Spine", (-0.11, 0.09, hub_height_m * 0.7)),
        "BackRight": MarkerAttachment("Spine", (-0.11, -0.09, hub_height_m * 0.7)),
        "HeadTop": MarkerAttachment("Head", (0.0, 0.0, 0.24)),
        "HeadFront": MarkerAttachment("Head", (0.10, 0.0, 0.16)),
        "HeadSide": MarkerAttachment("Head", (0.0, 0.08, 0.16)),
        "LShoulderTop": MarkerAttachment("LScap", (0.0, clav_m, 0.05)),
        "LShoulderBack": MarkerAttachment("LScap", (-0.06, clav_m - 0.01, 0.02)),
        "RShoulderTop": MarkerAttachment("RScap", (0.0, -clav_m, 0.05)),
        "RShoulderBack": MarkerAttachment("RScap", (-0.06, -clav_m + 0.01, 0.02)),
        "LUArmHigh": MarkerAttachment("LS", (0.01, 0.04, -upper_arm_m * 0.35)),
        "LElbowOut": MarkerAttachment("LS", (0.0, 0.04, -upper_arm_m)),
        "RUArmHigh": MarkerAttachment("RS", (0.01, -0.04, -upper_arm_m * 0.35)),
        "RElbowOut": MarkerAttachment("RS", (0.0, -0.04, -upper_arm_m)),
        "LWristTop": MarkerAttachment("LF", (0.03, 0.0, -forearm_m / 2 + 0.01)),
        "RWristTop": MarkerAttachment("RF", (0.03, 0.0, -forearm_m / 2 + 0.01)),
        **club,
    }
    for label in MARKER_SEGMENTS["left_leg"] + MARKER_SEGMENTS["right_leg"]:
        side = "l" if label.startswith("L") else "r"
        seeds[label] = MarkerAttachment(f"{LEG_MARKER_BODIES[label[1:]]}_{side}", None)
    missing = set(tracked_labels()) - set(seeds)
    if missing:
        raise ValueError(f"No seed for {sorted(missing)}")
    return {label: seeds[label] for label in tracked_labels()}


def scale_leg_mass(
    bodies: Mapping[str, dict[str, Any]], leg_mass_kg: float
) -> dict[str, dict[str, Any]]:
    """Scale the ``read_osim`` leg bodies to a total of ``leg_mass_kg`` (#12199).

    Every leg body's mass and inertia take one factor, so the Rajagopal mass
    distribution, centres of mass and geometry are kept (uniform density
    scaling). Other bodies are returned unchanged.

    Raises:
        ValueError: ``leg_mass_kg`` is not a positive finite number.
    """
    if not np.isfinite(leg_mass_kg) or leg_mass_kg <= 0:
        raise ValueError(f"leg mass must be positive and finite, got {leg_mass_kg}")
    legs = {f"{name}_{side}" for name in LEG_BODIES for side in ("r", "l")}
    # fsum is correctly rounded: builtin sum differs on 3.11 and 3.12+.
    factor = leg_mass_kg / math.fsum(bodies[name]["mass"] for name in legs)
    return {
        name: (
            {**body, "mass": body["mass"] * factor, "inertia": body["inertia"] * factor}
            if name in legs
            else body
        )
        for name, body in bodies.items()
    }


def _body_mass_without_club(document: Mapping[str, Any]) -> float:
    solids = math.fsum(s["mass_kg"] for b in document["bodies"] for s in b["solids"])
    return float(solids - document["club"]["total_mass_kg"])


def _build_aligned_leg_extension(
    osim_path: Path,
    stature_m: float,
    hip_frame: dict[str, Any],
    leg_mass_kg: float | None = None,
    settings: Mapping[str, Any] | None = None,
) -> tuple[Any, list[str], float]:
    """Aligned legs, the conversion notes and the hip half spacing, m.

    ``settings`` (#15296, each optional, m): ``thigh_length_m`` and
    ``shank_length_m`` are ``osim_leg_lengths.scale_leg_lengths``'s, and
    ``hip_half_spacing_m`` is half the distance between the hip joint centres
    instead of ``pelvis_alignment_for``'s stature fraction. Without them the
    legs are the OpenSim file's stored lengths, which do not follow stature.
    """
    settings = settings or {}
    thigh, shank = settings.get("thigh_length_m"), settings.get("shank_length_m")
    bodies, joints = scale_leg_lengths(*read_osim(osim_path), thigh, shank)
    if leg_mass_kg is not None:
        bodies = scale_leg_mass(bodies, leg_mass_kg)
    rotation, hip_half = pelvis_alignment_for(stature_m)
    if settings.get("hip_half_spacing_m") is not None:
        hip_half = positive_length_m(
            "hip_half_spacing_m", settings["hip_half_spacing_m"]
        )
    stated = "".join(
        f"; {name} scaled to {value:.4f} m on both sides"
        for name, value in (("thigh", thigh), ("shank", shank))
        if value is not None
    )
    hips = np.array(
        [joints["hip_r"]["parent_translation"], joints["hip_l"]["parent_translation"]]
    )
    mid = rotation @ hips.mean(axis=0)
    alignment = transform(rotation, -mid)
    extension, notes = leg_extension(
        dict(bodies),
        dict(joints),
        hip_frame["body"],
        np.asarray(hip_frame["placement"]) @ alignment,
        frozenset(s for s, sign in hip_axis_signs(osim_path).items() if sign < 0),
    )

    edges = []
    for joint in extension.joints:
        if joint.name in ("hip_r", "hip_l"):
            base = np.asarray(joint.parent_to_base, dtype=float)
            base[:3, 3] = [0.0, -hip_half if joint.name == "hip_r" else hip_half, 0.0]
            joint = type(joint)(
                joint.name,
                joint.parent,
                joint.child,
                base.tolist(),
                joint.child_to_follower,
                joint.primitives,
                joint.coordinates,
            )
        edges.append(joint)
    extension = type(extension)(
        extension.bodies,
        edges,
        extension.provenance
        + "; hip centres at the subject hip half width on the pelvis frame"
        + stated,
    )
    return extension, notes, hip_half


def _create_contact_spec() -> ContactSpec:
    return ContactSpec(
        law="hunt_crossley_coulomb",
        parameters={
            "stiffness_n_m": CONTACT_STIFFNESS_N_M,
            "dissipation_s_m": CONTACT_DISSIPATION_S_M,
            "static_friction": CONTACT_STATIC_FRICTION,
            "dynamic_friction": CONTACT_DYNAMIC_FRICTION,
            "viscous_friction": CONTACT_VISCOUS_FRICTION,
            "transition_velocity_m_s": CONTACT_TRANSITION_VELOCITY_M_S,
        },
        spheres=tuple(
            ContactSphere(f"{kind}_{side}", f"calcn_{side}", position, radius)
            for side in ("r", "l")
            for kind, position, radius in SPHERES
        ),
        ground_normal_policy="opposite_gravity",
        ground_height_m=None,
        provenance=(
            f"MS-20 (#10335) calibrated contact parameters; receipt SHA256 {CONTACT_RECEIPT_SHA256[:16]}"
        ),
    )


def _extract_marker_seeds(
    upper: dict[str, Any],
    native_candidate_path: Path,
    club_offsets: Mapping[str, Sequence[float]] | None = None,
) -> tuple[float, dict[str, Any]]:
    hub_frame = next(f for f in upper["frames"] if f["name"] == "Hub")
    hub_h = float(np.asarray(hub_frame["placement"])[2, 3])
    clav = float(
        np.linalg.norm(
            np.asarray(
                next(j for j in upper["joints"] if j["child"].endswith("LUpperArm"))[
                    "parent_to_base"
                ]
            )[:3, 3]
        )
    )
    upper_arm = float(
        np.linalg.norm(
            np.asarray(
                next(
                    j for j in upper["joints"] if j["child"].endswith("Spherical Solid")
                )["parent_to_base"]
            )[:3, 3]
        )
    )
    forearm = 2 * float(
        np.linalg.norm(
            np.asarray(
                next(
                    j for j in upper["joints"] if j["child"].endswith("LLowerForearm")
                )["parent_to_base"]
            )[:3, 3]
        )
    )
    candidate = json.loads(native_candidate_path.read_text(encoding="utf-8"))
    markers = marker_seeds(hub_h, clav, upper_arm, forearm, candidate, club_offsets)
    return hub_h, markers


def _club_marker_offsets(
    club: str, path: Path | str | None
) -> tuple[dict[str, Offset] | None, Path | None]:
    """The club's own ``Clubhead`` cluster, or ``None`` for the native one.

    Raises ``ValueError`` for a non-driver club without a cluster file, since
    the native candidate's cluster is the driver's (#12210).
    """
    resolved = get_club_markers(club, path)
    if resolved is None:
        if club != "driver":
            raise ValueError(
                f"no club marker file for {club!r}; the native candidate's "
                f"{CLUB_MARKER_BODY} cluster is the driver's"
            )
        return None, None
    markers = load_club_markers(resolved)
    if markers.club != club:
        raise ValueError(f"{resolved} is for {markers.club!r}, not {club!r}")
    return markers.offsets_m, resolved


def _seeded_markers(
    upper: dict[str, Any],
    extension: LowerLimbExtension,
    club: str,
    native_candidate_path: Path,
    kwargs: Mapping[str, Any],
) -> tuple[float, Mapping[str, MarkerAttachment], Path | None]:
    """Hub height, marker seeds and the club marker file of a build.

    ``marker_seed_overrides`` and ``dropped_markers`` (kwargs, #15296) are
    ``marker_seed_overrides.apply_marker_seed_overrides``'s; ``None`` keeps
    the builder's own seeds.
    """
    offsets, club_path = _club_marker_offsets(club, kwargs.get("club_markers_path"))
    hub_h, markers = _extract_marker_seeds(upper, native_candidate_path, offsets)
    joints = [*upper["joints"], *(j.as_document() for j in extension.joints)]
    return (
        hub_h,
        apply_marker_seed_overrides(
            markers,
            kwargs.get("marker_seed_overrides"),
            kwargs.get("dropped_markers"),
            joints,
            upper["frames"],
        ),
        club_path,
    )


def _write_outputs(
    document: dict[str, Any],
    output_dir: Path,
    spec_name: str,
    subject: dict[str, Any],
    input_paths: tuple[Path, ...],
    hip_half: float,
    hub_h: float,
    notes: list[str],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    spec_path = save_full_body_spec(document, output_dir / f"{spec_name}.json")
    receipt = {
        "spec_sha256": canonical_sha256(document),
        "spec_file_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        "de_leva_table_sha256": de_leva_table_sha256(),
        "subject": subject,
        "inputs": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in input_paths
        },
        "hip_half_width_m": hip_half,
        "hub_height_above_l5s1_m": hub_h,
        "total_mass_kg": float(
            sum(s["mass_kg"] for b in document["bodies"] for s in b["solids"])
        ),
        "simplifications": notes,
        "qualification": "anthropometric geometry; unqualified until Simscape parity (AN-1)",
    }
    receipt_suffix = spec_name.replace("full_body_spec_", "")
    receipt_path = output_dir / f"build_receipt_{receipt_suffix}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return spec_path, receipt_path


def _annotate_document(
    document: dict[str, Any], upper: dict[str, Any], mirror_arm: bool = True
) -> None:
    """Copy the subject, anthropometry, ranges and visual hints into ``document``.

    The build tables are native; ``mirror_arm`` (G1 #14746) mirrors the right-arm
    axes and limits, ``False`` keeps the native convention for consumers whose
    stored joint angles still use it (lifting).
    """
    document["subject"] = upper["subject"]
    document["de_leva_table_sha256"] = de_leva_table_sha256()
    document["anthropometry"] = {
        "table": "de_leva_1996_male",
        "de_leva_table_sha256": de_leva_table_sha256(),
        "segments": de_leva_table_dict(),
    }
    document["coordinate_ranges_deg"] = {
        **upper["coordinate_ranges_deg"],
        **as_document(
            {k: v for k, v in HUMAN_RANGES_DEG.items() if k[-2:] in ("_r", "_l")}
        ),
    }
    document["address_seed_deg"] = upper["address_seed_deg"]
    document["club"] = upper["club"]
    document["visual_hints"] = upper["visual_hints"]
    document["visual_hints"]["capsule_radius_m"].update(
        {
            f"{segment}_{side}": LEG_VISUAL_RADIUS_M[segment]
            for segment in LEG_VISUAL_RADIUS_M
            for side in ("r", "l")
        }
    )
    if mirror_arm:
        mirror_document_arm(document)


def with_shaft_model(
    document: dict[str, Any],
    club: str | ClubSpec,
    shaft_model: str | None,
    options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the ``club.shaft.model`` flag (#12686); ``"rigid"`` is the default.

    ``"rigid"`` (or ``None``) returns ``document`` itself, unchanged, so the
    default spec stays byte-identical; ``"chain"`` returns the flexible-shaft
    copy built by :func:`club_shaft_chain.apply_shaft_chain`. ``options``
    (#15150) are the ``build_shaft_chain`` flex, stiffness scale and link
    count (``checked_shaft_options``); empty gives the default chain, and the
    chain records them in ``club.shaft``.

    Raises:
        ValueError: ``shaft_model`` is not one of ``SHAFT_MODELS``, or the
            options are unknown, invalid or given for a rigid club.
    """
    model = DEFAULT_SHAFT_MODEL if shaft_model is None else shaft_model
    if model not in SHAFT_MODELS:
        raise ValueError(f"shaft_model must be one of {SHAFT_MODELS}, got {model!r}")
    chosen = checked_shaft_options(model, options)
    if model == "rigid":
        return document
    spec = recorded_club(document, club) if isinstance(club, str) else club
    return apply_shaft_chain(document, build_shaft_chain(spec, **chosen))


@dataclass(frozen=True)
class AnthropometricDocument:
    """A composed candidate document plus the values its receipt records."""

    document: dict[str, Any]
    subject: dict[str, Any]
    hip_half_m: float
    hub_height_m: float
    notes: list[str]


def _apply_kwarg_tibial_torsion(document: Any, kwargs: dict[str, Any]) -> Any:
    """Apply the static external tibial-torsion offset to both ankles.

    ``tibial_torsion_offset_deg`` (default
    ``tibial_torsion.DEFAULT_TIBIAL_TORSION_OFFSET_DEG``; R3b, #12789); ``0``
    keeps the generic Rajagopal leg bit-for-bit.
    """
    torsion = float(
        kwargs.get("tibial_torsion_offset_deg", DEFAULT_TIBIAL_TORSION_OFFSET_DEG)
    )
    return apply_tibial_torsion(document, {"r": torsion, "l": torsion})


def _provenance_notes(
    club: str, club_markers_path: Path | None, leg_mass_kg: float | None
) -> str:
    """Provenance suffix naming the club marker file and any leg mass scaling."""
    note = (
        f"; {club} club markers from {club_markers_path.name} (#12210)"
        if club_markers_path
        else ""
    )
    if leg_mass_kg is not None:
        note += (
            f"; Rajagopal legs scaled to {leg_mass_kg:.3f} kg so the body "
            "without the club matches the subject mass (#12199)"
        )
    return note


def _club_spec(club: str, kwargs: Mapping[str, Any]) -> ClubSpec:
    """The catalogue club, or the ``club_spec`` kwarg restating it (#15296)."""
    spec: ClubSpec = kwargs.get("club_spec") or CLUBS[club]
    if spec.name != club:
        raise ValueError(f"club_spec is a {spec.name!r}, not the {club!r} asked for")
    if kwargs.get("club_butt_to_sole_m") and "club_spec" not in kwargs:
        raise ValueError("club_butt_to_sole_m records a club_spec and needs one")
    return spec


def _record_preset(upper: dict[str, Any], kwargs: Mapping[str, Any]) -> None:
    """Name the preset in ``subject``; butt to sole length and the sole point
    (the address ground pin, ``checked_sole_point_m``) in ``club``."""
    if kwargs.get("subject_preset"):
        upper["subject"]["subject_preset"] = kwargs["subject_preset"]
    if kwargs.get("club_butt_to_sole_m"):
        upper["club"]["butt_to_sole_m"] = float(kwargs["club_butt_to_sole_m"])
    if kwargs.get("club_sole_point_m") is not None:
        point = checked_sole_point_m(kwargs["club_sole_point_m"])
        upper["club"]["sole_point_m"] = list(point)
        upper["club"]["sole_point_source"] = SOLE_POINT_SOURCE
    if kwargs.get("dropped_markers"):
        upper["subject"]["dropped_markers"] = sorted(kwargs["dropped_markers"])


def _apply_kwarg_model_options(
    document: Any, club_spec: Any, kwargs: dict[str, Any]
) -> Any:
    """Shaft model, tibial torsion and knee axial rotation from ``kwargs``.

    Without a ``shaft_model`` the composed club stays rigid here and
    :func:`with_current_golfer_model` gives it the current flexible shaft.
    """
    if kwargs.get("shaft_model") is not None:
        document = with_shaft_model(
            document, club_spec, kwargs["shaft_model"], kwargs.get("shaft_options")
        )
    document = _apply_kwarg_tibial_torsion(document, kwargs)
    return apply_knee_axial_rotation(document, kwargs.get(KNEE_AXIAL_KEY))


def compose_anthropometric_document(
    stature_m: float = 1.80,
    mass_kg: float = 80.0,
    club: str = "driver",
    *,
    native_path: Path | None = None,
    osim_path: Path | None = None,
    native_candidate_path: Path | None = None,
    **kwargs: Any,
) -> AnthropometricDocument:
    """Compose the candidate full-body document in memory (no files written).

    Single source for the spec content: ``build_anthropometric_spec`` writes
    it with a receipt and the character builder (CMB-1, #11652) compiles
    subject parameters through it. ``club_markers_path`` (kwarg) overrides
    the club's ``Clubhead`` marker file (``assets.get_club_markers``); a
    non-driver club without one raises ``ValueError``. Postcondition: the
    document passed ``validate_full_body_spec`` inside ``derive_full_body_spec``.

    ``match_subject_mass`` (kwarg, default ``False``): when ``False``, preserves
    the unscaled De Leva segment sum and pinned Simscape
    ``native_geometry_spec_9967.json`` hand masses (``0.5491357853472616 kg``
    per hand) for deterministic reference-capture parity. When ``True``, scales
    both the native hand solids (``0.0061 * mass_kg`` each) and the total character
    body mass so total subject mass equals ``mass_kg``, scaling the Rajagopal legs
    with :func:`scale_leg_mass` so body mass without the club equals ``mass_kg``
    (#12199, #12300); the legs carry what the upper body does not. Raises
    ``ValueError`` when the upper body alone reaches ``mass_kg``; see :func:`_apply_kwarg_tibial_torsion`.

    ``shoulder_girdle_mass`` (kwarg, default ``GIRDLE_MASS_ADDED``; #15150) is
    ``build_upper_body``'s rule for the clavicle links' mass, which validates it
    (``ValueError`` naming ``GIRDLE_MASS_RULES``) and records a non-default rule
    in ``subject``, so in the document and the build receipt.
    ``subject_preset.SubjectPreset`` kwargs (#15296): ``club_spec`` restates
    ``club`` (same name, ``ValueError`` otherwise),
    ``shoulder_below_cervicale_m`` is ``build_upper_body``'s; :func:`_seeded_markers`.
    ``knee_axial_rotation_deg`` (default off) is ``apply_knee_axial_rotation``'s.
    """
    native_path = native_path or get_native_geometry_spec(None)
    osim_path = osim_path or get_opensim_model(None)
    native_candidate_path = native_candidate_path or get_candidate_geometry_spec(None)
    match_subject_mass = bool(kwargs.get("match_subject_mass", False))
    upper_ranges = kwargs.get("upper_ranges_deg") or (
        _LIFT_UPPER_RANGES_DEG if not kwargs.get("mirror_arm", True) else None
    )
    native = json.loads(native_path.read_text(encoding="utf-8"))
    club_spec = _club_spec(club, kwargs)
    upper = build_upper_body(
        native,
        stature_m=stature_m,
        mass_kg=mass_kg,
        trunk_scale=float(kwargs.get("trunk_scale", 1.0)),
        arm_scale=float(kwargs.get("arm_scale", 1.0)),
        shoulder_scale=float(kwargs.get("shoulder_scale", 1.0)),
        club=club_spec,
        grip_roll_deg=float(kwargs.get("grip_roll_deg", 0.0)),
        grip_rotation_deg={
            "L": kwargs.get("lead_grip_rotation") or GRIP_ROTATION_DEG["L"],
            "R": kwargs.get("trail_grip_rotation") or GRIP_ROTATION_DEG["R"],
        },
        scale_hand_mass=match_subject_mass,
        upper_ranges_deg=upper_ranges,
        shoulder_girdle_mass=kwargs.get("shoulder_girdle_mass", GIRDLE_MASS_ADDED),
        shoulder_below_cervicale_m=kwargs.get("shoulder_below_cervicale_m"),
    )
    _record_preset(upper, kwargs)
    hip_frame = next(f for f in upper["frames"] if f["name"] == "Hip")
    leg_mass_kg = (
        mass_kg - _body_mass_without_club(upper) if match_subject_mass else None
    )
    extension, notes, hip_half = _build_aligned_leg_extension(
        osim_path, stature_m, hip_frame, leg_mass_kg, kwargs
    )
    contact = _create_contact_spec()
    hub_h, markers, club_markers_path = _seeded_markers(
        upper, extension, club, native_candidate_path, kwargs
    )
    club_note = _provenance_notes(club, club_markers_path, leg_mass_kg)
    document = derive_full_body_spec(
        upper,
        extension,
        contact,
        markers,
        provenance=(
            f"anthropometric full-body candidate v1 at {stature_m:.3f} m, {mass_kg:.1f} kg "
            "(spec_builder.py; AN-1 #10099); upper body unqualified; "
            "marker offsets are anatomical seeds to be calibrated" + club_note
        ),
    )
    _annotate_document(document, upper, kwargs.get("mirror_arm", True))
    document = _apply_kwarg_model_options(document, club_spec, kwargs)
    return AnthropometricDocument(
        document=document,
        subject=upper["subject"],
        hip_half_m=hip_half,
        hub_height_m=hub_h,
        notes=notes,
    )


def build_anthropometric_spec(
    *,
    native_path: Path,
    osim_path: Path,
    native_candidate_path: Path,
    stature_m: float,
    mass_kg: float,
    output_dir: Path,
    club: str = "driver",
    **kwargs: Any,
) -> tuple[Path, Path]:
    """Build candidate spec and receipt, writing both to output_dir.

    ``handedness="left"`` writes the sagittal mirror of the right-handed
    current-model spec (#15652).
    """
    name = kwargs.get("name")
    spec_name = name or f"full_body_spec_anthro_{club}"
    built = compose_anthropometric_document(
        native_path=native_path,
        osim_path=osim_path,
        native_candidate_path=native_candidate_path,
        stature_m=stature_m,
        mass_kg=mass_kg,
        club=club,
        **kwargs,
    )
    club_markers = get_club_markers(club, kwargs.get("club_markers_path"))
    document, subject = with_current_golfer_model(
        *with_grip_revision(
            built.document, built.subject, kwargs.get("grip_overrides")
        ),
        shaft_model=kwargs.get("shaft_model"),
        shaft_options=kwargs.get("shaft_options"),
        handedness=kwargs.get("handedness") or "right",
    )
    paths = _write_outputs(
        document=document,
        output_dir=output_dir,
        spec_name=spec_name,
        subject=subject,
        input_paths=(native_path, osim_path, native_candidate_path)
        + ((club_markers,) if club_markers else ()),
        hip_half=built.hip_half_m,
        hub_h=built.hub_height_m,
        notes=built.notes,
    )
    write_grip_sidecar(paths[0], kwargs.get("grip_overrides"))
    return paths


def build_parser(preset: Mapping[str, Any] | None = None) -> argparse.ArgumentParser:
    """The command line; ``preset`` values replace the defaults they name."""
    preset = dict(preset or {})
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subject-preset",
        default=None,
        help="evidence/subject_presets/<name>.json or a file; flags override it",
    )
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--osim", type=Path, required=True)
    parser.add_argument("--native-candidate", type=Path, required=True)
    parser.add_argument("--stature", type=float, required="stature" not in preset)
    parser.add_argument("--mass", type=float, required="mass" not in preset)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trunk-scale", type=float, default=1.0)
    parser.add_argument("--arm-scale", type=float, default=1.0)
    parser.add_argument("--shoulder-scale", type=float, default=1.0)
    parser.add_argument("--club", choices=sorted(CLUBS), default="driver")
    parser.add_argument("--handedness", choices=("right", "left"), default="right")
    parser.add_argument(
        "--grip-roll", type=float, default=0.0, help="hand roll about the shaft, deg"
    )
    parser.add_argument(
        "--lead-grip-rotation",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=None,
        help="lead hand rotation on wrist extrinsic x-y-z deg",
    )
    parser.add_argument(
        "--trail-grip-rotation",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=None,
        help="trail hand rotation on wrist",
    )
    parser.add_argument(
        "--name", default=None, help="default full_body_spec_anthro_<club>"
    )
    parser.add_argument(
        "--shaft-model",
        choices=SHAFT_MODELS,
        default=None,
        help=(
            "club.shaft.model; default: the current golfer model's flexible "
            "chain shaft (#15684); rigid is a ballpark fallback only"
        ),
    )
    add_shaft_arguments(parser)
    parser.add_argument(
        "--tibial-torsion",
        type=float,
        default=DEFAULT_TIBIAL_TORSION_OFFSET_DEG,
        help="static external tibial-torsion offset per ankle, deg (#12789)",
    )
    parser.add_argument(
        "--club-markers",
        type=Path,
        default=None,
        help="club-markers/1 file (default evidence/club_markers/<club>.json)",
    )
    parser.add_argument(
        "--shoulder-girdle-mass",
        choices=GIRDLE_MASS_RULES,
        default=GIRDLE_MASS_ADDED,
        help="where the clavicle links' mass comes from (#15279, #15150)",
    )
    parser.add_argument(
        "--knee-axial-rotation",
        type=float,
        nargs="?",
        const=DEFAULT_KNEE_AXIAL_RANGE_DEG[1],
        default=None,
        help="opt-in knee axial rotation, +- deg (off unless given; #15296)",
    )
    add_grip_arguments(parser, physical=True)
    parser.set_defaults(**preset)
    return parser


# Command-line destination -> builder kwarg a subject preset gives a default for.
_PRESET_FLAGS = {"stature": "stature_m", "mass": "mass_kg", "club": "club"} | {
    "club_markers": "club_markers_path",
    **{k: k for k in ("trunk_scale", "arm_scale", "shoulder_scale")},
    "knee_axial_rotation": KNEE_AXIAL_KEY,
}


def _preset_kwargs(argv: Sequence[str] | None) -> dict[str, Any]:
    """Builder kwargs of the ``--subject-preset`` named in ``argv``, if any."""
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--subject-preset", default=None)
    name = probe.parse_known_args(argv)[0].subject_preset
    return load_subject_preset(name).builder_kwargs() if name else {}


def main(argv: Sequence[str] | None = None) -> int:
    preset = _preset_kwargs(argv)
    flags = {k: preset.pop(v) for k, v in _PRESET_FLAGS.items() if preset}
    parser = build_parser(flags)
    args = parser.parse_args(argv)
    if flags and args.club != flags["club"]:
        parser.error(f"--club must be the preset's {flags['club']!r}")
    build_anthropometric_spec(
        native_path=args.native,
        osim_path=args.osim,
        native_candidate_path=args.native_candidate,
        stature_m=args.stature,
        mass_kg=args.mass,
        output_dir=args.output,
        trunk_scale=args.trunk_scale,
        arm_scale=args.arm_scale,
        shoulder_scale=args.shoulder_scale,
        club=args.club,
        grip_roll_deg=args.grip_roll,
        lead_grip_rotation=(
            tuple(args.lead_grip_rotation) if args.lead_grip_rotation else None
        ),
        trail_grip_rotation=(
            tuple(args.trail_grip_rotation) if args.trail_grip_rotation else None
        ),
        name=args.name,
        club_markers_path=args.club_markers,
        shaft_model=args.shaft_model,
        shaft_options=shaft_options(args),
        handedness=args.handedness,
        tibial_torsion_offset_deg=args.tibial_torsion,
        shoulder_girdle_mass=args.shoulder_girdle_mass,
        grip_overrides=overrides_from_args(args),
        **{KNEE_AXIAL_KEY: args.knee_axial_rotation},
        **preset,
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
