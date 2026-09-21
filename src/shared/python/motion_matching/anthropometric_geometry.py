"""Anthropometric native upper-body geometry built from a subject (AN-1, #10099).

The qualified native geometry places its only trunk joint at the base of the
neck and its shoulder hub above the shoulders. This builder writes a new
upper-body document with the same 27 coordinate names, the same body and
joint names, the same club, hands and grip closure, but an anatomical chain:

    world -(6 dof)-> pelvis (hip centres to L5/S1)
          -(torso Rz)-> UpperTorsoBase at L5/S1
          -(spine Rx, Ry)-> trunk to the shoulder centre
          -(neck Rx tilt, Ry nod, Rz turn at the cervicale)-> head
          -(scapula Rx elevation, Rz protraction at the hub)-> clavicle links
                                          to the shoulder joints
          -(shoulder Rx, Ry, Rz)-> upper arm -(elbow Ry)-> elbow ball
          -(forearm Rz)-> forearm -(wrist Rx, Ry)-> club / right-hand standoff

Zero pose: standing, x forward, y left, z up, arms straight forward. Lengths, masses,
centres of mass and inertias come from de Leva through ``anthropometry``.
The head is its own body on a three-axis neck at the cervicale (coordinates
``NeckInputX/Y/Z``, appended after the 27 native names; the Simscape model
has no neck, so this is a deliberate, documented departure). The scapula
coordinates keep their native names but the second primitive is
``Rz`` (protraction about the vertical), because a rotation about the link's
own axis would only spin the shoulder, duplicating the shoulder gimbal.
``COORDINATE_RANGES_DEG`` gives anatomical ranges in this document's sign
conventions (written to the document as ``coordinate_ranges_deg``) and
``ADDRESS_SEED_DEG`` a hands-forward start for pose solvers.
The document is unqualified until the Simscape model carries the same
numbers and R2025b frame parity is re-run; every consumer must say so.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.motion_matching.anthropometry import (
    DE_LEVA_MALE,
    de_leva_table_sha256,
    inertia_about_axis,
    segment_parameters,
)
from src.shared.python.motion_matching.club_models import DRIVER, ClubSpec, apply_club
from src.shared.python.motion_matching.grip_fit import grip_rotation
from src.shared.python.motion_matching.range_of_motion import UPPER_RANGES_DEG

Array = NDArray[np.float64]

PELVIS_HEIGHT_FRACTION = 0.23  # hip centres to L5/S1 as a fraction of trunk length
SHOULDER_BELOW_CERVICALE_M = 0.04
BIACROMIAL_FRACTION_OF_STATURE = 0.228
HIP_HALF_WIDTH_FRACTION_OF_STATURE = 0.052  # about 0.089 m at 1.71 m
COORDINATES = (
    "TranslationInputX", "TranslationInputY", "TranslationInputZ",
    "HipInputX", "HipInputY", "HipInputZ",
    "SpineInputX", "SpineInputY", "TorsoInput",
    "LEInput", "LFInput", "LScapInputX", "LScapInputY",
    "LSInputX", "LSInputY", "LSInputZ", "LWInputX", "LWInputY",
    "REInput", "RFInput", "RScapInputX", "RScapInputY",
    "RSInputX", "RSInputY", "RSInputZ", "RWInputX", "RWInputY",
)  # fmt: skip
NECK_COORDINATES = ("NeckInputX", "NeckInputY", "NeckInputZ")
HEAD_BODY = "GolfSwing3D_Kinetic/Head"
NECK_JOINT = "GolfSwing3D_Kinetic/Neck Joint"


# Anatomical ranges in this document's conventions come from the shared
# range-of-motion table; ball-joint Euler components stay unbounded.
COORDINATE_RANGES_DEG: dict[str, tuple[float, float]] = dict(UPPER_RANGES_DEG)
# Shoulder gimbal base: the upper arm points forward at zero pose so that the
# middle (Ry) rotation stays far from its +-90 deg singularity during a swing,
# where the arms hang forward-down at address and rise to about 45 deg above
# horizontal at the top; arms straight up or down never occur.
ARM_FORWARD = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
# Quarter turn about the forearm axis (z of the forearm frame, distal = -z)
# applied to the copied native wrist base: turns its cock axis from the pit
# direction onto the elbow axis so a positive cock lifts the hand toward the pit.
WRIST_ROLL = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
# Wrist primitives in the rolled base: Rx is radial/ulnar deviation (cock,
# positive toward the elbow pit) and Rz flexion/extension (about the palm
# normal). The native second primitive was a spin about the club, redundant
# with forearm pronation; a real wrist has no such freedom.
WRIST_PRIMITIVES = ("Rx", "Rz")
# A neutral wrist holds the shaft this far toward the little-finger (ulnar)
# side of the forearm line, so the wrist coordinates read zero in the grip
# position golfers call neutral. Applied as a fixed turn about the cock axis
# between the wrist base and the hand.
GRIP_ULNAR_OFFSET_DEG = 25.0
# Roll of the hands about the shaft relative to the native hand frames, the
# one free parameter of the hand-club attachment that the markers do not fix;
# calibrated so the wrist coordinates read human ranges over the swing
# (MM-2, #10104). Zero keeps the native roll.
GRIP_ROLL_DEG = 0.0
#: Constant rotation of each hand body on its wrist follower frame (extrinsic
#: x-y-z, degrees), fitted from the tour-average driver and 7-iron matches so
#: that the wrist cock, flexion and forearm pronation stay within the human
#: ranges over both swings (grip_fit.fit_grip_rotation; receipt
#: evidence/anthropometry/fit_grip_rotation_receipt.json). The capture has no
#: hand markers, so this is the one constant of the hand-club chain the
#: markers cannot fix directly. Lead ``L`` (club body), trail ``R`` (standoff).
GRIP_ROTATION_DEG: dict[str, tuple[float, float, float]] = {
    "L": (-90.0, 47.0, 0.0),
    "R": (-34.0, 46.0, 62.0),
}
ADDRESS_SEED_DEG: dict[str, float] = {  # arms 45 deg below horizontal, elbows soft
    "LSInputY": 45.0,
    "RSInputY": 45.0,
    "LEInput": -20.0,
    "REInput": -20.0,
}


def _grip_offset(
    child_to_follower: Any,
    grip_roll_deg: float = GRIP_ROLL_DEG,
    rotation_deg: Sequence[float] = (0.0, 0.0, 0.0),
) -> list[list[float]]:
    """Native hand-to-club transform with the grip roll about the shaft (base
    y), the neutral-grip ulnar turn about the wrist cock axis (base x) and the
    fitted constant hand rotation (``grip_fit``) applied on the base side."""
    extra = np.eye(4)
    extra[:3, :3] = grip_rotation(rotation_deg)
    angle = np.radians(GRIP_ULNAR_OFFSET_DEG)  # sign checked by the axis test
    c, s_ = np.cos(angle), np.sin(angle)
    roll = np.radians(grip_roll_deg)
    cr, sr = np.cos(roll), np.sin(roll)
    turn = np.eye(4)
    turn[:3, :3] = np.array([[1.0, 0.0, 0.0], [0.0, c, -s_], [0.0, s_, c]])
    about_shaft = np.eye(4)  # the child frame's y is the shaft axis
    about_shaft[:3, :3] = np.array([[cr, 0.0, sr], [0.0, 1.0, 0.0], [-sr, 0.0, cr]])
    # child_to_follower is expressed in the child (hand/club) frame, so a turn
    # in the follower (wrist base) frame composes on the right.
    c2f = np.asarray(child_to_follower, dtype=float)
    return (about_shaft @ c2f @ turn @ extra).tolist()


def _t(
    translation: Array | tuple[float, float, float], rotation: Array | None = None
) -> list[list[float]]:
    m = np.eye(4)
    if rotation is not None:
        m[:3, :3] = rotation
    m[:3, 3] = np.asarray(translation, dtype=float)
    return m.tolist()


def _solid(name: str, mass: float, com: Array, inertia: Array) -> dict[str, Any]:
    return {
        "name": name,
        "mass_kg": float(mass),
        "com_m": [float(v) for v in com],
        "inertia_com_kg_m2": np.asarray(inertia).tolist(),
        "placement": np.eye(4).tolist(),
    }


def _segment_solid(
    name: str,
    segment: str,
    stature: float,
    mass: float,
    start: Sequence[float] | Array,
    end: Sequence[float] | Array,
    share: float = 1.0,
) -> dict[str, Any]:
    """One de Leva solid for a segment running from ``start`` to ``end`` in the body frame."""
    params = segment_parameters(stature, mass, segment)
    start, end = np.asarray(start, float), np.asarray(end, float)
    axis = end - start
    length = float(np.linalg.norm(axis))
    axis /= length
    m = params.mass_kg * share
    com = start + axis * DE_LEVA_MALE[segment].com_fraction * length
    return _solid(
        name, m, com, inertia_about_axis(m, length, DE_LEVA_MALE[segment].radii, axis)
    )


def _copy(value: Any) -> Any:
    return json.loads(json.dumps(value))


def build_upper_body(
    native: Mapping[str, Any],
    *,
    stature_m: float,
    mass_kg: float,
    trunk_scale: float = 1.0,
    arm_scale: float = 1.0,
    shoulder_scale: float = 1.0,
    club: ClubSpec = DRIVER,
    grip_roll_deg: float = GRIP_ROLL_DEG,
    grip_rotation_deg: Mapping[str, Sequence[float]] | None = None,
) -> dict[str, Any]:
    """Return the anthropometric upper-body document for a subject.

    ``grip_rotation_deg`` maps ``"L"`` and ``"R"`` to the constant rotation
    of each hand on its wrist (default ``GRIP_ROTATION_DEG``).

    ``trunk_scale`` multiplies the de Leva trunk length (hips to shoulder
    centre), ``arm_scale`` the upper arm and forearm lengths and
    ``shoulder_scale`` the biacromial breadth: subjects depart from mean
    proportions and the capture decides these three.

    ``native`` supplies the body, joint and frame names, the club and
    right-hand standoff bodies, the wrist joint frames and the grip closure,
    all copied verbatim. Preconditions: positive stature and mass, native
    document with the 27 coordinates. Postconditions: same coordinate order,
    same body and joint names, total mass within 1 % of the de Leva sum for
    the bodies built here plus the copied club and hands.
    """
    if stature_m <= 0 or mass_kg <= 0:
        raise ValueError("Stature and mass must be positive")
    if trunk_scale <= 0 or arm_scale <= 0 or shoulder_scale <= 0:
        raise ValueError("Scale factors must be positive")
    if tuple(native["coordinate_order"]) != COORDINATES:
        raise ValueError("Native document must carry the 27 native coordinates")
    grip_rotation = {
        side: tuple(float(v) for v in angles)
        for side, angles in (grip_rotation_deg or GRIP_ROTATION_DEG).items()
    }
    if set(grip_rotation) != {"L", "R"} or any(
        len(v) != 3 for v in grip_rotation.values()
    ):
        raise ValueError("grip_rotation_deg needs three angles for L and R")
    names = {b["name"].rsplit("/", 1)[-1]: b["name"] for b in native["bodies"]}
    joints = {j["child"]: j for j in native["joints"]}
    jname = {j["child"].rsplit("/", 1)[-1]: j["name"] for j in native["joints"]}

    trunk = segment_parameters(stature_m, mass_kg, "trunk").length_m * trunk_scale
    pelvis_h = PELVIS_HEIGHT_FRACTION * trunk
    hub_h = trunk - SHOULDER_BELOW_CERVICALE_M - pelvis_h  # L5/S1 to shoulder centre
    cervicale_h = trunk - pelvis_h
    head_len = segment_parameters(stature_m, mass_kg, "head").length_m
    clav = BIACROMIAL_FRACTION_OF_STATURE * stature_m * shoulder_scale / 2
    upper_arm = segment_parameters(stature_m, mass_kg, "upper_arm").length_m * arm_scale
    forearm = segment_parameters(stature_m, mass_kg, "forearm").length_m * arm_scale

    bodies: list[dict[str, Any]] = [{"name": "world", "solids": []}]
    joints_out: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []

    def body(suffix: str, solids: list[dict[str, Any]]) -> str:
        full = names[suffix]
        bodies.append({"name": full, "solids": solids})
        return full

    def joint(
        child_suffix: str, parent: str, translation, primitives, rotation=None, c2f=None
    ) -> str:
        full = names[child_suffix]
        joints_out.append(
            {
                "name": jname[child_suffix],
                "parent": parent,
                "child": full,
                "parent_to_base": _t(translation, rotation),
                "child_to_follower": np.eye(4).tolist() if c2f is None else c2f,
                "primitives": [
                    {"primitive": p, "coordinate": c} for p, c in primitives
                ],
            }
        )
        return full

    def frame(name: str, body_full: str, translation) -> None:
        frames.append({"name": name, "body": body_full, "placement": _t(translation)})

    # Pelvis: origin at the hip-centre midpoint.
    pelvis = body(
        "LowerTorso",
        [
            _segment_solid(
                names["LowerTorso"] + "/pelvis",
                "lower_trunk",
                stature_m,
                mass_kg,
                (0, 0, 0),
                (0, 0, pelvis_h),
            )
        ],
    )
    joint(
        "LowerTorso",
        "world",
        (0, 0, 0),
        [
            ("Px", "TranslationInputX"),
            ("Py", "TranslationInputY"),
            ("Pz", "TranslationInputZ"),
            ("Rx", "HipInputX"),
            ("Ry", "HipInputY"),
            ("Rz", "HipInputZ"),
        ],
    )
    frame("Hip", pelvis, (0, 0, 0))
    # L5/S1: torso rotation then the spine universal joint, both at the same point.
    base = body(
        "UpperTorsoBase",
        [
            _solid(
                names["UpperTorsoBase"] + "/joint", 0.3, np.zeros(3), 1e-4 * np.eye(3)
            )
        ],
    )
    joint("UpperTorsoBase", pelvis, (0, 0, pelvis_h), [("Rz", "TorsoInput")])
    frame("Torso", base, (0, 0, 0))
    trunk_body = body(
        "COMRod",
        [
            _segment_solid(
                names["COMRod"] + "/middle_trunk",
                "middle_trunk",
                stature_m,
                mass_kg,
                (0, 0, 0),
                (0, 0, hub_h * 0.5),
            ),
            _segment_solid(
                names["COMRod"] + "/upper_trunk",
                "upper_trunk",
                stature_m,
                mass_kg,
                (0, 0, hub_h * 0.5),
                (0, 0, hub_h),
            ),
        ],
    )
    joint("COMRod", base, (0, 0, 0), [("Rx", "SpineInputX"), ("Ry", "SpineInputY")])
    frame("Spine", trunk_body, (0, 0, 0))
    frame("Hub", trunk_body, (0, 0, hub_h))
    # Head on a three-axis neck at the cervicale (not in the Simscape model).
    names["Head"] = HEAD_BODY
    jname["Head"] = NECK_JOINT
    head = body(
        "Head",
        [
            _segment_solid(
                HEAD_BODY + "/head",
                "head",
                stature_m,
                mass_kg,
                (0, 0, 0),
                (0, 0, head_len),
            )
        ],
    )
    joint(
        "Head",
        trunk_body,
        (0, 0, cervicale_h),
        [("Rx", "NeckInputX"), ("Ry", "NeckInputY"), ("Rz", "NeckInputZ")],
    )
    frame("Head", head, (0, 0, 0))
    for side, sign in (("L", 1.0), ("R", -1.0)):
        link = body(
            f"Hubto{side}S",
            [
                _segment_solid(
                    names[f"Hubto{side}S"] + "/clavicle",
                    "upper_arm",
                    stature_m,
                    mass_kg,
                    (0, 0, 0),
                    (0, sign * clav, 0),
                    share=0.2,
                )
            ],
        )
        joint(
            f"Hubto{side}S",
            trunk_body,
            (0, 0, hub_h),
            [("Rx", f"{side}ScapInputX"), ("Rz", f"{side}ScapInputY")],
        )
        frame(f"{side}Scap", link, (0, 0, 0))
        arm = body(
            f"{side}UpperArm",
            [
                _segment_solid(
                    names[f"{side}UpperArm"] + "/upper_arm",
                    "upper_arm",
                    stature_m,
                    mass_kg,
                    (0, 0, 0),
                    (0, 0, -upper_arm),
                )
            ],
        )
        joint(
            f"{side}UpperArm",
            link,
            (0, sign * clav, 0),
            [
                ("Rx", f"{side}SInputX"),
                ("Ry", f"{side}SInputY"),
                ("Rz", f"{side}SInputZ"),
            ],
            rotation=ARM_FORWARD,
        )
        frame(f"{side}S", arm, (0, 0, 0))
        ball_suffix = "Spherical Solid" if side == "L" else "Spherical Solid1"
        ball = body(
            ball_suffix,
            [
                _segment_solid(
                    names[ball_suffix] + "/forearm_proximal",
                    "forearm",
                    stature_m,
                    mass_kg,
                    (0, 0, 0),
                    (0, 0, -forearm / 2),
                    share=0.5,
                )
            ],
        )
        joint(ball_suffix, arm, (0, 0, -upper_arm), [("Ry", f"{side}EInput")])
        frame(f"{side}E", ball, (0, 0, 0))
        fore = body(
            f"{side}LowerForearm",
            [
                _segment_solid(
                    names[f"{side}LowerForearm"] + "/forearm_distal",
                    "forearm",
                    stature_m,
                    mass_kg,
                    (0, 0, 0),
                    (0, 0, -forearm / 2),
                    share=0.5,
                )
            ],
        )
        joint(
            f"{side}LowerForearm", ball, (0, 0, -forearm / 2), [("Rz", f"{side}FInput")]
        )
        frame(f"{side}F", fore, (0, 0, 0))
        hand_suffix = "Clubface Vector" if side == "L" else "RHandStandoff"
        native_wrist = joints[names[hand_suffix]]
        # The native wrist base is copied for its hand-to-club relation, then
        # rolled a quarter turn about the forearm so that the cock axis (Rx,
        # radial/ulnar deviation) is parallel to the elbow axis: with the
        # forearm unrotated, radial deviation moves the hand toward the elbow
        # pit, as it does anatomically. See WRIST_ROLL.
        rotation = (
            WRIST_ROLL @ np.asarray(native_wrist["parent_to_base"], float)[:3, :3]
        )
        hand = body(
            hand_suffix,
            _copy(
                next(b for b in native["bodies"] if b["name"] == names[hand_suffix])[
                    "solids"
                ]
            ),
        )
        joint(
            hand_suffix,
            fore,
            (0, 0, -forearm / 2),
            [
                (primitive, p["coordinate"])
                for primitive, p in zip(
                    WRIST_PRIMITIVES, native_wrist["primitives"], strict=True
                )
            ],
            rotation=rotation,
            c2f=_grip_offset(
                native_wrist["child_to_follower"], grip_roll_deg, grip_rotation[side]
            ),
        )
    for f in native["frames"]:
        if f["body"] in (names["Clubface Vector"], names["RHandStandoff"]):
            frames.append(_copy(f))
    order = list(native["coordinate_order"]) + list(NECK_COORDINATES)
    document = {
        "schema_version": native["schema_version"],
        "qualification": "anthropometric geometry (AN-1); unqualified until Simscape parity",
        "gravity_m_s2": _copy(native["gravity_m_s2"]),
        "coordinate_order": order,
        "bodies": bodies,
        "joints": joints_out,
        "frames": frames,
        "closure": _copy(native["closure"]),
        "subject": {
            "stature_m": stature_m,
            "mass_kg": mass_kg,
            "trunk_scale": trunk_scale,
            "arm_scale": arm_scale,
            "shoulder_scale": shoulder_scale,
            "grip_roll_deg": grip_roll_deg,
            "grip_rotation_deg": {k: list(v) for k, v in grip_rotation.items()},
            "source": "de Leva 1996 male table",
            "de_leva_table_sha256": de_leva_table_sha256(),
        },
        "coordinate_ranges_deg": {k: list(v) for k, v in COORDINATE_RANGES_DEG.items()},
        "address_seed_deg": dict(ADDRESS_SEED_DEG),
        "visual_hints": torso_visual_hints(names, stature_m, pelvis_h, hub_h, clav),
    }
    built = {b["name"] for b in bodies}
    missing = set(names.values()) - built
    if missing:
        raise ValueError(f"Bodies not built: {sorted(missing)}")
    return apply_club(document, club)


# Torso and shoulder visuals as fractions of stature (male means): the trunk
# is drawn as three ellipsoids (pelvis, abdomen, thorax) and the clavicle links
# as thick capsules, so the picture carries the trunk mass the solids carry.
TORSO_WIDTH_FRACTION = {"pelvis": 0.17, "abdomen": 0.155, "thorax": 0.185}
TORSO_DEPTH_FRACTION = {"pelvis": 0.115, "abdomen": 0.11, "thorax": 0.12}
CLAVICLE_VISUAL_RADIUS_M = 0.04
# Legs drawn at male mid-segment girths (thigh, calf) rather than the
# density-derived capsule radius.
LEG_VISUAL_RADIUS_M = {"femur": 0.072, "tibia": 0.055}


def torso_visual_hints(
    names: Mapping[str, str],
    stature_m: float,
    pelvis_h: float,
    hub_h: float,
    clav: float,
) -> dict[str, Any]:
    """Visual hints for the pelvis, abdomen, thorax and clavicle links."""

    def half(kind: str, height: float) -> list[float]:
        return [
            TORSO_DEPTH_FRACTION[kind] * stature_m / 2,
            TORSO_WIDTH_FRACTION[kind] * stature_m / 2,
            height / 2,
        ]

    return {
        "shapes": {
            names["LowerTorso"] + "/pelvis": {
                "shape": "ellipsoid",
                "half_size_m": half("pelvis", pelvis_h + 0.12),
                "center_m": [0.0, 0.0, pelvis_h / 2 - 0.01],
            },
            names["COMRod"] + "/middle_trunk": {
                "shape": "ellipsoid",
                "half_size_m": half("abdomen", hub_h * 0.62),
                "center_m": [0.0, 0.0, hub_h * 0.27],
            },
            names["COMRod"] + "/upper_trunk": {
                "shape": "ellipsoid",
                "half_size_m": half("thorax", hub_h * 0.62),
                "center_m": [0.0, 0.0, hub_h * 0.70],
            },
        },
        "capsule_radius_m": {
            names["HubtoLS"]: CLAVICLE_VISUAL_RADIUS_M,
            names["HubtoRS"]: CLAVICLE_VISUAL_RADIUS_M,
        },
    }


def pelvis_alignment_for(stature_m: float) -> tuple[Array, float]:
    """OpenSim pelvis frame -> this pelvis frame, and the hip half width.

    OpenSim: x anterior, y superior, z right. Here: x forward, y left, z up,
    origin at the hip-centre midpoint. The translation is fixed by the caller
    once the OpenSim hip centres are known; this returns the rotation.
    """
    rotation = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    return rotation, HIP_HALF_WIDTH_FRACTION_OF_STATURE * stature_m
