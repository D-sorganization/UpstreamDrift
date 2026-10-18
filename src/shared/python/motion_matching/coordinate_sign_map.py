"""Axis and sign table of the 44 full-body v1 coordinates, and the ISB basis map.

The table is *measured*, not transcribed. ``tests/unit/motion_matching/
test_full_body_coordinate_sign_map.py`` compiles the anthropometric full-body
spec to MuJoCo, perturbs each coordinate by +10 degrees from q = 0 and checks
every row against the child-body rotation axis (in the pelvis frame) and the
motion of a body-fixed probe point. The same axes are asserted again from
Drake and Pinocchio forward kinematics.

Model basis: the pelvis frame at q = 0 is the world frame, x forward, y left,
z up (right-handed). ISB (Wu et al.) uses X anterior, Y up, Z right, so
``isb = (x, z, -y)``; see ``MODEL_TO_ISB``.

Conventions that geometry cannot settle
---------------------------------------
The specification carries no palm or thumb landmark. The radial/ulnar and
flexion/extension labels of the wrist, and the pronation/supination labels of
the forearm, therefore assume the *neutral forearm at q = 0*: the thumb side
faces the direction in which elbow flexion moves the hand. That is the
convention ``WRIST_ROLL`` and ``HUMAN_RANGES_DEG`` are written for. What the
probe does settle is the relation between axes: ``*WInputX`` is parallel to
the elbow axis and ``*WInputY`` is perpendicular to both it and the forearm.

Valid for the hip-calibrated anthropometric family (``WRIST_PRIMITIVES`` =
Rx, Rz with ``WRIST_ROLL`` applied). Native v1/v2 specs differ; see
``wrist_roll_applied``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.shared.python.motion_matching.anthropometric_geometry import WRIST_ROLL

Array = NDArray[np.float64]

#: Largest angle between a measured and a pinned axis.
AXIS_TOLERANCE_DEG = 0.5
DIRECTIONS = ("forward", "back", "left", "right", "up", "down")
#: Probe point 0.1 m from the joint anchor along this pelvis axis, or the body
#: origin for sliding joints.
PROBES = ("forward", "up", "left", "origin")
_DIRECTION_AXIS = {
    "forward": (0, 1.0),
    "back": (0, -1.0),
    "left": (1, 1.0),
    "right": (1, -1.0),
    "up": (2, 1.0),
    "down": (2, -1.0),
}
_UNIT_TOLERANCE = 1e-2  # rows are pinned to three decimals

# --- ISB basis ---------------------------------------------------------------

#: Rows give ISB (X anterior, Y up, Z right) in terms of model (x forward,
#: y left, z up). A proper rotation: determinant +1.
MODEL_TO_ISB: Array = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=float
)
_ISB_NAMES = {
    0: ("X (anterior)", "-X (posterior)"),
    1: ("Y (up)", "-Y (down)"),
    2: ("Z (right)", "-Z (left)"),
}


def _vector3(value: ArrayLike, name: str) -> Array:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{name} must be a 3-vector, got shape {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite")
    return vector


def model_to_isb(vector: ArrayLike) -> Array:
    """Components of a model-basis vector in the ISB basis.

    Postcondition: the length is unchanged; ``isb_to_model`` inverts it.
    """
    return MODEL_TO_ISB @ _vector3(vector, "vector")


def isb_to_model(vector: ArrayLike) -> Array:
    """Components of an ISB-basis vector in the model basis."""
    return MODEL_TO_ISB.T @ _vector3(vector, "vector")


def model_rotation_to_isb(rotation: ArrayLike) -> Array:
    """The same physical rotation written in ISB components.

    Postcondition: angle and handedness are preserved (the map is proper).
    """
    matrix = _rotation(rotation)
    return MODEL_TO_ISB @ matrix @ MODEL_TO_ISB.T


def isb_axis_label(vector: ArrayLike) -> str:
    """Name of the ISB axis a vector (given in ISB components) lies along."""
    components = _vector3(vector, "vector")
    index = int(np.argmax(np.abs(components)))
    return _ISB_NAMES[index][0 if components[index] > 0 else 1]


def _rotation(value: ArrayLike) -> Array:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("rotation must be a finite 3x3 matrix")
    if (
        not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-8)
        or np.linalg.det(matrix) < 0.9
    ):
        raise ValueError("rotation must be orthonormal with determinant +1")
    return matrix


def axis_angle_from_rotation(rotation: ArrayLike) -> tuple[Array, float]:
    """Unit axis and angle in (0, pi) of a rotation matrix.

    Precondition: orthonormal, determinant +1, not the identity and not a
    half turn. Postcondition: rotating by ``angle`` about ``axis`` reproduces
    the matrix (right-hand rule).
    """
    matrix = _rotation(rotation)
    angle = float(np.arccos(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0)))
    if not 1e-9 < angle < np.pi - 1e-6:
        raise ValueError(f"rotation angle {angle!r} rad has no unique axis")
    vee = np.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ]
    )
    return vee / (2.0 * np.sin(angle)), angle


def dominant_direction(vector: ArrayLike) -> str:
    """Pelvis direction (forward/back/left/right/up/down) a vector points to."""
    components = _vector3(vector, "vector")
    if not np.any(components):
        raise ValueError("vector is zero: it has no direction")
    index = int(np.argmax(np.abs(components)))
    sign = 1.0 if components[index] > 0 else -1.0
    return next(k for k, v in _DIRECTION_AXIS.items() if v == (index, sign))


# --- the table ---------------------------------------------------------------


@dataclass(frozen=True)
class CoordinateSign:
    """Axis and anatomical sense of one single-DOF coordinate at q = 0.

    ``axis_model`` is the unit rotation axis (or slide direction) in the
    pelvis frame; ``probe`` names the body-fixed point whose motion under +10
    degrees was ``probe_motion``; ``meaning`` says what a positive value is.
    """

    coordinate: str
    parent_body: str
    body: str
    axis_model: tuple[float, float, float]
    probe: str
    probe_motion: str
    meaning: str

    def __post_init__(self) -> None:
        for field in ("coordinate", "parent_body", "body"):
            if not getattr(self, field).strip():
                raise ValueError(f"{field} must be a non-empty name")
        axis = _vector3(self.axis_model, "axis_model")
        if abs(float(np.linalg.norm(axis)) - 1.0) > _UNIT_TOLERANCE:
            raise ValueError(f"axis_model of {self.coordinate} must be a unit vector")
        if self.probe not in PROBES:
            raise ValueError(f"probe must be one of {PROBES}, got {self.probe!r}")
        if self.probe_motion not in DIRECTIONS:
            raise ValueError(
                f"probe_motion must be one of {DIRECTIONS}, got {self.probe_motion!r}"
            )
        if not self.meaning.strip():
            raise ValueError("meaning must say what a positive value is")


def _rows_part1() -> tuple[CoordinateSign, ...]:
    r = CoordinateSign
    return (
        r(
            "TranslationInputX",
            "world",
            "LowerTorso",
            (1.0, 0.0, 0.0),
            "origin",
            "forward",
            "pelvis translation forward",
        ),
        r(
            "TranslationInputY",
            "world",
            "LowerTorso",
            (0.0, 1.0, 0.0),
            "origin",
            "left",
            "pelvis translation to the left",
        ),
        r(
            "TranslationInputZ",
            "world",
            "LowerTorso",
            (0.0, 0.0, 1.0),
            "origin",
            "up",
            "pelvis translation upward",
        ),
        r(
            "HipInputX",
            "world",
            "LowerTorso",
            (1.0, 0.0, 0.0),
            "up",
            "right",
            "pelvis lateral tilt: left side up, right side down",
        ),
        r(
            "HipInputY",
            "world",
            "LowerTorso",
            (0.0, 1.0, 0.0),
            "forward",
            "down",
            "pelvis forward pitch (anterior tilt): front down, top forward",
        ),
        r(
            "HipInputZ",
            "world",
            "LowerTorso",
            (0.0, 0.0, 1.0),
            "forward",
            "left",
            "pelvis axial rotation to the left (counterclockwise seen from above)",
        ),
        r(
            "SpineInputX",
            "UpperTorsoBase",
            "COMRod",
            (1.0, 0.0, 0.0),
            "up",
            "right",
            "trunk lateral flexion to the right (top moves right)",
        ),
        r(
            "SpineInputY",
            "UpperTorsoBase",
            "COMRod",
            (0.0, 1.0, 0.0),
            "forward",
            "down",
            "trunk forward flexion (top moves forward)",
        ),
        r(
            "TorsoInput",
            "LowerTorso",
            "UpperTorsoBase",
            (0.0, 0.0, 1.0),
            "forward",
            "left",
            "axial rotation to the left of the upper torso on the pelvis (twist about pelvis z)",
        ),
    )


def _rows_part2() -> tuple[CoordinateSign, ...]:
    r = CoordinateSign
    return (
        r(
            "LEInput",
            "LUpperArm",
            "Spherical Solid",
            (0.0, 1.0, 0.0),
            "forward",
            "down",
            "left elbow extension (flexion is negative; forearm tip moves down)",
        ),
        r(
            "LFInput",
            "Spherical Solid",
            "LLowerForearm",
            (-1.0, 0.0, 0.0),
            "up",
            "left",
            "supination of the left forearm (about the proximal direction; neutral at zero)",
        ),
        r(
            "LScapInputX",
            "COMRod",
            "HubtoLS",
            (1.0, 0.0, 0.0),
            "up",
            "right",
            "elevation of the left shoulder",
        ),
        r(
            "LScapInputY",
            "COMRod",
            "HubtoLS",
            (0.0, 0.0, 1.0),
            "forward",
            "left",
            "retraction of the left shoulder (moves backward)",
        ),
        r(
            "LSInputX",
            "HubtoLS",
            "LUpperArm",
            (0.0, 0.0, 1.0),
            "forward",
            "left",
            "left shoulder horizontal abduction (arm swings away from the midline)",
        ),
        r(
            "LSInputY",
            "HubtoLS",
            "LUpperArm",
            (0.0, 1.0, 0.0),
            "forward",
            "down",
            "lowering of the forward-raised left arm (arm tip moves down)",
        ),
        r(
            "LSInputZ",
            "HubtoLS",
            "LUpperArm",
            (-1.0, 0.0, 0.0),
            "up",
            "left",
            "left humeral external rotation (at the forward-raised zero pose)",
        ),
        r(
            "LWInputX",
            "LLowerForearm",
            "Clubface Vector",
            (0.0, -1.0, 0.0),
            "forward",
            "up",
            "radial deviation of the left wrist (cock; toward the elbow pit; ulnar is negative)",
        ),
        r(
            "LWInputY",
            "LLowerForearm",
            "Clubface Vector",
            (0.0, 0.0, 1.0),
            "forward",
            "left",
            "extension of the left wrist (toward the dorsum; flexion is negative)",
        ),
    )


def _rows_part3() -> tuple[CoordinateSign, ...]:
    r = CoordinateSign
    return (
        r(
            "REInput",
            "RUpperArm",
            "Spherical Solid1",
            (0.0, 1.0, 0.0),
            "forward",
            "down",
            "right elbow extension (flexion is negative; forearm tip moves down)",
        ),
        r(
            "RFInput",
            "Spherical Solid1",
            "RLowerForearm",
            (-1.0, 0.0, 0.0),
            "up",
            "left",
            "pronation of the right forearm (about the proximal direction; neutral at zero)",
        ),
        r(
            "RScapInputX",
            "COMRod",
            "HubtoRS",
            (1.0, 0.0, 0.0),
            "up",
            "right",
            "depression of the right shoulder (elevation is negative)",
        ),
        r(
            "RScapInputY",
            "COMRod",
            "HubtoRS",
            (0.0, 0.0, 1.0),
            "forward",
            "left",
            "protraction of the right shoulder (moves forward)",
        ),
        r(
            "RSInputX",
            "HubtoRS",
            "RUpperArm",
            (0.0, 0.0, 1.0),
            "forward",
            "left",
            "right shoulder horizontal adduction (arm swings toward the midline)",
        ),
        r(
            "RSInputY",
            "HubtoRS",
            "RUpperArm",
            (0.0, 1.0, 0.0),
            "forward",
            "down",
            "lowering of the forward-raised right arm (arm tip moves down)",
        ),
        r(
            "RSInputZ",
            "HubtoRS",
            "RUpperArm",
            (-1.0, 0.0, 0.0),
            "up",
            "left",
            "right humeral internal rotation (at the forward-raised zero pose)",
        ),
        r(
            "RWInputX",
            "RLowerForearm",
            "RHandStandoff",
            (0.0, -1.0, 0.0),
            "forward",
            "up",
            "radial deviation of the right wrist (cock; toward the elbow pit; ulnar is negative)",
        ),
        r(
            "RWInputY",
            "RLowerForearm",
            "RHandStandoff",
            (0.0, 0.0, 1.0),
            "forward",
            "left",
            "flexion of the right wrist (toward the palm; extension is negative)",
        ),
    )


def _rows_part4() -> tuple[CoordinateSign, ...]:
    r = CoordinateSign
    return (
        r(
            "NeckInputX",
            "COMRod",
            "Head",
            (1.0, 0.0, 0.0),
            "up",
            "right",
            "head lateral flexion to the right",
        ),
        r(
            "NeckInputY",
            "COMRod",
            "Head",
            (0.0, 1.0, 0.0),
            "forward",
            "down",
            "head flexion (nod forward)",
        ),
        r(
            "NeckInputZ",
            "COMRod",
            "Head",
            (0.0, 0.0, 1.0),
            "forward",
            "left",
            "head axial rotation to the left",
        ),
        r(
            "hip_flexion_r",
            "LowerTorso",
            "femur_r",
            (-0.074, -0.981, 0.177),
            "forward",
            "up",
            "flexion of the right hip (thigh swings forward)",
        ),
        r(
            "hip_adduction_r",
            "LowerTorso",
            "femur_r",
            (0.997, -0.072, 0.018),
            "up",
            "right",
            "adduction of the right hip (thigh swings toward the midline)",
        ),
        r(
            "hip_rotation_r",
            "LowerTorso",
            "femur_r",
            (-0.005, 0.178, 0.984),
            "forward",
            "left",
            "internal rotation of the right hip (toes turn toward the midline)",
        ),
        r(
            "knee_angle_r",
            "femur_r",
            "tibia_r",
            (-0.075, -0.966, 0.246),
            "forward",
            "up",
            "extension of the right knee (flexion is negative; shank swings forward)",
        ),
        r(
            "ankle_angle_r",
            "tibia_r",
            "talus_r",
            (-0.342, -0.94, -0.008),
            "forward",
            "up",
            "dorsiflexion of the right ankle (forefoot rises)",
        ),
        r(
            "subtalar_angle_r",
            "talus_r",
            "calcn_r",
            (0.79, 0.033, 0.613),
            "left",
            "up",
            "inversion of the right subtalar joint (oblique axis; sole turns medially)",
        ),
    )


def _rows_part5() -> tuple[CoordinateSign, ...]:
    r = CoordinateSign
    return (
        r(
            "mtp_angle_r",
            "calcn_r",
            "toes_r",
            (0.764, 0.636, -0.111),
            "up",
            "right",
            "flexion of the right toes (oblique axis; toe tip moves down)",
        ),
        r(
            "hip_flexion_l",
            "LowerTorso",
            "femur_l",
            (0.453, -0.877, 0.161),
            "forward",
            "up",
            "flexion of the left hip (thigh swings forward)",
        ),
        r(
            "hip_adduction_l",
            "LowerTorso",
            "femur_l",
            (-0.891, -0.447, 0.076),
            "up",
            "left",
            "adduction of the left hip (thigh swings toward the midline)",
        ),
        r(
            "hip_rotation_l",
            "LowerTorso",
            "femur_l",
            (0.005, -0.178, -0.984),
            "forward",
            "right",
            "internal rotation of the left hip (toes turn toward the midline)",
        ),
        r(
            "knee_angle_l",
            "femur_l",
            "tibia_l",
            (0.452, -0.887, 0.091),
            "forward",
            "up",
            "extension of the left knee (flexion is negative; shank swings forward)",
        ),
        r(
            "ankle_angle_l",
            "tibia_l",
            "talus_l",
            (0.672, -0.677, 0.302),
            "up",
            "right",
            "dorsiflexion of the left ankle (forefoot rises)",
        ),
        r(
            "subtalar_angle_l",
            "talus_l",
            "calcn_l",
            (-0.699, -0.478, -0.532),
            "left",
            "down",
            "inversion of the left subtalar joint (oblique axis; sole turns medially)",
        ),
        r(
            "mtp_angle_l",
            "calcn_l",
            "toes_l",
            (-0.953, 0.296, -0.058),
            "up",
            "left",
            "flexion of the left toes (oblique axis; toe tip moves down)",
        ),
    )


def _rows() -> tuple[CoordinateSign, ...]:
    return (
        *_rows_part1(),
        *_rows_part2(),
        *_rows_part3(),
        *_rows_part4(),
        *_rows_part5(),
    )


COORDINATE_SIGN_MAP: Mapping[str, CoordinateSign] = {
    row.coordinate: row for row in _rows()
}
#: Proximal to distal: twist is applied before the two tilts, so the tilt axes
#: ride on the twist while the twist axis stays on the pelvis z axis.
TRUNK_CHAIN_ORDER = ("TorsoInput", "SpineInputX", "SpineInputY")


def sign_for(coordinate: str) -> CoordinateSign:
    """The pinned row of ``coordinate``; ``KeyError`` names unknown ones."""
    try:
        return COORDINATE_SIGN_MAP[coordinate]
    except KeyError:
        raise KeyError(f"No sign-map row for coordinate {coordinate!r}") from None


# --- spec family -------------------------------------------------------------


def _wrist_base(spec: Mapping[str, Any], coordinate: str) -> Array:
    for joint in spec["joints"]:
        if coordinate in [p["coordinate"] for p in joint["primitives"]]:
            return np.asarray(joint["parent_to_base"], dtype=float)[:3, :3]
    raise ValueError(f"spec has no joint driving {coordinate!r}")


def wrist_roll_applied(
    spec: Mapping[str, Any],
    native: Mapping[str, Any],
    *,
    coordinate: str = "LWInputX",
    atol: float = 1e-6,
) -> bool:
    """Whether ``spec``'s wrist base is the native one turned by ``WRIST_ROLL``.

    ``True`` for anthropometric specs (cock axis parallel to the elbow axis),
    ``False`` for native v1/v2 specs (the second wrist primitive spins about
    the forearm). Raises ``ValueError`` for any other basis rather than
    guessing.
    """
    base = _wrist_base(spec, coordinate)
    original = _wrist_base(native, coordinate)
    if np.allclose(base, WRIST_ROLL @ original, atol=atol):
        return True
    if np.allclose(base, original, atol=atol):
        return False
    raise ValueError(
        f"wrist base of {coordinate} is neither native nor WRIST_ROLL @ native"
    )


__all__: Sequence[str] = (
    "AXIS_TOLERANCE_DEG",
    "COORDINATE_SIGN_MAP",
    "CoordinateSign",
    "MODEL_TO_ISB",
    "TRUNK_CHAIN_ORDER",
    "axis_angle_from_rotation",
    "dominant_direction",
    "isb_axis_label",
    "isb_to_model",
    "model_rotation_to_isb",
    "model_to_isb",
    "sign_for",
    "wrist_roll_applied",
)
