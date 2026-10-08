"""Visual club assembly (shaft, grip, head) in the spec club-body frame.

One definition for every engine: MuJoCo, OpenSim, MeshCat and MyoSuite all
take the same three meshes. The club-body frame is the document convention in
``motion_matching.club_models``: origin at the sole point on the shaft axis,
shaft toward the grip along -y, shaft axis offset ``axis_offset_m`` along +z.
Everything here is visual only; masses and inertias stay as specified.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.shared.python.model_appearance.club_head_mesh import (
    DEFAULT_AXIS_OFFSET_M,
    club_face_normal,
    club_frame_mesh,
    face_centre,
    head_points_to_club,
    load_club_head,
)
from src.shared.python.model_appearance.club_shaft_mesh import tapered_tube
from src.shared.python.model_appearance.geometry import Mesh

CLUB_BODY_SUFFIX = "Clubface Vector"
DEFAULT_GRIP_LENGTH_M = 0.265
SHAFT_TIP_RADIUS_FRACTION = 0.65
SHAFT_BUTT_RADIUS_FRACTION = 0.95
GRIP_TIP_RADIUS_M = 0.0105
GRIP_BUTT_RADIUS_M = 0.0118
DRIVER_MIN_LENGTH_M = 1.05  # longer than this and the head is a driver
#: Roll of the head about the shaft, in the club-body frame, that squares the
#: face to the target at the captured address pose of each tour-average
#: capture (A: driver, B: 7-iron). The matched hand-club chain fixes the club
#: roll only through fitted wrist constants (``GRIP_ROTATION_DEG``), so the
#: club body ``+x`` axis is open 31 / 45 degrees there; this single constant
#: re-seats the head and the ``Clubface Vector`` for every engine. Solved from
#: forward kinematics (OSV-8, #11755); see test_clubface_square_at_address.
ADDRESS_SQUARE_FACE_ROLL_DEG = {"driver": -37.92, "iron7": -48.99}
PART_MATERIALS = {"shaft": None, "grip": "grip_rubber", "head": None}  # None: finish


@dataclass(frozen=True)
class ClubAssembly:
    """Geometry parameters of one club in the club-body frame (metres)."""

    head_alias: str
    length_m: float
    grip_length_m: float
    shaft_radius_m: float
    axis_offset_m: float = DEFAULT_AXIS_OFFSET_M
    face_roll_deg: float = 0.0  # head roll about the shaft axis (club frame)

    def __post_init__(self) -> None:
        if not math.isfinite(self.face_roll_deg):
            raise ValueError("face_roll_deg must be finite")
        for name in ("length_m", "grip_length_m", "shaft_radius_m"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if not math.isfinite(self.axis_offset_m):
            raise ValueError("axis_offset_m must be finite")
        if self.grip_length_m >= self.length_m:
            raise ValueError("the grip must be shorter than the club")


def _club_solids(spec: Mapping[str, Any]) -> dict[str, np.ndarray]:
    for body in spec["bodies"]:
        if str(body["name"]).endswith(CLUB_BODY_SUFFIX):
            return {
                str(s["name"]).rsplit("/", 1)[-1]: np.asarray(s["placement"], float)
                for s in body["solids"]
            }
    return {}


def assembly_from_spec(spec: Mapping[str, Any]) -> ClubAssembly | None:
    """The club assembly a native or full-body spec describes, else ``None``.

    Uses ``spec["club"]`` when present (name, length); otherwise infers the
    length from the Grip solid placement. A club longer than 1.05 m gets the
    driver head, a shorter one the 7-iron head.
    """
    solids = _club_solids(spec)
    if "Grip" not in solids:
        return None
    club = spec.get("club") or {}
    grip = DEFAULT_GRIP_LENGTH_M
    length = float(club.get("length_m") or (-solids["Grip"][1, 3] + grip / 2.0))
    alias = str(
        club.get("name") or ("driver" if length > DRIVER_MIN_LENGTH_M else "iron7")
    )
    offset = (
        float(solids["Clubhead"][2, 3])
        if "Clubhead" in solids
        else DEFAULT_AXIS_OFFSET_M
    )
    roll = club.get("face_roll_deg", ADDRESS_SQUARE_FACE_ROLL_DEG.get(alias, 0.0))
    return ClubAssembly(alias, length, grip, 0.0065, offset, float(roll))


def assembly_meshes(club: ClubAssembly) -> dict[str, Mesh]:
    """``{"shaft", "grip", "head"}`` meshes in the club-body frame.

    The head mesh includes the hosel; the shaft runs from the hosel top to the
    grip. Postcondition: every mesh is closed with outward winding.
    """
    head = load_club_head(club.head_alias)
    z = club.axis_offset_m
    shaft_top = -head.hosel_top_m
    grip_top = -(club.length_m - club.grip_length_m)
    shaft = tapered_tube(
        np.array([0.0, shaft_top, z]),
        np.array([0.0, grip_top, z]),
        SHAFT_TIP_RADIUS_FRACTION * club.shaft_radius_m,
        SHAFT_BUTT_RADIUS_FRACTION * club.shaft_radius_m,
    )
    grip = tapered_tube(
        np.array([0.0, grip_top, z]),
        np.array([0.0, -club.length_m, z]),
        GRIP_TIP_RADIUS_M,
        GRIP_BUTT_RADIUS_M,
    )
    return {
        "shaft": shaft,
        "grip": grip,
        "head": club_frame_mesh(
            head, axis_offset_m=z, face_roll_deg=club.face_roll_deg
        ),
    }


def clubface_vector(club: ClubAssembly) -> np.ndarray:
    """The spec ``Clubface Vector``: unit face normal (with loft) in the club frame.

    Rolled by the assembly's ``face_roll_deg``, so it is the direction the
    rendered head's face points in every engine.
    """
    return club_face_normal(load_club_head(club.head_alias), club.face_roll_deg)


def clubface_centre(club: ClubAssembly) -> np.ndarray:
    """Face-centre point (where the ball is struck) in the club-body frame.

    Same placement and roll as the rendered head, so it is the point every
    engine's head mesh shows at the middle of the face.
    """
    head = load_club_head(club.head_alias)
    centre = face_centre(head.head_mesh)[None, :]
    return head_points_to_club(
        head,
        centre,
        axis_offset_m=club.axis_offset_m,
        face_roll_deg=club.face_roll_deg,
    )[0]


def club_body_name(spec: Mapping[str, Any]) -> str | None:
    """Name of the spec body that carries the club solids, else ``None``."""
    for body in spec["bodies"]:
        if str(body["name"]).endswith(CLUB_BODY_SUFFIX):
            return str(body["name"])
    return None
