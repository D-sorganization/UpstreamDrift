"""Typical golf clubs as mass, length and shape parameters (AN-1, #10099).

The native document carries one club (a 0.25 kg head on a 1.08 m shaft with
the hands modelled on the club body). Matching a driver swing and a 7-iron
swing needs each club's length, masses and inertia, and a recognisable head
shape for the visual layer. ``ClubSpec`` holds typical retail values
(driver: 45.5 in, 198 g head, 65 g graphite shaft, 50 g grip; 7-iron:
37 in, 268 g head, 110 g steel shaft, 50 g grip); ``apply_club`` rewrites a
document's club body to a spec and moves the hands along the shaft so the
wrist-to-head distance follows the club length, leaving the hand-to-club
relation (wrist base, closure weld) untouched.

Club body frame convention (from the native document): origin at the head,
shaft along -y toward the grip, the ``Clubhead`` frame 0.064 m above the
origin along z, hands near y = -(length - butt offset).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

CLUB_BODY_SUFFIX = "Clubface Vector"
HAND_STANDOFF_SUFFIX = "RHandStandoff"
HEAD_HEIGHT_M = 0.064  # Clubhead frame above the club body origin (native)
WRIST_TO_BUTT_M = 0.032  # native: wrist joint 32 mm below the butt end


@dataclass(frozen=True)
class ClubSpec:
    """Typical values for one club type; lengths in metres, masses in kg."""

    name: str
    length_m: float  # butt end to sole along the shaft
    head_mass_kg: float
    shaft_mass_kg: float
    grip_mass_kg: float
    grip_length_m: float
    shaft_radius_m: float
    head_shape: str  # "ellipsoid" (driver) or "box" (iron)
    head_half_size_m: tuple[float, float, float]  # x (face normal), y (shaft), z
    head_gyration_m: tuple[float, float, float]  # radii of gyration about x, y, z

    def __post_init__(self) -> None:
        for value in (
            self.length_m,
            self.head_mass_kg,
            self.shaft_mass_kg,
            self.grip_mass_kg,
            self.grip_length_m,
            self.shaft_radius_m,
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Club dimensions and masses must be positive")
        if self.grip_length_m >= self.length_m:
            raise ValueError("Grip must be shorter than the club")
        if self.head_shape not in ("ellipsoid", "box"):
            raise ValueError("Head shape must be ellipsoid or box")

    @property
    def total_mass_kg(self) -> float:
        return self.head_mass_kg + self.shaft_mass_kg + self.grip_mass_kg

    @property
    def wrist_to_head_m(self) -> float:
        """Distance from the wrist joint to the club body origin along the shaft."""
        return self.length_m - WRIST_TO_BUTT_M


DRIVER = ClubSpec(
    name="driver",
    length_m=1.156,  # 45.5 in
    head_mass_kg=0.198,
    shaft_mass_kg=0.065,
    grip_mass_kg=0.050,
    grip_length_m=0.265,
    shaft_radius_m=0.0065,
    head_shape="ellipsoid",
    head_half_size_m=(0.060, 0.032, 0.050),  # heel-toe, along the shaft, face-back
    head_gyration_m=(0.040, 0.045, 0.045),
)
IRON_7 = ClubSpec(
    name="iron7",
    length_m=0.940,  # 37 in
    head_mass_kg=0.268,
    shaft_mass_kg=0.110,
    grip_mass_kg=0.050,
    grip_length_m=0.265,
    shaft_radius_m=0.0065,
    head_shape="box",
    head_half_size_m=(0.008, 0.040, 0.025),
    head_gyration_m=(0.030, 0.020, 0.030),
)
CLUBS: dict[str, ClubSpec] = {DRIVER.name: DRIVER, IRON_7.name: IRON_7}
# Shape hints per catalogue club type; masses and lengths come from the club
# database so there is one source of club numbers in the repository.
HEAD_SHAPES: dict[str, tuple[str, tuple[float, float, float]]] = {
    "Driver": ("ellipsoid", (0.060, 0.032, 0.050)),
    "Wood": ("ellipsoid", (0.050, 0.028, 0.042)),
    "Hybrid": ("ellipsoid", (0.040, 0.026, 0.034)),
    "Iron": ("box", (0.008, 0.040, 0.025)),
    "Wedge": ("box", (0.008, 0.040, 0.028)),
}


def from_database(club_id: str) -> ClubSpec:
    """A ``ClubSpec`` from the repository's club database entry ``club_id``
    (``ClubDatabase`` in the MuJoCo humanoid package: lengths in inches,
    masses in grams, head MOI in g cm^2 about the vertical axis). The head
    shape comes from ``HEAD_SHAPES`` by club type; the vertical radius of
    gyration from the MOI, the others scaled as in the typical constants.
    Raises ``ValueError`` for an unknown club or club type.
    """
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.club_configurations import (  # noqa: E501
        ClubDatabase,
    )

    entry = ClubDatabase.get_club(club_id)
    if entry is None:
        raise ValueError(f"Unknown club {club_id!r} in the club database")
    if entry.club_type not in HEAD_SHAPES:
        raise ValueError(f"No head shape for club type {entry.club_type!r}")
    shape, half = HEAD_SHAPES[entry.club_type]
    head_mass = entry.head_mass_grams / 1000.0
    k_vertical = math.sqrt(entry.moment_of_inertia * 1e-7 / head_mass)
    return ClubSpec(
        name=club_id,
        length_m=entry.length_inches * 0.0254,
        head_mass_kg=head_mass,
        shaft_mass_kg=entry.shaft_mass_grams / 1000.0,
        grip_mass_kg=entry.grip_mass_grams / 1000.0,
        grip_length_m=DRIVER.grip_length_m,
        shaft_radius_m=DRIVER.shaft_radius_m,
        head_shape=shape,
        head_half_size_m=half,
        head_gyration_m=(0.85 * k_vertical, k_vertical, k_vertical),
    )


def _placement(translation: tuple[float, float, float]) -> list[list[float]]:
    m = np.eye(4)
    m[:3, 3] = translation
    return m.tolist()


def _rod_inertia(mass: float, length: float, radius: float) -> list[list[float]]:
    """Solid cylinder along y: transverse m(3r^2+L^2)/12, axial m r^2/2."""
    transverse = mass * (3 * radius**2 + length**2) / 12.0
    return np.diag([transverse, mass * radius**2 / 2.0, transverse]).tolist()


def club_solids(prefix: str, club: ClubSpec) -> list[dict[str, Any]]:
    """Head, shaft and grip solids in the club body frame for ``club``."""
    head_inertia = np.diag(
        [club.head_mass_kg * g**2 for g in club.head_gyration_m]
    ).tolist()
    shaft_length = club.length_m - club.grip_length_m
    return [
        {
            "name": f"{prefix}/Clubhead",
            "mass_kg": club.head_mass_kg,
            "placement": _placement((0.0, 0.0, HEAD_HEIGHT_M)),
            "com_m": [0.0, 0.0, 0.0],
            "inertia_com_kg_m2": head_inertia,
        },
        {
            "name": f"{prefix}/Rigid Shaft",
            "mass_kg": club.shaft_mass_kg,
            "placement": _placement((0.0, -shaft_length / 2.0, HEAD_HEIGHT_M)),
            "com_m": [0.0, 0.0, 0.0],
            "inertia_com_kg_m2": _rod_inertia(
                club.shaft_mass_kg, shaft_length, club.shaft_radius_m
            ),
        },
        {
            "name": f"{prefix}/Grip",
            "mass_kg": club.grip_mass_kg,
            "placement": _placement(
                (0.0, -(club.length_m - club.grip_length_m / 2.0), HEAD_HEIGHT_M)
            ),
            "com_m": [0.0, 0.0, 0.0],
            "inertia_com_kg_m2": _rod_inertia(
                club.grip_mass_kg, club.grip_length_m, club.shaft_radius_m + 0.006
            ),
        },
    ]


def visual_hints(prefix: str, club: ClubSpec) -> dict[str, Any]:
    """Visual-layer hints: the head shape at the head solid, a thin shaft."""
    return {
        "shapes": {
            f"{prefix}/Clubhead": {
                "shape": club.head_shape,
                "half_size_m": list(club.head_half_size_m),
                "center_m": [0.0, 0.0, HEAD_HEIGHT_M],
            }
        },
        "capsule_radius_m": {prefix: club.shaft_radius_m},
    }


def _shift_y(matrix: Any, delta: float) -> list[list[float]]:
    m = np.asarray(matrix, dtype=float).copy()
    m[1, 3] += delta
    return m.tolist()


def apply_club(document: Mapping[str, Any], club: ClubSpec) -> dict[str, Any]:
    """Return a copy of ``document`` carrying ``club``.

    Replaces the head, shaft and grip solids of the club body (hand solids
    stay), records the club under ``club`` and its visual hints, and shifts
    every hand-side placement on the club body (wrist follower, closure weld,
    hand solids, ``LW`` and ``Grip`` frames) along the shaft so the wrist sits
    ``club.wrist_to_head_m`` from the head. Precondition: the document has a
    club body whose wrist follower lies on the shaft (x, z small, y < 0).
    Postcondition: the wrist-to-head distance equals the club's.
    """
    doc = json_copy(document)
    body = next(
        (b for b in doc["bodies"] if b["name"].endswith(CLUB_BODY_SUFFIX)), None
    )
    if body is None:
        raise ValueError("Document has no club body")
    joint = next(j for j in doc["joints"] if j["child"] == body["name"])
    follower = np.asarray(joint["child_to_follower"], dtype=float)
    wrist_y = float(follower[1, 3])
    if wrist_y >= 0:
        raise ValueError("Wrist follower must lie down the shaft (y < 0)")
    delta = -club.wrist_to_head_m - wrist_y  # move hands to the new wrist point
    prefix = body["name"]
    hand_solids = [s for s in body["solids"] if "Hand" in s["name"].rsplit("/", 1)[-1]]
    for solid in hand_solids:
        solid["placement"] = _shift_y(solid["placement"], delta)
    body["solids"] = club_solids(prefix, club) + hand_solids
    joint["child_to_follower"] = _shift_y(follower, delta)
    closure = doc["closure"]
    closure["placement_b"] = _shift_y(closure["placement_b"], delta)
    for frame in doc["frames"]:
        if frame["body"] == prefix and frame["name"] in ("LW", "Grip"):
            frame["placement"] = _shift_y(frame["placement"], delta)
    doc["club"] = {
        "name": club.name,
        "length_m": club.length_m,
        "head_mass_kg": club.head_mass_kg,
        "shaft_mass_kg": club.shaft_mass_kg,
        "grip_mass_kg": club.grip_mass_kg,
        "total_mass_kg": club.total_mass_kg,
        "wrist_to_head_m": club.wrist_to_head_m,
    }
    hints = doc.setdefault("visual_hints", {"shapes": {}, "capsule_radius_m": {}})
    club_hints = visual_hints(prefix, club)
    hints.setdefault("shapes", {}).update(club_hints["shapes"])
    hints.setdefault("capsule_radius_m", {}).update(club_hints["capsule_radius_m"])
    return doc


def json_copy(document: Mapping[str, Any]) -> dict[str, Any]:
    import json

    return json.loads(json.dumps(document))
