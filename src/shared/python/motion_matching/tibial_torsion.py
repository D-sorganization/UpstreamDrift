"""Static tibial-torsion foot-yaw offset in the shared full-body spec (R3b, #12789).

The Rajagopal leg chain has no shank or foot yaw coordinate, so all address
toe-out has to come from ``hip_rotation_*``. On the 7-iron the trail hip then
sits on its -40 deg external-rotation limit and the trail foot stops near
13.5 deg, short of the owner's 20 deg style target (R3, #12683). The ruling on
#12683 forbids widening the hip limit; the missing yaw belongs to the shank.

Real shanks carry external tibial torsion: the ankle (bimalleolar) axis is
turned outward against the knee flexion axis. The generic Rajagopal leg encodes
only about 6 deg of it (``GENERIC_LEG_TORSION_DEG``, measured from the spec's
own knee and ankle hinge axes and pinned by a test). This module adds a static
per-side offset, a fixed rotation of the ankle joint's base frame about the
shank long axis (knee centre to ankle centre) through the ankle centre. The
whole foot (talus, calcaneus, toes, their markers and contact spheres) turns
with it. It is not a degree of freedom, so the coordinate order is unchanged.

Every engine emitter (MuJoCo, Drake, Pinocchio, OpenSim, MyoSuite) already
builds the ankle from ``joints[ankle_<side>].parent_to_base``, so the offset
lives in exactly one place, the spec document, and no engine code changes.

Source for the anatomical range: Seber S, Hazer B, Kose N, Gokturk E, Gunal I,
Turgut A. Rotational profile of the lower extremity and foot progression angle:
computerized tomographic examination of 50 male adults. Arch Orthop Trauma
Surg. 2000;120(5-6):255-258. doi:10.1007/s004020050459. Tibiofibular torsion
mean 30.9 deg (right) / 29.1 deg (left), range 16-50 deg.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
import math
from typing import Any

import numpy as np

__all__ = [
    "ADULT_TIBIAL_TORSION_RANGE_DEG",
    "DEFAULT_TIBIAL_TORSION_OFFSET_DEG",
    "GENERIC_LEG_TORSION_DEG",
    "OFFSET_RANGE_DEG",
    "SIDES",
    "TIBIAL_TORSION_KEY",
    "TIBIAL_TORSION_SOURCE",
    "apply_tibial_torsion",
    "malleolar_torsion_deg",
    "validate_tibial_torsion",
]

SIDES: tuple[str, ...] = ("r", "l")
#: Spec document key recording the applied offset (absent when it is zero).
TIBIAL_TORSION_KEY = "tibial_torsion_deg"
TIBIAL_TORSION_SOURCE = (
    "Seber et al. 2000, Arch Orthop Trauma Surg 120:255-258 "
    "(doi:10.1007/s004020050459): CT tibiofibular torsion of 50 male adults, "
    "range 16-50 deg"
)
#: Documented adult external tibial torsion range, deg (Seber et al. 2000).
ADULT_TIBIAL_TORSION_RANGE_DEG: tuple[float, float] = (16.0, 50.0)
#: External torsion already in the generic Rajagopal leg: the angle about the
#: shank between the knee and ankle hinge axes (both sides, 6.06 deg).
GENERIC_LEG_TORSION_DEG: float = 6.06
#: Allowed offset, deg: zero keeps the generic leg; the upper bound keeps the
#: model's total torsion inside the documented adult range.
OFFSET_RANGE_DEG: tuple[float, float] = (
    0.0,
    ADULT_TIBIAL_TORSION_RANGE_DEG[1] - GENERIC_LEG_TORSION_DEG,
)
#: Default offset, deg: the smallest round value that puts the generic leg's
#: total torsion (6.06 + 10 = 16.06 deg) inside the documented adult range.
#: Applied identically to both legs (no side asymmetry is claimed).
DEFAULT_TIBIAL_TORSION_OFFSET_DEG: float = 10.0


def _joint(document: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    for joint in document["joints"]:
        if joint["name"] == name:
            return joint
    raise ValueError(f"spec has no {name} joint")


def _shank_axis(document: Mapping[str, Any], side: str) -> np.ndarray:
    """Unit shank long axis in the tibia frame, ankle centre -> knee centre."""
    knee = np.asarray(_joint(document, f"knee_{side}")["child_to_follower"], float)
    ankle = np.asarray(_joint(document, f"ankle_{side}")["parent_to_base"], float)
    axis = knee[:3, 3] - ankle[:3, 3]
    norm = float(np.linalg.norm(axis))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"knee_{side} and ankle_{side} centres coincide")
    return axis / norm


def _lateral_sign(document: Mapping[str, Any], side: str) -> float:
    """+1 when a positive turn about the shank axis moves the toes laterally.

    The toes point along the ankle base +x. The knee hinge axis points to the
    subject's right on both legs (the spec knee convention, #12057), so it is
    lateral on the right leg and medial on the left leg.
    """
    up = _shank_axis(document, side)
    ankle = np.asarray(_joint(document, f"ankle_{side}")["parent_to_base"], float)
    knee = np.asarray(_joint(document, f"knee_{side}")["child_to_follower"], float)
    toes = ankle[:3, 0]
    lateral = knee[:3, 2] if side == "r" else -knee[:3, 2]
    turn = np.cross(up, toes)  # toe velocity for a positive turn about ``up``
    return 1.0 if float(turn @ lateral) > 0.0 else -1.0


def _rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    k = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return np.eye(3) + math.sin(angle_rad) * k + (1.0 - math.cos(angle_rad)) * k @ k


def _check_offsets(offsets_deg: Mapping[str, float]) -> dict[str, float]:
    if set(offsets_deg) != set(SIDES):
        raise ValueError(f"offsets need exactly the sides {SIDES}, got {offsets_deg}")
    low, high = OFFSET_RANGE_DEG
    out: dict[str, float] = {}
    for side in SIDES:
        value = float(offsets_deg[side])
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(
                f"tibial torsion offset for side {side!r} must be finite in "
                f"[{low}, {high}] deg (generic leg {GENERIC_LEG_TORSION_DEG} deg "
                f"plus offset within {ADULT_TIBIAL_TORSION_RANGE_DEG}), got {value}"
            )
        out[side] = value
    return out


def validate_tibial_torsion(document: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` unless the document's recorded offset is valid.

    A document without ``tibial_torsion_deg`` is the generic leg (valid).
    """
    record = document.get(TIBIAL_TORSION_KEY)
    if record is None:
        return
    if not isinstance(record, Mapping):
        raise ValueError(f"{TIBIAL_TORSION_KEY} must be a mapping")
    _check_offsets({side: record.get(side, math.nan) for side in SIDES})
    if not str(record.get("source", "")).strip():
        raise ValueError(f"{TIBIAL_TORSION_KEY} needs a cited source")


def malleolar_torsion_deg(document: Mapping[str, Any], side: str) -> float:
    """External torsion (deg) between the knee and ankle hinge axes about the shank.

    Positive is external (the ankle axis' lateral end turned posteriorly).
    """
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    up = _shank_axis(document, side)
    knee = np.asarray(_joint(document, f"knee_{side}")["child_to_follower"], float)
    ankle = np.asarray(_joint(document, f"ankle_{side}")["parent_to_base"], float)

    def flat(v: np.ndarray) -> np.ndarray:
        w = v - (v @ up) * up
        return w / np.linalg.norm(w)

    a, b = flat(knee[:3, 2]), flat(ankle[:3, 2])
    angle = math.degrees(math.atan2(float(np.cross(a, b) @ up), float(a @ b)))
    # external = the same turn direction that moves the toes laterally
    return angle * _lateral_sign(document, side)


def apply_tibial_torsion(
    document: Mapping[str, Any], offsets_deg: Mapping[str, float]
) -> dict[str, Any]:
    """Copy of ``document`` with each foot turned out by its torsion offset.

    The ankle base frame of each side is rotated about the shank long axis
    through the ankle centre (translation unchanged). DbC: offsets are finite
    and within ``OFFSET_RANGE_DEG``; a document that already records an
    offset raises (no double application). All-zero offsets return an
    unmodified deep copy without the record key, so the spec is bit-identical.
    """
    offsets = _check_offsets(offsets_deg)
    if TIBIAL_TORSION_KEY in document:
        raise ValueError(f"document already records {TIBIAL_TORSION_KEY}")
    out = copy.deepcopy(dict(document))
    if all(value == 0.0 for value in offsets.values()):
        return out
    for side in SIDES:
        angle = math.radians(offsets[side]) * _lateral_sign(document, side)
        joint = next(j for j in out["joints"] if j["name"] == f"ankle_{side}")
        base = np.asarray(joint["parent_to_base"], dtype=float)
        base[:3, :3] = _rotation(_shank_axis(document, side), angle) @ base[:3, :3]
        joint["parent_to_base"] = base.tolist()
    out[TIBIAL_TORSION_KEY] = {**offsets, "source": TIBIAL_TORSION_SOURCE}
    return out
