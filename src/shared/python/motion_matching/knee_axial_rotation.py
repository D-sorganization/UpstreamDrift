"""Opt-in knee axial (tibial internal/external) rotation coordinate (#15296).

The spec knee is a pure hinge, so every turn of the foot against the pelvis
comes from ``hip_rotation_*``; in the driver fit the lead hip sat on its
internal-rotation limit near impact. A flexed knee also turns about the shank:
the typical passive range at 90 deg flexion is about 20-30 deg. That is a best
estimate stated without a citation and nothing is measured on the golfer.

Off by default: the document is then unchanged bit for bit. Given a range, each
``knee_<side>`` gains a second primitive, ``Rx`` driving ``knee_rotation_<side>``
after the hinge. Both joint frames are turned about the hinge by one constant
angle so ``x`` is the shank long axis (ankle to knee centre) projected
perpendicular to the hinge; at zero the model is the hinge knee at every
flexion angle. Positive is internal rotation (toes to the midline) on both
sides, as ``hip_rotation_*``. The range goes to ``coordinate_ranges_deg``.

Limitations: the projected axis is a few degrees off the knee to ankle line, so
the ankle centre moves slightly; the range does not narrow toward full
extension as a real knee's does; no passive stiffness; no row in
``HUMAN_RANGES_DEG`` or the coordinate sign map.

Engines: the MuJoCo, Drake, Pinocchio and OpenSim emitters build every joint
from its primitive list and carry the coordinate without engine code. Consumers
that map knee coordinates by name call :func:`require_hinge_knee` and refuse it.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from src.shared.python.motion_matching.tibial_torsion import (
    SIDES,
    _lateral_sign,
    _shank_axis,
)

#: Builder kwarg, subject-preset setting and document record key.
KNEE_AXIAL_KEY = "knee_axial_rotation_deg"
#: Suggested range when the setting is turned on, deg (best estimate).
DEFAULT_KNEE_AXIAL_RANGE_DEG: tuple[float, float] = (-20.0, 20.0)
#: Refusal bound on either limit, deg: a sanity bound, not an anatomical claim.
MAX_KNEE_AXIAL_DEG: float = 45.0
#: Largest accepted angle between the rotation axis and the shank line, deg.
MAX_AXIS_OFFSET_DEG: float = 10.0
KNEE_AXIAL_BASIS = (
    "best estimate, not measured: typical passive knee axial rotation at "
    "90 deg flexion is about 20-30 deg; no citation"
)


def knee_axial_coordinates() -> tuple[str, ...]:
    """Coordinate names the setting adds, right then left."""
    return tuple(f"knee_rotation_{side}" for side in SIDES)


def resolve_knee_axial_range(value: Any) -> tuple[float, float] | None:
    """The ``(low, high)`` range in deg of a setting value, ``None`` when off.

    ``None`` is off; a number ``h`` is ``(-h, h)``; a pair is ``(low, high)``.
    Raises ``TypeError`` for another type and ``ValueError`` unless the limits
    are finite, ``low <= 0 <= high``, ``low < high``, within ``MAX_KNEE_AXIAL_DEG``.
    """
    if value is None:
        return None
    if isinstance(value, (bool, str)):
        raise TypeError(f"{KNEE_AXIAL_KEY} must be a number or a pair, got {value!r}")
    pair = (-value, value) if isinstance(value, (int, float)) else tuple(value)
    if len(pair) != 2:
        raise ValueError(f"{KNEE_AXIAL_KEY} needs (low, high), got {value!r}")
    low, high = float(pair[0]), float(pair[1])
    if not (
        math.isfinite(low)
        and math.isfinite(high)
        and -MAX_KNEE_AXIAL_DEG <= low <= 0.0 <= high <= MAX_KNEE_AXIAL_DEG
        and low < high
    ):
        raise ValueError(
            f"{KNEE_AXIAL_KEY} must be finite with low <= 0 <= high, low < high "
            f"and within +-{MAX_KNEE_AXIAL_DEG} deg, got ({low}, {high})"
        )
    return low, high


def _axis_turn(document: Mapping[str, Any], side: str) -> np.ndarray:
    """Constant turn about the hinge that puts the rotation axis on joint ``x``."""
    knee = next(j for j in document["joints"] if j["name"] == f"knee_{side}")
    follower = np.asarray(knee["child_to_follower"], dtype=float)[:3, :3]
    shank = _shank_axis(document, side)
    # A turn about ``shank`` moves the toes laterally when the sign is +1;
    # internal rotation is the opposite turn.
    axis = -_lateral_sign(document, side) * (follower.T @ shank)
    off_deg = math.degrees(math.asin(min(1.0, abs(float(axis[2])))))
    if off_deg > MAX_AXIS_OFFSET_DEG:
        raise ValueError(f"knee_{side} hinge is {off_deg:.1f} deg off the shank normal")
    cos, sin = axis[:2] / np.hypot(axis[0], axis[1])
    turn = np.eye(4)
    turn[:2, :2] = [[cos, -sin], [sin, cos]]
    return turn


def apply_knee_axial_rotation(
    document: Mapping[str, Any], range_deg: Any
) -> dict[str, Any]:
    """Copy of ``document`` with a bounded axial rotation on each knee.

    ``range_deg`` is :func:`resolve_knee_axial_range`'s; ``None`` returns an
    unmodified deep copy. Preconditions (``ValueError``): the document does not
    already record the setting and each knee is a single ``Rz`` hinge.
    Postcondition: at ``knee_rotation_* = 0`` every body pose is unchanged.
    """
    limits = resolve_knee_axial_range(range_deg)
    out = copy.deepcopy(dict(document))
    if limits is None:
        return out
    if KNEE_AXIAL_KEY in document:
        raise ValueError(f"document already records {KNEE_AXIAL_KEY}")
    order = out["coordinate_order"]
    for side, name in zip(SIDES, knee_axial_coordinates(), strict=True):
        joint = next(j for j in out["joints"] if j["name"] == f"knee_{side}")
        if [p["primitive"] for p in joint["primitives"]] != ["Rz"]:
            raise ValueError(f"knee_{side} is not a single Rz hinge")
        turn = _axis_turn(document, side)
        for key in ("parent_to_base", "child_to_follower"):
            joint[key] = (np.asarray(joint[key], dtype=float) @ turn).tolist()
        joint["primitives"].append({"primitive": "Rx", "coordinate": name})
        order.insert(order.index(f"knee_angle_{side}") + 1, name)
        out.setdefault("coordinate_ranges_deg", {})[name] = list(limits)
    out[KNEE_AXIAL_KEY] = {"range": list(limits), "basis": KNEE_AXIAL_BASIS}
    return out


def validate_knee_axial_rotation(document: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` unless the recorded setting matches the knees; a
    document without ``knee_axial_rotation_deg`` is the hinge knee (valid)."""
    record = document.get(KNEE_AXIAL_KEY)
    if record is None:
        return
    if not isinstance(record, Mapping) or not str(record.get("basis", "")).strip():
        raise ValueError(f"{KNEE_AXIAL_KEY} must be a mapping with a basis line")
    limits = resolve_knee_axial_range(record.get("range", ()))
    ranges = document.get("coordinate_ranges_deg", {})
    for side, name in zip(SIDES, knee_axial_coordinates(), strict=True):
        joint = next(j for j in document["joints"] if j["name"] == f"knee_{side}")
        primitives = [(p["primitive"], p["coordinate"]) for p in joint["primitives"]]
        if primitives != [("Rz", f"knee_angle_{side}"), ("Rx", name)]:
            raise ValueError(f"knee_{side} does not carry {name} after its hinge")
        if limits is None or tuple(ranges.get(name, ())) != limits:
            raise ValueError(f"{name} range differs from {KNEE_AXIAL_KEY}")


def knee_axial_bounds_rad(spec: Mapping[str, Any]) -> dict[str, tuple[float, float]]:
    """Radian solver bounds of the added coordinates (empty when off): the
    stated range exactly, since a range that is a setting is not widened."""
    if KNEE_AXIAL_KEY not in spec:
        return {}
    low, high = (math.radians(float(v)) for v in spec[KNEE_AXIAL_KEY]["range"])
    return dict.fromkeys(knee_axial_coordinates(), (low, high))


def require_hinge_knee(coordinate_order: Sequence[str], consumer: str) -> None:
    """Refuse a coordinate order that carries the opt-in knee rotation.

    For consumers that map knee coordinates by name (muscle grafts, retarget
    maps) and would otherwise drop it silently. Raises ``ValueError``.
    """
    if set(coordinate_order) & set(knee_axial_coordinates()):
        raise ValueError(
            f"{consumer} does not support the opt-in knee axial rotation "
            f"({KNEE_AXIAL_KEY}); rebuild the spec without the setting"
        )
