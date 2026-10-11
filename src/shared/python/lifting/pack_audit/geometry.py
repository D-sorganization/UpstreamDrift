"""Bar and hand geometry derived from body-origin positions (pure numpy).

Everything here is engine independent: an adapter supplies canonical-frame
body origins and these functions reduce them to the quantities the lift epic
tracks (bar centre, hand-to-bar distance, grip width, hand height above feet).

Unavailable is never zero: a quantity that cannot be computed is ``None``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

Positions = Mapping[str, Any]

_BAR_PARTS = ("barbell_shaft", "barbell_left_sleeve", "barbell_right_sleeve")


def _vec(positions: Positions, name: str) -> np.ndarray:
    if name not in positions:
        raise ValueError(f"body {name!r} missing from positions")
    arr = np.asarray(positions[name], dtype=float)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"body {name!r} position must be a finite 3-vector")
    return arr


def bar_frame(positions: Positions) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(centre, unit axis)`` of the bar, axis pointing left-sleeve ward.

    The axis is frame agnostic: it runs from the right sleeve origin to the
    left sleeve origin, so a pack's local bar axis convention is irrelevant.

    Raises:
        ValueError: If a bar part is missing or the sleeves coincide.
    """
    for part in _BAR_PARTS:
        _vec(positions, part)
    left = _vec(positions, "barbell_left_sleeve")
    right = _vec(positions, "barbell_right_sleeve")
    span = left - right
    length = float(np.linalg.norm(span))
    if length < 1e-9:
        raise ValueError("barbell sleeves coincide; bar axis is undefined")
    return _vec(positions, "barbell_shaft"), span / length


def hand_bar_metrics(positions: Positions) -> dict[str, dict[str, float]]:
    """Per-hand grip geometry relative to the bar.

    ``axis_distance_m`` is the perpendicular distance from the hand body origin
    to the bar centreline, ``lateral_m`` its signed offset along the bar from
    the bar centre (positive toward the left sleeve), and ``centre_distance_m``
    the straight-line distance to the bar centre.
    """
    centre, axis = bar_frame(positions)
    out: dict[str, dict[str, float]] = {}
    for side in ("l", "r"):
        rel = _vec(positions, f"hand_{side}") - centre
        lateral = float(rel @ axis)
        perp = rel - lateral * axis
        out[side] = {
            "axis_distance_m": float(np.linalg.norm(perp)),
            "lateral_m": lateral,
            "centre_distance_m": float(np.linalg.norm(rel)),
        }
    return out


def grip_width_m(positions: Positions) -> float:
    """Distance between the two hand body origins."""
    return float(np.linalg.norm(_vec(positions, "hand_l") - _vec(positions, "hand_r")))


def feet_midpoint(positions: Positions) -> np.ndarray:
    """Midpoint of the two foot body origins."""
    return 0.5 * (_vec(positions, "foot_l") + _vec(positions, "foot_r"))


def pose_summary(positions: Positions) -> dict[str, Any]:
    """Placement-independent reduction of one pose, all lengths in metres.

    Bar and hand positions are reported relative to the foot midpoint, so the
    pelvis placement an engine chooses cannot leak into the comparison.
    """
    feet = feet_midpoint(positions)
    centre, axis = bar_frame(positions)
    hands = hand_bar_metrics(positions)
    mid_hand = 0.5 * (_vec(positions, "hand_l") + _vec(positions, "hand_r"))
    return {
        "bar_centre_rel_feet_m": (centre - feet).tolist(),
        "bar_axis": axis.tolist(),
        "hand_mid_rel_feet_m": (mid_hand - feet).tolist(),
        "hand_l_rel_feet_m": (_vec(positions, "hand_l") - feet).tolist(),
        "hand_r_rel_feet_m": (_vec(positions, "hand_r") - feet).tolist(),
        "grip_width_m": grip_width_m(positions),
        "hand_bar": hands,
        "foot_separation_m": float(
            np.linalg.norm(_vec(positions, "foot_l") - _vec(positions, "foot_r"))
        ),
    }


def max_abs_difference(a: Positions, b: Positions, names: list[str]) -> float:
    """Largest component difference over *names* (both mappings must hold them)."""
    if not names:
        raise ValueError("names must be non-empty")
    return max(float(np.max(np.abs(_vec(a, n) - _vec(b, n)))) for n in names)
