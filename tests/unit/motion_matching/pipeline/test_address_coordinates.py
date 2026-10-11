"""Address coordinate report and position-based toe-out (OSV-6, #11737)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_matching.foot_progression import forward_axis
from src.shared.python.motion_matching.pipeline.address_feet import (
    NATIVE_TARGET_AXIS,
    feet_deg_from_positions,
    split_address_coordinates,
)

pytestmark = [pytest.mark.unit]


def test_split_separates_angles_in_degrees_from_translations_in_metres() -> None:
    names = ("TranslationInputX", "TranslationInputZ", "HipInputZ", "knee_angle_l")
    q = np.array([0.5, 1.1, np.radians(30.0), np.radians(-12.0)])
    angles, translations = split_address_coordinates(names, q)
    assert angles == pytest.approx({"HipInputZ": 30.0, "knee_angle_l": -12.0})
    assert translations == pytest.approx(
        {"TranslationInputX": 0.5, "TranslationInputZ": 1.1}
    )


def test_split_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="coordinate"):
        split_address_coordinates(("a", "b"), np.zeros(3))


def _foot(heel: np.ndarray, direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return heel, heel + 0.16 * direction


def test_feet_deg_from_positions_matches_known_toe_out() -> None:
    up = np.array([0.0, 0.0, 1.0])
    target = NATIVE_TARGET_AXIS
    fwd = forward_axis(target, up, "right")
    # Lead (left) foot turned 20 deg toward the target, trail foot 8 deg away.
    lead = np.cos(np.radians(20.0)) * fwd + np.sin(np.radians(20.0)) * target
    trail = np.cos(np.radians(8.0)) * fwd - np.sin(np.radians(8.0)) * target
    calcn_l, toes_l = _foot(np.array([0.0, -0.1, 0.0]), lead)
    calcn_r, toes_r = _foot(np.array([0.0, 0.1, 0.0]), trail)
    positions = {
        "calcn_l": calcn_l,
        "toes_l": toes_l,
        "calcn_r": calcn_r,
        "toes_r": toes_r,
    }
    out = feet_deg_from_positions(positions, "right", NATIVE_TARGET_AXIS)
    assert out["left"] == pytest.approx(20.0, abs=1e-6)
    assert out["right"] == pytest.approx(8.0, abs=1e-6)


def test_feet_deg_from_positions_requires_all_bodies() -> None:
    with pytest.raises(ValueError, match="calcn_l"):
        feet_deg_from_positions({"toes_l": np.zeros(3)}, "right", NATIVE_TARGET_AXIS)
