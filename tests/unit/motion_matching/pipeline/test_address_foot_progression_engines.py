"""Per-engine address toe-out scoring from body poses (OSV-6, #11737)."""

from __future__ import annotations

import numpy as np
import pytest

from scripts import address_foot_progression_engines as afe
from src.shared.python.motion_matching.foot_progression import forward_axis
from src.shared.python.motion_matching.pipeline.address_feet import (
    NATIVE_TARGET_AXIS,
    NATIVE_UP_AXIS,
)

pytestmark = [pytest.mark.unit]


def _points(lead_deg: float, trail_deg: float) -> dict[str, np.ndarray]:
    fwd = forward_axis(NATIVE_TARGET_AXIS, NATIVE_UP_AXIS, "right")
    lead = np.cos(np.radians(lead_deg)) * fwd + np.sin(np.radians(lead_deg)) * (
        NATIVE_TARGET_AXIS
    )
    trail = np.cos(np.radians(trail_deg)) * fwd - np.sin(np.radians(trail_deg)) * (
        NATIVE_TARGET_AXIS
    )
    left, right = np.array([0.0, -0.1, 0.0]), np.array([0.0, 0.1, 0.0])
    return {
        "calcn_l": left,
        "toes_l": left + 0.16 * lead,
        "calcn_r": right,
        "toes_r": right + 0.16 * trail,
    }


def test_score_reports_errors_against_targets() -> None:
    out = afe.score_engine(_points(18.0, 5.0), {"left": 16.4, "right": 4.0})
    assert out["available"] is True
    assert out["model_deg"] == pytest.approx({"left": 18.0, "right": 5.0})
    assert out["error_deg"] == pytest.approx({"left": 1.6, "right": 1.0})
    assert out["within_tolerance"] is True


def test_score_flags_error_beyond_two_degrees() -> None:
    out = afe.score_engine(_points(21.0, 4.0), {"left": 16.4, "right": 4.0})
    assert out["within_tolerance"] is False


def test_score_requires_both_targets() -> None:
    with pytest.raises(ValueError, match="right"):
        afe.score_engine(_points(10.0, 5.0), {"left": 10.0})


def test_unavailable_carries_reason_and_no_angle() -> None:
    out = afe.unavailable("no hip_rotation joint")
    assert out["available"] is False
    assert out["model_deg"] is None and out["error_deg"] is None
    assert "hip_rotation" in out["reason"]
    with pytest.raises(ValueError):
        afe.unavailable("")


def test_coordinate_vector_orders_by_plant_and_converts_units() -> None:
    q = afe.coordinate_vector_rad(
        ("knee_angle_l", "TranslationInputX"),
        {"knee_angle_l": 90.0},
        {"TranslationInputX": 0.25},
    )
    assert q == pytest.approx([np.pi / 2, 0.25])
    with pytest.raises(ValueError, match="mtp_angle_r"):
        afe.coordinate_vector_rad(("mtp_angle_r",), {}, {})


@pytest.mark.parametrize("sign_l", [1.0, -1.0])
def test_solve_hip_rotation_is_sign_safe(sign_l: float) -> None:
    def foot(a: dict[str, float]) -> dict[str, float]:
        return {
            "left": 3.0 + sign_l * 0.9 * a["hip_rotation_l"],
            "right": -2.0 - 1.1 * a["hip_rotation_r"],
        }

    base = {"hip_rotation_l": 0.0, "hip_rotation_r": 0.0}
    solved = afe.solve_hip_rotation(foot, base, {"left": 16.4, "right": 4.0})
    out = foot(solved)
    assert out["left"] == pytest.approx(16.4, abs=0.05)
    assert out["right"] == pytest.approx(4.0, abs=0.05)


def test_solve_hip_rotation_rejects_unresponsive_foot() -> None:
    base = {"hip_rotation_l": 0.0, "hip_rotation_r": 0.0}
    with pytest.raises(ValueError, match="barely"):
        afe.solve_hip_rotation(
            lambda a: {"left": 1.0, "right": 2.0}, base, {"left": 5.0, "right": 5.0}
        )


def test_pelvis_rotation_is_a_proper_rotation_about_z_for_yaw() -> None:
    r = afe.pelvis_rotation({"HipInputX": 0.0, "HipInputY": 0.0, "HipInputZ": 90.0})
    assert r @ np.array([1.0, 0.0, 0.0]) == pytest.approx([0.0, 1.0, 0.0], abs=1e-12)
    assert np.linalg.det(r) == pytest.approx(1.0)
