"""Foot progression (toe-out) angle: definitions, capture window, defaults (OSV-4, #11730)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_matching.foot_progression import (
    DEFAULT_TOE_OUT_DEG,
    FootProgression,
    address_window,
    capture_foot_progression,
    forward_axis,
    foot_role,
    marker_long_axis,
    model_long_axis,
    progression_angle_deg,
    resolve_toe_out_target,
)

pytestmark = [pytest.mark.unit]

UP = np.array([0.0, 0.0, 1.0])
TARGET = np.array([0.0, 1.0, 0.0])  # target toward +y; golfer faces +x (x_t cross z)


def _axis(angle_deg: float, *, out: np.ndarray, pitch_deg: float = 0.0) -> np.ndarray:
    """Foot long axis turned ``angle_deg`` from forward (+x) toward ``out``."""
    a = np.radians(angle_deg)
    p = np.radians(pitch_deg)
    flat = np.cos(a) * np.array([1.0, 0.0, 0.0]) + np.sin(a) * out
    return np.cos(p) * flat + np.sin(p) * UP


def test_forward_axis_points_toward_the_ball_for_right_handed() -> None:
    np.testing.assert_allclose(forward_axis(TARGET, UP), [1.0, 0.0, 0.0], atol=1e-12)


def test_forward_axis_mirrors_for_left_handed() -> None:
    np.testing.assert_allclose(
        forward_axis(TARGET, UP, handedness="left"), [-1.0, 0.0, 0.0], atol=1e-12
    )


@pytest.mark.parametrize("angle", [0.0, 20.0, -10.0])
@pytest.mark.parametrize("role", ["lead", "trail"])
def test_synthetic_foot_recovers_signed_angle(angle: float, role: str) -> None:
    out = TARGET if role == "lead" else -TARGET
    axis = _axis(angle, out=out)
    got = progression_angle_deg(axis, target_axis=TARGET, up=UP, foot_role=role)
    assert got == pytest.approx(angle, abs=1e-9)


@pytest.mark.parametrize("angle", [0.0, 20.0, -10.0])
def test_left_handed_mirror_gives_same_angle(angle: float) -> None:
    # Mirror the whole scene in the target-forward plane: left-handed golfer.
    forward = forward_axis(TARGET, UP, handedness="left")
    lead_out = TARGET
    a = np.radians(angle)
    axis = np.cos(a) * forward + np.sin(a) * lead_out
    got = progression_angle_deg(
        axis, target_axis=TARGET, up=UP, foot_role="lead", handedness="left"
    )
    assert got == pytest.approx(angle, abs=1e-9)


def test_ground_projection_removes_pitch() -> None:
    flat = _axis(20.0, out=TARGET)
    pitched = _axis(20.0, out=TARGET, pitch_deg=25.0)
    a = progression_angle_deg(flat, target_axis=TARGET, up=UP, foot_role="lead")
    b = progression_angle_deg(pitched, target_axis=TARGET, up=UP, foot_role="lead")
    assert a == pytest.approx(b, abs=1e-9)


def test_vertical_axis_is_rejected() -> None:
    with pytest.raises(ValueError, match="vertical"):
        progression_angle_deg(UP, target_axis=TARGET, up=UP, foot_role="lead")


def test_invalid_role_and_handedness_are_rejected() -> None:
    with pytest.raises(ValueError, match="foot_role"):
        progression_angle_deg(
            _axis(0.0, out=TARGET), target_axis=TARGET, up=UP, foot_role="x"
        )
    with pytest.raises(ValueError, match="handedness"):
        forward_axis(TARGET, UP, handedness="ambi")


def test_foot_role_by_handedness() -> None:
    assert foot_role("left", "right") == "lead"
    assert foot_role("right", "right") == "trail"
    assert foot_role("left", "left") == "trail"
    assert foot_role("right", "left") == "lead"
    with pytest.raises(ValueError, match="side"):
        foot_role("middle", "right")


def test_model_long_axis_is_calcn_to_toes() -> None:
    calcn = np.array([0.0, 0.0, 0.05])
    toes = calcn + 0.16 * _axis(15.0, out=TARGET, pitch_deg=5.0)
    axis = model_long_axis(calcn, toes, UP)
    got = progression_angle_deg(axis, target_axis=TARGET, up=UP, foot_role="lead")
    assert got == pytest.approx(15.0, abs=1e-9)


def _foot_markers(
    angle_deg: float, *, out: np.ndarray, centre: np.ndarray, lateral: float = 0.04
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Heel/ankle-out, toe-in and toe-out markers of a foot turned ``angle_deg``."""
    axis = _axis(angle_deg, out=out)
    lat = np.cross(UP, axis)
    lat = lat if lat @ out > 0 else -lat  # outward-pointing ground normal to the axis
    heel_centre = centre - 0.14 * axis
    ankle_out = heel_centre + lateral * lat + np.array([0, 0, 0.1])
    toe_mid = centre + 0.02 * axis
    toe_out = toe_mid + 0.05 * lat
    toe_in = toe_mid - 0.05 * lat
    return ankle_out, toe_in, toe_out


def test_marker_axis_without_correction_is_biased_by_lateral_ankle() -> None:
    ankle, t_in, t_out = _foot_markers(0.0, out=TARGET, centre=np.zeros(3))
    biased = marker_long_axis(ankle, t_in, t_out, up=UP)
    raw = progression_angle_deg(biased, target_axis=TARGET, up=UP, foot_role="lead")
    # the lateral malleolus sits outside the heel centre line: reads toe-in
    assert raw < -5.0


def test_marker_axis_with_lateral_correction_recovers_angle() -> None:
    for angle in (0.0, 20.0, -10.0):
        ankle, t_in, t_out = _foot_markers(angle, out=TARGET, centre=np.zeros(3))
        axis = marker_long_axis(
            ankle, t_in, t_out, up=UP, ankle_lateral_offset_m=0.04, out_dir=TARGET
        )
        got = progression_angle_deg(axis, target_axis=TARGET, up=UP, foot_role="lead")
        assert got == pytest.approx(angle, abs=0.5)


def test_marker_axis_requires_out_dir_with_correction() -> None:
    ankle, t_in, t_out = _foot_markers(0.0, out=TARGET, centre=np.zeros(3))
    with pytest.raises(ValueError, match="out_dir"):
        marker_long_axis(ankle, t_in, t_out, up=UP, ankle_lateral_offset_m=0.04)


# ---------------------------------------------------------------- capture


LABELS = (
    "LAnkleOut",
    "LToeIn",
    "LToeOut",
    "RAnkleOut",
    "RToeIn",
    "RToeOut",
    "LWristTop",
    "RWristTop",
)


def _capture(
    lead_deg: float, trail_deg: float, frames: int = 40, takeaway: int = 25
) -> tuple[np.ndarray, np.ndarray]:
    """Right-handed capture: left (lead) foot at +y, golfer facing +x, target +y."""
    lead = _foot_markers(
        lead_deg, out=TARGET, centre=np.array([0.0, 0.3, 0.0]), lateral=0.04
    )
    trail = _foot_markers(
        trail_deg, out=-TARGET, centre=np.array([0.0, -0.3, 0.0]), lateral=0.04
    )
    wrists = (np.array([0.3, 0.05, 0.8]), np.array([0.3, -0.05, 0.8]))
    pts = np.zeros((frames, len(LABELS), 3))
    for i, p in enumerate((*lead, *trail, *wrists)):
        pts[:, i] = p
    rng = np.random.default_rng(0)
    pts[:, :6] += rng.normal(0.0, 0.0005, size=(frames, 6, 3))
    # the wrists leave the address pose after ``takeaway``
    ramp = np.clip(np.arange(frames) - takeaway, 0, None)[:, None] * 0.03
    pts[:, 6:, 0] += ramp
    valid = np.ones((frames, len(LABELS)), dtype=bool)
    return pts, valid


def test_address_window_ends_at_takeaway() -> None:
    pts, valid = _capture(18.0, 2.0)
    window = address_window(pts, valid, LABELS)
    assert window[0] == 0
    assert 20 <= window[-1] <= 26


def test_address_window_requires_wrists() -> None:
    pts, valid = _capture(18.0, 2.0)
    with pytest.raises(ValueError, match="wrist"):
        address_window(pts[:, :6], valid[:, :6], LABELS[:6])


def test_capture_foot_progression_measures_both_feet() -> None:
    pts, valid = _capture(18.0, 4.0)
    result = capture_foot_progression(pts, valid, LABELS, up=UP)
    assert result["left"].angle_deg == pytest.approx(18.0, abs=1.0)
    assert result["right"].angle_deg == pytest.approx(4.0, abs=1.0)
    assert result["left"].role == "lead"
    assert result["right"].role == "trail"
    assert not result["left"].is_default
    assert result["left"].reliable


def test_capture_foot_progression_uses_explicit_target_axis() -> None:
    pts, valid = _capture(18.0, 4.0)
    result = capture_foot_progression(pts, valid, LABELS, up=UP, target_axis=TARGET)
    assert result["left"].angle_deg == pytest.approx(18.0, abs=1.0)


def test_missing_foot_markers_fall_back_to_flagged_default() -> None:
    pts, valid = _capture(18.0, 4.0)
    valid[:, LABELS.index("RToeIn")] = False
    result = capture_foot_progression(pts, valid, LABELS, up=UP)
    right = result["right"]
    assert right.is_default
    assert not right.reliable
    assert right.angle_deg == DEFAULT_TOE_OUT_DEG
    assert "marker" in right.reason


def test_noisy_markers_are_flagged_unreliable() -> None:
    pts, valid = _capture(18.0, 4.0)
    rng = np.random.default_rng(1)
    pts[:, :6] += rng.normal(0.0, 0.03, size=pts[:, :6].shape)
    result = capture_foot_progression(pts, valid, LABELS, up=UP)
    assert not result["left"].reliable


def test_resolve_target_prefers_reliable_measurement() -> None:
    fp = FootProgression(
        side="left",
        role="lead",
        angle_deg=15.0,
        raw_angle_deg=2.0,
        forefoot_angle_deg=17.0,
        frames=20,
        spread_deg=0.5,
        reliable=True,
        is_default=False,
        reason="measured",
    )
    assert resolve_toe_out_target(fp) == (15.0, False)
    assert resolve_toe_out_target(None) == (DEFAULT_TOE_OUT_DEG, True)
    unreliable = FootProgression(
        **{**fp.__dict__, "reliable": False, "is_default": True, "angle_deg": 20.0}
    )
    assert resolve_toe_out_target(unreliable) == (DEFAULT_TOE_OUT_DEG, True)
