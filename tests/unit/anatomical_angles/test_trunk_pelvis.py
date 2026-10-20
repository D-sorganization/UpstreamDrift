"""HRG-E3 (#13171): pelvis and thorax ISB lab-frame terms in golf language."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.shared.python.anatomical_angles.trunk_pelvis import (
    GOLF_WORDS,
    TRUNK_PELVIS_GLOSSARY,
    UNAVAILABLE_V1,
    golfer_lab_frame,
    segment_lab_angles,
)
from src.shared.python.spine import golf_terms

pytestmark = pytest.mark.unit

UP = np.array([0.0, 0.0, 1.0])
TARGET = np.array([1.0, 0.0, 0.0])
E3_TERMS = (
    "pelvis_turn_isb",
    "pelvis_anterior_tilt_isb",
    "pelvis_lead_hip_up_isb",
    "thorax_turn_isb",
    "thorax_forward_bend_isb",
    "thorax_lead_side_bend_isb",
)
_D = np.diag([1.0, 1.0, -1.0])


def _zxy(a: np.ndarray) -> np.ndarray:
    """Intrinsic Rz(a0) Rx(a1) Ry(a2), rows of ``a`` in radians."""
    return Rotation.from_euler("ZXY", a).as_matrix()


def _inputs(pelvis: np.ndarray, thorax: np.ndarray, up=UP, target=TARGET) -> dict:
    return {
        "pelvis_rotation_world": pelvis,
        "thorax_rotation_world": thorax,
        "up": up,
        "target": target,
    }


def _all_terms(inputs: dict, handedness: str) -> np.ndarray:
    return np.column_stack(
        [golf_terms.evaluate(t, inputs, handedness=handedness) for t in E3_TERMS]
    )


def test_lab_frame_is_proper_and_faces_the_ball() -> None:
    for hand in ("right", "left"):
        lab = golfer_lab_frame(UP, TARGET, hand)
        assert np.allclose(lab.T @ lab, np.eye(3), atol=1e-12)
        assert np.linalg.det(lab) == pytest.approx(1.0)
        assert np.allclose(lab[:, 1], UP)
    # right-handed golfer: target on the golfer's left, so Z (right) = -target
    assert np.allclose(golfer_lab_frame(UP, TARGET, "right")[:, 2], -TARGET)
    assert np.allclose(golfer_lab_frame(UP, TARGET, "left")[:, 2], TARGET)


def test_synthetic_recovery_within_1e_6_rad() -> None:
    rng = np.random.default_rng(13171)
    angles = rng.uniform(-1.2, 1.2, size=(60, 3))
    for hand in ("right", "left"):
        lab = golfer_lab_frame(UP, TARGET, hand)
        got = segment_lab_angles(lab @ _zxy(angles), UP, TARGET, hand)
        assert np.max(np.abs(got - angles)) < 1e-6


def test_neutral_pose_is_zero_not_nan() -> None:
    for hand in ("right", "left"):
        lab = golfer_lab_frame(UP, TARGET, hand)
        out = _all_terms(_inputs(lab, lab), hand)
        assert np.allclose(out, 0.0, atol=1e-12)


def test_backswing_turn_is_positive_for_both_hands() -> None:
    # right-handed backswing: clockwise from above = negative about up
    lab_r = golfer_lab_frame(UP, TARGET, "right")
    turned = Rotation.from_rotvec(-0.7 * UP).as_matrix() @ lab_r
    got = golf_terms.evaluate(
        "pelvis_turn_isb", _inputs(turned, turned), handedness="right"
    )
    assert got[0] == pytest.approx(np.rad2deg(0.7))
    lab_l = golfer_lab_frame(UP, TARGET, "left")
    turned_l = Rotation.from_rotvec(0.7 * UP).as_matrix() @ lab_l
    got_l = golf_terms.evaluate(
        "thorax_turn_isb", _inputs(turned_l, turned_l), handedness="left"
    )
    assert got_l[0] == pytest.approx(np.rad2deg(0.7))


def test_forward_bend_toward_ball_is_positive() -> None:
    lab = golfer_lab_frame(UP, TARGET, "right")
    # bend: rotate about the golfer's right (Z) so anterior X tips downward
    bent = lab @ _zxy(np.array([-0.5, 0.0, 0.0]))
    assert (bent[:, 0] @ UP) < 0  # anterior axis points down toward the ball
    got = golf_terms.evaluate(
        "thorax_forward_bend_isb", _inputs(bent, bent), handedness="right"
    )
    assert got[0] == pytest.approx(np.rad2deg(0.5))
    tilt = golf_terms.evaluate(
        "pelvis_anterior_tilt_isb", _inputs(bent, bent), handedness="right"
    )
    assert tilt[0] == pytest.approx(np.rad2deg(0.5))


def test_lead_hip_up_sign() -> None:
    for hand, lead_sign in (("right", -1.0), ("left", 1.0)):
        lab = golfer_lab_frame(UP, TARGET, hand)
        # lab Z is the golfer's right; the lead hip is at lead_sign * Z
        pelvis = lab @ _zxy(np.array([0.0, 0.3, 0.0]))
        lead_hip_height = lead_sign * (pelvis[:, 2] @ UP)
        got = golf_terms.evaluate(
            "pelvis_lead_hip_up_isb", _inputs(pelvis, pelvis), handedness=hand
        )[0]
        assert np.sign(got) == np.sign(lead_hip_height)
        assert abs(got) == pytest.approx(np.rad2deg(0.3))


def test_mirroring_right_to_left_gives_identical_terms() -> None:
    rng = np.random.default_rng(7)
    lab = golfer_lab_frame(UP, TARGET, "right")
    pelvis = lab @ _zxy(rng.uniform(-1.0, 1.0, size=(40, 3)))
    thorax = lab @ _zxy(rng.uniform(-1.0, 1.0, size=(40, 3)))
    right = _all_terms(_inputs(pelvis, thorax), "right")
    # reflect across the plane normal to the target line; anatomical right of
    # the mirrored body is -M z, hence the right-multiplication by diag(1,1,-1)
    m = np.eye(3) - 2.0 * np.outer(TARGET, TARGET)
    left = _all_terms(
        _inputs(m @ pelvis @ _D, m @ thorax @ _D, target=m @ TARGET), "left"
    )
    assert np.max(np.abs(right - left)) < 1e-9


def test_frame_independence_under_world_rotation() -> None:
    rng = np.random.default_rng(11)
    lab = golfer_lab_frame(UP, TARGET, "right")
    pelvis = lab @ _zxy(rng.uniform(-1.0, 1.0, size=(30, 3)))
    thorax = lab @ _zxy(rng.uniform(-1.0, 1.0, size=(30, 3)))
    base = _all_terms(_inputs(pelvis, thorax), "right")
    q = Rotation.random(random_state=4).as_matrix()
    moved = _all_terms(_inputs(q @ pelvis, q @ thorax, q @ UP, q @ TARGET), "right")
    assert np.max(np.abs(base - moved)) < 1e-9


def test_gimbal_lock_is_nan_with_middle_angle_kept() -> None:
    lab = golfer_lab_frame(UP, TARGET, "right")
    locked = lab @ _zxy(np.array([0.2, np.pi / 2, 0.1]))
    out = _all_terms(_inputs(locked, locked), "right")
    turn_cols = [E3_TERMS.index(t) for t in ("pelvis_turn_isb", "thorax_turn_isb")]
    tilt_cols = [
        E3_TERMS.index(t)
        for t in ("pelvis_anterior_tilt_isb", "thorax_forward_bend_isb")
    ]
    side_cols = [
        E3_TERMS.index(t)
        for t in ("pelvis_lead_hip_up_isb", "thorax_lead_side_bend_isb")
    ]
    assert np.isnan(out[0, turn_cols + tilt_cols]).all()
    assert np.isfinite(out[0, side_cols]).all()
    assert np.allclose(np.abs(out[0, side_cols]), 90.0)


def test_terms_registered_with_complete_glossary() -> None:
    terms = golf_terms.registered_terms()
    for t in E3_TERMS:
        assert t in terms
        spec = golf_terms.definition(t)
        assert spec.frame and spec.sign_convention and spec.reference
        entry = TRUNK_PELVIS_GLOSSARY[t]
        for field in ("formula", "sign", "frame", "reference"):
            assert getattr(entry, field).strip(), (t, field)
        assert entry.available


def test_golf_words_resolve_to_registered_terms() -> None:
    terms = set(golf_terms.registered_terms())
    assert GOLF_WORDS["hip turn"] == "pelvis_turn_isb"
    assert GOLF_WORDS["shoulder turn"] == "thorax_turn_isb"
    for word, term in GOLF_WORDS.items():
        assert term in terms, word


def test_thoracic_lumbar_split_is_unavailable_not_zero() -> None:
    assert UNAVAILABLE_V1
    for name, entry in UNAVAILABLE_V1.items():
        assert not entry.available
        assert "single trunk joint" in entry.reason, name
        assert name not in golf_terms.registered_terms()


def test_missing_inputs_and_bad_handedness_raise() -> None:
    lab = golfer_lab_frame(UP, TARGET, "right")
    with pytest.raises(ValueError, match="pelvis_rotation_world"):
        golf_terms.evaluate(
            "pelvis_turn_isb", {"up": UP, "target": TARGET}, handedness="right"
        )
    with pytest.raises(ValueError, match="handedness"):
        golf_terms.evaluate("pelvis_turn_isb", _inputs(lab, lab), handedness="both")
    with pytest.raises(ValueError, match="parallel"):
        golfer_lab_frame(UP, UP, "right")
    with pytest.raises(ValueError, match="proper rotations"):
        segment_lab_angles(np.diag([1.0, 1.0, -1.0]), UP, TARGET, "right")
