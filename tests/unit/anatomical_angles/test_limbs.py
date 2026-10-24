"""HRG-E3 (#13171): Limb (hip, knee, shoulder, elbow) and line turn terms."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.shared.python.anatomical_angles.limbs import LIMB_GLOSSARY, limb_term_specs
from src.shared.python.spine import golf_terms
from src.shared.python.swing_comparison.events import SwingEvents

pytestmark = pytest.mark.unit


def _rot(seq: str, angles: object) -> np.ndarray:
    return Rotation.from_euler(seq, angles).as_matrix()


def test_line_turn_pelvis_and_thorax_registered_and_handed() -> None:
    terms = golf_terms.registered_terms()
    assert "pelvis_turn_line" in terms and "thorax_turn_line" in terms
    t = np.linspace(0, 1.0, 10)
    events = SwingEvents(0, 0.0, 5, 0.5, 8, 0.8, 9, 0.9)
    points = {
        "WaistLeft": np.array([[-0.1, 0.0, 1.0]] * 10),
        "WaistRight": np.array([[0.1, 0.0, 1.0]] * 10),
        "BackLeft": np.array([[-0.2, 0.0, 1.5]] * 10),
        "BackRight": np.array([[0.2, 0.0, 1.5]] * 10),
    }
    inputs = {
        "points": points,
        "t": t,
        "events": events,
        "up": np.array([0.0, 0.0, 1.0]),
    }
    p_turn_r = golf_terms.evaluate("pelvis_turn_line", inputs, handedness="right")
    p_turn_l = golf_terms.evaluate("pelvis_turn_line", inputs, handedness="left")
    assert np.allclose(p_turn_r, -p_turn_l, equal_nan=True)


def test_limb_terms_registered_with_complete_glossary() -> None:
    registered, specs = golf_terms.registered_terms(), limb_term_specs()
    assert len(specs) == 18
    for s in specs:
        assert s.term_id in registered and s.term_id in LIMB_GLOSSARY
        defn, entry = golf_terms.definition(s.term_id), LIMB_GLOSSARY[s.term_id]
        assert (
            defn.frame.strip()
            and defn.sign_convention.strip()
            and defn.reference.strip()
        )
        assert entry.formula.strip() and entry.sign.strip() and entry.available


@pytest.mark.parametrize(
    ("term", "inp_name", "rot", "expected_deg"),
    [
        (
            "lead_hip_flexion",
            "femur_in_pelvis_rotation",
            _rot("ZXY", [0.4, 0.0, 0.0]),
            np.rad2deg(0.4),
        ),
        (
            "trail_hip_abduction",
            "femur_in_pelvis_rotation",
            _rot("ZXY", [0.0, -0.3, 0.0]),
            np.rad2deg(0.3),
        ),
        (
            "trail_hip_internal_rotation",
            "femur_in_pelvis_rotation",
            _rot("ZXY", [0.0, 0.0, 0.25]),
            np.rad2deg(0.25),
        ),
        (
            "lead_knee_flexion",
            "tibia_in_femur_rotation",
            _rot("XYZ", [0.6, 0.0, 0.0]),
            np.rad2deg(0.6),
        ),
        (
            "lead_shoulder_elevation",
            "humerus_in_thorax_rotation",
            _rot("YXY", [np.pi / 2, 0.5, 0.0]),
            np.rad2deg(0.5),
        ),
        (
            "lead_shoulder_horizontal_adduction",
            "humerus_in_thorax_rotation",
            _rot("Z", np.pi / 2),
            90.0,
        ),
        (
            "lead_elbow_flexion",
            "forearm_in_humerus_rotation",
            _rot("ZXY", [0.8, 0.0, 0.0]),
            np.rad2deg(0.8),
        ),
    ],
)
def test_limb_term_signs(
    term: str, inp_name: str, rot: np.ndarray, expected_deg: float
) -> None:
    got = golf_terms.evaluate(term, {inp_name: rot}, handedness="right")
    assert got[0] == pytest.approx(expected_deg, rel=1e-5)


def test_synthetic_recovery_within_1e_6_rad() -> None:
    rng = np.random.default_rng(13171)
    angles = rng.uniform(-0.8, 0.8, size=(50, 3))
    rots = _rot("ZXY", angles)
    for i in range(len(angles)):
        got = golf_terms.evaluate(
            "trail_hip_flexion",
            {"femur_in_pelvis_rotation": rots[i : i + 1]},
            handedness="right",
        )
        assert abs(np.deg2rad(got[0]) - angles[i, 0]) < 1e-6


def test_gimbal_lock_yields_nan_never_zero() -> None:
    locked = _rot("ZXY", [0.2, np.pi / 2, 0.3])
    flex = golf_terms.evaluate(
        "lead_hip_flexion", {"femur_in_pelvis_rotation": locked}, handedness="right"
    )
    abd = golf_terms.evaluate(
        "lead_hip_abduction", {"femur_in_pelvis_rotation": locked}, handedness="right"
    )
    assert np.isnan(flex[0]) and np.isfinite(abd[0])


def test_mirroring_right_to_left_gives_identical_terms() -> None:
    rng = np.random.default_rng(42)
    a_hip = rng.uniform(-0.4, 0.4, size=3)
    hip_r, hip_l = _rot("ZXY", a_hip), _rot("ZXY", a_hip * [1.0, -1.0, -1.0])
    for term in ("hip_flexion", "hip_abduction", "hip_internal_rotation"):
        vr = golf_terms.evaluate(
            f"trail_{term}", {"femur_in_pelvis_rotation": hip_r}, handedness="right"
        )
        vl = golf_terms.evaluate(
            f"trail_{term}", {"femur_in_pelvis_rotation": hip_l}, handedness="left"
        )
        assert vr[0] == pytest.approx(vl[0], abs=1e-9)

    a_knee = rng.uniform(-0.4, 0.4, size=3)
    kr = golf_terms.evaluate(
        "trail_knee_flexion",
        {"tibia_in_femur_rotation": _rot("XYZ", a_knee)},
        handedness="right",
    )
    kl = golf_terms.evaluate(
        "trail_knee_flexion",
        {"tibia_in_femur_rotation": _rot("XYZ", a_knee * [1.0, -1.0, -1.0])},
        handedness="left",
    )
    assert kr[0] == pytest.approx(kl[0], abs=1e-9)

    sr = golf_terms.evaluate(
        "trail_shoulder_elevation",
        {"humerus_in_thorax_rotation": _rot("YXY", [0.4, 0.8, 0.3])},
        handedness="right",
    )
    sl = golf_terms.evaluate(
        "trail_shoulder_elevation",
        {"humerus_in_thorax_rotation": _rot("YXY", [-0.4, 0.8, -0.3])},
        handedness="left",
    )
    assert sr[0] == pytest.approx(sl[0], abs=1e-9)
