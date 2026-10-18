"""HRG-E2 (#13169): engine-neutral wrist and forearm anatomical angles."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.shared.python.anatomical_angles import (
    OBSERVABILITY,
    WRIST_GOLF_ALIASES,
    BasisMap,
    ISB_FROM_MYOFULLBODY,
    alias_series,
    forearm_pronation,
    hand_frame_from_club,
    reconstruct_wrist_rotation,
    wrist_angles,
    wrist_forearm_angles,
    wu_wrist_frame,
)
from src.shared.python.biomechanics.joint_conventions import AnatomicalFrame
from src.shared.python.model_appearance.grip_pose import GripPose, GripStyle
from src.shared.python.motion_matching.grip_contract import GripSpec, HandAngles
from src.shared.python.spine import golf_terms

pytestmark = pytest.mark.unit

RECEIPT = "receipts/fit/receipt.json"


def _frame(rot: np.ndarray, cid: str = "t") -> AnatomicalFrame:
    return AnatomicalFrame(rot, np.zeros(np.shape(rot)[:-2] + (3,)), cid)


def _zxy(a: np.ndarray) -> np.ndarray:
    """Intrinsic Rz(a0) Rx(a1) Ry(a2), rows of ``a`` in radians."""
    return Rotation.from_euler("ZXY", a).as_matrix()


# Right-side signs: flexion +, ulnar +, pronation +.  The left-side proper frame
# (X volar, Y proximal, Z = X x Y = ulnar) makes the last two components flip.
_SIGN = {"R": np.array([1.0, 1.0, 1.0]), "L": np.array([1.0, -1.0, -1.0])}


def _hand_for(side: str, angles: np.ndarray, radius_rot: np.ndarray) -> AnatomicalFrame:
    local = _zxy(angles * _SIGN[side])
    return _frame(radius_rot @ local)


def test_recovers_known_angles_both_sides() -> None:
    rng = np.random.default_rng(3)
    angles = rng.uniform(-1.2, 1.2, size=(50, 3))
    radius_rot = Rotation.random(50, random_state=1).as_matrix()
    for side in ("R", "L"):
        out = wrist_angles(
            _frame(radius_rot),
            _hand_for(side, angles, radius_rot),
            side,
            receipt_path=RECEIPT,
        )
        got = np.column_stack(
            [out.flexion_rad, out.ulnar_deviation_rad, out.axial_rotation_rad]
        )
        assert np.max(np.abs(got - angles)) < 1e-6
        assert not np.isnan(got).any()


def test_rotation_reconstruction_within_1e_10() -> None:
    rng = np.random.default_rng(5)
    angles = rng.uniform(-1.2, 1.2, size=(40, 3))
    radius_rot = Rotation.random(40, random_state=2).as_matrix()
    for side in ("R", "L"):
        hand = _hand_for(side, angles, radius_rot)
        out = wrist_angles(_frame(radius_rot), hand, side, receipt_path=RECEIPT)
        rebuilt = reconstruct_wrist_rotation(out)
        target = np.swapaxes(radius_rot, -1, -2) @ hand.rotation_world
        assert np.max(np.abs(rebuilt - target)) < 1e-10


def test_left_right_mirroring_gives_identical_anatomical_values() -> None:
    rng = np.random.default_rng(9)
    angles = rng.uniform(-1.0, 1.0, size=(30, 3))
    radius_rot = Rotation.random(30, random_state=4).as_matrix()
    right = wrist_angles(
        _frame(radius_rot),
        _hand_for("R", angles, radius_rot),
        "R",
        receipt_path=RECEIPT,
    )
    # Reflect the right-side poses across the sagittal plane (world z -> -z):
    # a reflected proper frame keeps its axes' images but loses handedness, which
    # is restored by negating the Z column (volar, proximal -> ulnar for left).
    s = np.diag([1.0, 1.0, -1.0])
    refl = lambda r: s @ r @ np.diag([1.0, 1.0, -1.0])  # noqa: E731
    left = wrist_angles(
        _frame(refl(radius_rot)),
        _frame(refl(right_hand_rot(angles, radius_rot))),
        "L",
        receipt_path=RECEIPT,
    )
    np.testing.assert_allclose(left.flexion_rad, right.flexion_rad, atol=1e-9)
    np.testing.assert_allclose(
        left.ulnar_deviation_rad, right.ulnar_deviation_rad, atol=1e-9
    )
    np.testing.assert_allclose(
        left.axial_rotation_rad, right.axial_rotation_rad, atol=1e-9
    )


def right_hand_rot(angles: np.ndarray, radius_rot: np.ndarray) -> np.ndarray:
    return _hand_for("R", angles, radius_rot).rotation_world


def test_singularity_invalidates_only_indeterminate_components() -> None:
    angles = np.array([[0.3, 0.2, -0.4], [0.3, np.pi / 2, -0.4], [-0.1, 0.5, 0.2]])
    radius_rot = np.tile(np.eye(3), (3, 1, 1))
    out = wrist_angles(
        _frame(radius_rot),
        _hand_for("R", angles, radius_rot),
        "R",
        receipt_path=RECEIPT,
    )
    assert np.isnan(out.flexion_rad[1]) and np.isnan(out.axial_rotation_rad[1])
    assert out.ulnar_deviation_rad[1] == pytest.approx(np.pi / 2, abs=1e-6)
    assert not np.isnan(out.flexion_rad[[0, 2]]).any()
    assert out.flexion_rad[1] != 0.0  # never zero-filled: it is NaN
    assert "gimbal" in out.reasons["flexion"][1]
    assert "gimbal" in out.reasons["axial_rotation"][1]
    assert out.reasons["ulnar_deviation"][1] == ""
    assert out.reasons["flexion"][0] == ""


def test_series_carries_observability_and_receipt() -> None:
    r = np.eye(3)
    out = wrist_angles(_frame(r), _frame(r), "R", receipt_path=RECEIPT)
    assert out.observability == OBSERVABILITY == "fitted_grip_rotation"
    assert out.receipt_path == RECEIPT
    with pytest.raises(ValueError, match="receipt"):
        wrist_angles(_frame(r), _frame(r), "R", receipt_path="")
    with pytest.raises(ValueError, match="side"):
        wrist_angles(_frame(r), _frame(r), "X", receipt_path=RECEIPT)


def test_forearm_pronation_is_separate_from_wrist_third_angle() -> None:
    humerus = _frame(np.eye(3))
    forearm_rot = _zxy(np.array([[0.9, 0.05, 0.6]]))
    out = forearm_pronation(humerus, _frame(forearm_rot), "R", receipt_path=RECEIPT)
    assert out.pronation_rad[0] == pytest.approx(0.6, abs=1e-9)
    assert out.flexion_rad[0] == pytest.approx(0.9, abs=1e-9)
    wrist = wrist_angles(
        _frame(forearm_rot),
        _frame(forearm_rot @ _zxy(np.array([[0.1, 0.2, 0.3]]))),
        "R",
        receipt_path=RECEIPT,
    )
    assert wrist.axial_rotation_rad[0] == pytest.approx(0.3, abs=1e-9)
    combined = wrist_forearm_angles(
        humerus,
        _frame(forearm_rot),
        _frame(forearm_rot @ _zxy(np.array([[0.1, 0.2, 0.3]]))),
        "R",
        receipt_path=RECEIPT,
    )
    assert combined.forearm.pronation_rad[0] != combined.wrist.axial_rotation_rad[0]


def test_left_forearm_pronation_sign_matches_right_for_mirrored_motion() -> None:
    a = np.array([[0.4, 0.1, 0.7]])
    right = forearm_pronation(
        _frame(np.eye(3)), _frame(_zxy(a)), "R", receipt_path=RECEIPT
    )
    left = forearm_pronation(
        _frame(np.eye(3)), _frame(_zxy(a * _SIGN["L"])), "L", receipt_path=RECEIPT
    )
    assert left.pronation_rad[0] == pytest.approx(right.pronation_rad[0], abs=1e-9)


# -- hand frame ---------------------------------------------------------------


def _club_pose(n: int = 4) -> tuple[np.ndarray, np.ndarray]:
    return Rotation.random(n, random_state=7).as_matrix(), np.zeros((n, 3))


def test_hand_frame_from_club_and_physical_grip() -> None:
    rot, org = _club_pose()
    spec = GripSpec.preset("strong", "palm")
    a = hand_frame_from_club(rot, org, np.eye(3), spec.lead, calibration_id="g")
    b = hand_frame_from_club(rot, org, np.eye(3), HandAngles(15.0, 65.0, 0.0), "g")
    np.testing.assert_array_equal(a.rotation_world, b.rotation_world)
    weak = hand_frame_from_club(
        rot, org, np.eye(3), GripSpec.preset("weak").lead, calibration_id="g"
    )
    assert not np.allclose(a.rotation_world, weak.rotation_world)


def test_hand_frame_rejects_visual_grip_pose() -> None:
    rot, org = _club_pose()
    with pytest.raises(TypeError, match="visual"):
        hand_frame_from_club(rot, org, np.eye(3), GripPose(), calibration_id="g")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="HandAngles"):
        hand_frame_from_club(rot, org, np.eye(3), {"alpha_deg": 0}, calibration_id="g")  # type: ignore[arg-type]


def test_appearance_and_visual_grip_do_not_change_wrist_angles() -> None:
    rot, org = _club_pose(6)
    radius_rot = Rotation.random(6, random_state=11).as_matrix()
    base = GripSpec.preset("neutral", "neutral")
    varied = dataclasses.replace(
        base,
        style=GripStyle.INTERLOCK,
        lead_thumb_offset_deg=base.lead_thumb_offset_deg + 17.0,
    )
    document = {"appearance": {"skin_tone": "skin_6", "hair": "hair_black"}}
    document_b = {"appearance": {"skin_tone": "skin_1", "hair": "hair_blond"}}
    assert document != document_b

    def run(spec: GripSpec) -> np.ndarray:
        hand = hand_frame_from_club(rot, org, np.eye(3), spec.lead, "g")
        out = wrist_angles(_frame(radius_rot), hand, "L", receipt_path=RECEIPT)
        return np.column_stack(
            [out.flexion_rad, out.ulnar_deviation_rad, out.axial_rotation_rad]
        )

    assert run(base).tobytes() == run(varied).tobytes()


# -- golf aliases and registry ------------------------------------------------


def test_aliases_and_no_release_sign_alias() -> None:
    assert "release" not in WRIST_GOLF_ALIASES
    assert WRIST_GOLF_ALIASES["cupped"].component == "flexion"
    assert WRIST_GOLF_ALIASES["cupped"].sign == -1.0  # extension
    assert WRIST_GOLF_ALIASES["bowed"].sign == 1.0  # flexion
    assert WRIST_GOLF_ALIASES["hinge"].component == "ulnar_deviation"
    assert WRIST_GOLF_ALIASES["hinge"].sign == -1.0  # radial deviation
    assert WRIST_GOLF_ALIASES["cock"] is WRIST_GOLF_ALIASES["hinge"]
    with pytest.raises(ValueError, match="release"):
        alias_series(
            wrist_angles(_frame(np.eye(3)), _frame(np.eye(3)), "L", receipt_path="r"),
            "release",
        )


def test_alias_series_cupped_is_extension() -> None:
    radius = np.eye(3)
    hand = _zxy(np.array([-0.2, 0.0, 0.0]))
    out = wrist_angles(_frame(radius), _frame(hand), "L", receipt_path=RECEIPT)
    assert alias_series(out, "cupped")[0] == pytest.approx(np.rad2deg(0.2))
    assert alias_series(out, "bowed")[0] == pytest.approx(-np.rad2deg(0.2))


def test_terms_extend_existing_registry_with_handedness() -> None:
    terms = golf_terms.registered_terms()
    for t in (
        "lead_wrist_flexion_extension",
        "trail_wrist_ulnar_deviation",
        "lead_forearm_pronation",
    ):
        assert t in terms
        assert "fitted_grip_rotation" in golf_terms.definition(t).description
    angles = np.array([[0.2, 0.3, -0.1]])
    rel = _zxy(angles * _SIGN["R"])
    # right-handed golfer: lead = left hand; the term uses lead_side("right") = L
    left_expected = angles * 1.0
    got = golf_terms.evaluate(
        "lead_wrist_flexion_extension",
        {"hand_in_radius_rotation": _zxy(angles * _SIGN["L"])},
        handedness="right",
    )
    assert got[0] == pytest.approx(np.rad2deg(left_expected[0, 0]))
    trail = golf_terms.evaluate(
        "trail_wrist_ulnar_deviation",
        {"hand_in_radius_rotation": rel},
        handedness="right",
    )
    assert trail[0] == pytest.approx(np.rad2deg(0.3))
    swapped = golf_terms.evaluate(
        "lead_wrist_ulnar_deviation",
        {"hand_in_radius_rotation": rel},
        handedness="left",
    )
    assert swapped[0] == pytest.approx(np.rad2deg(0.3))


def test_term_gimbal_is_nan_not_zero() -> None:
    rel = _zxy(np.array([[0.3, np.pi / 2, 0.2]]))
    out = golf_terms.evaluate(
        "trail_wrist_flexion_extension",
        {"hand_in_radius_rotation": rel},
        handedness="right",
    )
    assert np.isnan(out[0])


# -- engine basis map ---------------------------------------------------------


def test_basis_map_myofullbody_to_isb() -> None:
    # MyoFullBody: x right, y up, z posterior -> ISB: X anterior, Y up, Z right.
    assert ISB_FROM_MYOFULLBODY.apply_vector([0, 0, -1]).tolist() == [1, 0, 0]
    assert ISB_FROM_MYOFULLBODY.apply_vector([1, 0, 0]).tolist() == [0, 0, 1]
    assert ISB_FROM_MYOFULLBODY.apply_vector([0, 1, 0]).tolist() == [0, 1, 0]
    r = Rotation.from_euler("y", 0.4).as_matrix()
    out = ISB_FROM_MYOFULLBODY.apply_rotation(r)
    np.testing.assert_allclose(out @ out.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(out), 1.0, atol=1e-12)
    assert out[1, 1] == pytest.approx(1.0)  # yaw about up stays about up
    with pytest.raises(ValueError, match="proper rotation"):
        BasisMap(np.diag([1.0, 1.0, -1.0]), "mirror")


def test_wu_frame_builder_is_proper_and_side_aware() -> None:
    for side in ("R", "L"):
        f = wu_wrist_frame(
            side,
            volar_world=[1, 0, 0],
            proximal_world=[0, 1, 0.0],
            origin_world=[0, 0, 0],
            calibration_id="wu",
        )
        assert np.linalg.det(f.rotation_world) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="parallel"):
        wu_wrist_frame("R", [0, 1, 0], [0, 1, 0], [0, 0, 0], "wu")


def test_anatomical_senses_from_physical_motions() -> None:
    """Flexion moves the fingers volar; ulnar deviation moves them ulnar; the
    pronation sense carries the thumb (radial side) across the volar side."""
    theta = 0.35
    for side in ("R", "L"):
        radius = wu_wrist_frame(side, [1, 0, 0], [0, 1, 0], [0, 0, 0], "wu")
        # Right: Z = radial.  Left: Z = ulnar.  Ulnar direction in world:
        ulnar_world = np.array([0, 0, -1.0 if side == "R" else 1.0])
        radial_world = -ulnar_world
        distal = np.array([0, -1.0, 0])
        for name, axis_world in (
            ("flexion", np.array([0, 0, 1.0])),  # fingers toward volar (+x)
            ("ulnar", np.cross(distal, ulnar_world) * 1.0),  # fingers toward ulnar
            ("pron", np.array([0, 1.0, 0])),
        ):
            if name == "pron":
                # thumb toward volar: rotation about +y moves radial -> volar
                # iff radial x ... pick the sign that does so physically.
                for s in (1.0, -1.0):
                    moved = Rotation.from_rotvec(s * theta * axis_world).apply(
                        radial_world
                    )
                    if moved @ np.array([1.0, 0, 0]) > 0:
                        sign = s
                        break
                axis_world = sign * axis_world
            rw = Rotation.from_rotvec(theta * axis_world).as_matrix()
            hand = _frame(rw @ radius.rotation_world)
            out = wrist_angles(radius, hand, side, receipt_path=RECEIPT)
            got = {
                "flexion": out.flexion_rad[0],
                "ulnar": out.ulnar_deviation_rad[0],
                "pron": out.axial_rotation_rad[0],
            }[name]
            moved_fingers = Rotation.from_rotvec(theta * axis_world).apply(distal)
            if name == "flexion":
                assert moved_fingers[0] > 0  # volar
            if name == "ulnar":
                assert moved_fingers @ ulnar_world > 0
            assert got == pytest.approx(theta, abs=1e-9), (side, name)


def test_myofullbody_segment_rotation_uses_explicit_basis() -> None:
    from src.shared.python.myofullbody.anatomy import (
        SegmentFrames,
        segment_rotation_isb,
    )

    k = Rotation.from_euler("x", 0.3).as_matrix()
    frames = SegmentFrames({"hand_r": 0}, {"hand_r": k}, np.zeros(3), np.eye(3))
    body = Rotation.from_euler("y", 0.5).as_matrix()
    out = segment_rotation_isb(frames, "hand_r", body, ISB_FROM_MYOFULLBODY)
    np.testing.assert_allclose(out, ISB_FROM_MYOFULLBODY.apply_rotation(body @ k))
    with pytest.raises(ValueError):
        segment_rotation_isb(frames, "nope", body, ISB_FROM_MYOFULLBODY)
