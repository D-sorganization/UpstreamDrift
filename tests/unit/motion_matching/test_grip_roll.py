"""Golf-language address grip roll (SHOW-A1 #14550, re-zeroed by SHOW-A7 #14731).

Owner convention (#14731): 0 deg = the back of the wrist points straight up, on
top of the shaft (12 o'clock on the face-on clock); + = counterclockwise on that
clock, toward the target. Synthetic hands are built at a known roll about a tilted
shaft, so the recovered value is checked against the input, not the implementation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.shared.python.contracts import ContractViolationError
from src.shared.python.impact_parameters.target_frame import TargetFrame
from src.shared.python.motion_matching.grip_roll import (
    GripPoseState,
    address_grip_roll_deg,
    grip_pose_state_from_spec,
    reference_axes,
)

pytestmark = pytest.mark.unit

# A realistic address shaft: from the hands down and toward the ball, in the
# face-on plane (x-z), golfer facing -x, target along -y (UD default frame).
SHAFT = np.array([-0.718, 0.028, -0.696])
SHAFT = SHAFT / np.linalg.norm(SHAFT)


def _club_rotation(shaft: np.ndarray = SHAFT, spin_deg: float = 0.0) -> np.ndarray:
    """World <- club frame with column 1 (club y) along ``shaft``."""
    y = shaft / np.linalg.norm(shaft)
    x = np.cross(y, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    z = np.cross(x, y)
    base = np.column_stack([x, y, z])
    spin = Rotation.from_rotvec(np.radians(spin_deg) * y).as_matrix()
    return spin @ base


def _state(lead_deg: float, trail_deg: float, spin_deg: float = 0.0) -> GripPoseState:
    """Hands whose wrist axis (-dorsal, the lead hand's back) sits at the roll."""
    base_rot = _club_rotation()  # dorsal axes are fixed in the club frame
    rot = _club_rotation(spin_deg=spin_deg)
    e_h, e_v = reference_axes(SHAFT, TargetFrame())

    def dorsal_club(roll_deg: float) -> np.ndarray:
        r = np.radians(roll_deg)
        normal = np.cos(r) * e_v + np.sin(r) * e_h  # world, perpendicular to shaft
        return -(base_rot.T @ normal)  # wrist axis b = -dorsal, one rule for both hands

    return GripPoseState(
        club_rotation=rot,
        lead_dorsal_club=dorsal_club(lead_deg),
        trail_dorsal_club=dorsal_club(trail_deg),
    )


@pytest.mark.parametrize("deg", [0.0, 15.0, 45.0, -15.0, -45.0, 90.0, 99.4])
def test_recovers_synthetic_roll_within_half_degree(deg: float) -> None:
    lead, trail = address_grip_roll_deg(_state(deg, -deg / 2.0))
    assert lead == pytest.approx(deg, abs=0.5)
    assert trail == pytest.approx(-deg / 2.0, abs=0.5)


def test_zero_is_the_back_of_the_wrist_straight_up_on_the_shaft() -> None:
    e_h, e_v = reference_axes(SHAFT, TargetFrame())
    state = _state(0.0, 0.0)
    back_of_wrist = -(state.club_rotation @ state.lead_dorsal_club)
    assert back_of_wrist == pytest.approx(e_v, abs=1e-9)  # 12 o'clock
    assert address_grip_roll_deg(state) == pytest.approx((0.0, 0.0), abs=1e-9)


def test_positive_is_counterclockwise_toward_the_target_on_the_face_on_clock() -> None:
    # Face-on, the target is on the viewer's left, so 12 -> 9 o'clock (the target
    # side) is counterclockwise: +90 deg puts the back of the wrist at 9 o'clock.
    e_h, e_v = reference_axes(SHAFT, TargetFrame())
    state = _state(30.0, 30.0)
    back_of_wrist = -(state.club_rotation @ state.lead_dorsal_club)
    assert back_of_wrist @ e_h > 0.0  # toward the target
    assert back_of_wrist @ e_v > 0.0  # still above the horizontal
    assert address_grip_roll_deg(state)[0] == pytest.approx(30.0, abs=1e-6)
    assert address_grip_roll_deg(_state(90.0, 90.0))[0] == pytest.approx(90.0, abs=1e-6)
    assert address_grip_roll_deg(_state(-30.0, -30.0))[0] == pytest.approx(-30.0)


def test_one_rule_reads_both_hands_from_the_same_wrist_axis() -> None:
    """Lead and trail share one rule: roll = angle of b = -dorsal, no per-hand sign.

    Hands whose dorsal axis is the same club-frame vector read the same roll; the
    trail hand's back is the opposite end of that axis (180 deg away), as in a real
    grip where the two backs face opposite ways.
    """
    state = _state(25.0, 25.0)
    assert state.lead_dorsal_club == pytest.approx(state.trail_dorsal_club)
    lead, trail = address_grip_roll_deg(state)
    assert lead == pytest.approx(trail, abs=1e-9)
    assert lead == pytest.approx(25.0, abs=1e-6)
    flipped = GripPoseState(
        club_rotation=state.club_rotation,
        lead_dorsal_club=-state.lead_dorsal_club,
        trail_dorsal_club=-state.trail_dorsal_club,
    )
    for before, after in zip(
        (lead, trail), address_grip_roll_deg(flipped), strict=True
    ):  # the opposite end of the axis reads 180 deg away, for both hands alike
        assert abs((after - before + 180.0) % 360.0 - 180.0) == pytest.approx(
            180.0, abs=1e-6
        )


def test_reference_axes_are_horizontal_and_orthonormal() -> None:
    e_h, e_v = reference_axes(SHAFT, TargetFrame())
    assert abs(e_h[2]) < 1e-12  # horizontal
    assert e_h @ SHAFT == pytest.approx(0.0, abs=1e-12)
    assert e_v @ SHAFT == pytest.approx(0.0, abs=1e-12)
    assert e_h @ e_v == pytest.approx(0.0, abs=1e-12)
    assert e_h @ np.array(TargetFrame().target_dir) > 0.0  # points at the target
    assert e_v[2] > 0.0  # points up


def test_spin_about_the_shaft_changes_the_roll_one_for_one() -> None:
    # The dorsal axes stay fixed in the club, so spinning the club about the
    # shaft by d degrees changes the roll by exactly -d (alpha is a rotation
    # about the shaft, positive stronger; the face-on clock counts the other way).
    base = address_grip_roll_deg(_state(10.0, 10.0))[0]
    spun = address_grip_roll_deg(_state(10.0, 10.0, spin_deg=7.0))[0]
    assert spun - base == pytest.approx(-7.0, abs=0.5)


def test_rejects_a_non_orthonormal_club_frame() -> None:
    state = _state(10.0, 10.0)
    bad = GripPoseState.__new__(GripPoseState)
    object.__setattr__(bad, "club_rotation", state.club_rotation * 1.1)
    object.__setattr__(bad, "lead_dorsal_club", state.lead_dorsal_club)
    object.__setattr__(bad, "trail_dorsal_club", state.trail_dorsal_club)
    object.__setattr__(bad, "target", TargetFrame())
    with pytest.raises(ContractViolationError):
        address_grip_roll_deg(bad)


def test_rejects_a_dorsal_axis_along_the_shaft() -> None:
    rot = _club_rotation()
    with pytest.raises(ContractViolationError):
        address_grip_roll_deg(
            GripPoseState(
                club_rotation=rot,
                lead_dorsal_club=np.array([0.0, 1.0, 0.0]),
                trail_dorsal_club=np.array([1.0, 0.0, 0.0]),
            )
        )


def test_rejects_a_vertical_shaft() -> None:
    with pytest.raises(ContractViolationError):
        reference_axes(np.array([0.0, 0.0, -1.0]), TargetFrame())


def test_rejects_left_handed_frames_until_mirroring_is_specified() -> None:
    with pytest.raises(ContractViolationError):
        address_grip_roll_deg(
            GripPoseState(
                club_rotation=_club_rotation(),
                lead_dorsal_club=np.array([1.0, 0.0, 0.0]),
                trail_dorsal_club=np.array([1.0, 0.0, 0.0]),
                target=TargetFrame(handedness="left"),
            )
        )


SPEC = (
    Path(__file__).resolve().parents[3]
    / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
)
SPEC_V2 = SPEC.with_name("full_body_spec_anthro_driver_v2.json")
# Native club frame at the flagship driver address, from the solved MuJoCo pose
# (hipcal-scaled spec, IK frame 0; columns are the club x, y (shaft), z axes).
FLAGSHIP_CLUB_ROTATION = Rotation.from_matrix(
    [[-0.5417, -0.7177, -0.4376], [0.6055, 0.0279, -0.7953], [0.5831, -0.6958, 0.4195]]
).as_matrix()  # nearest rotation to the 4-decimal print


# the club_face fixture capture's address club rotation, measured on the pre-A2
# fixture pair (capture grip) from origin/main before the neutral re-solve
FIXTURE_CLUB_ROTATION = np.array(
    [
        [-0.53711, -0.72233, -0.43561],
        [0.60706, 0.02755, -0.79418],
        [0.58565, -0.691, 0.4237],
    ]
)

EVIDENCE = SPEC.parent / "evidence/grip_roll"
# the neutral-grip address frame solved by solve_neutral_grip.py (showpiece spec hash)
IK = EVIDENCE / "address_frame_neutral.npz"


def test_flagship_driver_grip_is_neutral_15_degrees_face_on() -> None:
    """Flagship driver reads 15 deg on the A7 dial (#14736)."""
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    state = grip_pose_state_from_spec(spec, FLAGSHIP_CLUB_ROTATION)
    lead, trail = address_grip_roll_deg(state)
    assert lead == pytest.approx(15.0, abs=1.0)
    assert trail == pytest.approx(-126.5, abs=3.0)


@pytest.mark.parametrize("spec_path", [SPEC, SPEC_V2], ids=["v1", "v2"])
def test_driver_trail_hand_is_not_a_copy_of_the_lead_hand(spec_path: Path) -> None:
    """Option (b15), #14751 comment 6689879332: trail = lead psi/gamma at alpha 15."""
    from src.shared.python.motion_matching.grip_contract import trail_grip_euler

    totals = json.loads(spec_path.read_text(encoding="utf-8"))["subject"][
        "grip_rotation_deg"
    ]
    assert totals["L"] != totals["R"]
    assert np.allclose(totals["R"], trail_grip_euler(totals["L"]), atol=1e-3)


def test_before_receipt_pins_the_owner_observed_90_degree_grip() -> None:
    """Pre-A2 spec (grip_roll_before.json): top of lead hand, underside of trail."""
    before = json.loads((EVIDENCE / "grip_roll_before.json").read_text("utf-8"))
    assert before["lead_roll_deg"] == pytest.approx(97.9, abs=0.5)
    assert before["trail_roll_deg"] == pytest.approx(108.0, abs=0.5)


@pytest.mark.requires_mujoco
def test_resolved_address_pose_reads_15_degrees_and_keeps_the_club_pose() -> None:
    """The resolved IK address frame, posed in MuJoCo, reads 15 +- 1 deg face-on (#14736)."""
    mujoco = pytest.importorskip("mujoco")
    from docs.development.full_body_models.evidence.grip_roll.solve_neutral_grip import (
        Rig,
        resolve_swing,
    )
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )
    from src.shared.python.motion_matching.execution.grip_rotation import (
        apply_grip_rotation,
    )

    spec_bytes = SPEC.read_bytes()
    xml, meta = export_full_body_mjcf(spec_bytes, visual=True)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    ik_data = np.load(IK)
    q_src = ik_data["q"]
    pre_a7_doc, _ = apply_grip_rotation(
        json.loads(spec_bytes), {"L": [-75.3894, -9.7161, 4.5795]}
    )
    q_solved = resolve_swing(
        Rig(json.dumps(pre_a7_doc).encode()), Rig(spec_bytes), q_src
    )
    q0 = q_solved[0]

    for value, name in zip(q0, meta["coordinate_order"], strict=True):
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[joint]] = value
    mujoco.mj_forward(model, data)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, meta["frame_sites"]["LW"])
    rot = data.site_xmat[site].reshape(3, 3)
    spec = json.loads(spec_bytes)
    lead, trail = address_grip_roll_deg(grip_pose_state_from_spec(spec, rot))
    assert lead == pytest.approx(15.0, abs=1.0)
    assert trail == pytest.approx(-126.5, abs=3.0)
    # the club is where it was before the grip change (shaft and face unchanged)
    assert rot == pytest.approx(FLAGSHIP_CLUB_ROTATION, abs=1e-3)


@pytest.mark.requires_mujoco
def test_club_face_fixture_address_pose_is_posed_on_the_neutral_grip_model() -> None:
    """The club_face fixture pair (spec_driver.json + address pose) stays consistent.

    The fixture swing and address pose were re-solved for the neutral grip, so
    the fixture's own model must carry that grip: posed on it, the address pose
    reads 75 +- 3 deg on the A7 dial and leaves the club where the pre-change fixture put it.
    """
    mujoco = pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )

    fixtures = Path(__file__).resolve().parents[2] / "fixtures/club_face"
    spec_bytes = (fixtures / "spec_driver.json").read_bytes()
    pose = json.loads((fixtures / "address_poses.json").read_text("utf-8"))["poses"][
        "driver"
    ]
    xml, meta = export_full_body_mjcf(spec_bytes, visual=True)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    for name, value in zip(pose["coordinate_order"], pose["q_rad"], strict=True):
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[joint]] = value
    mujoco.mj_forward(model, data)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, meta["frame_sites"]["LW"])
    rot = data.site_xmat[site].reshape(3, 3)
    lead, trail = address_grip_roll_deg(
        grip_pose_state_from_spec(json.loads(spec_bytes), rot)
    )
    assert lead == pytest.approx(75.0, abs=3.0)
    assert trail == pytest.approx(75.0, abs=3.0)
    assert rot == pytest.approx(FIXTURE_CLUB_ROTATION, abs=1e-4)


def test_neutral_grip_roll_tracks_alpha_deg_one_for_one() -> None:
    """Pins roll == offset - alpha for the HX-canonical grip (psi 40, gamma 0).

    With face-on-referenced alpha (#14633), the offset at ``alpha = 0`` is 90 deg
    on the A7 clock; ``d_alpha`` moves the measured roll by ``-d_alpha``, and
    ``alpha = 15`` deg reads 75 deg face-on.
    """
    from src.shared.python.motion_matching.execution.grip_rotation import (
        apply_grip_rotation,
    )
    from src.shared.python.motion_matching.grip_contract import (
        HandAngles,
        hand_to_euler,
    )

    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    psi = 40.0

    def roll(alpha: float) -> np.ndarray:
        triple = list(hand_to_euler(HandAngles(alpha, psi, 0.0)))
        doc, _ = apply_grip_rotation(spec, {"L": triple, "R": triple})
        state = grip_pose_state_from_spec(doc, FLAGSHIP_CLUB_ROTATION)
        return np.array(address_grip_roll_deg(state))

    offset = roll(0.0)
    assert offset == pytest.approx([90.0, 90.0], abs=0.5)
    for alpha in (-30.0, 0.0, 20.0):
        assert roll(alpha) == pytest.approx(offset - alpha, abs=1e-6)
    assert roll(15.0) == pytest.approx([75.0, 75.0], abs=0.1)


def test_hx_alpha_zero_reads_zero_degrees_face_on_at_flagship_address() -> None:
    """HX alpha is face-on referenced (#14633): alpha 0 is palm-vertical (90 deg on A7 clock)."""
    from src.shared.python.motion_matching.grip_contract import (
        ALPHA_FACE_ON_ZERO_OFFSET_DEG,
        HandAngles,
        grip_matrix,
    )

    native_path = (
        Path(__file__).resolve().parents[3]
        / "docs/development/simscape_tour_matching/native_evidence/native_geometry_spec_9967.json"
    )
    native_doc = json.loads(native_path.read_text(encoding="utf-8"))
    native_l = next(
        j
        for j in native_doc["joints"]
        if str(j.get("child", "")).endswith("Clubface Vector")
    )
    n = np.asarray(native_l["child_to_follower"], float)[:3, :3]
    closure = np.asarray(native_doc["closure"]["placement_b"], float)[:3, :3]
    native_r = next(
        j
        for j in native_doc["joints"]
        if str(j.get("child", "")).endswith("RHandStandoff")
    )
    n_r = closure @ np.asarray(native_r["child_to_follower"], float)[:3, :3]

    club_rot = FLAGSHIP_CLUB_ROTATION
    shaft = club_rot @ np.array([0.0, 1.0, 0.0])
    e_h, e_v = reference_axes(shaft, TargetFrame())

    def roll_hx(side_native: np.ndarray, angles: HandAngles) -> float:
        follower = side_native @ grip_matrix(
            angles.alpha_deg, angles.psi_deg, angles.gamma_deg
        )
        dorsal = follower[:, 0]
        dorsal = dorsal - float(dorsal @ np.array([0.0, 1.0, 0.0])) * np.array(
            [0.0, 1.0, 0.0]
        )
        dorsal /= np.linalg.norm(dorsal)
        normal = -(club_rot @ dorsal)
        from src.shared.python.motion_matching.grip_roll import _roll_deg

        return _roll_deg(normal, e_h, e_v, shaft)

    neutral_hx = HandAngles(0.0, 40.0, 0.0)
    assert roll_hx(n, neutral_hx) == pytest.approx(90.0, abs=0.5)
    assert roll_hx(n_r, neutral_hx) == pytest.approx(90.0, abs=0.5)
    strong_hx = HandAngles(15.0, 40.0, 0.0)
    assert roll_hx(n, strong_hx) == pytest.approx(75.0, abs=0.5)
    assert pytest.approx(25.733, abs=0.01) == ALPHA_FACE_ON_ZERO_OFFSET_DEG
