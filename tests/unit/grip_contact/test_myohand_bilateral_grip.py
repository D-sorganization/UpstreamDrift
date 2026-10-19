"""Tests for MyoHub myoHand articulated bilateral grip fixture (HRG-GS-M8, #13330).

Covers:
1. Pure-function scene composer generating driver/iron bilateral myoHand MJCFs
   and committed receipt JSON with SHA-256 provenance and pin verification
   without copying meshes from shared/models/myosuite/myo_sim.
2. Sagittal reflection verification for the mirrored left myoHand (body
   positions, COMs, proper rotations, masses, world inertia tensors, joint axes).
3. Segment-mapped grip-binding/v1 manifests with 100% coverage of all
   ContactTopology sites onto distph/midph/proxph/metacarpal/thumb bodies
   across overlap, interlock, and ten_finger (with and without trigger_finger).
4. Frozen-finger mode: GripPose-to-myoHand joint mapping, equality joint-locking,
   inactive muscle actuators, mutual exclusion of native vs shared contact, and
   M3 qualification harness parity against Pinocchio within 5% peak / 2% RMS.
5. Grasp-resultant inverse-dynamics check (m(a - g) and moment within 5% peak /
   2% RMS), Newton-Euler balance, honest friction-cone ratios, and hyperstatic
   INDETERMINATE labels for per-hand split and squeeze.
6. Design-by-Contract negative tests and AST/LoD/file-size budget checks.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.grip_contact.binding import (
    SchemaFeatureUnavailableError,
)
from src.shared.python.grip_contact.binding_manifest import (
    binding_from_manifest,
    validate_manifest,
    validate_resolved_binding,
)
from src.shared.python.grip_contact.contact_topology import (
    GripStyle,
    contact_topology,
)
from src.shared.python.grip_contact.interface import GripInterface
from src.shared.python.grip_contact.parity import (
    PEAK_TOLERANCE,
    RMS_TOLERANCE,
    compare_topology_series,
)
from src.shared.python.grip_contact.topology_series import (
    FIXTURE_DT_S,
    law_level_series,
)
from src.shared.python.model_appearance.grip_pose import (
    DEFAULT_GRIP_POSE,
    GripPose,
    Hand,
)
from src.shared.python.motion_matching.grip_contract import GripSpec

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
MANIFESTS_DIR = ROOT / "src/config/grip_bindings"
MYO_SIM_HAND_XML = ROOT / "shared/models/myosuite/myo_sim/hand/myohand.xml"
HAND_SCENE_DIR = ROOT / "shared/models/myosuite/golf/hand"
SCENE_PY = (
    ROOT / "src/engines/physics_engines/myosuite/python/myohand_bilateral_scene.py"
)
ADAPTER_PY = (
    ROOT / "src/engines/physics_engines/myosuite/python/myohand_bilateral_adapter.py"
)
CONTROLLER_PY = (
    ROOT / "src/engines/physics_engines/myosuite/python/myohand_grip_controller.py"
)

ALLOWED_HAND_SEGMENT_TOKENS = (
    "distph",
    "midph",
    "proxph",
    "firstmc",
    "secondmc",
    "thirdmc",
    "fourthmc",
    "fifthmc",
    "proximal_thumb",
    "distal_thumb",
)


def _require_myohand_assets() -> None:
    if not MYO_SIM_HAND_XML.is_file():
        pytest.skip(
            "shared/models/myosuite/myo_sim/hand/myohand.xml not populated; "
            "run scripts/setup_myosuite_models.sh"
        )


@pytest.fixture(scope="module")
def interface() -> GripInterface:
    spec_dict = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    return GripInterface.from_spec(spec_dict)


# ---------------------------------------------------------------------------
# 1. Pure-function scene composer & receipt provenance
# ---------------------------------------------------------------------------


@pytest.mark.requires_mujoco
def test_scene_generator_creates_driver_iron_and_receipt() -> None:
    """Scene generator writes driver/iron MJCFs and receipt without copying meshes."""
    pytest.importorskip("mujoco")
    _require_myohand_assets()
    from src.engines.physics_engines.myosuite.python.myohand_bilateral_scene import (
        MYO_SIM_PIN_SHA,
        generate_myohand_bilateral_scene,
    )

    paths = generate_myohand_bilateral_scene(repo_root=ROOT, output_root=HAND_SCENE_DIR)
    assert paths.driver.is_file()
    assert paths.iron.is_file()
    assert paths.receipt.is_file()

    # Verify no .stl or .png files were copied into HAND_SCENE_DIR
    copied_meshes = list(HAND_SCENE_DIR.rglob("*.stl"))
    copied_textures = list(HAND_SCENE_DIR.rglob("*.png"))
    assert copied_meshes == [], f"Unexpected copied meshes: {copied_meshes}"
    assert copied_textures == [], f"Unexpected copied textures: {copied_textures}"

    receipt = json.loads(paths.receipt.read_text(encoding="utf-8"))
    assert receipt["issue"] == 13330
    assert receipt["myo_sim_pin"] == MYO_SIM_PIN_SHA
    assert "hand/myohand.xml" in receipt["source_xml_sha256"]
    assert "hand/assets/myohand_assets.xml" in receipt["source_xml_sha256"]
    assert "hand/assets/myohand_body.xml" in receipt["source_xml_sha256"]
    assert "driver" in receipt["clubs"] and "iron" in receipt["clubs"]
    assert receipt["left_hand_mirror"]["supported"] is True
    assert receipt["qualification"]["is_contact_fixture"] is True
    assert receipt["qualification"]["muscle_recruitment_in_scope"] is False
    assert receipt["qualification"]["actuators_active"] is False


# ---------------------------------------------------------------------------
# 2. Left-hand sagittal reflection verification
# ---------------------------------------------------------------------------


@pytest.mark.requires_mujoco
def test_left_hand_is_exact_sagittal_reflection_of_right_hand() -> None:
    """Left myoHand is a sagittal reflection (S_W = diag(-1, 1, 1)) of right myoHand."""
    pytest.importorskip("mujoco")
    _require_myohand_assets()
    from src.engines.physics_engines.myosuite.python.myohand_bilateral_scene import (
        verify_left_hand_sagittal_reflection,
    )

    report = verify_left_hand_sagittal_reflection(repo_root=ROOT)
    assert report.supported is True
    assert report.body_count == 37
    assert report.joint_count == 23
    assert report.max_xpos_error_m <= 1e-6
    assert report.max_xipos_error_m <= 1e-6
    assert report.max_rotation_error <= 1e-6
    assert report.max_mass_error_kg <= 1e-12
    assert report.max_world_inertia_error_kg_m2 <= 1e-6
    assert report.max_articulated_xpos_error_m <= 1e-6
    assert report.max_articulated_xipos_error_m <= 1e-6
    assert report.max_normal_reflection_error <= 1e-6


# ---------------------------------------------------------------------------
# 3. Segment-mapped manifests & 100% topology coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_id",
    [
        "myosuite_myohand_bilateral_driver",
        "myosuite_myohand_bilateral_iron",
    ],
)
@pytest.mark.parametrize(
    "style", [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER]
)
@pytest.mark.parametrize("trigger_finger", [True, False])
def test_segment_mapped_manifests_100_percent_coverage_all_styles(
    interface: GripInterface,
    model_id: str,
    style: GripStyle,
    trigger_finger: bool,
) -> None:
    """Every requested topology site is mapped onto a distinct phalanx/metacarpal/thumb body."""
    manifest_path = MANIFESTS_DIR / f"{model_id}.json"
    assert manifest_path.is_file(), f"Missing manifest: {manifest_path}"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    assert manifest["attachment_mode"] == "separated_v2"

    spec = GripSpec(style=style, trigger_finger=trigger_finger)
    topo = contact_topology(spec)
    binding = binding_from_manifest(manifest, topo, interface)
    validate_resolved_binding(binding, topo, interface)

    expected_sites = set(topo.site_names())
    bound_sites = {sb.site_id for sb in binding.sites}
    assert bound_sites == expected_sites
    assert len(binding.unavailable_sites) == 0

    body_ids = [sb.body_id for sb in binding.sites]
    assert len(set(body_ids)) == len(body_ids), (
        f"Expected distinct segment bodies per site, got duplicates in {body_ids}"
    )
    for sb in binding.sites:
        assert sb.representation == "phalange"
        expected_prefix = "lh_" if sb.site_id.startswith("lead_") else "rh_"
        assert sb.body_id.startswith(expected_prefix)
        assert any(tok in sb.body_id for tok in ALLOWED_HAND_SEGMENT_TOKENS), (
            f"Site {sb.site_id} mapped to {sb.body_id}, not in {ALLOWED_HAND_SEGMENT_TOKENS}"
        )


# ---------------------------------------------------------------------------
# 4. Frozen-finger mode: pose mapping, joint-locking, inactive actuators
# ---------------------------------------------------------------------------


@pytest.mark.requires_mujoco
@pytest.mark.parametrize(
    "style", [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER]
)
def test_grip_pose_maps_to_myohand_joints_and_freezes_without_muscles(
    style: GripStyle,
) -> None:
    """Style finger pose maps to myoHand joints, locks via equality, and leaves muscles inactive."""
    pytest.importorskip("mujoco")
    _require_myohand_assets()
    from src.engines.physics_engines.myosuite.python.myohand_bilateral_adapter import (
        FINGER_POSE_JOINT_MAP,
        MyoHandBilateralGripAdapter,
        pose_to_myohand_joint_targets,
    )

    assert len(FINGER_POSE_JOINT_MAP) == 23
    spec = GripSpec(style=style, trigger_finger=True)
    pose = GripPose(style=style)
    targets = pose_to_myohand_joint_targets(pose, trigger_finger=spec.trigger_finger)
    assert len(targets[Hand.LEAD]) == 23
    assert len(targets[Hand.TRAIL]) == 23

    # Also verify an explicit custom GripPose (simulating a G5 preset) plugs in directly
    custom_pose = GripPose(
        style=style,
        lead_v_azimuth_deg=28.0,
        trail_v_offset_deg=4.0,
        lead_thumb_offset_deg=12.0,
        palm_diagonal_deg=42.0,
        grip_strength_deg=8.0,
        lead_roll_offset_deg=3.0,
        trail_roll_offset_deg=-3.0,
    )
    adapter = MyoHandBilateralGripAdapter(
        "myosuite_myohand_bilateral_driver",
        grip_spec=spec,
        grip_pose=custom_pose,
        contact_mode="shared",
        frozen_fingers=True,
    )
    caps = adapter.capabilities()
    assert caps.supports_physical_pairs is True
    assert caps.supports_wrench_application is True
    assert caps.native_contact is False
    assert adapter.actuators_active() is False

    q_before = adapter.finger_joint_positions().copy()
    report = adapter.step_with_contacts(FIXTURE_DT_S)
    q_after = adapter.finger_joint_positions().copy()
    assert np.allclose(q_before, q_after, atol=1e-9)
    assert report.applied_allocated_force_n() == 0.0
    assert adapter.max_actuator_force_n() == 0.0


# ---------------------------------------------------------------------------
# 5. Native vs shared contact exclusion & physical hand-hand wrenches
# ---------------------------------------------------------------------------


@pytest.mark.requires_mujoco
@pytest.mark.parametrize(
    "style", [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER]
)
def test_shared_vs_native_contact_mutual_exclusion_and_hand_hand_pairs(
    style: GripStyle,
) -> None:
    """Native and shared contact are never applied on the same pair; hand-hand pairs are physical."""
    pytest.importorskip("mujoco")
    _require_myohand_assets()
    from src.engines.physics_engines.myosuite.python.myohand_bilateral_adapter import (
        MyoHandBilateralGripAdapter,
    )

    spec = GripSpec(style=style, trigger_finger=True)
    shared_adapter = MyoHandBilateralGripAdapter(
        "myosuite_myohand_bilateral_driver",
        grip_spec=spec,
        contact_mode="shared",
        frozen_fingers=True,
    )
    assert shared_adapter.registered_grip_collision_count() == 0
    report = shared_adapter.evaluate_contacts()
    assert all(lbl == "physical" for lbl in report.pair_labels.values())
    assert report.applied_allocated_force_n() == 0.0

    hh_names = [p.name for p in shared_adapter.topology().hand_hand_pairs()]
    assert "trail_palm_lead_thumb" in hh_names
    for hh_name in hh_names:
        pw = report.pair_wrenches[hh_name]
        assert pw.normal_force_mag_n > 0.0, (
            f"Expected positive normal force on {hh_name}"
        )

    shared_adapter.apply_report(report)
    net_xfrc_force = shared_adapter.total_applied_xfrc_force_n()
    assert float(np.linalg.norm(net_xfrc_force)) <= 1e-10

    native_adapter = MyoHandBilateralGripAdapter(
        "myosuite_myohand_bilateral_driver",
        grip_spec=spec,
        contact_mode="native",
        frozen_fingers=True,
    )
    assert native_adapter.capabilities().native_contact is True
    assert native_adapter.capabilities().supports_wrench_application is False
    assert native_adapter.registered_grip_collision_count() > 0
    with pytest.raises(SchemaFeatureUnavailableError, match="native"):
        native_adapter.apply_report(report)


# ---------------------------------------------------------------------------
# 6. Frozen-finger M3 law-level parity on identical site states
# ---------------------------------------------------------------------------


@pytest.mark.requires_mujoco
@pytest.mark.parametrize(
    "model_id",
    [
        "myosuite_myohand_bilateral_driver",
        "myosuite_myohand_bilateral_iron",
    ],
)
@pytest.mark.parametrize(
    "style", [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER]
)
def test_frozen_finger_law_level_parity_on_identical_site_states(
    model_id: str,
    style: GripStyle,
) -> None:
    """Law-level parity (frozen-finger law on identical site states, not engine dynamics parity)."""
    pytest.importorskip("mujoco")
    _require_myohand_assets()
    from src.engines.physics_engines.myosuite.python.myohand_bilateral_adapter import (
        MyoHandBilateralGripAdapter,
        build_prescribed_grip_trajectory,
        run_myohand_frozen_qualification_series,
    )
    from src.engines.physics_engines.pinocchio.python.grip_binding_adapter import (
        PinocchioGripAdapter,
    )
    from src.shared.python.grip_contact.binding import (
        GripStepReport,
        evaluate_contacts,
    )
    from src.shared.python.grip_contact.topology_series import (
        TopologyKineticsSeries,
        topology_mapping_hash,
    )

    spec = GripSpec(style=style, trigger_finger=True)
    adapter = MyoHandBilateralGripAdapter(
        model_id,
        grip_spec=spec,
        contact_mode="shared",
        frozen_fingers=True,
    )
    times, hand_frames_seq, club_frames_seq = build_prescribed_grip_trajectory(
        adapter.interface(), n_steps=21, dt_s=FIXTURE_DT_S
    )
    candidate_series = run_myohand_frozen_qualification_series(
        adapter, hand_frames_seq, club_frames_seq, times
    )
    assert candidate_series.applied_allocated_force_n == 0.0

    # Law-level parity check (frozen-finger shared law on identical site states)
    topo_model = adapter.topology_model()
    m_hash = topology_mapping_hash(topo_model)
    law_series = law_level_series(
        topo_model,
        hand_frames_seq,
        club_frames_seq,
        times,
        model_id=model_id,
        mapping_hash=m_hash,
    )
    law_report = compare_topology_series(law_series, candidate_series)
    assert law_report.passed is True, f"Law-level parity failed: {law_report.to_dict()}"
    for ch, m in law_report.channels.items():
        assert m.error is not None
        assert m.error.peak_error <= PEAK_TOLERANCE, f"{ch} peak {m.error.peak_error}"
        assert m.error.rms_error <= RMS_TOLERANCE, f"{ch} rms {m.error.rms_error}"

    # Law-level parity against PinocchioGripAdapter (both evaluate the same shared
    # frozen-finger contact law on identical site states; this is law-level parity,
    # not independent engine forward-dynamics parity).
    pin_adapter = PinocchioGripAdapter(adapter.interface(), model_hash=m_hash)
    pin_binding = pin_adapter.resolve_binding(adapter.topology())
    closure = adapter.synthesise_closure_from_net_wrench()
    pin_reports: list[GripStepReport] = []
    for h_frames, c_frames in zip(hand_frames_seq, club_frames_seq, strict=True):
        club_state = adapter._current_club_state(c_frames)
        pin_adapter.set_prescribed_states(hand_states=h_frames, club_state=club_state)
        pin_states = pin_adapter.sample_states(pin_binding)
        pin_rep = evaluate_contacts(
            topo_model, pin_binding, pin_states, closure=closure
        )
        pin_reports.append(pin_rep)
    pin_series = TopologyKineticsSeries.from_step_reports(
        times,
        pin_reports,
        engine="pinocchio_reference",
        mapping_hash=m_hash,
        binding=pin_binding,
    )
    pin_report = compare_topology_series(pin_series, candidate_series)
    assert pin_report.passed is True, (
        f"Pinocchio law-level parity failed: {pin_report.to_dict()}"
    )
    for ch, m in pin_report.channels.items():
        assert m.error is not None
        assert m.error.peak_error <= PEAK_TOLERANCE, f"{ch} peak {m.error.peak_error}"
        assert m.error.rms_error <= RMS_TOLERANCE, f"{ch} rms {m.error.rms_error}"


# ---------------------------------------------------------------------------
# 7. Prescribed club inverse-dynamics check, non-circularity, slip & cone ratios
# ---------------------------------------------------------------------------


@pytest.mark.requires_mujoco
@pytest.mark.parametrize(
    "model_id",
    [
        "myosuite_myohand_bilateral_driver",
        "myosuite_myohand_bilateral_iron",
    ],
)
def test_mujoco_free_joint_mj_inverse_frame_conversion_convention(
    model_id: str,
) -> None:
    """Verify MuJoCo freejoint mj_inverse conversion (world linear DOFs, body angular DOFs about xpos)."""
    mujoco = pytest.importorskip("mujoco")
    _require_myohand_assets()
    from scipy.spatial.transform import Rotation

    from src.engines.physics_engines.myosuite.python.myohand_bilateral_adapter import (
        MyoHandBilateralGripAdapter,
    )

    adapter = MyoHandBilateralGripAdapter(model_id, contact_mode="shared")
    rot_club = Rotation.from_euler("XYZ", [0.32, -0.41, 0.68]).as_matrix()
    q_xyzw = Rotation.from_matrix(rot_club).as_quat()
    q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float)
    p_origin = np.array([0.12, -0.18, 0.45], dtype=float)
    v_origin_w = np.array([0.35, -0.25, 0.15], dtype=float)
    omega_w = np.array([1.1, -0.75, 0.55], dtype=float)
    a_origin_w = np.array([-1.8, 1.4, 0.65], dtype=float)
    alpha_w = np.array([2.8, -2.2, 3.6], dtype=float)

    qpos_club = np.concatenate([p_origin, q_wxyz])
    qvel_club = np.concatenate([v_origin_w, rot_club.T @ omega_w])
    qacc_club = np.concatenate([a_origin_w, rot_club.T @ alpha_w])

    f_mj, m_mj_com, m_mj_origin, c_norm = adapter.club_mj_inverse_requirement(
        qpos_club=qpos_club,
        qvel_club=qvel_club,
        qacc_club=qacc_club,
    )
    assert c_norm == 0.0

    # 1. Exact 3D Newton-Euler using scene mass/inertia matches mj_inverse to machine precision
    scene_props = adapter.club_scene_inertial_properties()
    f_sc, m_sc_com, m_sc_origin = scene_props.inverse_dynamics_at_phase(
        0.0,
        0.0,
        0.0,
        rotation=rot_club,
        origin_accel_m_s2=a_origin_w,
        alpha_rad_s2=alpha_w,
        omega_rad_s=omega_w,
    )
    assert np.allclose(f_mj, f_sc, atol=1e-13, rtol=1e-13)
    assert np.allclose(m_mj_com, m_sc_com, atol=1e-13, rtol=1e-13)
    assert np.allclose(m_mj_origin, m_sc_origin, atol=1e-13, rtol=1e-13)

    # 2. Analytic ClubSpec properties match mass, COM, I_xx, I_zz; iron I_yy reflects boundinertia=1e-4
    f_ana, m_ana_com, m_ana_origin = adapter.club_inverse_dynamics_requirement(
        origin_accel_m_s2=a_origin_w,
        alpha_rad_s2=alpha_w,
        omega_rad_s=omega_w,
        rotation=rot_club,
    )
    assert np.allclose(f_mj, f_ana, atol=1e-13, rtol=1e-13)
    if model_id == "myosuite_myohand_bilateral_driver":
        assert np.allclose(m_mj_com, m_ana_com, atol=1e-13, rtol=1e-13)
        assert np.allclose(m_mj_origin, m_ana_origin, atol=1e-13, rtol=1e-13)
    else:
        assert scene_props.inertia_com_kg_m2[1, 1] == pytest.approx(1e-4, abs=1e-16)

    # 3. Round-trip (f_mj, m_mj_com) through xfrc_applied + mj_forward recovers qacc_club
    bid = adapter._club_body_id()
    qadr, vadr = adapter._club_joint_addrs()
    adapter._data.qpos[qadr : qadr + 7] = qpos_club
    adapter._data.qvel[:] = 0.0
    adapter._data.qvel[vadr : vadr + 6] = qvel_club
    adapter._data.qacc[:] = 0.0
    adapter._data.xfrc_applied[:] = 0.0
    adapter._data.xfrc_applied[bid, :3] = f_mj
    adapter._data.xfrc_applied[bid, 3:6] = m_mj_com
    mujoco.mj_forward(adapter._model, adapter._data)
    assert np.allclose(
        adapter._data.qacc[vadr : vadr + 6], qacc_club, atol=1e-11, rtol=1e-11
    )


@pytest.mark.requires_mujoco
@pytest.mark.parametrize(
    "model_id",
    [
        "myosuite_myohand_bilateral_driver",
        "myosuite_myohand_bilateral_iron",
    ],
)
@pytest.mark.parametrize(
    "style", [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER]
)
def test_club_inverse_dynamics_requirement_is_non_circular_and_sliding_cone_ratios(
    model_id: str,
    style: GripStyle,
) -> None:
    """Both analytic and MuJoCo mj_inverse requirements are independent of applied hand wrenches."""
    pytest.importorskip("mujoco")
    _require_myohand_assets()
    from src.engines.physics_engines.myosuite.python.myohand_bilateral_adapter import (
        MyoHandBilateralGripAdapter,
        build_prescribed_grip_trajectory,
        evaluate_grasp_resultant_newton_check,
    )

    spec = GripSpec(style=style, trigger_finger=True)
    adapter = MyoHandBilateralGripAdapter(
        model_id,
        grip_spec=spec,
        contact_mode="shared",
        frozen_fingers=True,
    )
    times, hand_seq, club_seq = build_prescribed_grip_trajectory(
        adapter.interface(), n_steps=21, dt_s=FIXTURE_DT_S
    )
    adapter.set_prescribed_grip_state(hand_seq[-1], club_seq[-1])
    rep = adapter.evaluate_contacts()
    adapter.apply_report(rep)
    f_req_1, m_req_com_1, m_req_orig_1 = adapter.club_inverse_dynamics_requirement(
        t_s=float(times[-1]), n_steps=21, dt_s=FIXTURE_DT_S
    )
    f_mj_1, m_mj_com_1, m_mj_orig_1, c_norm_1 = adapter.club_mj_inverse_requirement(
        t_s=float(times[-1]), n_steps=21, dt_s=FIXTURE_DT_S
    )
    assert c_norm_1 == 0.0

    # Apply a deliberately wrong (scaled x2) contact wrench: neither analytic nor
    # MuJoCo mj_inverse requirement may change.
    wrenches_x2 = {
        body_id: (2.0 * f, 2.0 * m) for body_id, (f, m) in rep.body_wrenches().items()
    }
    adapter.apply_wrenches(wrenches_x2)
    f_req_2, m_req_com_2, m_req_orig_2 = adapter.club_inverse_dynamics_requirement(
        t_s=float(times[-1]), n_steps=21, dt_s=FIXTURE_DT_S
    )
    f_mj_2, m_mj_com_2, m_mj_orig_2, c_norm_2 = adapter.club_mj_inverse_requirement(
        t_s=float(times[-1]), n_steps=21, dt_s=FIXTURE_DT_S
    )
    assert c_norm_2 == 0.0
    assert np.allclose(f_req_1, f_req_2, atol=0.0, rtol=0.0)
    assert np.allclose(m_req_com_1, m_req_com_2, atol=0.0, rtol=0.0)
    assert np.allclose(m_req_orig_1, m_req_orig_2, atol=0.0, rtol=0.0)
    assert np.allclose(f_mj_1, f_mj_2, atol=0.0, rtol=0.0)
    assert np.allclose(m_mj_com_1, m_mj_com_2, atol=0.0, rtol=0.0)
    assert np.allclose(m_mj_orig_1, m_mj_orig_2, atol=0.0, rtol=0.0)

    # Full trajectory report at 1x vs 2x scaled contact wrench:
    # both MuJoCo mj_inverse and analytic requirements are unchanged while measured errors change.
    report_1x = evaluate_grasp_resultant_newton_check(
        adapter, n_steps=21, dt_s=FIXTURE_DT_S, wrench_scale=1.0
    )
    report_2x = evaluate_grasp_resultant_newton_check(
        adapter, n_steps=21, dt_s=FIXTURE_DT_S, wrench_scale=2.0
    )
    assert report_1x.required_force_n == report_2x.required_force_n
    assert report_1x.required_moment_com_nm == report_2x.required_moment_com_nm
    assert report_1x.analytic_required_force_n == report_2x.analytic_required_force_n
    assert (
        report_1x.analytic_required_moment_com_nm
        == report_2x.analytic_required_moment_com_nm
    )
    assert not np.isclose(
        report_1x.force_peak_rel_error, report_2x.force_peak_rel_error, rtol=1e-3
    )
    assert not np.isclose(
        report_1x.force_rms_rel_error, report_2x.force_rms_rel_error, rtol=1e-3
    )
    assert not np.isclose(
        report_1x.moment_peak_rel_error, report_2x.moment_peak_rel_error, rtol=1e-3
    )
    assert not np.isclose(
        report_1x.moment_rms_rel_error, report_2x.moment_rms_rel_error, rtol=1e-3
    )
    assert not np.isclose(
        report_1x.mj_central_diff_force_rms_rel_error,
        report_2x.mj_central_diff_force_rms_rel_error,
        rtol=1e-3,
    )
    assert not np.isclose(
        report_1x.mj_central_diff_moment_rms_rel_error,
        report_2x.mj_central_diff_moment_rms_rel_error,
        rtol=1e-3,
    )
    assert not np.isclose(
        report_1x.newton_force_rel_error, report_2x.newton_force_rel_error, rtol=1e-3
    )
    assert not np.isclose(
        report_1x.newton_moment_com_residual_nm,
        report_2x.newton_moment_com_residual_nm,
        rtol=1e-3,
    )

    assert report_1x.grasp_synthesis.residual_inf_n <= 1e-9
    assert report_1x.per_hand_split_status == "INDETERMINATE"
    assert report_1x.squeeze_status == "INDETERMINATE"
    assert report_1x.contact_regime == "quasi_stick"
    assert report_1x.min_squeeze_for_stick_n > 0.0
    assert report_1x.lead_squeeze_normal_force_n > report_1x.min_squeeze_for_stick_n
    assert report_1x.trail_squeeze_normal_force_n > report_1x.min_squeeze_for_stick_n
    assert len(report_1x.nullspace_split_vector) == 6
    assert float(np.linalg.norm(report_1x.nullspace_split_vector)) > 0.0

    # Every hand-club and hand-hand pair is held in bilateral compression and
    # regularized quasi-stick (0 < |v_t| < v_tr = 1.0 mm/s, 0 < |f_t|/(mu_s*f_n) < 1).
    law = adapter.topology_model().law
    v_tr = float(law.transition_velocity_m_s)
    assert len(report_1x.friction_cone_ratios) == len(adapter.topology().pairs)
    for pair_name, ratio in report_1x.friction_cone_ratios.items():
        v_slip = report_1x.slip_velocities_m_s[pair_name]
        assert ratio is not None, f"Expected active compression on {pair_name}"
        assert 0.0 < ratio < 0.25, (
            f"Expected quasi-stick cone ratio < 0.25 on {pair_name}, got {ratio}"
        )
        assert 0.0 < v_slip < v_tr, (
            f"Expected quasi-stick micro-slip < {v_tr} m/s on {pair_name}, got {v_slip}"
        )


@pytest.mark.requires_mujoco
@pytest.mark.parametrize(
    "model_id",
    [
        "myosuite_myohand_bilateral_driver",
        "myosuite_myohand_bilateral_iron",
    ],
)
@pytest.mark.parametrize(
    "style", [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER]
)
def test_grasp_synthesis_solver_closure_residual(
    model_id: str,
    style: GripStyle,
) -> None:
    """Grasp-synthesis closure (solver residual + adapter site-mapping check, not acceptance evidence)."""
    pytest.importorskip("mujoco")
    _require_myohand_assets()
    from src.engines.physics_engines.myosuite.python.myohand_bilateral_adapter import (
        MyoHandBilateralGripAdapter,
        evaluate_grasp_resultant_newton_check,
    )

    spec = GripSpec(style=style, trigger_finger=True)
    adapter = MyoHandBilateralGripAdapter(
        model_id,
        grip_spec=spec,
        contact_mode="shared",
        frozen_fingers=True,
    )
    report = evaluate_grasp_resultant_newton_check(adapter, n_steps=21)
    assert report.solver_closure_force_peak_rel_error <= 1e-10
    assert report.solver_closure_force_rms_rel_error <= 1e-10
    assert report.solver_closure_moment_peak_rel_error <= 1e-10
    assert report.solver_closure_moment_rms_rel_error <= 1e-10


@pytest.mark.requires_mujoco
@pytest.mark.parametrize(
    "model_id",
    [
        "myosuite_myohand_bilateral_driver",
        "myosuite_myohand_bilateral_iron",
    ],
)
@pytest.mark.parametrize(
    "style", [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER]
)
def test_grasp_resultant_matches_prescribed_inverse_dynamics_within_m3_tolerances(
    model_id: str,
    style: GripStyle,
) -> None:
    """Acceptance gate: MuJoCo mj_inverse requirement (analytic & central-diff qacc) + forward replay."""
    pytest.importorskip("mujoco")
    _require_myohand_assets()
    from src.engines.physics_engines.myosuite.python.myohand_bilateral_adapter import (
        MyoHandBilateralGripAdapter,
        evaluate_grasp_resultant_newton_check,
        replay_club_forward_trajectory,
    )

    spec = GripSpec(style=style, trigger_finger=True)
    adapter = MyoHandBilateralGripAdapter(
        model_id,
        grip_spec=spec,
        contact_mode="shared",
        frozen_fingers=True,
    )
    newton_report = evaluate_grasp_resultant_newton_check(adapter, n_steps=21)

    # Step 2a: Hand contact resultant vs MuJoCo mj_inverse (analytic qacc)
    assert newton_report.max_club_constraint_force_n == 0.0
    assert newton_report.force_peak_rel_error <= PEAK_TOLERANCE
    assert newton_report.force_rms_rel_error <= RMS_TOLERANCE
    assert newton_report.moment_peak_rel_error <= PEAK_TOLERANCE
    assert newton_report.moment_rms_rel_error <= RMS_TOLERANCE
    assert newton_report.newton_force_rel_error <= PEAK_TOLERANCE
    assert newton_report.newton_moment_com_residual_nm <= 1e-3

    # Step 2b: Hand contact resultant vs MuJoCo mj_inverse (central-difference qacc of prescribed qvel)
    assert newton_report.mj_central_diff_force_peak_rel_error <= PEAK_TOLERANCE
    assert newton_report.mj_central_diff_force_rms_rel_error <= RMS_TOLERANCE
    assert newton_report.mj_central_diff_moment_peak_rel_error <= PEAK_TOLERANCE
    assert newton_report.mj_central_diff_moment_rms_rel_error <= RMS_TOLERANCE

    # Step 2c: Analytic ClubInertialProperties vs MuJoCo mj_inverse discrepancy (mass/inertia check)
    assert newton_report.analytic_vs_mj_force_peak_rel_error <= PEAK_TOLERANCE
    assert newton_report.analytic_vs_mj_force_rms_rel_error <= RMS_TOLERANCE
    assert newton_report.analytic_vs_mj_moment_peak_rel_error <= PEAK_TOLERANCE
    assert newton_report.analytic_vs_mj_moment_rms_rel_error <= RMS_TOLERANCE

    # Step 3: Open-loop forward replay with mj_step under solved hand contact wrenches + gravity
    replay_report = replay_club_forward_trajectory(adapter, n_steps=21, substeps=10)
    assert replay_report.integrator == "semi_implicit_euler"
    assert replay_report.integrator_order == 1
    assert replay_report.max_active_club_weld_count == 0
    assert replay_report.max_club_constraint_force_n == 0.0
    assert replay_report.com_pos_peak_rel_error <= PEAK_TOLERANCE
    assert replay_report.com_pos_rms_rel_error <= RMS_TOLERANCE
    assert replay_report.orientation_peak_rel_error <= PEAK_TOLERANCE
    assert replay_report.orientation_rms_rel_error <= RMS_TOLERANCE
    assert replay_report.passed is True


# ---------------------------------------------------------------------------
# 8. DbC negative tests & AST / LoD / file-size budgets
# ---------------------------------------------------------------------------


@pytest.mark.requires_mujoco
def test_dbc_preconditions_and_missing_assets(tmp_path: Path) -> None:
    """Verify DbC preconditions and missing myo_sim failure mode."""
    pytest.importorskip("mujoco")
    _require_myohand_assets()
    from src.engines.physics_engines.myosuite.python.myohand_bilateral_adapter import (
        ClubInertialProperties,
        MyoHandBilateralGripAdapter,
        solve_force_regulated_hand_frames,
    )
    from src.engines.physics_engines.myosuite.python.myohand_bilateral_scene import (
        check_myohand_assets_available,
    )

    with pytest.raises(SchemaFeatureUnavailableError, match="myohand.xml"):
        check_myohand_assets_available(repo_root=tmp_path)

    with pytest.raises(ValueError, match="Unsupported model_id"):
        MyoHandBilateralGripAdapter("unknown_model")

    with pytest.raises(ValueError, match="contact_mode"):
        MyoHandBilateralGripAdapter(
            "myosuite_myohand_bilateral_driver",
            contact_mode="invalid_mode",  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="squeeze_n"):
        MyoHandBilateralGripAdapter(
            "myosuite_myohand_bilateral_driver",
            squeeze_n=-10.0,
        )

    with pytest.raises(ValueError, match="mass_kg"):
        ClubInertialProperties(
            mass_kg=0.0,
            com_local_m=(0.0, -0.2, 0.0),
            inertia_com_kg_m2=np.eye(3),
        )

    with pytest.raises(TypeError, match="topology_model"):
        solve_force_regulated_hand_frames(
            "not_a_model",  # type: ignore[arg-type]
            {},
            np.zeros(6),
        )


def _attribute_chain_depth(node: ast.AST) -> int:
    depth = 0
    cur = node
    while True:
        if isinstance(cur, ast.Attribute):
            depth += 1
            cur = cur.value
        elif isinstance(cur, ast.Call):
            cur = cur.func
        elif isinstance(cur, ast.Subscript):
            cur = cur.value
        else:
            break
    return depth


def test_ast_file_function_and_lod_budgets() -> None:
    """Enforce <= 1200 lines/file, <= 100 lines and <= 8 params/fn, <= 2 dots LoD."""
    for path in (SCENE_PY, ADAPTER_PY, CONTROLLER_PY):
        assert path.is_file(), f"Missing implementation file: {path}"
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()
        assert len(lines) <= 1200, f"{path.name} has {len(lines)} lines > 1200"

        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                depth = _attribute_chain_depth(node)
                assert depth <= 2, (
                    f"{path.name}:{node.lineno} has attribute chain depth {depth} > 2: "
                    f"{ast.unparse(node)}"
                )
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                assert node.end_lineno is not None
                length = node.end_lineno - node.lineno + 1
                assert length <= 100, f"{path.name}:{node.name} is {length} lines > 100"
                args = node.args
                n_params = (
                    len(args.posonlyargs)
                    + len(args.args)
                    + len(args.kwonlyargs)
                    + (1 if args.vararg else 0)
                    + (1 if args.kwarg else 0)
                )
                if args.args and args.args[0].arg in ("self", "cls"):
                    n_params -= 1
                assert n_params <= 8, (
                    f"{path.name}:{node.name} has {n_params} params > 8"
                )
