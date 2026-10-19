"""Unit tests for the native MuJoCo humanoid golf grip adapter (HRG-GS-M5, #13327).

Acceptance criteria:
- 3 styles (overlap, interlock, ten_finger), both hands bound across all 5 admitted
  MuJoCo humanoid golf MJCFs;
- Visual style from GripSpec attached as zero-mass non-colliding proxy geoms without
  altering body mass, inertia, or default welded dynamics;
- Weld-mode hand-hand outputs are extracted from the equality weld constraint force
  (efc_force) and labelled "allocated" with the prior name;
- GripCapabilities honestly declare supports_physical_pairs=False, with an
  unavailable_reason, for welded bodies, and apply_wrenches / apply_report raise
  for welded bodies;
- Behind an explicit flag (attachment_redesign="off" by default), optional attachment
  redesign supports "free_club_shared" (free club with shared-law xfrc_applied) and
  "free_club_native" (native pad geoms with contype="0"/exclude and explicit <pair>),
  and never applies both native and shared contact on the same pair;
- Manifests under src/config/grip_bindings/ for each admitted MJCF validate;
- M3 qualification harness passes against the Pinocchio reference (PinocchioGripAdapter)
  and law_level_series on identical site states across all 3 styles.
"""

from __future__ import annotations

import ast
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.grip_contact.binding import (
    GripStepReport,
    HandClubBinding,
    ModelGripAdapter,
    SchemaFeatureUnavailableError,
    apply_report,
    evaluate_contacts,
    validate_binding,
)
from src.shared.python.grip_contact.binding_manifest import (
    binding_from_manifest,
    validate_manifest,
    validate_resolved_binding,
)
from src.shared.python.grip_contact.bushing_law import BushingState
from src.shared.python.grip_contact.contact_topology import (
    ContactTopology,
    GripStyle,
    contact_topology,
)
from src.shared.python.grip_contact.hand_hand_allocation import AllocationPrior
from src.shared.python.grip_contact.interface import GripInterface
from src.shared.python.grip_contact.parity import (
    PEAK_TOLERANCE,
    RMS_TOLERANCE,
    compare_topology_series,
)
from src.shared.python.grip_contact.prescribed_motion import (
    RigidBodyState,
    hand_frame_states,
)
from src.shared.python.grip_contact.topology_contact import (
    TopologyContactModel,
    build_topology_model,
)
from src.shared.python.grip_contact.topology_series import (
    SAMPLE_DT_S,
    TopologyKineticsSeries,
    law_level_series,
    topology_mapping_hash,
    total_applied_allocated_force_n,
)
from src.shared.python.motion_matching.grip_contract import GripSpec

pytestmark = [pytest.mark.unit, pytest.mark.requires_mujoco]

ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
MANIFESTS_DIR = ROOT / "src/config/grip_bindings"
ADAPTER_PY = ROOT / "src/engines/physics_engines/mujoco/python/humanoid_grip_adapter.py"

ADMITTED_MODEL_IDS: tuple[str, ...] = (
    "mujoco_golfer",
    "mujoco_upper_body_swing",
    "mujoco_full_body_swing",
    "mujoco_advanced_biomechanical_swing",
    "mujoco_canonical_golf_humanoid",
)
SQUEEZE_N = 1100.0


def _require_mujoco() -> Any:
    try:
        import mujoco  # noqa: PLC0415
    except ImportError as exc:
        pytest.skip(f"mujoco not installed: {exc}")
    return mujoco


@pytest.fixture(scope="module")
def spec_dict() -> dict[str, Any]:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def interface(spec_dict: dict[str, Any]) -> GripInterface:
    return GripInterface.from_spec(spec_dict)


@pytest.fixture(scope="module")
def overlap_topology() -> ContactTopology:
    return contact_topology(GripSpec(style=GripStyle.OVERLAP))


@pytest.fixture(scope="module")
def overlap_model(
    interface: GripInterface, overlap_topology: ContactTopology
) -> TopologyContactModel:
    return build_topology_model(
        interface, overlap_topology, squeeze_per_hand_n=SQUEEZE_N
    )


@pytest.mark.parametrize("model_id", ADMITTED_MODEL_IDS)
@pytest.mark.parametrize(
    "style", [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER]
)
@pytest.mark.parametrize("trigger_finger", [False, True])
def test_manifests_exist_and_resolve_all_sites_across_styles(
    model_id: str,
    style: GripStyle,
    trigger_finger: bool,
    interface: GripInterface,
) -> None:
    """Every admitted MuJoCo humanoid MJCF has a valid manifest resolving 100% of sites."""
    _require_mujoco()
    from src.engines.physics_engines.mujoco.python.humanoid_grip_adapter import (
        ADMITTED_HUMANOID_MODELS,
        MANIFEST_PATHS,
        MuJoCoHumanoidGripAdapter,
    )

    assert model_id in ADMITTED_HUMANOID_MODELS
    manifest_path = MANIFEST_PATHS[model_id]
    assert manifest_path.exists(), f"Missing manifest file: {manifest_path}"

    raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(raw_manifest)
    assert raw_manifest["model_id"] == model_id
    assert raw_manifest["attachment_mode"] == "weld"

    grip_spec = GripSpec(style=style, trigger_finger=trigger_finger)
    topo = contact_topology(grip_spec)
    adapter = MuJoCoHumanoidGripAdapter.from_model_id(
        model_id, grip_spec=grip_spec, interface=interface
    )
    assert isinstance(adapter, ModelGripAdapter)

    binding = adapter.resolve_binding(topo)
    assert isinstance(binding, HandClubBinding)
    validate_resolved_binding(binding, topo, interface)

    topo_site_names = set(topo.site_names())
    bound_site_names = {sb.site_id for sb in binding.sites}
    assert bound_site_names == topo_site_names
    assert len(binding.unavailable_sites) == 0

    lead_sites = {s.name for s in topo.sites_for_hand("lead")}
    trail_sites = {s.name for s in topo.sites_for_hand("trail")}
    site_map = binding.site_binding_map()
    for s_name in lead_sites:
        assert site_map[s_name].body_id == binding.lead_hand_body
    for s_name in trail_sites:
        assert site_map[s_name].body_id == binding.trail_hand_body


@pytest.mark.parametrize(
    "style", [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER]
)
def test_visual_style_from_grip_spec_leaves_welded_dynamics_bit_identical(
    style: GripStyle,
) -> None:
    """Visual style geoms from GripSpec reflect style without altering mass, inertia, or dynamics."""
    mujoco = _require_mujoco()
    from src.engines.physics_engines.mujoco.python.humanoid_grip_adapter import (
        MuJoCoHumanoidGripAdapter,
        load_humanoid_mjcf_xml,
    )

    grip_spec = GripSpec(style=style, trigger_finger=True)
    raw_xml = load_humanoid_mjcf_xml("mujoco_golfer")
    raw_model = mujoco.MjModel.from_xml_string(raw_xml)
    raw_data = mujoco.MjData(raw_model)

    adapter = MuJoCoHumanoidGripAdapter.from_model_id(
        "mujoco_golfer", grip_spec=grip_spec
    )
    assert adapter.grip_spec == grip_spec
    assert adapter.visual_pose.style == style

    vis_geoms = adapter.visual_style_geom_names()
    assert len(vis_geoms) >= len(adapter.topology.sites)
    # Style-specific marker geom name reflects the GripStyle
    assert any(style.value in g for g in vis_geoms)

    # Every visual style geom must have zero mass and contype=0, conaffinity=0
    m = adapter.model
    for g_name in vis_geoms:
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g_name)
        assert gid >= 0
        assert int(m.geom_contype[gid]) == 0
        assert int(m.geom_conaffinity[gid]) == 0

    # Body mass, inertia, and COM are bit-for-bit identical
    np.testing.assert_allclose(m.body_mass, raw_model.body_mass, atol=0.0)
    np.testing.assert_allclose(m.body_inertia, raw_model.body_inertia, atol=0.0)
    np.testing.assert_allclose(m.body_ipos, raw_model.body_ipos, atol=0.0)

    # Multi-step forward dynamics trajectories are bit-for-bit identical
    for _ in range(15):
        mujoco.mj_step(raw_model, raw_data)
        mujoco.mj_step(adapter.model, adapter.data)
    np.testing.assert_allclose(adapter.data.qpos, raw_data.qpos, atol=0.0)
    np.testing.assert_allclose(adapter.data.qvel, raw_data.qvel, atol=0.0)
    np.testing.assert_allclose(adapter.data.efc_force, raw_data.efc_force, atol=0.0)


@pytest.mark.parametrize("model_id", ADMITTED_MODEL_IDS)
def test_welded_mode_capabilities_and_apply_raises(
    model_id: str,
    overlap_topology: ContactTopology,
    overlap_model: TopologyContactModel,
) -> None:
    """Welded bodies declare supports_physical_pairs=False with unavailable_reason and apply raises."""
    _require_mujoco()
    from src.engines.physics_engines.mujoco.python.humanoid_grip_adapter import (
        MuJoCoHumanoidGripAdapter,
        WeldedGripBodyError,
    )

    adapter = MuJoCoHumanoidGripAdapter.from_model_id(model_id)
    caps = adapter.capabilities()
    assert caps.supports_physical_pairs is False
    assert caps.supports_wrench_application is False
    assert caps.native_contact is False
    assert caps.attachment_modes == ("weld",)
    assert caps.unavailable_reason is not None
    assert len(caps.unavailable_reason) > 10
    assert adapter.club_body in caps.unavailable_reason

    # Requesting a physical attachment mode on welded bodies raises SchemaFeatureUnavailableError
    for phys_mode in ("contact", "contact_v1", "bushing"):
        with pytest.raises(SchemaFeatureUnavailableError, match=adapter.club_body):
            adapter.resolve_binding(overlap_topology, attachment_mode=phys_mode)

    binding = adapter.resolve_binding(overlap_topology)
    assert binding.attachment_mode == "weld"
    validate_binding(overlap_topology, binding, caps)

    report = adapter.evaluate_contacts(overlap_model, binding)
    assert report.applied_allocated_force_n() == 0.0
    assert report.body_wrenches() == {}

    # Both apply_report and apply_wrenches raise for welded bodies
    with pytest.raises(SchemaFeatureUnavailableError):
        apply_report(adapter, report)

    with pytest.raises(
        (SchemaFeatureUnavailableError, ValueError, WeldedGripBodyError)
    ):
        adapter.apply_wrenches(
            {
                binding.lead_hand_body: (
                    np.array([25.0, -10.0, 5.0]),
                    np.array([0.5, -0.2, 0.1]),
                ),
                binding.club_body: (
                    np.array([-25.0, 10.0, -5.0]),
                    np.array([-0.5, 0.2, -0.1]),
                ),
            }
        )


@pytest.mark.parametrize("model_id", ADMITTED_MODEL_IDS)
def test_weld_mode_hand_hand_allocated_from_efc_force_with_prior_name(
    model_id: str,
    overlap_topology: ContactTopology,
    overlap_model: TopologyContactModel,
) -> None:
    """Weld constraint force (efc_force) drives allocated hand-hand reporting with prior name."""
    mujoco = _require_mujoco()
    from src.engines.physics_engines.mujoco.python.humanoid_grip_adapter import (
        MuJoCoHumanoidGripAdapter,
    )

    adapter = MuJoCoHumanoidGripAdapter.from_model_id(model_id)
    mujoco.mj_step(adapter.model, adapter.data)

    closure = adapter.extract_weld_closure()
    assert closure.available is True
    assert len(closure.wrenches) == 1
    w0 = closure.wrenches[0]
    assert np.isfinite(w0.force_n).all()
    assert np.isfinite(w0.torque_nm).all()
    assert float(np.linalg.norm(w0.force_n)) > 0.0

    # 1. Default prior ("equal_split")
    report_default = adapter.evaluate_contacts(overlap_model)
    assert report_default.allocated_contacts is not None
    assert report_default.allocated_contacts.prior_name == "equal_split"
    assert report_default.applied_allocated_force_n() == 0.0

    for hh_pair in overlap_topology.hand_hand_pairs():
        assert report_default.pair_labels[hh_pair.name] == "allocated"
        assert report_default.allocated_labels[hh_pair.name] == hh_pair.allocated_label
        pf = report_default.allocated_contacts.pairs[hh_pair.name]
        assert pf.label == hh_pair.allocated_label
        assert pf.method == "allocation"
        assert pf.pair_name == hh_pair.name
        assert pf.available is True
        assert np.isfinite(report_default.normal_force_n[hh_pair.name])
        assert np.isfinite(report_default.tangential_force_n[hh_pair.name])
        assert report_default.normal_force_n[hh_pair.name] >= 0.0
        assert (
            report_default.normal_force_n[hh_pair.name]
            + report_default.tangential_force_n[hh_pair.name]
            > 0.0
        )
        assert hh_pair.name not in report_default.pair_wrenches

    # 2. Custom named prior ("palm_dominant_75_25")
    pair_names = tuple(p.name for p in overlap_topology.hand_hand_pairs())
    weights = {pair_names[0]: 0.75}
    for extra in pair_names[1:]:
        weights[extra] = 0.25 / max(1, len(pair_names) - 1)
    custom_prior = AllocationPrior(name="palm_dominant_75_25", weights=weights)

    report_custom = adapter.evaluate_contacts(overlap_model, prior=custom_prior)
    assert report_custom.allocated_contacts is not None
    assert report_custom.allocated_contacts.prior_name == "palm_dominant_75_25"
    for hh_pair in overlap_topology.hand_hand_pairs():
        assert report_custom.pair_labels[hh_pair.name] == "allocated"
        pf_c = report_custom.allocated_contacts.pairs[hh_pair.name]
        assert pf_c.label == hh_pair.allocated_label
        assert pf_c.method == "allocation"
        assert pf_c.pair_name == hh_pair.name


def test_optional_attachment_redesign_shared_vs_native_never_duplicate(
    interface: GripInterface,
    overlap_topology: ContactTopology,
    overlap_model: TopologyContactModel,
) -> None:
    """Attachment redesign supports free_club_shared and free_club_native and never duplicates contact."""
    mujoco = _require_mujoco()
    from src.engines.physics_engines.mujoco.python.humanoid_grip_adapter import (
        MuJoCoHumanoidGripAdapter,
    )

    # 1. Shared-law free-club redesign ("free_club_shared")
    shared_adapter = MuJoCoHumanoidGripAdapter.from_model_id(
        "mujoco_golfer",
        interface=interface,
        attachment_redesign="free_club_shared",
    )
    shared_caps = shared_adapter.capabilities()
    assert shared_caps.supports_physical_pairs is True
    assert shared_caps.supports_wrench_application is True
    assert shared_caps.native_contact is False
    assert "contact" in shared_caps.attachment_modes
    assert shared_adapter.registered_grip_collision_count() == 0

    shared_binding = shared_adapter.resolve_binding(
        overlap_topology, attachment_mode="contact"
    )
    validate_binding(overlap_topology, shared_binding, shared_caps)

    # Set prescribed hold states and evaluate + apply shared-law wrenches via xfrc_applied
    club_state = RigidBodyState(
        rotation=np.eye(3),
        position_m=np.array([0.0, 0.22, 1.47]),
        velocity_m_s=np.zeros(3),
        omega_rad_s=np.zeros(3),
    )
    hands = hand_frame_states(club_state, interface)
    shared_adapter.set_prescribed_grip_states(hand_states=hands, club_state=club_state)
    rep_shared = shared_adapter.evaluate_contacts(overlap_model, shared_binding)
    assert rep_shared.action_reaction_residual_n() <= 1e-12
    assert rep_shared.applied_allocated_force_n() == 0.0
    assert len(rep_shared.pair_wrenches) == len(overlap_topology.hand_grip_pairs())

    apply_report(shared_adapter, rep_shared)
    assert set(shared_adapter.applied_wrenches.keys()) == {
        shared_binding.club_body,
        shared_binding.lead_hand_body,
        shared_binding.trail_hand_body,
    }
    # Verify xfrc_applied on the MuJoCo data buffer has zero net world force
    cid = mujoco.mj_name2id(
        shared_adapter.model, mujoco.mjtObj.mjOBJ_BODY, shared_binding.club_body
    )
    lid = mujoco.mj_name2id(
        shared_adapter.model, mujoco.mjtObj.mjOBJ_BODY, shared_binding.lead_hand_body
    )
    rid = mujoco.mj_name2id(
        shared_adapter.model, mujoco.mjtObj.mjOBJ_BODY, shared_binding.trail_hand_body
    )
    net_xfrc_force = (
        shared_adapter.data.xfrc_applied[cid, :3]
        + shared_adapter.data.xfrc_applied[lid, :3]
        + shared_adapter.data.xfrc_applied[rid, :3]
    )
    np.testing.assert_allclose(net_xfrc_force, np.zeros(3), atol=1e-11)

    # 2. Native pad-geom redesign ("free_club_native")
    native_adapter = MuJoCoHumanoidGripAdapter.from_model_id(
        "mujoco_golfer",
        interface=interface,
        topology_model=overlap_model,
        attachment_redesign="free_club_native",
    )
    native_caps = native_adapter.capabilities()
    assert native_caps.supports_physical_pairs is True
    assert native_caps.supports_wrench_application is False
    assert native_caps.native_contact is True
    assert native_adapter.registered_grip_collision_count() == len(
        overlap_topology.hand_grip_pairs()
    )

    native_binding = native_adapter.resolve_binding(
        overlap_topology, attachment_mode="contact"
    )
    validate_binding(overlap_topology, native_binding, native_caps)

    # Native and shared contact MUST NEVER be applied on the same pair:
    # applying shared-law report or wrenches to a native_contact adapter raises!
    with pytest.raises(SchemaFeatureUnavailableError, match="native_contact"):
        apply_report(native_adapter, rep_shared)

    with pytest.raises(SchemaFeatureUnavailableError, match="native_contact"):
        native_adapter.apply_wrenches(rep_shared.body_wrenches())


@pytest.mark.parametrize(
    "style", [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER]
)
@pytest.mark.parametrize("redesign_mode", ["off", "free_club_shared"])
def test_m3_harness_passes_against_pinocchio_reference_on_identical_site_states(
    interface: GripInterface,
    style: GripStyle,
    redesign_mode: str,
) -> None:
    """M3 qualification harness passes against PinocchioGripAdapter and law_level_series
    across all 3 styles on identical site states within 5% peak / 2% RMS.
    """
    mujoco = _require_mujoco()
    from src.engines.physics_engines.mujoco.python.humanoid_grip_adapter import (
        MuJoCoHumanoidGripAdapter,
    )
    from src.engines.physics_engines.pinocchio.python.grip_binding_adapter import (
        PinocchioGripAdapter,
    )

    grip_spec = GripSpec(style=style)
    topo = contact_topology(grip_spec)
    topo_model = build_topology_model(interface, topo, squeeze_per_hand_n=SQUEEZE_N)
    m_hash = topology_mapping_hash(topo_model)

    adapter = MuJoCoHumanoidGripAdapter.from_model_id(
        "mujoco_golfer",
        grip_spec=grip_spec,
        interface=interface,
        attachment_redesign=redesign_mode,
    )
    mujoco.mj_step(adapter.model, adapter.data)
    closure = (
        adapter.extract_weld_closure()
        if redesign_mode == "off"
        else adapter.synthesise_closure_from_net_wrench()
    )

    att_mode = "weld" if redesign_mode == "off" else "contact"
    binding = adapter.resolve_binding(topo, attachment_mode=att_mode)
    assert len(binding.sites) == len(topo.sites)

    pin_adapter = PinocchioGripAdapter(interface, model_hash=m_hash)
    pin_binding = replace(
        pin_adapter.resolve_binding(topo),
        attachment_mode=att_mode,
        club_grip_frames=binding.club_grip_frames,
    )

    times = np.arange(21, dtype=float) * SAMPLE_DT_S  # 0.00 s .. 0.04 s (21 steps)
    mj_reports: list[GripStepReport] = []
    pin_reports: list[GripStepReport] = []
    hand_seq: dict[str, list[BushingState]] = {"L": [], "R": []}
    club_seq: dict[str, list[BushingState]] = {"L": [], "R": []}

    for _k, t_s in enumerate(times):
        # Smoothly varying non-trivial site state trajectory
        phase = 2.0 * np.pi * (t_s / 0.04)
        club_pos = np.array([0.01 * np.sin(phase), 0.22, 1.47 + 0.005 * np.cos(phase)])
        club_vel = np.array([0.05 * np.cos(phase), -0.02 * np.sin(phase), 0.01])
        club_omega = np.array([0.1 * np.sin(phase), 0.2 * np.cos(phase), -0.15])
        club_state = RigidBodyState(
            rotation=np.eye(3),
            position_m=club_pos,
            velocity_m_s=club_vel,
            omega_rad_s=club_omega,
        )
        hands = hand_frame_states(club_state, interface)
        # Small micro-slip perturbation on lead and trail hands
        slip_v = np.array([0.002 * np.sin(phase + 0.3), -0.001 * np.cos(phase), 0.0])
        hands_perturbed = {
            "L": replace(hands["L"], velocity_m_s=hands["L"].velocity_m_s + slip_v),
            "R": replace(hands["R"], velocity_m_s=hands["R"].velocity_m_s - slip_v),
        }
        club_frames = {
            "L": club_state.frame(binding.club_grip_frames["L"]),
            "R": club_state.frame(binding.club_grip_frames["R"]),
        }
        hand_seq["L"].append(hands_perturbed["L"])
        hand_seq["R"].append(hands_perturbed["R"])
        club_seq["L"].append(club_frames["L"])
        club_seq["R"].append(club_frames["R"])

        adapter.set_prescribed_grip_states(
            hand_states=hands_perturbed, club_state=club_state
        )
        mj_rep = adapter.evaluate_contacts(topo_model, binding, closure=closure)
        assert mj_rep.applied_allocated_force_n() == 0.0
        if redesign_mode == "free_club_shared":
            apply_report(adapter, mj_rep)
        mj_reports.append(mj_rep)

        pin_adapter.set_prescribed_states(
            hand_states=hands_perturbed, club_state=club_state
        )
        pin_states = pin_adapter.sample_states(pin_binding)
        pin_rep = evaluate_contacts(
            topo_model, pin_binding, pin_states, closure=closure
        )
        assert pin_rep.applied_allocated_force_n() == 0.0
        pin_reports.append(pin_rep)

    assert total_applied_allocated_force_n(mj_reports) == 0.0

    mj_series = TopologyKineticsSeries.from_step_reports(
        times,
        mj_reports,
        engine=f"mujoco_humanoid_{redesign_mode}",
        mapping_hash=m_hash,
        binding=binding,
    )
    pin_series = TopologyKineticsSeries.from_step_reports(
        times,
        pin_reports,
        engine="pinocchio_reference",
        mapping_hash=m_hash,
        binding=pin_binding,
    )
    law_series = law_level_series(
        topo_model,
        hand_seq,
        club_seq,
        times,
        mapping_hash=m_hash,
        mode=att_mode,
    )

    parity_vs_pin = compare_topology_series(pin_series, mj_series)
    assert parity_vs_pin.status == "passed", (
        f"M3 parity vs Pinocchio failed for {style.value} ({redesign_mode}): "
        f"{parity_vs_pin.to_dict()}"
    )
    assert parity_vs_pin.passed is True
    for ch_name, ch_res in parity_vs_pin.channels.items():
        assert ch_res.passed is True, f"Channel {ch_name} failed: {ch_res}"
        assert ch_res.error is not None
        assert ch_res.error.peak_error <= PEAK_TOLERANCE
        assert ch_res.error.rms_error <= RMS_TOLERANCE

    parity_vs_law = compare_topology_series(law_series, mj_series)
    assert parity_vs_law.status == "passed", (
        f"M3 parity vs law_level_series failed for {style.value} ({redesign_mode}): "
        f"{parity_vs_law.to_dict()}"
    )
    assert parity_vs_law.passed is True


@pytest.mark.slow
@pytest.mark.parametrize("model_id", ADMITTED_MODEL_IDS)
@pytest.mark.parametrize(
    "style", [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER]
)
def test_m3_harness_native_fk_hold_across_all_admitted_mjcfs_and_styles(
    model_id: str,
    style: GripStyle,
    interface: GripInterface,
) -> None:
    """M3 harness across all 5 admitted MJCFs and 3 styles sampling native MuJoCo FK states."""
    mujoco = _require_mujoco()
    from src.engines.physics_engines.mujoco.python.humanoid_grip_adapter import (
        MuJoCoHumanoidGripAdapter,
    )
    from src.engines.physics_engines.pinocchio.python.grip_binding_adapter import (
        PinocchioGripAdapter,
    )

    grip_spec = GripSpec(style=style)
    topo = contact_topology(grip_spec)
    topo_model = build_topology_model(interface, topo, squeeze_per_hand_n=SQUEEZE_N)
    m_hash = topology_mapping_hash(topo_model)

    adapter = MuJoCoHumanoidGripAdapter.from_model_id(model_id, grip_spec=grip_spec)
    binding = adapter.resolve_binding(topo)
    assert binding.attachment_mode == "weld"

    pin_adapter = PinocchioGripAdapter(interface, model_hash=m_hash)
    pin_binding = replace(
        pin_adapter.resolve_binding(topo),
        attachment_mode="weld",
        club_grip_frames=binding.club_grip_frames,
    )

    times = np.arange(21, dtype=float) * SAMPLE_DT_S
    mj_reports: list[GripStepReport] = []
    pin_reports: list[GripStepReport] = []
    hand_seq: dict[str, list[BushingState]] = {"L": [], "R": []}
    club_seq: dict[str, list[BushingState]] = {"L": [], "R": []}

    q0 = adapter.data.qpos.copy()
    v0 = np.zeros(adapter.model.nv, dtype=float)
    mujoco.mj_step(adapter.model, adapter.data)
    closure = adapter.extract_weld_closure()

    for _ in times:
        adapter.data.qpos[:] = q0
        adapter.data.qvel[:] = v0
        states = adapter.sample_states(binding)
        mj_rep = evaluate_contacts(topo_model, binding, states, closure=closure)
        assert mj_rep.applied_allocated_force_n() == 0.0
        assert mj_rep.body_wrenches() == {}
        mj_reports.append(mj_rep)

        c_pose = states[binding.club_body]
        l_pose = states[binding.lead_hand_body]
        r_pose = states[binding.trail_hand_body]
        c_vel, c_omega = states.velocity_of(binding.club_body)
        l_vel, l_omega = states.velocity_of(binding.lead_hand_body)
        r_vel, r_omega = states.velocity_of(binding.trail_hand_body)

        c_state = RigidBodyState(c_pose[:3, :3], c_pose[:3, 3], c_vel, c_omega)
        b_lead = BushingState(l_pose[:3, :3], l_pose[:3, 3], l_vel, l_omega)
        b_trail = BushingState(r_pose[:3, :3], r_pose[:3, 3], r_vel, r_omega)
        hand_seq["L"].append(b_lead)
        hand_seq["R"].append(b_trail)
        club_seq["L"].append(c_state.frame(binding.club_grip_frames["L"]))
        club_seq["R"].append(c_state.frame(binding.club_grip_frames["R"]))

        pin_adapter.set_prescribed_states(
            hand_states={"L": b_lead, "R": b_trail},
            club_state=c_state,
        )
        pin_states = pin_adapter.sample_states(pin_binding)
        pin_rep = evaluate_contacts(
            topo_model, pin_binding, pin_states, closure=closure
        )
        pin_reports.append(pin_rep)

    mj_series = TopologyKineticsSeries.from_step_reports(
        times, mj_reports, engine=model_id, mapping_hash=m_hash, binding=binding
    )
    pin_series = TopologyKineticsSeries.from_step_reports(
        times,
        pin_reports,
        engine="pinocchio_reference",
        mapping_hash=m_hash,
        binding=pin_binding,
    )
    law_series = law_level_series(
        topo_model, hand_seq, club_seq, times, mapping_hash=m_hash, mode="weld"
    )

    parity_pin = compare_topology_series(pin_series, mj_series)
    assert parity_pin.status == "passed"
    assert parity_pin.passed is True
    for ch_res in parity_pin.allocated_channels.values():
        assert ch_res.passed is True
        assert ch_res.error is not None
        assert ch_res.error.peak_error <= PEAK_TOLERANCE
        assert ch_res.error.rms_error <= RMS_TOLERANCE

    parity_law = compare_topology_series(law_series, mj_series)
    assert parity_law.status == "passed"
    assert parity_law.passed is True


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


def test_adapter_ast_budgets_and_lod() -> None:
    """AST check: file <= 1200 lines, functions <= 100 lines, <= 8 params, LoD <= 2 dots."""
    assert ADAPTER_PY.exists(), f"Missing adapter module: {ADAPTER_PY}"
    source = ADAPTER_PY.read_text(encoding="utf-8")
    lines = source.splitlines()
    assert len(lines) <= 1200, f"{ADAPTER_PY.name} has {len(lines)} lines > 1200"

    tree = ast.parse(source, filename=str(ADAPTER_PY))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            depth = _attribute_chain_depth(node)
            assert depth <= 2, (
                f"{ADAPTER_PY.name}:{node.lineno} has attribute chain depth {depth} > 2: "
                f"{ast.unparse(node)}"
            )
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            assert node.end_lineno is not None
            length = node.end_lineno - node.lineno + 1
            assert length <= 100, (
                f"{ADAPTER_PY.name}:{node.name} is {length} lines > 100"
            )
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
                f"{ADAPTER_PY.name}:{node.name} has {n_params} params > 8"
            )


def test_xml_parsing_uses_defusedxml_in_grip_adapters() -> None:
    """XML parsing in grip adapters and bilateral scene uses DefusedET and marks stdlib ET build-only."""
    expected_comment = (
        "# noqa: S405  # nosemgrep: python.lang.security.use-defused-xml.use-defused-xml"
        "  # build-only; parse via DefusedET"
    )
    targets = (
        ROOT / "src/engines/physics_engines/mujoco/python/grip_lab_adapter.py",
        ROOT / "src/engines/physics_engines/mujoco/python/humanoid_grip_adapter.py",
        ROOT / "src/engines/physics_engines/myosuite/python/myohand_bilateral_scene.py",
    )
    for path in targets:
        text = path.read_text(encoding="utf-8")
        assert "import defusedxml.ElementTree as DefusedET" in text, path.name
        for line in text.splitlines():
            if line.startswith("import xml.etree.ElementTree"):
                assert expected_comment in line, (
                    f"{path.name}: missing build-only nosemgrep comment on {line!r}"
                )
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in ("ET", "StdET")
            ):
                assert node.func.attr not in ("fromstring", "parse"), (
                    f"{path.name}:{node.lineno}: must not call "
                    f"{node.func.value.id}.{node.func.attr}"
                )
