"""Unit and live-simulation tests for the MyoSuite golfer grip adapter (HRG-GS-M7, #13329).

Acceptance criteria:
- Admitted scenes (driver, iron) load with nbody=33, neq=20 and bind hand_l, hand_r, golf_club;
- 3 styles (overlap, interlock, ten_finger) x 2 modes (weld, contact) run;
- Site-weld mode (default) reports allocated pad and hand-hand forces from the weld
  constraint forces (efc_force of grip_weld_l and grip_weld_r), labelled allocated
  with the prior name;
- GripCapabilities honestly declares supports_physical_pairs=False with an
  unavailable_reason in site-weld mode, and apply_wrenches / apply_report raise;
- Contact mode (contact_mode=True) disables grip_weld_l and grip_weld_r via eq_active
  (leaving knee and foot-contact equalities active) and applies shared-law pad-grip
  wrenches via xfrc_applied;
- Never applies both the weld and shared contact on the same pair (re-enabling
  eq_active on a grip weld causes apply_wrenches to raise);
- Missing bodies raise SchemaFeatureUnavailableError before stepping;
- Bimanual grasp synthesis ||G y - w_club||_inf <= 1e-9 N across 3 styles x 2 modes;
- M3 qualification harness passes against the Pinocchio rigid-hand reference
  (PinocchioGripAdapter) and law_level_series on identical site states within
  PEAK_TOLERANCE (5%) and RMS_TOLERANCE (2%);
- AST check: file <= 1200 lines, functions <= 100 lines and <= 8 params, LoD <= 2 dots.
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

pytestmark = [
    pytest.mark.unit,
    pytest.mark.requires_mujoco,
    pytest.mark.requires_myosuite,
]

ROOT = Path(__file__).resolve().parents[3]
DRIVER_SPEC_PATH = (
    ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
)
IRON_SPEC_PATH = (
    ROOT / "docs/development/full_body_models/full_body_spec_anthro_iron7.json"
)
MYO_SIM_BODY = ROOT / "shared/models/myosuite/myo_sim/body/myobody_simpleupper.xml"
GOLFER_SCENE_PY = ROOT / "src/engines/physics_engines/myosuite/python/golfer_scene.py"

ADMITTED_SCENE_IDS: tuple[str, ...] = (
    "myosuite_golfer_driver",
    "myosuite_golfer_iron",
)
SQUEEZE_N = 1100.0


def _require_mujoco_and_myo_sim() -> Any:
    try:
        import mujoco  # noqa: PLC0415
    except ImportError as exc:
        pytest.skip(f"mujoco not installed: {exc}")
    if not MYO_SIM_BODY.is_file():
        pytest.skip(
            f"shared/models/myosuite/myo_sim not populated at {MYO_SIM_BODY}; "
            "run scripts/setup_myosuite_models.sh"
        )
    return mujoco


@pytest.fixture(scope="module")
def driver_interface() -> GripInterface:
    spec = json.loads(DRIVER_SPEC_PATH.read_text(encoding="utf-8"))
    return GripInterface.from_spec(spec)


@pytest.fixture(scope="module")
def iron_interface() -> GripInterface:
    spec = json.loads(IRON_SPEC_PATH.read_text(encoding="utf-8"))
    return GripInterface.from_spec(spec)


@pytest.fixture(scope="module")
def overlap_topology() -> ContactTopology:
    return contact_topology(GripSpec(style=GripStyle.OVERLAP))


@pytest.fixture(scope="module")
def overlap_model(
    driver_interface: GripInterface,
    overlap_topology: ContactTopology,
) -> TopologyContactModel:
    return build_topology_model(
        driver_interface, overlap_topology, squeeze_per_hand_n=SQUEEZE_N
    )


@pytest.mark.parametrize("model_id", ADMITTED_SCENE_IDS)
@pytest.mark.parametrize(
    "style", [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER]
)
@pytest.mark.parametrize("trigger_finger", [False, True])
def test_manifests_and_scene_topology_bind_all_sites(
    model_id: str,
    style: GripStyle,
    trigger_finger: bool,
    driver_interface: GripInterface,
    iron_interface: GripInterface,
) -> None:
    """Both admitted MyoSuite golfer scenes have valid manifests and bind hand_l, hand_r, golf_club."""
    _require_mujoco_and_myo_sim()
    from src.engines.physics_engines.myosuite.python.golfer_scene import (
        ADMITTED_MYOSUITE_GOLFER_MODELS,
        MYOSUITE_MANIFEST_PATHS,
        MyoSuiteGolferGripAdapter,
    )

    assert model_id in ADMITTED_MYOSUITE_GOLFER_MODELS
    manifest_path = MYOSUITE_MANIFEST_PATHS[model_id]
    assert manifest_path.is_file(), f"Missing manifest: {manifest_path}"

    raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(raw_manifest)
    assert raw_manifest["model_id"] == model_id
    assert raw_manifest["club_body"] == "golf_club"
    assert raw_manifest["lead_hand_body"] == "hand_l"
    assert raw_manifest["trail_hand_body"] == "hand_r"
    assert raw_manifest["attachment_mode"] == "weld"

    iface = driver_interface if "driver" in model_id else iron_interface
    grip_spec = GripSpec(style=style, trigger_finger=trigger_finger)
    topo = contact_topology(grip_spec)

    adapter = MyoSuiteGolferGripAdapter.from_model_id(
        model_id, grip_spec=grip_spec, interface=iface
    )
    assert isinstance(adapter, ModelGripAdapter)
    assert adapter.model.nbody == 33
    assert adapter.model.neq == 20
    assert adapter.club_body == "golf_club"
    assert adapter.lead_hand_body == "hand_l"
    assert adapter.trail_hand_body == "hand_r"

    binding = adapter.resolve_binding(topo)
    assert isinstance(binding, HandClubBinding)
    validate_resolved_binding(binding, topo, iface)
    assert {sb.site_id for sb in binding.sites} == set(topo.site_names())
    assert len(binding.unavailable_sites) == 0


@pytest.mark.parametrize("model_id", ADMITTED_SCENE_IDS)
def test_site_weld_mode_capabilities_and_allocated_forces_from_efc(
    model_id: str,
    overlap_topology: ContactTopology,
    overlap_model: TopologyContactModel,
) -> None:
    """Site-weld mode reports allocated pad and hand-hand forces from efc_force with prior name,
    declares supports_physical_pairs=False with unavailable_reason, and apply raises.
    """
    mujoco = _require_mujoco_and_myo_sim()
    from src.engines.physics_engines.myosuite.python.golfer_scene import (
        MyoSuiteGolferGripAdapter,
        WeldedGripBodyError,
    )

    adapter = MyoSuiteGolferGripAdapter.from_model_id(model_id)
    assert adapter.contact_mode is False
    assert adapter.active_grip_weld_names() == ("grip_weld_l", "grip_weld_r")

    caps = adapter.capabilities()
    assert caps.supports_physical_pairs is False
    assert caps.supports_wrench_application is False
    assert caps.native_contact is False
    assert caps.attachment_modes == ("weld",)
    assert caps.unavailable_reason is not None
    assert "golf_club" in caps.unavailable_reason
    assert "hand_l" in caps.unavailable_reason
    assert "hand_r" in caps.unavailable_reason

    for phys_mode in ("contact", "contact_v1", "bushing"):
        with pytest.raises(SchemaFeatureUnavailableError, match="golf_club"):
            adapter.resolve_binding(overlap_topology, attachment_mode=phys_mode)

    mujoco.mj_step(adapter.model, adapter.data)
    closure = adapter.extract_weld_closure()
    assert closure.available is True
    assert len(closure.wrenches) == 2
    for w in closure.wrenches:
        assert np.isfinite(w.force_n).all()
        assert np.isfinite(w.torque_nm).all()
        assert float(np.linalg.norm(w.force_n)) > 0.0

    # Allocated pad and hand-hand forces with default prior ("equal_split")
    report = adapter.evaluate_contacts(overlap_model)
    assert report.applied_allocated_force_n() == 0.0
    assert report.body_wrenches() == {}
    assert report.allocated_contacts is not None
    assert report.allocated_contacts.available is True
    assert report.allocated_contacts.prior_name == "equal_split"

    for pair in overlap_topology.pairs:
        assert report.pair_labels[pair.name] == "allocated"
        assert report.allocated_labels[pair.name] == pair.allocated_label
        assert np.isfinite(report.normal_force_n[pair.name])
        assert np.isfinite(report.tangential_force_n[pair.name])

    pad_alloc = adapter.allocate_weld_pad_forces(overlap_topology)
    assert len(pad_alloc) == len(overlap_topology.hand_grip_pairs())
    for hg_pair in overlap_topology.hand_grip_pairs():
        pf = pad_alloc[hg_pair.name]
        assert pf.available is True
        assert pf.method == "allocation"
        assert pf.label == hg_pair.allocated_label
        assert pf.normal_force_n is not None and math.isfinite(pf.normal_force_n)
        assert pf.tangential_force_n is not None and math.isfinite(
            pf.tangential_force_n
        )

    # Custom named prior ("palm_dominant_75_25")
    hh_names = tuple(p.name for p in overlap_topology.hand_hand_pairs())
    weights = {hh_names[0]: 0.75}
    for extra in hh_names[1:]:
        weights[extra] = 0.25 / max(1, len(hh_names) - 1)
    custom_prior = AllocationPrior(name="palm_dominant_75_25", weights=weights)
    report_custom = adapter.evaluate_contacts(overlap_model, prior=custom_prior)
    assert report_custom.allocated_contacts is not None
    assert report_custom.allocated_contacts.prior_name == "palm_dominant_75_25"

    # Applying wrenches in site-weld mode raises WeldedGripBodyError / SchemaFeatureUnavailableError
    with pytest.raises(SchemaFeatureUnavailableError):
        apply_report(adapter, report)

    with pytest.raises(
        (SchemaFeatureUnavailableError, ValueError, WeldedGripBodyError)
    ):
        adapter.apply_wrenches(
            {
                "hand_l": (np.array([10.0, 0.0, 0.0]), np.zeros(3)),
                "golf_club": (np.array([-10.0, 0.0, 0.0]), np.zeros(3)),
            }
        )


import math  # noqa: E402


def test_contact_mode_disables_grip_welds_and_never_duplicates_weld_and_shared_contact(
    driver_interface: GripInterface,
    overlap_topology: ContactTopology,
    overlap_model: TopologyContactModel,
) -> None:
    """Contact mode disables grip_weld_l/r via eq_active, applies shared-law wrenches via
    xfrc_applied, and raises if any grip weld is active when applying wrenches.
    """
    mujoco = _require_mujoco_and_myo_sim()
    from src.engines.physics_engines.myosuite.python.golfer_scene import (
        MyoSuiteGolferGripAdapter,
        WeldedGripBodyError,
    )

    adapter = MyoSuiteGolferGripAdapter.from_model_id(
        "myosuite_golfer_driver",
        interface=driver_interface,
        contact_mode=True,
    )
    assert adapter.contact_mode is True
    assert adapter.active_grip_weld_names() == ()

    # Verify grip_weld_l and grip_weld_r have eq_active == 0, while knee/foot equalities remain active
    for side_key, eq_id in adapter.grip_weld_ids.items():
        assert int(adapter.data.eq_active[eq_id]) == 0, (
            f"grip weld {side_key} (eq {eq_id}) must be inactive in contact_mode"
        )
    active_other = [
        eq_id
        for eq_id in range(int(adapter.model.neq))
        if eq_id not in adapter.grip_weld_ids.values()
        and int(adapter.data.eq_active[eq_id]) != 0
    ]
    assert len(active_other) == 18

    caps = adapter.capabilities()
    assert caps.supports_physical_pairs is True
    assert caps.supports_wrench_application is True
    assert caps.native_contact is False

    binding = adapter.resolve_binding(overlap_topology, attachment_mode="contact")
    validate_binding(overlap_topology, binding, caps)

    club_state = RigidBodyState(
        rotation=np.eye(3),
        position_m=np.array([0.3, 0.0, 1.0]),
        velocity_m_s=np.zeros(3),
        omega_rad_s=np.zeros(3),
    )
    hands = hand_frame_states(club_state, driver_interface)
    adapter.set_prescribed_grip_states(hand_states=hands, club_state=club_state)

    report = adapter.evaluate_contacts(overlap_model, binding)
    assert report.action_reaction_residual_n() <= 1e-12
    assert report.applied_allocated_force_n() == 0.0
    assert len(report.pair_wrenches) == len(overlap_topology.hand_grip_pairs())

    apply_report(adapter, report)
    assert set(adapter.applied_wrenches.keys()) == {"golf_club", "hand_l", "hand_r"}

    cid = mujoco.mj_name2id(adapter.model, mujoco.mjtObj.mjOBJ_BODY, "golf_club")
    lid = mujoco.mj_name2id(adapter.model, mujoco.mjtObj.mjOBJ_BODY, "hand_l")
    rid = mujoco.mj_name2id(adapter.model, mujoco.mjtObj.mjOBJ_BODY, "hand_r")
    net_xfrc = (
        adapter.data.xfrc_applied[cid, :3]
        + adapter.data.xfrc_applied[lid, :3]
        + adapter.data.xfrc_applied[rid, :3]
    )
    np.testing.assert_allclose(net_xfrc, np.zeros(3), atol=1e-11)

    # Guard against simultaneous weld and shared contact on the same pair:
    # if a caller re-activates eq_active on grip_weld_l or grip_weld_r, apply_wrenches raises!
    eq_l = adapter.grip_weld_ids["L"]
    adapter.data.eq_active[eq_l] = 1
    with pytest.raises((SchemaFeatureUnavailableError, WeldedGripBodyError)):
        adapter.apply_wrenches(report.body_wrenches())


def test_missing_bodies_raise_schema_feature_unavailable_before_stepping(
    overlap_topology: ContactTopology,
) -> None:
    """Missing bodies in manifest/binding raise SchemaFeatureUnavailableError before stepping."""
    _require_mujoco_and_myo_sim()
    from src.engines.physics_engines.myosuite.python.golfer_scene import (
        MYOSUITE_MANIFEST_PATHS,
        MyoSuiteGolferGripAdapter,
    )

    base = MyoSuiteGolferGripAdapter.from_model_id("myosuite_golfer_driver")
    raw_manifest = json.loads(
        MYOSUITE_MANIFEST_PATHS["myosuite_golfer_driver"].read_text(encoding="utf-8")
    )
    bad_manifest = dict(raw_manifest)
    bad_manifest["lead_hand_body"] = "nonexistent_hand_l"

    with pytest.raises(SchemaFeatureUnavailableError, match="nonexistent_hand_l"):
        MyoSuiteGolferGripAdapter(
            base.model,
            base.data,
            manifest=bad_manifest,
        )


@pytest.mark.parametrize(
    "style", [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER]
)
@pytest.mark.parametrize("contact_mode", [False, True])
def test_three_styles_by_two_modes_run_and_bimanual_grasp_synthesis(
    driver_interface: GripInterface,
    style: GripStyle,
    contact_mode: bool,
) -> None:
    """3 styles x 2 modes run and satisfy bimanual grasp synthesis ||G y - w_club||_inf <= 1e-9 N."""
    mujoco = _require_mujoco_and_myo_sim()
    from src.engines.physics_engines.myosuite.python.golfer_scene import (
        MyoSuiteGolferGripAdapter,
    )

    grip_spec = GripSpec(style=style)
    topo = contact_topology(grip_spec)
    topo_model = build_topology_model(
        driver_interface, topo, squeeze_per_hand_n=SQUEEZE_N
    )

    adapter = MyoSuiteGolferGripAdapter.from_model_id(
        "myosuite_golfer_driver",
        grip_spec=grip_spec,
        interface=driver_interface,
        contact_mode=contact_mode,
    )
    att_mode = "contact" if contact_mode else "weld"
    binding = adapter.resolve_binding(topo, attachment_mode=att_mode)

    if contact_mode:
        club_state = RigidBodyState(
            rotation=np.eye(3),
            position_m=np.array([0.3, 0.0, 1.0]),
            velocity_m_s=np.array([0.04, -0.02, 0.01]),
            omega_rad_s=np.array([0.1, -0.15, 0.05]),
        )
        hands = hand_frame_states(club_state, driver_interface)
        adapter.set_prescribed_grip_states(hand_states=hands, club_state=club_state)
        report = adapter.evaluate_contacts(topo_model, binding)
        apply_report(adapter, report)
        mujoco.mj_step(adapter.model, adapter.data)
    else:
        mujoco.mj_step(adapter.model, adapter.data)
        report = adapter.evaluate_contacts(topo_model, binding)

    synthesis = adapter.synthesize_bimanual_grasp(report, binding=binding)
    assert synthesis.grasp_matrix.shape == (6, 12)
    assert synthesis.hand_wrench_vector.shape == (12,)
    assert synthesis.club_wrench_vector.shape == (6,)
    assert float(np.linalg.norm(synthesis.club_wrench_vector[:3])) > 0.0
    assert synthesis.residual_inf_n <= 1e-9, (
        f"Bimanual grasp synthesis residual {synthesis.residual_inf_n:.3e} N > 1e-9 N "
        f"for style={style.value}, contact_mode={contact_mode}"
    )


@pytest.mark.parametrize(
    "style", [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER]
)
@pytest.mark.parametrize("contact_mode", [False, True])
def test_m3_harness_passes_against_pinocchio_rigid_hand_reference(
    driver_interface: GripInterface,
    style: GripStyle,
    contact_mode: bool,
) -> None:
    """M3 qualification harness passes against PinocchioGripAdapter and law_level_series
    across 3 styles x 2 modes on identical site states within 5% peak / 2% RMS.
    """
    mujoco = _require_mujoco_and_myo_sim()
    from src.engines.physics_engines.myosuite.python.golfer_scene import (
        MyoSuiteGolferGripAdapter,
    )
    from src.engines.physics_engines.pinocchio.python.grip_binding_adapter import (
        PinocchioGripAdapter,
    )

    grip_spec = GripSpec(style=style)
    topo = contact_topology(grip_spec)
    topo_model = build_topology_model(
        driver_interface, topo, squeeze_per_hand_n=SQUEEZE_N
    )
    m_hash = topology_mapping_hash(topo_model)

    adapter = MyoSuiteGolferGripAdapter.from_model_id(
        "myosuite_golfer_driver",
        grip_spec=grip_spec,
        interface=driver_interface,
        contact_mode=contact_mode,
    )
    if not contact_mode:
        mujoco.mj_step(adapter.model, adapter.data)
    closure = (
        adapter.extract_weld_closure()
        if not contact_mode
        else adapter.synthesise_closure_from_net_wrench()
    )

    att_mode = "contact" if contact_mode else "weld"
    binding = adapter.resolve_binding(topo, attachment_mode=att_mode)

    pin_adapter = PinocchioGripAdapter(driver_interface, model_hash=m_hash)
    pin_binding = replace(
        pin_adapter.resolve_binding(topo),
        attachment_mode=att_mode,
        club_grip_frames=binding.club_grip_frames,
    )

    times = np.arange(21, dtype=float) * SAMPLE_DT_S
    myo_reports: list[GripStepReport] = []
    pin_reports: list[GripStepReport] = []
    hand_seq: dict[str, list[BushingState]] = {"L": [], "R": []}
    club_seq: dict[str, list[BushingState]] = {"L": [], "R": []}

    for t_s in times:
        phase = 2.0 * np.pi * (t_s / 0.04)
        club_pos = np.array(
            [0.3 + 0.01 * np.sin(phase), 0.0, 1.0 + 0.005 * np.cos(phase)]
        )
        club_vel = np.array([0.05 * np.cos(phase), -0.02 * np.sin(phase), 0.01])
        club_omega = np.array([0.1 * np.sin(phase), 0.2 * np.cos(phase), -0.15])
        club_state = RigidBodyState(
            rotation=np.eye(3),
            position_m=club_pos,
            velocity_m_s=club_vel,
            omega_rad_s=club_omega,
        )
        hands = hand_frame_states(club_state, driver_interface)
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
        myo_rep = adapter.evaluate_contacts(topo_model, binding, closure=closure)
        assert myo_rep.applied_allocated_force_n() == 0.0
        if contact_mode:
            apply_report(adapter, myo_rep)
        myo_reports.append(myo_rep)

        pin_adapter.set_prescribed_states(
            hand_states=hands_perturbed, club_state=club_state
        )
        pin_states = pin_adapter.sample_states(pin_binding)
        pin_rep = evaluate_contacts(
            topo_model, pin_binding, pin_states, closure=closure
        )
        assert pin_rep.applied_allocated_force_n() == 0.0
        pin_reports.append(pin_rep)

    assert total_applied_allocated_force_n(myo_reports) == 0.0

    myo_series = TopologyKineticsSeries.from_step_reports(
        times,
        myo_reports,
        engine=f"myosuite_golfer_{att_mode}",
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

    parity_vs_pin = compare_topology_series(pin_series, myo_series)
    assert parity_vs_pin.status == "passed", (
        f"M3 parity vs Pinocchio failed for {style.value} ({att_mode}): "
        f"{parity_vs_pin.to_dict()}"
    )
    assert parity_vs_pin.passed is True
    for ch_name, ch_res in parity_vs_pin.channels.items():
        assert ch_res.passed is True, f"Channel {ch_name} failed: {ch_res}"
        assert ch_res.error is not None
        assert ch_res.error.peak_error <= PEAK_TOLERANCE
        assert ch_res.error.rms_error <= RMS_TOLERANCE

    parity_vs_law = compare_topology_series(law_series, myo_series)
    assert parity_vs_law.status == "passed"
    assert parity_vs_law.passed is True


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


def test_golfer_scene_ast_budgets_and_lod() -> None:
    """AST check: file <= 1200 lines, functions <= 100 lines, <= 8 params, LoD <= 2 dots."""
    assert GOLFER_SCENE_PY.is_file()
    source = GOLFER_SCENE_PY.read_text(encoding="utf-8")
    lines = source.splitlines()
    assert len(lines) <= 1200, f"{GOLFER_SCENE_PY.name} has {len(lines)} lines > 1200"

    tree = ast.parse(source, filename=str(GOLFER_SCENE_PY))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            depth = _attribute_chain_depth(node)
            assert depth <= 2, (
                f"{GOLFER_SCENE_PY.name}:{node.lineno} has attribute chain depth {depth} > 2: "
                f"{ast.unparse(node)}"
            )
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            assert node.end_lineno is not None
            length = node.end_lineno - node.lineno + 1
            assert length <= 100, (
                f"{GOLFER_SCENE_PY.name}:{node.name} is {length} lines > 100"
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
                f"{GOLFER_SCENE_PY.name}:{node.name} has {n_params} params > 8"
            )
