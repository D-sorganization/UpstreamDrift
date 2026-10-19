"""Unit and integration tests for the native OpenSim MSK grip adapter (HRG-GS-M4, #13326).

Acceptance criteria:
- Extend ``opensim/python/msk_club.py`` (do not fork it) with ``grip_style``,
  ``trigger_finger`` and an OpenSim ``ModelGripAdapter`` for both club modes:
  - lead ``WeldJoint`` + trail ``WeldConstraint``: allocated/reporting only;
    ``GripCapabilities`` declares ``supports_physical_pairs=False`` with a
    non-empty ``unavailable_reason``, and ``apply_wrenches`` / ``apply_report``
    raises for welded bodies (never silently cancels).
  - free club + two bushings: physical, via shared-law point forces on
    ``hand_l`` / ``hand_r`` (and ``Club``).
- Admit ``bruno2015_fullbody_os4_v2`` individually (locked forearm/wrist noted)
  with a manifest under ``src/config/grip_bindings/``; lumbar-only spine models
  fail closed with ``SchemaFeatureUnavailableError``.
- 3 styles x 2 attachment modes run; neutral preload penetration within +-0.5 mm.
- Net club wrench vs the existing ``msk_club`` bushing path <= 5 % peak / <= 2 % RMS;
  M3 qualification harness passes.
"""

from __future__ import annotations

import ast
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from defusedxml import ElementTree as SafeET

from src.engines.physics_engines.opensim.python import msk_club as mc
from src.engines.physics_engines.opensim.python import msk_grip_adapter as ga
from src.shared.python.grip_contact import BushingState, GripInterface
from src.shared.python.grip_contact.binding import (
    GripCapabilities,
    GripStepReport,
    HandClubBinding,
    ModelGripAdapter,
    apply_report,
    evaluate_contacts,
)
from src.shared.python.grip_contact.binding_manifest import (
    binding_from_manifest,
    validate_manifest,
    validate_resolved_binding,
)
from src.shared.python.grip_contact.contact_topology import (
    DEFAULT_TOPOLOGY_PRELOAD_M,
    ContactTopology,
    GripStyle,
    contact_topology,
)
from src.shared.python.grip_contact.parity import (
    PEAK_TOLERANCE,
    RMS_TOLERANCE,
    compare_topology_series,
)
from src.shared.python.grip_contact.topology_contact import (
    TopologyContactModel,
    build_topology_model,
    topology_pair_wrench,
)
from src.shared.python.grip_contact.topology_series import (
    TopologyKineticsSeries,
    law_level_series,
    topology_mapping_hash,
    total_applied_allocated_force_n,
)
from src.shared.python.motion_matching.full_body_schema import (
    SchemaFeatureUnavailableError,
)
from src.shared.python.motion_matching.grip_contract import GripSpec
from tests.helpers.opensim_support import require_opensim

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
MANIFESTS_DIR = ROOT / "src" / "config" / "grip_bindings"
OPENSIM_PY = ROOT / "src" / "engines" / "physics_engines" / "opensim" / "python"
GUARDED_MODULES = (OPENSIM_PY / "msk_club.py", OPENSIM_PY / "msk_grip_adapter.py")
NEUTRAL_PENETRATION_TOL_M = 5.0e-4  # +-0.5 mm
SQUEEZE_N = 1100.0

ADMITTED_MANIFESTS = {
    "golf_humanoid": MANIFESTS_DIR / "opensim_golf_humanoid.json",
    "golf_humanoid_scaled": MANIFESTS_DIR / "opensim_golf_humanoid_scaled.json",
    "bruno2015_fullbody_os4_v2": MANIFESTS_DIR / "bruno2015_fullbody_os4_v2.json",
}


def _require_opensim_runtime() -> Any:
    osim = require_opensim()
    if not hasattr(osim, "Model"):
        pytest.skip("opensim in sys.modules is a test double")
    return osim


@pytest.mark.parametrize(
    "style", [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER]
)
@pytest.mark.parametrize("trigger_finger", [False, True])
def test_msk_club_grip_style_and_trigger_finger(
    style: GripStyle, trigger_finger: bool
) -> None:
    """load_msk_club and msk_club accept grip_style and trigger_finger and build topology."""
    club = mc.load_msk_club("driver", grip_style=style, trigger_finger=trigger_finger)
    assert club.grip_style == style
    assert club.trigger_finger is trigger_finger
    expected_topo = contact_topology(
        GripSpec(style=style, trigger_finger=trigger_finger)
    )
    assert club.topology == expected_topo

    # String style coercion also works
    club_from_str = mc.load_msk_club(
        "driver", grip_style=style.value, trigger_finger=trigger_finger
    )
    assert club_from_str.grip_style == style
    assert club_from_str.trigger_finger is trigger_finger


def test_msk_club_grip_style_and_trigger_finger_preconditions() -> None:
    """Invalid grip_style or trigger_finger raises TypeError/ValueError (DbC)."""
    with pytest.raises(ValueError, match="grip_style"):
        mc.load_msk_club("driver", grip_style="baseball_v99")
    with pytest.raises(TypeError, match="grip_style"):
        mc.load_msk_club("driver", grip_style=123)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="trigger_finger"):
        mc.load_msk_club("driver", trigger_finger="yes")  # type: ignore[arg-type]


@pytest.mark.parametrize("model_id,manifest_path", sorted(ADMITTED_MANIFESTS.items()))
@pytest.mark.parametrize(
    "style", [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER]
)
@pytest.mark.parametrize("trigger_finger", [False, True])
def test_admitted_manifests_validate_and_resolve_all_sites(
    model_id: str,
    manifest_path: Path,
    style: GripStyle,
    trigger_finger: bool,
) -> None:
    """Every admitted OpenSim manifest validates and resolves 100 % of topology sites."""
    assert manifest_path.is_file(), f"Missing manifest: {manifest_path}"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    assert manifest["model_id"] == model_id
    assert manifest["club_body"] == "Club"

    club = mc.load_msk_club("driver", grip_style=style, trigger_finger=trigger_finger)
    topo = club.topology
    binding = binding_from_manifest(manifest, topo, club.interface)
    validate_resolved_binding(binding, topo, club.interface)

    topo_sites = set(topo.site_names())
    bound_sites = {s.site_id for s in binding.sites}
    unavail_sites = set(binding.unavailable_sites.keys())
    assert bound_sites | unavail_sites == topo_sites
    assert len(bound_sites) == len(topo_sites)


def test_bruno2015_admission_notes_locked_forearm_and_wrist() -> None:
    """bruno2015_fullbody_os4_v2 is admitted individually with its locked forearm/wrist noted."""
    admission = ga.admit_opensim_grip_model("bruno2015_fullbody_os4_v2")
    assert admission.model_id == "bruno2015_fullbody_os4_v2"
    assert admission.lead_hand_body == "hand_L"
    assert admission.trail_hand_body == "hand_R"
    assert admission.club_body == "Club"
    expected_locked = {
        "pro_sup_l",
        "wrist_flex_l",
        "wrist_dev_l",
        "pro_sup_r",
        "wrist_flex_r",
        "wrist_dev_r",
    }
    assert expected_locked.issubset(set(admission.locked_forearm_wrist_coordinates))
    assert "forearm" in admission.note.lower() and "wrist" in admission.note.lower()

    # Adapter can resolve binding for bruno2015_fullbody_os4_v2 without requiring private SimTK file
    adapter = ga.OpenSimMskGripAdapter.from_model_id("bruno2015_fullbody_os4_v2")
    topo = contact_topology(GripSpec(style=GripStyle.OVERLAP))
    binding = adapter.resolve_binding(topo)
    assert binding.model_id == "bruno2015_fullbody_os4_v2"
    assert binding.lead_hand_body == "hand_L"
    assert binding.trail_hand_body == "hand_R"


@pytest.mark.parametrize(
    "lumbar_model_id",
    [
        "senteler2016_lumbspine_3dof",
        "senteler2016_lumbspine_30dof",
        "lumbar_only_spine_stub",
    ],
)
def test_lumbar_only_spine_models_fail_closed(lumbar_model_id: str) -> None:
    """Lumbar-only spine models lack hand bodies and fail closed with SchemaFeatureUnavailableError."""
    with pytest.raises(SchemaFeatureUnavailableError):
        ga.admit_opensim_grip_model(lumbar_model_id)
    with pytest.raises(SchemaFeatureUnavailableError):
        ga.OpenSimMskGripAdapter.from_model_id(lumbar_model_id)


def test_xml_model_without_hands_fails_closed() -> None:
    """An OpenSim XML tree missing hand bodies raises SchemaFeatureUnavailableError before stepping."""
    tree = SafeET.parse(str(mc.MODELS_DIR / "golf_humanoid.osim"))
    model_el = tree.getroot().find("Model")
    assert model_el is not None
    mc.strip_club(model_el)
    body_objects = model_el.find("BodySet/objects")
    assert body_objects is not None
    for body in list(body_objects):
        if body.get("name") in ("hand_l", "hand_r", "hand_L", "hand_R"):
            body_objects.remove(body)

    with pytest.raises(SchemaFeatureUnavailableError):
        ga.OpenSimMskGripAdapter.from_xml_element(model_el, model_id="stripped_lumbar")


@pytest.mark.requires_opensim
@pytest.mark.parametrize(
    "style", [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER]
)
@pytest.mark.parametrize("trigger_finger", [False, True])
def test_weld_mode_three_styles_allocated_only_and_apply_raises(
    style: GripStyle, trigger_finger: bool
) -> None:
    """Weld mode (lead WeldJoint + trail WeldConstraint): allocated/reporting only, apply raises."""
    _require_opensim_runtime()
    adapter = ga.OpenSimMskGripAdapter.from_model_id(
        "golf_humanoid",
        grip_model="weld",
        grip_style=style,
        trigger_finger=trigger_finger,
    )
    assert isinstance(adapter, ModelGripAdapter)
    caps = adapter.capabilities()
    assert isinstance(caps, GripCapabilities)
    assert caps.supports_physical_pairs is False
    assert caps.unavailable_reason is not None and len(caps.unavailable_reason) > 0
    assert "WeldJoint" in caps.unavailable_reason
    assert "WeldConstraint" in caps.unavailable_reason
    assert caps.attachment_modes == ("weld",)

    topo = adapter.club.topology
    binding = adapter.resolve_binding(topo)
    assert binding.attachment_mode == "weld"

    topo_model = build_topology_model(
        adapter.club.interface, topo, squeeze_per_hand_n=SQUEEZE_N
    )
    states = adapter.sample_states(binding)
    closure_stub = SimpleNamespace(
        available=True,
        reason="",
        wrenches=[
            SimpleNamespace(
                force_n=np.array([180.0, -90.0, 60.0]),
                torque_nm=np.array([5.0, -3.0, 1.5]),
                point_m=np.zeros(3),
            )
        ],
    )
    report = evaluate_contacts(topo_model, binding, states, closure=closure_stub)

    for pair in topo.hand_grip_pairs():
        assert report.pair_labels[pair.name] == "allocated"
        assert np.isfinite(report.normal_force_n[pair.name])
        assert report.normal_force_n[pair.name] > 0.0
    for hh in topo.hand_hand_pairs():
        assert report.pair_labels[hh.name] == "allocated"
        assert report.allocated_labels[hh.name] == hh.allocated_label
        assert np.isfinite(report.normal_force_n[hh.name])
        assert report.normal_force_n[hh.name] >= 0.0
        assert report.normal_force_n[hh.name] + report.tangential_force_n[hh.name] > 0.0

    assert report.pair_wrenches == {}
    assert report.body_wrenches() == {}
    assert report.applied_allocated_force_n() == 0.0

    # Applying wrenches to welded bodies must raise ValueError (never silently cancel)
    for body_name in (
        binding.lead_hand_body,
        binding.trail_hand_body,
        binding.club_body,
    ):
        with pytest.raises(ValueError, match="welded|Weld"):
            adapter.apply_wrenches(
                {body_name: (np.array([10.0, 0.0, 0.0]), np.zeros(3))}
            )

    # Tampering with report to inject a physical wrench in weld mode also raises
    lead_pose = states[binding.lead_hand_body]
    b_lead = (
        lead_pose
        if isinstance(lead_pose, BushingState)
        else BushingState(lead_pose[:3, :3], lead_pose[:3, 3], np.zeros(3), np.zeros(3))
    )
    sample_pw = topology_pair_wrench(
        topo_model, topo.hand_grip_pairs()[0], b_lead, b_lead, mode="contact"
    )
    tampered = replace(
        report,
        _body_wrenches={binding.lead_hand_body: (sample_pw.force_a_n, np.zeros(3))},
    )
    with pytest.raises(ValueError, match="welded|Weld"):
        apply_report(adapter, tampered)


@pytest.mark.requires_opensim
@pytest.mark.parametrize(
    "style", [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER]
)
@pytest.mark.parametrize("trigger_finger", [False, True])
def test_bushing_mode_three_styles_physical_preload_and_point_forces(
    style: GripStyle, trigger_finger: bool
) -> None:
    """Free club + two bushings mode: physical pairs, neutral preload within +-0.5 mm, point forces applied."""
    _require_opensim_runtime()
    adapter = ga.OpenSimMskGripAdapter.from_model_id(
        "golf_humanoid",
        grip_model="bushing",
        grip_style=style,
        trigger_finger=trigger_finger,
    )
    caps = adapter.capabilities()
    assert caps.supports_physical_pairs is True
    assert caps.supports_wrench_application is True
    assert caps.native_contact is False
    assert "contact" in caps.attachment_modes

    topo = adapter.club.topology
    binding = adapter.resolve_binding(topo)
    assert binding.attachment_mode == "contact"

    topo_model = build_topology_model(
        adapter.club.interface, topo, squeeze_per_hand_n=SQUEEZE_N
    )
    states = adapter.sample_states(binding)
    report = evaluate_contacts(topo_model, binding, states)

    # Every physical hand-grip pair has neutral preload penetration within +-0.5 mm
    for pair in topo.hand_grip_pairs():
        assert report.pair_labels[pair.name] == "physical"
        pw = report.pair_wrenches[pair.name]
        assert abs(pw.penetration_m) <= NEUTRAL_PENETRATION_TOL_M, (
            f"{pair.name} penetration {pw.penetration_m * 1000:.4f} mm outside +-0.5 mm"
        )
        assert pw.penetration_m == pytest.approx(DEFAULT_TOPOLOGY_PRELOAD_M, abs=1e-6)
        assert report.normal_force_n[pair.name] > 0.0

    # Hand-hand pairs remain allocated and never enter pair_wrenches
    for hh in topo.hand_hand_pairs():
        assert report.pair_labels[hh.name] == "allocated"
        assert hh.name not in report.pair_wrenches

    assert report.applied_allocated_force_n() == 0.0
    assert report.action_reaction_residual_n() <= 1e-12

    # Apply report: applies shared-law point forces on hand_l, hand_r, and Club,
    # and disables native BushingForce elements so forces are never applied twice.
    apply_report(adapter, report)
    applied = adapter.applied_wrenches
    assert set(applied) == {
        binding.lead_hand_body,
        binding.trail_hand_body,
        binding.club_body,
    }
    assert adapter.native_bushings_enabled() is False


@pytest.mark.requires_opensim
@pytest.mark.parametrize(
    "style", [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER]
)
def test_net_club_wrench_vs_msk_club_bushing_and_m3_harness(
    style: GripStyle,
) -> None:
    """Net club wrench vs existing msk_club bushing path <= 5 % peak / <= 2 % RMS, and M3 harness passes."""
    _require_opensim_runtime()
    sim_res = ga.simulate_msk_grip_hold(
        model_id="golf_humanoid",
        club="driver",
        grip_style=style,
        trigger_finger=False,
    )

    # 1. Net club wrench parity vs the existing msk_club bushing path
    assert sim_res.force_error.peak_error <= PEAK_TOLERANCE, (
        f"{style.value} net force peak_error={sim_res.force_error.peak_error:.5f} > {PEAK_TOLERANCE}"
    )
    assert sim_res.force_error.rms_error <= RMS_TOLERANCE, (
        f"{style.value} net force rms_error={sim_res.force_error.rms_error:.5f} > {RMS_TOLERANCE}"
    )
    assert sim_res.moment_error.peak_error <= PEAK_TOLERANCE, (
        f"{style.value} net moment peak_error={sim_res.moment_error.peak_error:.5f} > {PEAK_TOLERANCE}"
    )
    assert sim_res.moment_error.rms_error <= RMS_TOLERANCE, (
        f"{style.value} net moment rms_error={sim_res.moment_error.rms_error:.5f} > {RMS_TOLERANCE}"
    )
    assert sim_res.force_error.passes() is True
    assert sim_res.moment_error.passes() is True

    # 2. M3 qualification harness passes for the physical contact series
    assert total_applied_allocated_force_n(sim_res.reports) == 0.0
    m_hash = topology_mapping_hash(sim_res.topology_model)
    cand_series = TopologyKineticsSeries.from_step_reports(
        sim_res.time_s,
        sim_res.reports,
        engine="opensim_msk_contact",
        mapping_hash=m_hash,
        binding=sim_res.binding,
    )
    ref_series = law_level_series(
        sim_res.topology_model,
        sim_res.hand_states,
        sim_res.club_states,
        sim_res.time_s,
        mapping_hash=m_hash,
        mode="contact",
    )
    parity = compare_topology_series(ref_series, cand_series)
    assert parity.status == "passed", (
        f"M3 harness failed for {style.value}: {parity.to_dict()}"
    )
    assert parity.passed is True
    for ch_name, ch_res in parity.channels.items():
        assert ch_res.passed is True, f"Channel {ch_name} failed: {ch_res}"
        assert ch_res.error is not None
        assert ch_res.error.peak_error <= PEAK_TOLERANCE
        assert ch_res.error.rms_error <= RMS_TOLERANCE

    # 3. M3 qualification harness also passes for the allocated weld-mode series
    weld_adapter = ga.OpenSimMskGripAdapter.from_model_id(
        "golf_humanoid",
        grip_model="weld",
        grip_style=style,
    )
    weld_binding = weld_adapter.resolve_binding(sim_res.topology_model.topology)
    weld_states = weld_adapter.sample_states(weld_binding)
    weld_reports = [
        evaluate_contacts(sim_res.topology_model, weld_binding, weld_states)
        for _ in sim_res.time_s
    ]
    assert total_applied_allocated_force_n(weld_reports) == 0.0
    weld_cand = TopologyKineticsSeries.from_step_reports(
        sim_res.time_s,
        weld_reports,
        engine="opensim_msk_weld",
        mapping_hash=m_hash,
        binding=weld_binding,
    )
    lead_pose = weld_states[weld_binding.lead_hand_body]
    trail_pose = weld_states[weld_binding.trail_hand_body]
    club_pose = weld_states[weld_binding.club_body]
    assert isinstance(lead_pose, BushingState)
    assert isinstance(trail_pose, BushingState)
    assert isinstance(club_pose, BushingState)
    weld_hand_seq = {
        "L": [lead_pose for _ in sim_res.time_s],
        "R": [trail_pose for _ in sim_res.time_s],
    }
    weld_club_seq = {
        side: [
            ga._club_side_bushing_state(
                club_pose.rotation,
                club_pose.position_m,
                club_pose.velocity_m_s,
                club_pose.omega_rad_s,
                weld_binding.club_grip_frames[side],
            )
            for _ in sim_res.time_s
        ]
        for side in ("L", "R")
    }
    weld_ref = law_level_series(
        sim_res.topology_model,
        weld_hand_seq,
        weld_club_seq,
        sim_res.time_s,
        mapping_hash=m_hash,
        mode="weld",
    )
    weld_parity = compare_topology_series(weld_ref, weld_cand)
    assert weld_parity.status == "passed"
    assert weld_parity.passed is True


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


@pytest.mark.parametrize("module_path", GUARDED_MODULES, ids=lambda p: p.name)
def test_ast_lod_and_function_budgets(module_path: Path) -> None:
    """AST gate: <= 2 dot chain depth, <= 100 lines/function, <= 8 params, <= 1200 lines/file."""
    source = module_path.read_text(encoding="utf-8")
    lines = source.splitlines()
    assert len(lines) <= 1200, f"{module_path.name} has {len(lines)} lines > 1200"
    tree = ast.parse(source, filename=str(module_path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            depth = _attribute_chain_depth(node)
            assert depth <= 2, (
                f"{module_path.name}:{node.lineno} has attribute chain depth {depth} > 2: "
                f"{ast.unparse(node)}"
            )
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            assert node.end_lineno is not None
            length = node.end_lineno - node.lineno + 1
            assert length <= 100, (
                f"{module_path.name}:{node.name} is {length} lines > 100"
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
                f"{module_path.name}:{node.name} has {n_params} params > 8"
            )
