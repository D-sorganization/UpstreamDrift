"""Unit tests for the native Pinocchio golfer two-hand variant and grip adapter (HRG-GS-M9, #13331).

Acceptance criteria:
- Both hands are addressable (distinct frames ``lead_hand`` and ``trail_hand``,
  with separation equal to the topology hand spacing within 1 mm) BEFORE any
  dynamics claim.
- The default-flag model (``hands="merged"``) is byte- and numerically
  unchanged (``nq``, ``nv``, frames, masses, and FK of every existing frame).
- The manifest ``src/config/grip_bindings/pinocchio_native_golfer.json``
  validates, and the resolved binding covers 100 % of topology sites.
- The M3 harness passes: on an identical prescribed 0.04 s hold trajectory,
  the adapter's ``evaluate_contacts`` series compared with ``law_level_series``
  via ``compare_topology_series`` has ``status == "passed"``.
- Allocated force applied is 0 (``apply_report`` guard).
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

from src.shared.python.grip_contact.binding import (
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
from src.shared.python.grip_contact.bushing_law import BushingState
from src.shared.python.grip_contact.contact_topology import (
    ContactTopology,
    GripStyle,
    contact_topology,
)
from src.shared.python.grip_contact.interface import GripInterface
from src.shared.python.grip_contact.parity import (
    PEAK_TOLERANCE,
    RMS_TOLERANCE,
    compare_topology_series,
)
from src.shared.python.grip_contact.topology_contact import (
    TopologyContactModel,
    build_topology_model,
    hand_x_origin,
)
from src.shared.python.grip_contact.topology_series import (
    SAMPLE_DT_S,
    TopologyKineticsSeries,
    law_level_series,
    topology_mapping_hash,
    total_applied_allocated_force_n,
)
from src.shared.python.motion_matching.grip_contract import GripSpec

pytestmark = [pytest.mark.unit, pytest.mark.requires_pinocchio]

ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
CANONICAL_YAML = (
    ROOT / "src/engines/physics_engines/pinocchio/models/spec/golfer_canonical.yaml"
)
URDF_PATH = ROOT / "src/engines/physics_engines/pinocchio/models/generated/golfer.urdf"
NATIVE_MANIFEST_PATH = ROOT / "src/config/grip_bindings/pinocchio_native_golfer.json"
ADAPTER_PY = (
    ROOT / "src/engines/physics_engines/pinocchio/python/native_golfer_grip_adapter.py"
)
SIMULATE_PY = (
    ROOT / "src/engines/physics_engines/pinocchio/python/motion_matching/simulate.py"
)
URDF_EXPORTER_PY = ROOT / "src/tools/model_converter/urdf_exporter.py"


def _require_pinocchio() -> Any:
    try:
        import pinocchio as pin  # noqa: PLC0415
    except ImportError as exc:
        pytest.skip(f"pinocchio not installed: {exc}")
    return pin


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
    return build_topology_model(interface, overlap_topology, squeeze_per_hand_n=1100.0)


@pytest.fixture(scope="module")
def native_manifest() -> dict[str, Any]:
    assert NATIVE_MANIFEST_PATH.exists(), f"Missing manifest: {NATIVE_MANIFEST_PATH}"
    return json.loads(NATIVE_MANIFEST_PATH.read_text(encoding="utf-8"))


def test_both_hands_addressable_and_separated_by_topology_spacing(
    overlap_topology: ContactTopology,
) -> None:
    """AC 1: Both hands are distinct frames separated by topology hand spacing within 1 mm."""
    pin = _require_pinocchio()
    from src.engines.physics_engines.pinocchio.python.motion_matching.simulate import (
        GRIP_FRAME_NAME,
        LEAD_HAND_FRAME_NAME,
        TRAIL_HAND_FRAME_NAME,
        build_golfer_model,
    )

    model = build_golfer_model(hands="separate")
    assert model.existFrame(GRIP_FRAME_NAME)
    assert model.existFrame(LEAD_HAND_FRAME_NAME)
    assert model.existFrame(TRAIL_HAND_FRAME_NAME)

    mid_id = int(model.getFrameId(GRIP_FRAME_NAME))
    lead_id = int(model.getFrameId(LEAD_HAND_FRAME_NAME))
    trail_id = int(model.getFrameId(TRAIL_HAND_FRAME_NAME))
    assert len({mid_id, lead_id, trail_id}) == 3

    data = model.createData()
    q0 = pin.neutral(model)
    pin.framesForwardKinematics(model, data, q0)

    p_mid = np.asarray(data.oMf[mid_id].translation, dtype=float)
    p_lead = np.asarray(data.oMf[lead_id].translation, dtype=float)
    p_trail = np.asarray(data.oMf[trail_id].translation, dtype=float)

    separation_m = float(np.linalg.norm(p_trail - p_lead))
    expected_spacing_m = float(
        hand_x_origin(overlap_topology, "trail")
        - hand_x_origin(overlap_topology, "lead")
    )
    assert abs(separation_m - expected_spacing_m) <= 1e-3, (
        f"Hand separation {separation_m:.6f} m differs from topology spacing "
        f"{expected_spacing_m:.6f} m by more than 1 mm"
    )
    # Midpoint of lead_hand and trail_hand coincides with mid_hands
    np.testing.assert_allclose(0.5 * (p_lead + p_trail), p_mid, atol=1e-12)


def test_default_flag_model_is_byte_and_numerically_unchanged() -> None:
    """AC 2: Default flag leaves URDF bytes, nq, nv, frames, masses, and FK identical."""
    pin = _require_pinocchio()
    from src.engines.physics_engines.pinocchio.python.motion_matching.simulate import (
        SimOptions,
        build_golfer_model,
        simulate_with_coefficients,
    )
    from src.tools.model_converter.schema_validator import validate_canonical_model
    from src.tools.model_converter.urdf_exporter import export_urdf

    canonical_model = validate_canonical_model(CANONICAL_YAML)
    exported_default = export_urdf(canonical_model)
    exported_merged = export_urdf(canonical_model, hands="merged")
    disk_urdf = URDF_PATH.read_text(encoding="utf-8")
    assert exported_default == disk_urdf
    assert exported_merged == disk_urdf

    exported_separate = export_urdf(canonical_model, hands="separate")
    assert 'link name="lead_hand"' in exported_separate
    assert 'link name="trail_hand"' in exported_separate
    assert 'link name="mid_hands"' in exported_separate

    raw_model = pin.buildModelFromUrdf(str(URDF_PATH))
    default_model = build_golfer_model()
    separate_model = build_golfer_model(hands="separate")

    assert default_model.nq == raw_model.nq
    assert default_model.nv == raw_model.nv
    assert default_model.nframes == raw_model.nframes
    assert not default_model.existFrame("lead_hand")
    assert not default_model.existFrame("trail_hand")

    # Separate-hands model has identical generalized coordinates and inertia
    assert separate_model.nq == raw_model.nq
    assert separate_model.nv == raw_model.nv
    for j_idx in range(raw_model.njoints):
        np.testing.assert_allclose(
            default_model.inertias[j_idx].mass,
            raw_model.inertias[j_idx].mass,
            atol=0.0,
        )
        np.testing.assert_allclose(
            separate_model.inertias[j_idx].mass,
            raw_model.inertias[j_idx].mass,
            atol=0.0,
        )

    # Check FK of every existing frame at a non-trivial joint configuration
    rng = np.random.default_rng(13331)
    q_test = pin.neutral(raw_model).copy()
    q_test[7:] = rng.uniform(-0.25, 0.25, size=raw_model.nq - 7)

    d_raw = raw_model.createData()
    d_def = default_model.createData()
    d_sep = separate_model.createData()
    pin.framesForwardKinematics(raw_model, d_raw, q_test)
    pin.framesForwardKinematics(default_model, d_def, q_test)
    pin.framesForwardKinematics(separate_model, d_sep, q_test)

    for f_idx in range(raw_model.nframes):
        assert default_model.frames[f_idx].name == raw_model.frames[f_idx].name
        np.testing.assert_allclose(
            d_def.oMf[f_idx].homogeneous,
            d_raw.oMf[f_idx].homogeneous,
            atol=0.0,
        )
        np.testing.assert_allclose(
            d_sep.oMf[f_idx].homogeneous,
            d_raw.oMf[f_idx].homogeneous,
            atol=0.0,
        )

    # Forward dynamics trajectory with SimOptions(hands="separate") vs default is identical
    theta = np.zeros(raw_model.nv * 7, dtype=np.float64)
    out_def = simulate_with_coefficients(
        theta, SimOptions(t_final=0.01, dt=0.002, hands="merged")
    )
    out_sep = simulate_with_coefficients(
        theta, SimOptions(t_final=0.01, dt=0.002, hands="separate")
    )
    np.testing.assert_allclose(out_sep.q, out_def.q, atol=0.0)
    np.testing.assert_allclose(out_sep.grip, out_def.grip, atol=0.0)
    np.testing.assert_allclose(out_sep.clubhead, out_def.clubhead, atol=0.0)


@pytest.mark.parametrize(
    "style", [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER]
)
@pytest.mark.parametrize("trigger_finger", [True, False])
def test_manifest_validates_and_resolves_all_sites(
    native_manifest: dict[str, Any],
    style: GripStyle,
    trigger_finger: bool,
) -> None:
    """AC 3: Manifest validates and adapter resolves 100 % of topology sites."""
    _require_pinocchio()
    from src.engines.physics_engines.pinocchio.python.native_golfer_grip_adapter import (
        NativePinocchioGolferGripAdapter,
    )

    validate_manifest(native_manifest)
    assert native_manifest["schema"] == "grip-binding/v1"
    assert native_manifest["model_id"] == "pinocchio_native_golfer"
    assert native_manifest["lead_hand_body"] == "lead_hand"
    assert native_manifest["trail_hand_body"] == "trail_hand"

    adapter = NativePinocchioGolferGripAdapter()
    assert isinstance(adapter, ModelGripAdapter)

    topology = contact_topology(GripSpec(style=style, trigger_finger=trigger_finger))
    binding = adapter.resolve_binding(topology)
    assert isinstance(binding, HandClubBinding)

    site_names_in_topology = set(topology.site_names())
    bound_site_names = {s.site_id for s in binding.sites}
    unavailable_names = set(binding.unavailable_sites.keys())

    assert bound_site_names | unavailable_names == site_names_in_topology
    assert len(bound_site_names & unavailable_names) == 0
    assert len(bound_site_names) == len(site_names_in_topology)

    validate_resolved_binding(binding, topology)


def test_capabilities_declare_weld_only_and_reject_physical_contact_mode(
    overlap_topology: ContactTopology,
) -> None:
    """Req 1: Capabilities declare supports_physical_pairs=False and attachment_modes=('weld',);
    requesting physical contact mode raises SchemaFeatureUnavailableError naming thorax2_to_thorax3.
    """
    _require_pinocchio()
    from src.engines.physics_engines.pinocchio.python.native_golfer_grip_adapter import (
        NativePinocchioGolferGripAdapter,
    )
    from src.shared.python.grip_contact.binding import validate_binding
    from src.shared.python.grip_contact.contact_topology import PHYSICAL_HAND_GRIP_MODES
    from src.shared.python.motion_matching.full_body_schema import (
        SchemaFeatureUnavailableError,
    )

    adapter = NativePinocchioGolferGripAdapter()
    caps = adapter.capabilities()
    assert caps.supports_physical_pairs is False
    assert caps.attachment_modes == ("weld",)
    for phys_mode in PHYSICAL_HAND_GRIP_MODES:
        assert phys_mode not in caps.attachment_modes

    binding = adapter.resolve_binding(overlap_topology)
    assert binding.attachment_mode == "weld"
    validate_binding(overlap_topology, binding, caps)

    for phys_mode in PHYSICAL_HAND_GRIP_MODES:
        phys_binding = replace(binding, attachment_mode=phys_mode)
        with pytest.raises(
            SchemaFeatureUnavailableError, match="thorax2_to_thorax3"
        ) as exc_info:
            validate_binding(overlap_topology, phys_binding, caps)
        assert phys_mode in str(exc_info.value)

        with pytest.raises(
            SchemaFeatureUnavailableError, match="thorax2_to_thorax3"
        ) as exc_resolve:
            adapter.resolve_binding(overlap_topology, attachment_mode=phys_mode)
        assert phys_mode in str(exc_resolve.value)


def test_adapter_fk_velocity_sampling_allocated_reporting_and_shared_joint_refusal(
    overlap_topology: ContactTopology,
    overlap_model: TopologyContactModel,
) -> None:
    """Req 2: Hand/club pad pairs are evaluated at FK states as allocated reporting channels,
    never enter apply_wrenches, and apply_wrenches refuses wrenches on bodies sharing a parent joint.
    """
    pin = _require_pinocchio()
    from src.engines.physics_engines.pinocchio.python.native_golfer_grip_adapter import (
        NativePinocchioGolferGripAdapter,
    )

    adapter = NativePinocchioGolferGripAdapter()
    binding = adapter.resolve_binding(overlap_topology)

    model = adapter.model
    q = pin.neutral(model).copy()
    v = np.zeros(model.nv, dtype=float)
    v[3:6] = np.array([0.2, -0.3, 0.4])  # non-zero angular velocity at root
    adapter.set_joint_state(q, v)

    states = adapter.sample_states(binding)
    for body_name in (
        binding.club_body,
        binding.lead_hand_body,
        binding.trail_hand_body,
    ):
        fid = int(model.getFrameId(body_name))
        np.testing.assert_allclose(
            states[body_name], adapter.data.oMf[fid].homogeneous, atol=1e-12
        )
        v_lin, v_ang = states.velocity_of(body_name)
        f_vel = pin.getFrameVelocity(
            model, adapter.data, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        np.testing.assert_allclose(v_lin, f_vel.linear, atol=1e-12)
        np.testing.assert_allclose(v_ang, f_vel.angular, atol=1e-12)

    report = evaluate_contacts(overlap_model, binding, states)
    # Every hand-grip pair is evaluated kinematically and labelled allocated (and external:<pair>_allocated)
    for pair in overlap_topology.hand_grip_pairs():
        assert report.pair_labels[pair.name] == "allocated"
        assert report.allocated_labels[pair.name] == pair.allocated_label
        assert report.allocated_labels[pair.name] == f"external:{pair.name}_allocated"
        assert np.isfinite(report.normal_force_n[pair.name])
        assert np.isfinite(report.tangential_force_n[pair.name])
        assert np.isfinite(report.spin_moment_nm[pair.name])
        assert report.normal_force_n[pair.name] > 0.0

    # Allocated pairs never enter pair_wrenches, body_wrenches(), or apply_wrenches
    assert len(report.pair_wrenches) == 0
    assert report.body_wrenches() == {}
    assert report.applied_allocated_force_n() == 0.0
    apply_report(adapter, report)
    assert adapter.applied_wrenches == {}
    assert adapter.frame_joint_forces == {}

    # apply_wrenches must refuse a physical pair wrench for bodies sharing one parent joint
    cancelling_pair_wrenches = {
        binding.lead_hand_body: (
            np.array([120.0, -45.0, 30.0]),
            np.array([2.0, -1.0, 0.5]),
        ),
        binding.club_body: (
            np.array([-120.0, 45.0, -30.0]),
            np.array([-2.0, 1.0, -0.5]),
        ),
    }
    with pytest.raises(ValueError, match="thorax2_to_thorax3"):
        adapter.apply_wrenches(cancelling_pair_wrenches)

    # Also refuses a wrench on a single welded grip body sharing thorax2_to_thorax3 with club_shaft
    with pytest.raises(ValueError, match="thorax2_to_thorax3"):
        adapter.apply_wrenches(
            {
                binding.trail_hand_body: (
                    np.array([50.0, 0.0, 0.0]),
                    np.zeros(3),
                )
            }
        )


def test_allocated_force_applied_is_zero_and_guard_blocks_tampering(
    overlap_topology: ContactTopology,
    overlap_model: TopologyContactModel,
) -> None:
    """AC 5: Allocated force applied is 0 and apply_report rejects non-zero allocated force."""
    _require_pinocchio()
    from src.engines.physics_engines.pinocchio.python.native_golfer_grip_adapter import (
        NativePinocchioGolferGripAdapter,
    )
    from src.shared.python.grip_contact.topology_contact import topology_pair_wrench

    adapter = NativePinocchioGolferGripAdapter()
    binding = adapter.resolve_binding(overlap_topology)
    states = adapter.sample_states(binding)

    closure_stub = SimpleNamespace(
        available=True,
        reason="",
        wrenches=[
            SimpleNamespace(
                force_n=np.array([350.0, -180.0, 120.0]),
                torque_nm=np.array([9.0, -6.0, 3.0]),
                point_m=np.zeros(3),
            )
        ],
    )

    report = evaluate_contacts(overlap_model, binding, states, closure=closure_stub)
    for hh in overlap_topology.hand_hand_pairs():
        assert report.pair_labels[hh.name] == "allocated"
        assert report.allocated_labels[hh.name] == hh.allocated_label
        assert np.isfinite(report.normal_force_n[hh.name])
        assert report.normal_force_n[hh.name] > 0.0

    assert report.applied_allocated_force_n() == 0.0
    apply_report(adapter, report)

    # Tamper with report by inserting an allocated pair into pair_wrenches
    lead_pose = states[binding.lead_hand_body]
    b_lead = BushingState(lead_pose[:3, :3], lead_pose[:3, 3], np.zeros(3), np.zeros(3))
    sample_pw = topology_pair_wrench(
        overlap_model,
        overlap_topology.hand_grip_pairs()[0],
        b_lead,
        b_lead,
        mode="contact",
    )
    alloc_name = overlap_topology.hand_hand_pairs()[0].name
    tampered_pw = {alloc_name: sample_pw}
    tampered_report = replace(report, pair_wrenches=tampered_pw)
    assert tampered_report.applied_allocated_force_n() > 0.0
    with pytest.raises(ValueError, match="allocated"):
        apply_report(adapter, tampered_report)


@pytest.mark.slow
def test_m3_harness_hold_trajectory_passes_against_law_level_series(
    overlap_topology: ContactTopology,
    overlap_model: TopologyContactModel,
) -> None:
    """Req 3: On a 0.04 s prescribed hold trajectory, kinematic pair evaluation matches
    law_level_series at the same states as an allocated reporting-channel check.
    """
    pin = _require_pinocchio()
    from src.engines.physics_engines.pinocchio.python.native_golfer_grip_adapter import (
        NativePinocchioGolferGripAdapter,
    )

    adapter = NativePinocchioGolferGripAdapter()
    binding = adapter.resolve_binding(overlap_topology)
    assert binding.attachment_mode == "weld"

    model = adapter.model
    q_hold = pin.neutral(model).copy()
    v_hold = np.zeros(model.nv, dtype=float)

    times = np.arange(21, dtype=float) * SAMPLE_DT_S  # 0.00 s .. 0.04 s (21 steps)
    reports: list[GripStepReport] = []
    hand_seq: dict[str, list[BushingState]] = {"L": [], "R": []}
    club_seq: dict[str, list[BushingState]] = {"L": [], "R": []}

    for _ in times:
        adapter.set_joint_state(q_hold, v_hold)
        states = adapter.sample_states(binding)
        rep = evaluate_contacts(overlap_model, binding, states)
        assert rep.applied_allocated_force_n() == 0.0
        assert rep.body_wrenches() == {}
        apply_report(adapter, rep)
        reports.append(rep)

        lead_pose = states[binding.lead_hand_body]
        trail_pose = states[binding.trail_hand_body]
        lead_v, lead_w = states.velocity_of(binding.lead_hand_body)
        trail_v, trail_w = states.velocity_of(binding.trail_hand_body)

        b_lead = BushingState(lead_pose[:3, :3], lead_pose[:3, 3], lead_v, lead_w)
        b_trail = BushingState(trail_pose[:3, :3], trail_pose[:3, 3], trail_v, trail_w)
        hand_seq["L"].append(b_lead)
        hand_seq["R"].append(b_trail)
        club_seq["L"].append(b_lead)
        club_seq["R"].append(b_trail)

    assert total_applied_allocated_force_n(reports) == 0.0

    m_hash = topology_mapping_hash(overlap_model)
    candidate_series = TopologyKineticsSeries.from_step_reports(
        times,
        reports,
        engine="pinocchio_native_golfer",
        mapping_hash=m_hash,
        binding=binding,
    )
    reference_series = law_level_series(
        overlap_model,
        hand_seq,
        club_seq,
        times,
        mapping_hash=m_hash,
        mode="weld",
    )

    parity = compare_topology_series(reference_series, candidate_series)
    assert parity.status == "passed", f"M3 parity failed: {parity.to_dict()}"
    assert parity.passed is True
    for ch_name, ch_res in parity.channels.items():
        assert ch_res.status == "passed", f"Channel {ch_name} failed: {ch_res}"
        assert ch_res.passed is True
        assert ch_res.error is not None
        assert ch_res.error.peak_error <= PEAK_TOLERANCE
        assert ch_res.error.rms_error <= RMS_TOLERANCE

    # Verify every hand-grip pad pair is checked in allocated_channels as a reporting channel
    for pair in overlap_topology.hand_grip_pairs():
        for kind in ("normal", "tangential", "spin"):
            ch_key = f"{kind}:{pair.name}"
            assert ch_key in parity.allocated_channels
            alloc_res = parity.allocated_channels[ch_key]
            assert alloc_res.label == "allocated"
            assert alloc_res.status == "passed", (
                f"Reporting channel {ch_key} failed: {alloc_res}"
            )
            assert alloc_res.passed is True
            assert alloc_res.error is not None
            assert alloc_res.error.peak_error <= PEAK_TOLERANCE
            assert alloc_res.error.rms_error <= RMS_TOLERANCE

    # Physical-mode reference series cannot pass against the welded adapter's allocated series
    physical_ref_series = law_level_series(
        overlap_model,
        hand_seq,
        club_seq,
        times,
        mapping_hash=m_hash,
        mode="contact",
    )
    phys_parity = compare_topology_series(physical_ref_series, candidate_series)
    assert phys_parity.status == "unavailable"
    assert phys_parity.passed is False


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


def test_ast_lod_and_function_budgets() -> None:
    """AST gate: <= 2 dot chain depth, <= 100 lines/function, <= 8 params, <= 1200 lines/file."""
    for path in (ADAPTER_PY, SIMULATE_PY, URDF_EXPORTER_PY):
        assert path.exists(), f"Missing source file: {path}"
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
