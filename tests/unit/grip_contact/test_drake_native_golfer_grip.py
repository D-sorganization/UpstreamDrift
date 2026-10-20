"""Unit tests for the native Drake golfer bilateral attachment and grip adapter (HRG-GS-M10, #13332).

Acceptance criteria:
- The default-flag model (attachment="unilateral") is byte- and numerically
  unchanged from the canonical golfer.urdf plant.
- The bilateral model (attachment="bilateral") loads and finalizes with a
  LinearBushingRollPitchYaw (reusing grip_contact.parameters.BushingParameters)
  placed at the grip-frame lead-hand centre from contact_topology (hand_x_origin),
  and the lead hand stays within 1 mm of its grip site over a 0.04 s prescribed hold.
- The manifest src/config/grip_bindings/drake_native_golfer.json validates via
  binding_manifest, and the binding resolves 100 % of sites or reason-codes them.
- Physics honesty:
  - capabilities().supports_physical_pairs is True only for a hand that is a
    DISTINCT body from the club and is not welded to it (lead / left_hand under
    bilateral attachment; False for trail / right_hand welded to club_shaft, and
    False when both hands are welded or unilateral).
  - Trail (right_hand) is welded to club_shaft: its hand-club pairs act within
    one rigid body; they are reporting/allocated only, never applied;
    apply_wrenches raises ValueError when given a wrench on a body welded to the
    club (or a pair whose two bodies are welded together) rather than silently
    cancelling.
  - Lead (left_hand) under the bilateral bushing is a distinct body, so its pad
    pairs can be physical in parallel with the bushing via
    ExternallyAppliedSpatialForce, with no native grip collision geometry
    registered (never applied twice).
  - Hand-hand pairs are allocated (NaN plus a reason) and zero allocated force
    is applied.
- Contact normal is recorded as the projection onto nhat_BA_W (G0 convention)
  and remains invariant (<= 1e-9 N) under pure tangential axial slip.
- M3 harness: for all 3 styles (overlap, interlock, ten_finger) on an identical
  prescribed 0.04 s hold, the adapter's evaluate_contacts series compared with
  the Pinocchio reference / law_level_series via compare_topology_series has
  status "passed" under the unchanged 5 % / 2 % tolerances.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.shared.python.grip_contact import (
    BushingState,
    GripInterface,
    load_coordinate_swing,
)
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
from src.shared.python.grip_contact.contact_topology import (
    ContactTopology,
    GripStyle,
    contact_topology,
)
from src.shared.python.grip_contact.parameters import BushingParameters, default_bushing
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
    TopologyKineticsSeries,
    law_level_series,
    topology_mapping_hash,
    total_applied_allocated_force_n,
)
from src.shared.python.motion_matching.grip_contract import GripSpec

pytestmark = [pytest.mark.unit, pytest.mark.requires_drake]

ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
FIXTURES = ROOT / "tests/fixtures/club_face"
MANIFEST_PATH = ROOT / "src/config/grip_bindings/drake_native_golfer.json"
HUMANOID_URDF_PY = (
    ROOT / "src/engines/physics_engines/drake/python/motion_matching/humanoid_urdf.py"
)
ADAPTER_PY = (
    ROOT / "src/engines/physics_engines/drake/python/native_golfer_grip_adapter.py"
)
SQUEEZE_N = 1100.0
HOLD_DURATION_S = 0.04


def _require_drake() -> Any:
    try:
        from pydrake.multibody.plant import MultibodyPlant
    except ImportError as exc:
        pytest.skip(f"pydrake not installed: {exc}")
    return MultibodyPlant


@pytest.fixture(scope="module")
def spec_dict() -> dict[str, Any]:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def interface(spec_dict: dict[str, Any]) -> GripInterface:
    return GripInterface.from_spec(spec_dict)


def test_default_flag_model_is_byte_and_numerically_unchanged() -> None:
    """AC 1: Default attachment='unilateral' leaves URDF and plant dynamics identical."""
    MultibodyPlant = _require_drake()
    from pydrake.multibody.parsing import Parser

    from src.engines.physics_engines.drake.python.motion_matching.humanoid_urdf import (
        EXPECTED_NUM_VELOCITIES,
        load_humanoid_dimensions,
        load_humanoid_into_plant,
        render_urdf_string,
    )

    # 1. The default model is exactly the generator's rendered humanoid (the
    #    right-hand-weld model the bilateral bushing targets).  Since #13312 the
    #    on-disk models/generated/golfer.urdf is the canonical golfer built by
    #    tools/model_converter (drift-gated by test_drake_urdf_drift), a
    #    different body tree, so it is not the default here (#13466).
    plant_rendered = MultibodyPlant(0.0)
    Parser(plant_rendered).AddModelsFromString(
        render_urdf_string(load_humanoid_dimensions()), "urdf"
    )
    plant_rendered.Finalize()

    # 2. Loading with default args vs explicit attachment="unilateral" produces
    #    numerically identical plant state, mass matrix, and time derivatives.
    plant_default = MultibodyPlant(0.0)
    load_humanoid_into_plant(plant_default)
    plant_default.Finalize()
    assert plant_default.HasBodyNamed("left_hand")
    assert plant_default.HasBodyNamed("right_hand")
    assert plant_default.num_positions() == plant_rendered.num_positions()

    plant_unilat = MultibodyPlant(0.0)
    load_humanoid_into_plant(plant_unilat, attachment="unilateral")
    plant_unilat.Finalize()

    assert plant_default.num_positions() == plant_unilat.num_positions()
    assert plant_default.num_velocities() == EXPECTED_NUM_VELOCITIES
    assert plant_unilat.num_velocities() == EXPECTED_NUM_VELOCITIES

    rng = np.random.default_rng(13332)
    ctx_def = plant_default.CreateDefaultContext()
    ctx_uni = plant_unilat.CreateDefaultContext()
    q = plant_default.GetPositions(ctx_def).copy()
    q[7:] = rng.uniform(-0.2, 0.2, size=plant_default.num_positions() - 7)
    v = rng.uniform(-0.5, 0.5, size=plant_default.num_velocities())

    plant_default.SetPositions(ctx_def, q)
    plant_default.SetVelocities(ctx_def, v)
    plant_unilat.SetPositions(ctx_uni, q)
    plant_unilat.SetVelocities(ctx_uni, v)

    m_def = plant_default.CalcMassMatrix(ctx_def)
    m_uni = plant_unilat.CalcMassMatrix(ctx_uni)
    np.testing.assert_allclose(m_def, m_uni, atol=0.0, rtol=0.0)

    xdot_def = plant_default.EvalTimeDerivatives(ctx_def).CopyToVector()
    xdot_uni = plant_unilat.EvalTimeDerivatives(ctx_uni).CopyToVector()
    np.testing.assert_allclose(xdot_def, xdot_uni, atol=0.0, rtol=0.0)

    ctx_ren = plant_rendered.CreateDefaultContext()
    plant_rendered.SetPositions(ctx_ren, q)
    plant_rendered.SetVelocities(ctx_ren, v)
    np.testing.assert_allclose(
        plant_rendered.CalcMassMatrix(ctx_ren), m_def, atol=0.0, rtol=0.0
    )
    np.testing.assert_allclose(
        plant_rendered.EvalTimeDerivatives(ctx_ren).CopyToVector(),
        xdot_def,
        atol=0.0,
        rtol=0.0,
    )

    # Invalid attachment flag raises ValueError
    plant_bad = MultibodyPlant(0.0)
    with pytest.raises(ValueError, match="attachment"):
        load_humanoid_into_plant(plant_bad, attachment="invalid_mode")


def test_bilateral_model_loads_finalizes_and_holds_lead_hand_within_1mm() -> None:
    """AC 2: attachment='bilateral' attaches left_hand via LinearBushingRollPitchYaw at hand_x_origin
    and keeps left_hand within 1 mm of its grip site over a 0.04 s prescribed hold.
    """
    MultibodyPlant = _require_drake()
    from src.engines.physics_engines.drake.python.motion_matching.humanoid_urdf import (
        EXPECTED_NUM_VELOCITIES,
        load_humanoid_into_plant,
    )
    from src.engines.physics_engines.drake.python.native_golfer_grip_adapter import (
        CLUB_LEAD_GRIP_FRAME,
        LEAD_HAND_GRIP_FRAME,
        NativeDrakeGolferGripAdapter,
        simulate_bilateral_hold,
    )

    topo = contact_topology(GripSpec(style=GripStyle.OVERLAP))
    bushing_params: BushingParameters = default_bushing()

    plant = MultibodyPlant(1.0e-3)
    load_humanoid_into_plant(
        plant,
        attachment="bilateral",
        topology=topo,
        bushing=bushing_params,
    )
    plant.Finalize()

    # Closed loop uses a compliant bushing, NEVER a second tree joint (DOF count unchanged)
    assert plant.num_velocities() == EXPECTED_NUM_VELOCITIES

    # Verify the grip frames exist and the club lead grip frame sits at hand_x_origin offset
    club_lead_frame = plant.GetFrameByName(CLUB_LEAD_GRIP_FRAME)
    lead_hand_frame = plant.GetFrameByName(LEAD_HAND_GRIP_FRAME)
    assert club_lead_frame.body().name() == "club_shaft"
    assert lead_hand_frame.body().name() == "left_hand"

    expected_dx = hand_x_origin(topo, "lead") - hand_x_origin(topo, "trail")
    tf_in_club = club_lead_frame.GetFixedPoseInBodyFrame()
    np.testing.assert_allclose(
        tf_in_club.translation(),
        np.array([expected_dx, 0.0, 0.10]),
        atol=1e-12,
    )

    # Simulate a 0.04 s prescribed hold with the bilateral model and physical lead-hand pads
    hold_result = simulate_bilateral_hold(
        style=GripStyle.OVERLAP,
        duration_s=HOLD_DURATION_S,
        dt_s=0.002,
    )
    assert hold_result.max_lead_hand_gap_m < 1.0e-3, (
        f"Lead hand drifted {hold_result.max_lead_hand_gap_m * 1000:.4f} mm > 1.0 mm"
    )
    assert isinstance(hold_result.adapter, NativeDrakeGolferGripAdapter)


@pytest.mark.parametrize(
    "style", [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER]
)
@pytest.mark.parametrize("trigger_finger", [False, True])
def test_manifest_validates_and_resolves_all_sites(
    interface: GripInterface,
    style: GripStyle,
    trigger_finger: bool,
) -> None:
    """AC 3: drake_native_golfer.json validates and resolves 100 % of sites or reason-codes them."""
    assert MANIFEST_PATH.exists(), f"Missing manifest at {MANIFEST_PATH}"
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    validate_manifest(manifest)

    assert manifest["model_id"] == "drake_native_golfer"
    assert manifest["club_body"] == "club_shaft"
    assert manifest["lead_hand_body"] == "left_hand"
    assert manifest["trail_hand_body"] == "right_hand"

    topo = contact_topology(GripSpec(style=style, trigger_finger=trigger_finger))
    binding = binding_from_manifest(manifest, topo, interface)
    validate_resolved_binding(binding, topo, interface)

    topo_sites = set(topo.site_names())
    bound_sites = {s.site_id for s in binding.sites}
    unavail_sites = set(binding.unavailable_sites.keys())
    assert bound_sites | unavail_sites == topo_sites
    assert len(bound_sites & unavail_sites) == 0
    assert len(bound_sites) == len(topo_sites)


def test_physics_honesty_capabilities_welded_trail_and_unwelded_lead(
    interface: GripInterface,
) -> None:
    """AC 4 (Physics Honesty):
    - supports_physical_pairs is True ONLY for a hand that is a distinct body from
      the club and not welded to it (left_hand under bilateral bushing).
    - Trail (right_hand) is welded to club_shaft: supports_physical_pairs is False;
      its pairs are reporting/allocated only, never applied; apply_wrenches raises
      ValueError if called for right_hand or any welded pair.
    - Lead (left_hand) pad pairs are applied via ExternallyAppliedSpatialForce with
      action-reaction balance, and zero collision geometries are registered for grip pairs.
    - Hand-hand pairs are allocated (NaN + reason) and zero allocated force is applied.
    """
    MultibodyPlant = _require_drake()
    from src.engines.physics_engines.drake.python.motion_matching.humanoid_urdf import (
        load_humanoid_into_plant,
    )
    from src.engines.physics_engines.drake.python.native_golfer_grip_adapter import (
        NativeDrakeGolferGripAdapter,
        solve_bilateral_grip_pose,
    )

    topo = contact_topology(GripSpec(style=GripStyle.OVERLAP, trigger_finger=True))
    topo_model = build_topology_model(interface, topo, squeeze_per_hand_n=SQUEEZE_N)

    # 1. Bilateral plant: left_hand is distinct and unwelded; right_hand is welded to club_shaft
    plant = MultibodyPlant(0.0)
    load_humanoid_into_plant(plant, attachment="bilateral", topology=topo)
    plant.Finalize()
    ctx = plant.CreateDefaultContext()
    q_bi = solve_bilateral_grip_pose(plant, ctx, topology=topo)
    plant.SetPositions(ctx, q_bi)

    adapter = NativeDrakeGolferGripAdapter(plant, ctx, topology=topo)
    assert isinstance(adapter, ModelGripAdapter)

    # Check welded status directly from plant.GetBodiesWeldedTo(club_shaft)
    assert adapter.is_body_welded_to_club("right_hand") is True
    assert adapter.is_body_welded_to_club("club_shaft") is True
    assert adapter.is_body_welded_to_club("left_hand") is False

    # Per-hand capability table
    lead_caps = adapter.hand_capabilities("lead")
    trail_caps = adapter.hand_capabilities("trail")
    assert lead_caps.supports_physical_pairs is True
    assert lead_caps.supports_wrench_application is True
    assert trail_caps.supports_physical_pairs is False
    assert trail_caps.supports_wrench_application is False
    assert adapter.capabilities("L").supports_physical_pairs is True
    assert adapter.capabilities("R").supports_physical_pairs is False
    assert adapter.capabilities().supports_physical_pairs is True

    # If a plant has left_hand welded to club_shaft as well, supports_physical_pairs is False
    from pydrake.multibody.tree import SpatialInertia

    plant_both_welded = MultibodyPlant(0.0)
    inertia = SpatialInertia.SolidSphereWithMass(0.4, 0.04)
    b_club = plant_both_welded.AddRigidBody("club_shaft", inertia)
    b_left = plant_both_welded.AddRigidBody("left_hand", inertia)
    b_right = plant_both_welded.AddRigidBody("right_hand", inertia)
    plant_both_welded.WeldFrames(b_club.body_frame(), b_left.body_frame())
    plant_both_welded.WeldFrames(b_club.body_frame(), b_right.body_frame())
    plant_both_welded.Finalize()
    ctx_bw = plant_both_welded.CreateDefaultContext()
    adapter_bw = NativeDrakeGolferGripAdapter(plant_both_welded, ctx_bw, topology=topo)
    assert adapter_bw.hand_capabilities("lead").supports_physical_pairs is False
    assert adapter_bw.hand_capabilities("trail").supports_physical_pairs is False
    assert adapter_bw.capabilities().supports_physical_pairs is False

    # 2. No native collision geometry is registered for the grip pairs (shared law only)
    assert adapter.registered_grip_collision_count() == 0

    # 3. Physical-only binding reason-codes all welded trail sites and binds only lead sites
    phys_binding = adapter.resolve_binding(topo, physical_only=True)
    for site in topo.sites_for_hand("lead"):
        assert site.name in phys_binding.site_binding_map()
    for site in topo.sites_for_hand("trail"):
        assert site.name in phys_binding.unavailable_sites
        assert "welded" in phys_binding.unavailable_sites[site.name].lower()

    # 4. Evaluating with physical_only binding produces NaN + reason for welded trail pairs
    #    and hand-hand allocated pairs, and finite forces for physical lead pairs.
    states = adapter.sample_states(phys_binding)
    phys_report = evaluate_contacts(topo_model, phys_binding, states)
    for pair in topo_model.physical_grip_pairs("L"):
        assert np.isfinite(phys_report.normal_force_n[pair.name])
        assert phys_report.normal_force_n[pair.name] > 0.0
    for pair in topo_model.physical_grip_pairs("R"):
        assert np.isnan(phys_report.normal_force_n[pair.name])
        assert pair.name in phys_report.unavailable_reasons
    for hh in topo.hand_hand_pairs():
        assert phys_report.pair_labels[hh.name] == "allocated"
        assert np.isnan(phys_report.normal_force_n[hh.name])
        assert hh.name in phys_report.unavailable_reasons
    assert phys_report.applied_allocated_force_n() == 0.0

    # 5. Applying physical_only report succeeds and applies ONLY left_hand and club_shaft
    #    ExternallyAppliedSpatialForces with exact action-reaction balance.
    apply_report(adapter, phys_report)
    assert set(adapter.applied_wrenches.keys()) == {"left_hand", "club_shaft"}
    assert len(adapter.last_applied_spatial_forces) == 2
    net_f = np.zeros(3)
    for sf in adapter.last_applied_spatial_forces:
        net_f += np.asarray(sf.F_Bq_W.translational())
    np.testing.assert_allclose(net_f, np.zeros(3), atol=1e-12)

    # 6. Attempting to apply a wrench on welded right_hand (or a full report that
    #    includes non-zero trail hand wrenches, or a welded pair) MUST raise ValueError!
    full_binding = adapter.resolve_binding(topo)
    full_report = evaluate_contacts(
        topo_model, full_binding, adapter.sample_states(full_binding)
    )
    with pytest.raises(ValueError, match="welded|right_hand|rigid body"):
        apply_report(adapter, full_report)

    with pytest.raises(ValueError, match="welded|right_hand|rigid body"):
        adapter.apply_wrenches(
            {
                "right_hand": (np.array([10.0, 0.0, 0.0]), np.zeros(3)),
                "club_shaft": (np.array([-10.0, 0.0, 0.0]), np.zeros(3)),
            }
        )

    with pytest.raises(ValueError, match="welded|right_hand|rigid body"):
        adapter.apply_pair_wrench(
            "trail_palm_proximal_grip",
            full_report.pair_wrenches["trail_palm_proximal_grip"],
            full_binding,
        )


def test_contact_normal_projected_onto_nhat_ba_w() -> None:
    """AC 5: Contact normal is recorded as the projection onto nhat_BA_W (G0 convention)
    and remains invariant (<= 1e-9 N) under pure tangential slip along the grip axis.
    """
    MultibodyPlant = _require_drake()
    from src.engines.physics_engines.drake.python.motion_matching.humanoid_urdf import (
        load_humanoid_into_plant,
    )
    from src.engines.physics_engines.drake.python.native_golfer_grip_adapter import (
        NativeDrakeGolferGripAdapter,
        project_pair_contact_normal,
        solve_bilateral_grip_pose,
    )
    from src.shared.python.grip_contact.binding import BodyStates

    topo = contact_topology(GripSpec(style=GripStyle.OVERLAP))
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    interface = GripInterface.from_spec(spec)
    topo_model = build_topology_model(interface, topo, squeeze_per_hand_n=SQUEEZE_N)

    plant = MultibodyPlant(0.0)
    load_humanoid_into_plant(plant, attachment="bilateral", topology=topo)
    plant.Finalize()
    ctx = plant.CreateDefaultContext()
    q_bi = solve_bilateral_grip_pose(plant, ctx, topology=topo)
    plant.SetPositions(ctx, q_bi)
    plant.SetVelocities(ctx, np.zeros(plant.num_velocities()))

    adapter = NativeDrakeGolferGripAdapter(plant, ctx, topology=topo)
    binding = adapter.resolve_binding(topo)
    base_states = adapter.sample_states(binding)
    rep_static = evaluate_contacts(topo_model, binding, base_states)

    # Introduce a pure tangential axial velocity (0.05 m/s along the club grip +x axis) on left_hand
    club_pose = base_states["club_shaft"]
    grip_x_world = club_pose[:3, 0]
    vel_map = {
        "club_shaft": (np.zeros(3), np.zeros(3)),
        "left_hand": (0.05 * grip_x_world, np.zeros(3)),
        "right_hand": (np.zeros(3), np.zeros(3)),
    }
    sliding_states = BodyStates(dict(base_states), velocities=vel_map)
    rep_sliding = evaluate_contacts(topo_model, binding, sliding_states)

    for pair in topo_model.physical_grip_pairs("L"):
        pw_stat = rep_static.pair_wrenches[pair.name]
        pw_slide = rep_sliding.pair_wrenches[pair.name]
        f_norm_stat, f_tang_stat, nhat_ba_w = project_pair_contact_normal(pw_stat)
        f_norm_slide, f_tang_slide, _ = project_pair_contact_normal(pw_slide)

        # nhat_BA_W is unit length and points from club (B) to hand site (A)
        assert np.isclose(np.linalg.norm(nhat_ba_w), 1.0, atol=1e-12)
        np.testing.assert_allclose(nhat_ba_w, -pw_stat.normal, atol=1e-12)

        # Projection onto -nhat_BA_W equals recorded normal_force_mag_n to <= 1e-12 N
        assert abs(f_norm_stat - rep_static.normal_force_n[pair.name]) <= 1e-12
        assert abs(f_norm_slide - rep_sliding.normal_force_n[pair.name]) <= 1e-12
        assert abs(f_tang_slide - rep_sliding.tangential_force_n[pair.name]) <= 1e-12

        # Under pure tangential sliding, total force magnitude increases due to friction,
        # but the projected normal force stays constant to <= 1e-9 N!
        assert f_tang_stat <= 1e-12
        assert f_tang_slide > 1.0
        assert np.linalg.norm(pw_slide.force_b_n) > f_norm_slide + 0.1
        assert abs(f_norm_slide - f_norm_stat) <= 1e-9


@pytest.mark.parametrize(
    "style", [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER]
)
def test_m3_harness_bilateral_hold_passes_across_all_styles(
    interface: GripInterface,
    style: GripStyle,
) -> None:
    """AC 6 (M3 harness, plant-level): For all 3 styles on an identical prescribed 0.04 s hold,
    the adapter's evaluate_contacts series compared with the Pinocchio reference adapter and
    law_level_series via compare_topology_series has status 'passed' under 5 % / 2 % tolerances,
    and zero allocated force is applied.
    """
    from dataclasses import replace

    _require_drake()
    from src.engines.physics_engines.drake.python.native_golfer_grip_adapter import (
        simulate_bilateral_hold,
    )
    from src.engines.physics_engines.pinocchio.python.grip_binding_adapter import (
        PinocchioGripAdapter,
    )

    topo = contact_topology(GripSpec(style=style))
    topo_model = build_topology_model(interface, topo, squeeze_per_hand_n=SQUEEZE_N)
    mapping_hash = topology_mapping_hash(topo_model)

    hold_res = simulate_bilateral_hold(
        style=style,
        duration_s=HOLD_DURATION_S,
        dt_s=0.002,
        squeeze_per_hand_n=SQUEEZE_N,
    )
    assert hold_res.max_lead_hand_gap_m < 1.0e-3
    assert total_applied_allocated_force_n(hold_res.reports) == 0.0

    cand_series = TopologyKineticsSeries.from_step_reports(
        hold_res.time_s,
        hold_res.reports,
        engine="drake_native_golfer",
        mapping_hash=mapping_hash,
        binding=hold_res.binding,
    )
    assert cand_series.applied_allocated_force_n == 0.0

    # 1. Compare against law_level_series at the identical 0.04 s prescribed hold states
    law_ref = law_level_series(
        topo_model,
        hold_res.hand_frames,
        hold_res.club_frames,
        hold_res.time_s,
        mapping_hash=mapping_hash,
    )
    parity_vs_law = compare_topology_series(law_ref, cand_series)
    assert parity_vs_law.status == "passed", (
        f"M3 parity vs law_level_series failed for {style.value}: {parity_vs_law.to_dict()}"
    )
    assert parity_vs_law.passed is True

    # 2. Compare against PinocchioGripAdapter evaluated on the identical 0.04 s hold states
    pin_adapter = PinocchioGripAdapter(interface, model_hash=mapping_hash)
    pin_binding = replace(
        pin_adapter.resolve_binding(topo),
        club_grip_frames=hold_res.binding.club_grip_frames,
    )
    pin_reports: list[GripStepReport] = []
    closure_stub = SimpleNamespace(
        available=True,
        reason="",
        wrenches=[
            SimpleNamespace(
                force_n=np.array([250.0, -120.0, 80.0]),
                torque_nm=np.array([6.0, -4.0, 2.0]),
                point_m=np.zeros(3),
            )
        ],
    )
    for k in range(len(hold_res.time_s)):
        pin_adapter.set_prescribed_states(
            hand_states={
                "L": hold_res.hand_frames["L"][k],
                "R": hold_res.hand_frames["R"][k],
            },
            club_state=hold_res.club_states[k],
        )
        p_states = pin_adapter.sample_states(pin_binding)
        p_rep = evaluate_contacts(
            topo_model, pin_binding, p_states, closure=closure_stub
        )
        assert p_rep.applied_allocated_force_n() == 0.0
        pin_reports.append(p_rep)

    pin_series = TopologyKineticsSeries.from_step_reports(
        hold_res.time_s,
        pin_reports,
        engine="pinocchio_reference",
        mapping_hash=mapping_hash,
        binding=pin_binding,
    )
    parity_vs_pin = compare_topology_series(pin_series, cand_series)
    assert parity_vs_pin.status == "passed", (
        f"M3 parity vs Pinocchio reference failed for {style.value}: {parity_vs_pin.to_dict()}"
    )
    assert parity_vs_pin.passed is True
    for ch_name, ch_res in parity_vs_pin.channels.items():
        assert ch_res.passed is True, f"{ch_name} failed: {ch_res}"
        assert ch_res.error is not None
        assert ch_res.error.peak_error <= PEAK_TOLERANCE
        assert ch_res.error.rms_error <= RMS_TOLERANCE


@pytest.mark.slow
@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    "style", [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER]
)
def test_m3_harness_dynamic_hold_vs_pinocchio_sim(
    spec_dict: dict[str, Any],
    interface: GripInterface,
    style: GripStyle,
) -> None:
    """AC 6 (M3 harness, full dynamic hold vs Pinocchio ClubInHands simulation):
    For all 3 styles on the 0.04 s dynamic hold, NativeDrakeGolferGripAdapter's
    evaluate_contacts series passes compare_topology_series against Pinocchio's
    simulate_grip_contact and law_level_series within 5 % peak / 2 % RMS.
    """
    MultibodyPlant = _require_drake()
    try:
        import pinocchio  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"pinocchio not installed: {exc}")

    from src.engines.physics_engines.drake.python import grip_contact_sim as drake_sim
    from src.engines.physics_engines.drake.python.motion_matching.humanoid_urdf import (
        load_humanoid_into_plant,
    )
    from src.engines.physics_engines.drake.python.native_golfer_grip_adapter import (
        NativeDrakeGolferGripAdapter,
    )
    from src.engines.physics_engines.pinocchio.python import grip_contact_sim as pin_sim

    spec_bytes = SPEC_PATH.read_bytes()
    names = spec_dict["coordinate_order"]
    swing = load_coordinate_swing(
        FIXTURES / "swing_q_driver.npz",
        FIXTURES / "address_poses.json",
        "driver",
        names,
    )
    topo = contact_topology(GripSpec(style=style))
    topo_model = build_topology_model(interface, topo, squeeze_per_hand_n=SQUEEZE_N)
    mapping_hash = topology_mapping_hash(topo_model)

    drake_run = drake_sim.simulate_grip_contact(
        spec_bytes,
        swing,
        interface=interface,
        t_end_s=HOLD_DURATION_S,
        hold=True,
        topology_model=topo_model,
    )
    pin_run = pin_sim.simulate_grip_contact(
        spec_bytes,
        swing,
        interface=interface,
        t_end_s=HOLD_DURATION_S,
        hold=True,
        topology_model=topo_model,
    )
    assert drake_run.hand_states is not None
    assert drake_run.club_states is not None

    # Evaluate every step through NativeDrakeGolferGripAdapter
    plant = MultibodyPlant(0.0)
    load_humanoid_into_plant(plant, attachment="bilateral", topology=topo)
    plant.Finalize()
    ctx = plant.CreateDefaultContext()
    adapter = NativeDrakeGolferGripAdapter(
        plant, ctx, topology=topo, interface=interface
    )
    binding = adapter.resolve_binding(topo)

    reports: list[GripStepReport] = []
    times = drake_run.series.time_s
    for k in range(len(times)):
        adapter.set_prescribed_grip_states(
            hand_states={
                "L": drake_run.hand_states["L"][k],
                "R": drake_run.hand_states["R"][k],
            },
            club_frames={
                "L": drake_run.club_states["L"][k],
                "R": drake_run.club_states["R"][k],
            },
        )
        rep = evaluate_contacts(topo_model, binding, adapter.sample_states(binding))
        assert rep.applied_allocated_force_n() == 0.0
        reports.append(rep)

    assert total_applied_allocated_force_n(reports) == 0.0
    cand_series = TopologyKineticsSeries.from_step_reports(
        times,
        reports,
        engine="drake_native_golfer",
        mapping_hash=mapping_hash,
        binding=binding,
    )
    pin_series = TopologyKineticsSeries.from_contact_run(
        pin_run,
        topo_model,
        engine="pinocchio",
        mapping_hash=mapping_hash,
    )
    law_series = law_level_series(
        topo_model,
        drake_run.hand_states,
        drake_run.club_states,
        times,
        mapping_hash=mapping_hash,
    )

    rep_vs_pin = compare_topology_series(pin_series, cand_series)
    rep_vs_law = compare_topology_series(law_series, cand_series)
    assert rep_vs_pin.status == "passed", (
        f"Parity vs Pinocchio failed for {style.value}: {rep_vs_pin.to_dict()}"
    )
    assert rep_vs_law.status == "passed", (
        f"Parity vs law_level_series failed for {style.value}: {rep_vs_law.to_dict()}"
    )
    max_pk = max(
        c.error.peak_error for c in rep_vs_pin.channels.values() if c.error is not None
    )
    max_rms = max(
        c.error.rms_error for c in rep_vs_pin.channels.values() if c.error is not None
    )
    assert max_pk <= PEAK_TOLERANCE
    assert max_rms <= RMS_TOLERANCE


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


def test_ast_budgets_on_changed_files() -> None:
    """AST gate: functions <= 100 lines, <= 8 parameters, files <= 1200 lines, LoD <= 2 on new adapter."""
    for path in (HUMANOID_URDF_PY, ADAPTER_PY):
        assert path.exists(), f"Missing {path}"
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()
        assert len(lines) <= 1200, f"{path.name} has {len(lines)} lines > 1200"
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if path == ADAPTER_PY and isinstance(node, ast.Attribute):
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
