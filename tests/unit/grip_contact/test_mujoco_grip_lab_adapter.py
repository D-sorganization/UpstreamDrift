"""Unit and live-simulation tests for the MuJoCo Shadow/Allegro grip laboratory adapter (HRG-GS-M6, #13328).

This suite tests a contact fixture, not a qualified golfer:
- Segment-mapped ``grip-binding/v1`` manifests under ``src/config/grip_bindings/``
  for the Shadow Hand (``mujoco_shadow_hand_lab``) and Wonik Allegro Hand
  (``mujoco_allegro_hand_lab``) where every topology site maps to a DISTINCT
  phalanx body;
- Physical hand-hand pairs on phalanges evaluated with the G2 kernels
  (``sphere_capsule`` and ``capsule_capsule``):
  - ``overlap_little_finger_lead_index``,
  - ``interlock_little_finger_lead_middle`` and ``interlock_lead_index_wrap``,
  - ``ten_finger_little_finger_grip`` (hand-grip),
  - ``trail_palm_lead_thumb`` (sphere-capsule);
- Honest ``GripCapabilities`` declaration and equal-and-opposite wrench
  application on the interacting phalanx and handle bodies without duplicating
  native and shared contact on the same pair;
- Frozen-finger mode reproducing the rigid-proxy law results within the M3
  qualification tolerances (<= 5% peak, <= 2% RMS) vs Pinocchio and
  ``law_level_series``;
- Physics invariants on the fixture: true zero force at separation, unilateral
  normal force (never pulling), friction cone compliance, and action-reaction
  residual <= 1e-9 N;
- Explicit labelling everywhere as "a contact fixture, not a qualified golfer".
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
ADAPTER_PY = ROOT / "src/engines/physics_engines/mujoco/python/grip_lab_adapter.py"
BINDING_PY = ROOT / "src/shared/python/grip_contact/binding.py"
BINDING_MANIFEST_PY = ROOT / "src/shared/python/grip_contact/binding_manifest.py"
TOPOLOGY_CONTACT_PY = ROOT / "src/shared/python/grip_contact/topology_contact.py"

LAB_MODEL_IDS: tuple[str, ...] = (
    "mujoco_shadow_hand_lab",
    "mujoco_allegro_hand_lab",
)
SQUEEZE_N = 1100.0
FIXTURE_PHRASE = "a contact fixture, not a qualified golfer"


def _require_mujoco() -> Any:
    try:
        import mujoco  # noqa: PLC0415
    except ImportError as exc:
        pytest.skip(f"mujoco not installed: {exc}")
    return mujoco


def _require_lab_adapter_and_assets(model_id: str) -> Any:
    _require_mujoco()
    from src.engines.physics_engines.mujoco.python.grip_lab_adapter import (
        MuJoCoGripLabAdapter,
        check_hand_assets_available,
    )

    try:
        check_hand_assets_available(model_id)
    except SchemaFeatureUnavailableError as exc:
        pytest.skip(f"hand assets unavailable for {model_id}: {exc}")
    return MuJoCoGripLabAdapter


@pytest.fixture(scope="module")
def spec_dict() -> dict[str, Any]:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def interface(spec_dict: dict[str, Any]) -> GripInterface:
    return GripInterface.from_spec(spec_dict)


@pytest.mark.parametrize("model_id", LAB_MODEL_IDS)
@pytest.mark.parametrize(
    "style", [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER]
)
@pytest.mark.parametrize("trigger_finger", [False, True])
def test_segment_mapped_manifests_map_every_site_to_distinct_phalanx_body(
    model_id: str,
    style: GripStyle,
    trigger_finger: bool,
    interface: GripInterface,
) -> None:
    """Every topology site maps to a DISTINCT phalanx body in the grip-binding/v1 manifest."""
    mujoco = _require_mujoco()
    adapter_cls = _require_lab_adapter_and_assets(model_id)
    from src.engines.physics_engines.mujoco.python.grip_lab_adapter import (
        LAB_MANIFEST_PATHS,
    )

    manifest_path = LAB_MANIFEST_PATHS[model_id]
    assert manifest_path.exists(), f"Missing manifest file: {manifest_path}"

    raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(raw_manifest)
    assert raw_manifest["schema"] == "grip-binding/v1"
    assert raw_manifest["model_id"] == model_id
    assert raw_manifest["attachment_mode"] == "separated_v2"
    assert FIXTURE_PHRASE in json.dumps(raw_manifest["provenance"]).lower()

    grip_spec = GripSpec(style=style, trigger_finger=trigger_finger)
    topo = contact_topology(grip_spec)
    adapter = adapter_cls.from_model_id(
        model_id, grip_spec=grip_spec, interface=interface
    )
    mj_model = adapter.model
    assert isinstance(adapter, ModelGripAdapter)

    binding = adapter.resolve_binding(topo)
    assert isinstance(binding, HandClubBinding)
    validate_resolved_binding(binding, topo, interface)

    # Every topology site must be bound with representation='phalange'
    topo_site_names = set(topo.site_names())
    bound_site_names = {sb.site_id for sb in binding.sites}
    assert bound_site_names == topo_site_names
    assert len(binding.unavailable_sites) == 0

    bound_bodies = [sb.body_id for sb in binding.sites]
    assert len(set(bound_bodies)) == len(bound_bodies), (
        f"Expected distinct phalanx bodies for every site in {model_id}, "
        f"got duplicates in {bound_bodies}"
    )
    for sb in binding.sites:
        assert sb.representation == "phalange"
        bid = int(mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, sb.body_id))
        assert bid >= 0, f"Phalanx body {sb.body_id!r} not found in MjModel"
        site = topo.site(sb.site_id)
        expected_prefix = "lh_" if site.hand == "lead" else "rh_"
        assert sb.body_id.startswith(expected_prefix)


def test_missing_hand_assets_fails_closed_with_schema_feature_unavailable() -> None:
    """When Shadow/Allegro hand assets are absent, fail closed with SchemaFeatureUnavailableError."""
    _require_mujoco()
    from src.engines.physics_engines.mujoco.python.grip_lab_adapter import (
        MuJoCoGripLabAdapter,
        check_hand_assets_available,
    )

    nonexistent_dir = ROOT / "src/engines/physics_engines/mujoco/python/no_such_assets"
    with pytest.raises(SchemaFeatureUnavailableError, match="hand assets"):
        check_hand_assets_available(
            "mujoco_shadow_hand_lab", assets_root=nonexistent_dir
        )

    with pytest.raises(SchemaFeatureUnavailableError, match="hand assets"):
        MuJoCoGripLabAdapter.from_model_id(
            "mujoco_shadow_hand_lab", assets_root=nonexistent_dir
        )


@pytest.mark.parametrize("model_id", LAB_MODEL_IDS)
def test_manifest_model_hash_is_reproducible_from_the_hand_assets(
    model_id: str,
) -> None:
    """The committed ``model_hash`` is recomputed from the XML files the lab builds from.

    A provenance hash nobody can reproduce is not provenance: the manifest
    value must equal :func:`lab_model_hash` over the exact hand-asset XML
    files that :func:`build_grip_lab_mjcf` consumes, and the provenance
    source must be an existing path.
    """
    _require_lab_adapter_and_assets(model_id)
    from src.engines.physics_engines.mujoco.python.grip_lab_adapter import (
        lab_model_hash,
    )

    manifest = json.loads((MANIFESTS_DIR / f"{model_id}.json").read_text("utf-8"))
    assert manifest["model_hash"] == lab_model_hash(model_id)
    assert (ROOT / manifest["provenance"]["source"]).exists()


def test_lab_model_hash_changes_with_any_consumed_xml(tmp_path: Path) -> None:
    """Editing any consumed XML file changes the hash; unknown ids are rejected."""
    _require_lab_adapter_and_assets("mujoco_allegro_hand_lab")
    import shutil  # noqa: PLC0415

    from src.engines.physics_engines.mujoco.python.grip_lab_adapter import (
        _DEFAULT_ASSETS_ROOT,
        lab_model_hash,
    )

    shutil.copytree(_DEFAULT_ASSETS_ROOT / "wonik_allegro", tmp_path / "wonik_allegro")
    base = lab_model_hash("mujoco_allegro_hand_lab", assets_root=tmp_path)
    right = tmp_path / "wonik_allegro" / "right_hand.xml"
    right.write_text(right.read_text("utf-8") + "\n<!-- edit -->\n", "utf-8")
    assert lab_model_hash("mujoco_allegro_hand_lab", assets_root=tmp_path) != base
    with pytest.raises(ValueError):
        lab_model_hash("not_a_lab_model")


@pytest.mark.live_simulation
@pytest.mark.parametrize("model_id", LAB_MODEL_IDS)
@pytest.mark.parametrize(
    "style,expected_hh_pairs",
    [
        (
            GripStyle.OVERLAP,
            ("trail_palm_lead_thumb", "overlap_little_finger_lead_index"),
        ),
        (
            GripStyle.INTERLOCK,
            (
                "trail_palm_lead_thumb",
                "interlock_little_finger_lead_middle",
                "interlock_lead_index_wrap",
            ),
        ),
        (
            GripStyle.TEN_FINGER,
            ("trail_palm_lead_thumb",),
        ),
    ],
)
def test_physical_hand_hand_pairs_on_phalanges_and_capabilities(
    model_id: str,
    style: GripStyle,
    expected_hh_pairs: tuple[str, ...],
    interface: GripInterface,
) -> None:
    """Physical hand-hand pairs on phalanges evaluate via G2 kernels and apply equal-and-opposite wrenches."""
    mujoco = _require_mujoco()
    adapter_cls = _require_lab_adapter_and_assets(model_id)

    grip_spec = GripSpec(style=style, trigger_finger=True)
    topo = contact_topology(grip_spec)
    topo_model = build_topology_model(interface, topo, squeeze_per_hand_n=SQUEEZE_N)

    adapter = adapter_cls.from_model_id(
        model_id, grip_spec=grip_spec, interface=interface
    )
    caps = adapter.capabilities()
    assert caps.supports_physical_pairs is True
    assert caps.supports_wrench_application is True
    assert caps.native_contact is False
    assert "separated_v2" in caps.attachment_modes

    binding = adapter.resolve_binding(topo)
    assert binding.attachment_mode == "separated_v2"
    validate_binding(topo, binding, caps)

    report = adapter.evaluate_contacts(topo_model, binding)
    assert report.applied_allocated_force_n() == 0.0
    assert report.allocated_contacts is None
    assert report.allocated_labels == {}

    # Every pair in the topology (both hand-grip and hand-hand) is PHYSICAL in separated_v2
    for pair in topo.pairs:
        assert report.pair_labels[pair.name] == "physical"
        assert pair.name in report.pair_wrenches
        assert np.isfinite(report.normal_force_n[pair.name])
        assert np.isfinite(report.tangential_force_n[pair.name])
        assert np.isfinite(report.spin_moment_nm[pair.name])
        assert report.normal_force_n[pair.name] > 0.0

    hh_names = tuple(p.name for p in topo.hand_hand_pairs())
    assert hh_names == expected_hh_pairs
    if style is GripStyle.TEN_FINGER:
        assert "ten_finger_little_finger_grip" in report.pair_wrenches
        assert report.normal_force_n["ten_finger_little_finger_grip"] > 0.0

    # Verify equal-and-opposite forces on the two phalanx bodies for each hand-hand pair
    site_map = binding.site_binding_map()
    for hh_name in expected_hh_pairs:
        hh_pair = topo.pair(hh_name)
        pw = report.pair_wrenches[hh_name]
        body_a = site_map[hh_pair.site_a].body_id
        body_b = site_map[hh_pair.site_b].body_id
        assert body_a != body_b
        np.testing.assert_allclose(pw.force_a_n + pw.force_b_n, np.zeros(3), atol=1e-12)
        assert float(np.linalg.norm(pw.force_a_n)) > 0.0

    # Apply wrenches to MuJoCo data.xfrc_applied and verify action-reaction <= 1e-9 N
    apply_report(adapter, report)
    assert report.action_reaction_residual_n() <= 1e-9
    net_xfrc_force = np.sum(adapter.data.xfrc_applied[:, :3], axis=0)
    assert float(np.linalg.norm(net_xfrc_force)) <= 1e-9

    # Articulating a finger joint in MjData changes the phalanx pose and hand-hand contact force
    f_before = report.normal_force_n["trail_palm_lead_thumb"]
    adapter.articulate_site_phalanx(
        "lead_thumb", delta_pos_m=np.array([0.0, 0.015, 0.015])
    )
    report_after = adapter.evaluate_contacts(topo_model, binding)
    f_after = report_after.normal_force_n["trail_palm_lead_thumb"]
    assert f_after != f_before
    mujoco.mj_step(adapter.model, adapter.data)


@pytest.mark.parametrize("model_id", LAB_MODEL_IDS)
def test_native_vs_shared_contact_never_duplicated(
    model_id: str,
    interface: GripInterface,
) -> None:
    """Native contact mode disables wrench application and refuses shared-law wrenches."""
    adapter_cls = _require_lab_adapter_and_assets(model_id)

    grip_spec = GripSpec(style=GripStyle.OVERLAP)
    topo = contact_topology(grip_spec)
    topo_model = build_topology_model(interface, topo, squeeze_per_hand_n=SQUEEZE_N)

    shared_adapter = adapter_cls.from_model_id(
        model_id, grip_spec=grip_spec, interface=interface, contact_mode="shared"
    )
    rep = shared_adapter.evaluate_contacts(topo_model)

    native_adapter = adapter_cls.from_model_id(
        model_id, grip_spec=grip_spec, interface=interface, contact_mode="native"
    )
    native_caps = native_adapter.capabilities()
    assert native_caps.native_contact is True
    assert native_caps.supports_wrench_application is False

    with pytest.raises(SchemaFeatureUnavailableError, match="native_contact"):
        apply_report(native_adapter, rep)

    with pytest.raises(SchemaFeatureUnavailableError, match="native_contact"):
        native_adapter.apply_wrenches(rep.body_wrenches())


@pytest.mark.live_simulation
@pytest.mark.parametrize("model_id", LAB_MODEL_IDS)
def test_fixture_physics_separation_unilateral_friction_cone_and_action_reaction(
    model_id: str,
    interface: GripInterface,
) -> None:
    """Physics tests on the fixture:
    1. No force at separation (a true zero, 0.0);
    2. Unilateral normal (never pulling, f_n >= 0 even when receding);
    3. Friction cone respected (|f_t| <= mu * f_n + viscous regularisation allowance);
    4. Action-reaction residual <= 1e-9 N.
    """
    adapter_cls = _require_lab_adapter_and_assets(model_id)

    for style in (GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER):
        grip_spec = GripSpec(style=style, trigger_finger=True)
        topo = contact_topology(grip_spec)
        topo_model = build_topology_model(interface, topo, squeeze_per_hand_n=SQUEEZE_N)
        law = topo_model.law
        mu_s = float(law.static_friction)
        mu_v = float(law.viscous_friction)

        adapter = adapter_cls.from_model_id(
            model_id, grip_spec=grip_spec, interface=interface
        )
        binding = adapter.resolve_binding(topo)

        # 1. Separation test: displace both hands far from club and from each other
        club_state = RigidBodyState(
            rotation=np.eye(3),
            position_m=np.zeros(3),
            velocity_m_s=np.zeros(3),
            omega_rad_s=np.zeros(3),
        )
        hands_sep = {
            "L": BushingState(
                rotation=np.eye(3),
                position_m=np.array([0.0, 0.50, 0.50]),
                velocity_m_s=np.array([0.1, -0.2, 0.05]),
                omega_rad_s=np.zeros(3),
            ),
            "R": BushingState(
                rotation=np.eye(3),
                position_m=np.array([0.0, -0.50, -0.50]),
                velocity_m_s=np.array([-0.1, 0.2, -0.05]),
                omega_rad_s=np.zeros(3),
            ),
        }
        adapter.set_prescribed_grip_states(hand_states=hands_sep, club_state=club_state)
        rep_sep = adapter.evaluate_contacts(topo_model, binding)
        for p_name in topo.pair_names():
            assert rep_sep.normal_force_n[p_name] == 0.0
            assert rep_sep.tangential_force_n[p_name] == 0.0
            assert rep_sep.spin_moment_nm[p_name] == 0.0
            pw_sep = rep_sep.pair_wrenches[p_name]
            assert float(np.linalg.norm(pw_sep.force_a_n)) == 0.0
            assert float(np.linalg.norm(pw_sep.force_b_n)) == 0.0

        # 2. Unilateral normal & 3. Friction cone & 4. Action-reaction under sliding + receding motion
        hands_nom = hand_frame_states(club_state, interface)
        # Test both sliding under compression and rapid receding velocity (attempting to pull)
        for v_slide, v_recede in (
            (np.array([0.08, -0.04, 0.0]), 0.0),
            (np.array([0.03, 0.02, 0.0]), 25.0),
        ):
            adapter.clear_prescribed_grip_states()
            hands_dyn = {
                "L": replace(
                    hands_nom["L"],
                    velocity_m_s=hands_nom["L"].velocity_m_s + v_slide,
                ),
                "R": replace(
                    hands_nom["R"],
                    velocity_m_s=hands_nom["R"].velocity_m_s - v_slide,
                ),
            }
            adapter.set_prescribed_grip_states(
                hand_states=hands_dyn, club_state=club_state
            )
            if v_recede > 0.0:
                adapter.articulate_site_phalanx(
                    "trail_little_finger",
                    delta_vel_m_s=np.array([0.0, v_recede, v_recede]),
                )
            rep_dyn = adapter.evaluate_contacts(topo_model, binding)
            apply_report(adapter, rep_dyn)

            # Action-reaction <= 1e-9 N
            assert rep_dyn.action_reaction_residual_n() <= 1e-9
            net_xfrc = np.sum(adapter.data.xfrc_applied[:, :3], axis=0)
            assert float(np.linalg.norm(net_xfrc)) <= 1e-9

            v_mag = float(np.linalg.norm(2.0 * v_slide)) + v_recede
            for p_name, pw in rep_dyn.pair_wrenches.items():
                fn = float(pw.normal_force_mag_n)
                ft = float(pw.tangential_force_mag_n)
                # Unilateral normal: never negative (never pulling)
                assert fn >= 0.0, (
                    f"Pair {p_name} pulled with negative normal force {fn}"
                )
                # Friction cone with documented viscous regularisation allowance
                cone_bound = (mu_s + mu_v * v_mag) * fn + 1e-12
                assert ft <= cone_bound, (
                    f"Pair {p_name} violated friction cone: |f_t|={ft} > {cone_bound}"
                )
                # Pair-level action-reaction
                assert float(np.linalg.norm(pw.force_a_n + pw.force_b_n)) <= 1e-12


@pytest.mark.live_simulation
@pytest.mark.parametrize("model_id", LAB_MODEL_IDS)
@pytest.mark.parametrize(
    "style", [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER]
)
def test_frozen_finger_mode_reproduces_rigid_proxy_law_parity_vs_pinocchio(
    model_id: str,
    style: GripStyle,
    interface: GripInterface,
) -> None:
    """Frozen-finger mode (finger joints locked) reproduces rigid-proxy law results
    within <= 5% peak and <= 2% RMS vs PinocchioGripAdapter and law_level_series.
    """
    adapter_cls = _require_lab_adapter_and_assets(model_id)
    from src.engines.physics_engines.pinocchio.python.grip_binding_adapter import (
        PinocchioGripAdapter,
    )

    grip_spec = GripSpec(style=style, trigger_finger=True)
    topo = contact_topology(grip_spec)
    topo_model = build_topology_model(interface, topo, squeeze_per_hand_n=SQUEEZE_N)
    m_hash = topology_mapping_hash(topo_model)

    adapter = adapter_cls.from_model_id(
        model_id,
        grip_spec=grip_spec,
        interface=interface,
        frozen_fingers=True,
    )
    assert adapter.frozen_fingers is True

    # Attempt to articulate a finger in frozen_fingers mode must be ignored or locked
    binding = adapter.resolve_binding(topo, attachment_mode="contact")
    closure = adapter.synthesise_closure_from_net_wrench()

    pin_adapter = PinocchioGripAdapter(interface, model_hash=m_hash)
    pin_binding = replace(
        pin_adapter.resolve_binding(topo),
        attachment_mode="contact",
        club_grip_frames=binding.club_grip_frames,
    )

    times = np.arange(21, dtype=float) * SAMPLE_DT_S
    lab_reports: list[GripStepReport] = []
    pin_reports: list[GripStepReport] = []
    hand_seq: dict[str, list[BushingState]] = {"L": [], "R": []}
    club_seq: dict[str, list[BushingState]] = {"L": [], "R": []}

    for t_s in times:
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
        lab_rep = adapter.evaluate_contacts(topo_model, binding, closure=closure)
        assert lab_rep.applied_allocated_force_n() == 0.0
        apply_report(adapter, lab_rep)
        lab_reports.append(lab_rep)

        pin_adapter.set_prescribed_states(
            hand_states=hands_perturbed, club_state=club_state
        )
        pin_states = pin_adapter.sample_states(pin_binding)
        pin_rep = evaluate_contacts(
            topo_model, pin_binding, pin_states, closure=closure
        )
        pin_reports.append(pin_rep)

    assert total_applied_allocated_force_n(lab_reports) == 0.0

    lab_series = TopologyKineticsSeries.from_step_reports(
        times,
        lab_reports,
        engine=f"{model_id}_frozen",
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
        mode="contact",
    )

    parity_vs_pin = compare_topology_series(pin_series, lab_series)
    assert parity_vs_pin.status == "passed", (
        f"Frozen-finger parity vs Pinocchio failed for {model_id} ({style.value}): "
        f"{parity_vs_pin.to_dict()}"
    )
    assert parity_vs_pin.passed is True
    for ch_name, ch_res in parity_vs_pin.channels.items():
        assert ch_res.passed is True, f"Channel {ch_name} failed: {ch_res}"
        assert ch_res.error is not None
        assert ch_res.error.peak_error <= PEAK_TOLERANCE
        assert ch_res.error.rms_error <= RMS_TOLERANCE

    parity_vs_law = compare_topology_series(law_series, lab_series)
    assert parity_vs_law.status == "passed"
    assert parity_vs_law.passed is True


def test_fixture_labelling_everywhere() -> None:
    """Fixture must be explicitly labelled 'a contact fixture, not a qualified golfer'."""
    assert ADAPTER_PY.exists(), f"Missing adapter file: {ADAPTER_PY}"
    source = ADAPTER_PY.read_text(encoding="utf-8").lower()
    assert FIXTURE_PHRASE in source

    from src.engines.physics_engines.mujoco.python.grip_lab_adapter import (
        FIXTURE_DISCLAIMER,
        LAB_MANIFEST_PATHS,
    )

    assert FIXTURE_PHRASE in FIXTURE_DISCLAIMER.lower()
    for model_id, mf_path in LAB_MANIFEST_PATHS.items():
        assert mf_path.exists(), f"Missing manifest for {model_id}: {mf_path}"
        mf_text = mf_path.read_text(encoding="utf-8").lower()
        assert FIXTURE_PHRASE in mf_text

    tex_path = ROOT / "docs/research/human_realistic_golfer/human_realistic_golfer.tex"
    assert tex_path.exists(), f"Missing research reference: {tex_path}"
    assert FIXTURE_PHRASE in tex_path.read_text(encoding="utf-8").lower()


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


def test_adapter_and_shared_ast_budgets_and_lod() -> None:
    """AST check: file <= 1200 lines, functions <= 100 lines, <= 8 params, LoD <= 2 dots."""
    for path in (ADAPTER_PY, BINDING_PY, BINDING_MANIFEST_PY, TOPOLOGY_CONTACT_PY):
        assert path.exists(), f"Missing module: {path}"
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
