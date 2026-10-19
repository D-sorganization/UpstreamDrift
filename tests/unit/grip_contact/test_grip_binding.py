"""Unit tests for the shared model grip binding contract (HRG-GS-M1, #13323).

Acceptance criteria:
- Table-driven test: 100 % of missing-body, missing-site, and unsupported-mode
  cases raise SchemaFeatureUnavailableError before evaluate/step (>= 6 rows).
- Binding rotations are proper within 1e-9 (and an improper rotation raises).
- Action-reaction residual (sum of body_wrenches forces) is <= 1e-12 N.
- Virtual work on a synthetic two-body fixture: for a random small rigid twist
  applied to the club, power from body_wrenches equals sum over pairs of
  f . v_contact to <= 1e-12.
- Spy adapter records apply_wrenches calls; allocated pairs never appear, and
  unavailable channels are NaN, never 0.
- PinocchioGripAdapter reproduces topology_side_wrench to <= 1e-12 N for each style.
- AST test: no attribute chain deeper than two dots in binding.py and
  grip_binding_adapter.py, and all functions <= 100 lines and <= 8 parameters.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.engines.physics_engines.pinocchio.python.grip_binding_adapter import (
    PinocchioGripAdapter,
)
from src.engines.protocols import GripProvider
from src.shared.python.grip_contact import BushingState, GripInterface
from src.shared.python.grip_contact.binding import (
    BodyStates,
    GripCapabilities,
    GripStepReport,
    HandClubBinding,
    ModelGripAdapter,
    SiteBinding,
    apply_report,
    bind_rigid_hand_sites,
    evaluate_contacts,
    validate_binding,
)
from src.shared.python.grip_contact.contact_topology import (
    ContactTopology,
    GripStyle,
    contact_topology,
)
from src.shared.python.grip_contact.prescribed_motion import RigidBodyState
from src.shared.python.grip_contact.topology_contact import (
    TopologyContactModel,
    build_topology_model,
    topology_side_wrench,
)
from src.shared.python.motion_matching.full_body_schema import (
    SchemaFeatureUnavailableError,
)
from src.shared.python.motion_matching.grip_contract import GripSpec

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
BINDING_PY = ROOT / "src/shared/python/grip_contact/binding.py"
ADAPTER_PY = (
    ROOT / "src/engines/physics_engines/pinocchio/python/grip_binding_adapter.py"
)
PROTOCOLS_PY = ROOT / "src/engines/protocols.py"


@pytest.fixture(scope="module")
def spec_dict() -> dict[str, Any]:
    return json.loads(SPEC_PATH.read_text())


@pytest.fixture(scope="module")
def interface(spec_dict: dict[str, Any]) -> GripInterface:
    return GripInterface.from_spec(spec_dict)


@pytest.fixture(scope="module")
def default_topology() -> ContactTopology:
    return contact_topology(GripSpec(style=GripStyle.OVERLAP))


@pytest.fixture(scope="module")
def default_topology_model(
    interface: GripInterface, default_topology: ContactTopology
) -> TopologyContactModel:
    return build_topology_model(interface, default_topology, squeeze_per_hand_n=1100.0)


@pytest.fixture(scope="module")
def default_capabilities() -> GripCapabilities:
    return GripCapabilities(
        attachment_modes=("contact", "contact_v1"),
        supports_physical_pairs=True,
        supports_wrench_application=True,
        native_contact=False,
        available_bodies=("club", "hand_l", "hand_r"),
    )


@pytest.fixture(scope="module")
def default_binding(default_topology: ContactTopology) -> HandClubBinding:
    sites = bind_rigid_hand_sites(
        default_topology,
        lead_hand_body="hand_l",
        trail_hand_body="hand_r",
    )
    return HandClubBinding(
        model_id="test_golfer_v1",
        model_hash="sha256:test1234",
        handedness="right",
        attachment_mode="contact",
        club_body="club",
        lead_hand_body="hand_l",
        trail_hand_body="hand_r",
        sites=sites,
        unavailable_sites={},
    )


@pytest.fixture(scope="module")
def neutral_states() -> dict[str, np.ndarray]:
    return {
        "club": np.eye(4),
        "hand_l": np.eye(4),
        "hand_r": np.eye(4),
    }


def _mutate_missing_club_body(
    topo: ContactTopology,
    binding: HandClubBinding,
    caps: GripCapabilities,
    states: dict[str, np.ndarray],
) -> tuple[ContactTopology, HandClubBinding, GripCapabilities, dict[str, np.ndarray]]:
    new_caps = replace(caps, available_bodies=("hand_l", "hand_r"))
    new_states = {k: v for k, v in states.items() if k != "club"}
    return topo, binding, new_caps, new_states


def _mutate_missing_lead_hand_body(
    topo: ContactTopology,
    binding: HandClubBinding,
    caps: GripCapabilities,
    states: dict[str, np.ndarray],
) -> tuple[ContactTopology, HandClubBinding, GripCapabilities, dict[str, np.ndarray]]:
    new_caps = replace(caps, available_bodies=("club", "hand_r"))
    new_states = {k: v for k, v in states.items() if k != "hand_l"}
    return topo, binding, new_caps, new_states


def _mutate_missing_trail_hand_body(
    topo: ContactTopology,
    binding: HandClubBinding,
    caps: GripCapabilities,
    states: dict[str, np.ndarray],
) -> tuple[ContactTopology, HandClubBinding, GripCapabilities, dict[str, np.ndarray]]:
    new_caps = replace(caps, available_bodies=("club", "hand_l"))
    new_states = {k: v for k, v in states.items() if k != "hand_r"}
    return topo, binding, new_caps, new_states


def _mutate_missing_phalange_body(
    topo: ContactTopology,
    binding: HandClubBinding,
    caps: GripCapabilities,
    states: dict[str, np.ndarray],
) -> tuple[ContactTopology, HandClubBinding, GripCapabilities, dict[str, np.ndarray]]:
    mutated_sites = (
        SiteBinding(
            site_id=binding.sites[0].site_id,
            body_id="missing_distal_phalanx",
            body_to_site=binding.sites[0].body_to_site,
            representation="phalange",
        ),
        *binding.sites[1:],
    )
    new_binding = replace(binding, sites=mutated_sites)
    return topo, new_binding, caps, states


def _mutate_missing_lead_physical_site(
    topo: ContactTopology,
    binding: HandClubBinding,
    caps: GripCapabilities,
    states: dict[str, np.ndarray],
) -> tuple[ContactTopology, HandClubBinding, GripCapabilities, dict[str, np.ndarray]]:
    filtered = tuple(s for s in binding.sites if s.site_id != "lead_palm_proximal")
    new_binding = replace(binding, sites=filtered)
    return topo, new_binding, caps, states


def _mutate_missing_trail_physical_site(
    topo: ContactTopology,
    binding: HandClubBinding,
    caps: GripCapabilities,
    states: dict[str, np.ndarray],
) -> tuple[ContactTopology, HandClubBinding, GripCapabilities, dict[str, np.ndarray]]:
    filtered = tuple(s for s in binding.sites if s.site_id != "trail_finger_distal")
    new_binding = replace(binding, sites=filtered)
    return topo, new_binding, caps, states


def _mutate_unsupported_mode_in_capabilities(
    topo: ContactTopology,
    binding: HandClubBinding,
    caps: GripCapabilities,
    states: dict[str, np.ndarray],
) -> tuple[ContactTopology, HandClubBinding, GripCapabilities, dict[str, np.ndarray]]:
    new_binding = replace(binding, attachment_mode="separated_v2")
    return topo, new_binding, caps, states


def _mutate_unknown_attachment_mode(
    topo: ContactTopology,
    binding: HandClubBinding,
    caps: GripCapabilities,
    states: dict[str, np.ndarray],
) -> tuple[ContactTopology, HandClubBinding, GripCapabilities, dict[str, np.ndarray]]:
    new_binding = replace(binding, attachment_mode="nonexistent_attachment_mode")
    new_caps = replace(
        caps, attachment_modes=("contact", "nonexistent_attachment_mode")
    )
    return topo, new_binding, new_caps, states


def _mutate_physical_mode_without_physical_support(
    topo: ContactTopology,
    binding: HandClubBinding,
    caps: GripCapabilities,
    states: dict[str, np.ndarray],
) -> tuple[ContactTopology, HandClubBinding, GripCapabilities, dict[str, np.ndarray]]:
    new_caps = replace(caps, supports_physical_pairs=False)
    return topo, binding, new_caps, states


VALIDATION_TABLE = [
    ("missing_club_body", _mutate_missing_club_body, "club"),
    ("missing_lead_hand_body", _mutate_missing_lead_hand_body, "hand_l"),
    ("missing_trail_hand_body", _mutate_missing_trail_hand_body, "hand_r"),
    ("missing_phalange_body", _mutate_missing_phalange_body, "missing_distal_phalanx"),
    (
        "missing_lead_physical_site",
        _mutate_missing_lead_physical_site,
        "lead_palm_proximal",
    ),
    (
        "missing_trail_physical_site",
        _mutate_missing_trail_physical_site,
        "trail_finger_distal",
    ),
    (
        "unsupported_mode_in_capabilities",
        _mutate_unsupported_mode_in_capabilities,
        "separated_v2",
    ),
    (
        "unknown_attachment_mode",
        _mutate_unknown_attachment_mode,
        "nonexistent_attachment_mode",
    ),
    (
        "physical_mode_without_physical_support",
        _mutate_physical_mode_without_physical_support,
        "physical",
    ),
]


@pytest.mark.parametrize(
    ("case_id", "mutate_fn", "expected_match"),
    VALIDATION_TABLE,
    ids=[row[0] for row in VALIDATION_TABLE],
)
def test_validate_binding_table_driven(
    case_id: str,
    mutate_fn: Any,
    expected_match: str,
    default_topology_model: TopologyContactModel,
    default_binding: HandClubBinding,
    default_capabilities: GripCapabilities,
    neutral_states: dict[str, np.ndarray],
) -> None:
    """Every missing-body, missing-site, and unsupported-mode case raises before step."""
    assert len(VALIDATION_TABLE) >= 6
    topo, binding, caps, states = mutate_fn(
        default_topology_model.topology,
        default_binding,
        default_capabilities,
        neutral_states,
    )
    with pytest.raises(SchemaFeatureUnavailableError, match=expected_match):
        validate_binding(topo, binding, caps)

    if case_id not in (
        "unsupported_mode_in_capabilities",
        "physical_mode_without_physical_support",
    ):
        with pytest.raises(SchemaFeatureUnavailableError, match=expected_match):
            evaluate_contacts(default_topology_model, binding, states)


def test_site_and_hand_club_binding_dbc_invariants(
    default_topology: ContactTopology,
    default_binding: HandClubBinding,
) -> None:
    """DbC checks on proper rotations (within 1e-9), bottom row, IDs, uniqueness, and overlap."""
    valid_pose = np.eye(4)
    sb = SiteBinding("lead_palm_proximal", "hand_l", valid_pose, "rigid_proxy")
    np.testing.assert_allclose(sb.rotation, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(sb.translation_m, np.zeros(3), atol=1e-12)

    # Improper rotation (reflection det == -1) must raise ValueError
    reflection = np.diag([1.0, 1.0, -1.0, 1.0])
    with pytest.raises(ValueError, match="proper rotation"):
        SiteBinding("lead_palm_proximal", "hand_l", reflection, "rigid_proxy")

    # Non-orthonormal matrix beyond 1e-9 must raise ValueError
    scaled = np.eye(4)
    scaled[0, 0] = 1.0 + 1e-6
    with pytest.raises(ValueError, match="proper rotation"):
        SiteBinding("lead_palm_proximal", "hand_l", scaled, "rigid_proxy")

    # Orthonormal within 5e-10 succeeds
    nearly_ortho = np.eye(4)
    nearly_ortho[0, 0] = 1.0 + 2e-10
    SiteBinding("lead_palm_proximal", "hand_l", nearly_ortho, "rigid_proxy")

    # Bad bottom row raises ValueError
    bad_bottom = np.eye(4)
    bad_bottom[3, 2] = 0.01
    with pytest.raises(ValueError, match="bottom row"):
        SiteBinding("lead_palm_proximal", "hand_l", bad_bottom, "rigid_proxy")

    # Empty IDs raise ValueError
    with pytest.raises(ValueError, match="site_id"):
        SiteBinding("   ", "hand_l", valid_pose, "rigid_proxy")
    with pytest.raises(ValueError, match="body_id"):
        SiteBinding("lead_palm_proximal", "", valid_pose, "rigid_proxy")

    # Unknown representation raises ValueError
    with pytest.raises(ValueError, match="representation"):
        SiteBinding(
            "lead_palm_proximal",
            "hand_l",
            valid_pose,
            "invalid_rep",  # type: ignore[arg-type]
        )

    # Duplicate site_id raises ValueError
    with pytest.raises(ValueError, match="unique|duplicate"):
        replace(default_binding, sites=(sb, sb))

    # Site both bound and unavailable raises ValueError
    with pytest.raises(ValueError, match="both bound and unavailable"):
        replace(
            default_binding,
            unavailable_sites={sb.site_id: "marked unavailable while bound"},
        )

    # Unknown handedness raises ValueError
    with pytest.raises(ValueError, match="handedness"):
        replace(default_binding, handedness="ambidextrous")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "style", [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER]
)
@pytest.mark.parametrize("trigger_finger", [False, True])
def test_action_reaction_residual(
    interface: GripInterface,
    style: GripStyle,
    trigger_finger: bool,
) -> None:
    """Action-reaction residual (sum of body_wrenches forces and moments) is <= 1e-12 N."""
    topo = contact_topology(GripSpec(style=style, trigger_finger=trigger_finger))
    model = build_topology_model(interface, topo, squeeze_per_hand_n=1100.0)
    sites = bind_rigid_hand_sites(
        topo, lead_hand_body="hand_l", trail_hand_body="hand_r"
    )
    binding = HandClubBinding(
        model_id="test_golfer",
        model_hash="sha256:ar",
        handedness="right",
        attachment_mode="contact",
        club_body="club",
        lead_hand_body="hand_l",
        trail_hand_body="hand_r",
        sites=sites,
        unavailable_sites={},
    )
    t_club = np.eye(4)
    t_club[:3, 3] = np.array([0.0001, -0.0001, 0.00005])
    states = {
        "club": t_club,
        "hand_l": np.eye(4),
        "hand_r": np.eye(4),
    }

    report = evaluate_contacts(model, binding, states)
    wrenches = report.body_wrenches()
    assert set(wrenches.keys()) == {"club", "hand_l", "hand_r"}

    net_force = sum((f for f, _ in wrenches.values()), start=np.zeros(3))
    assert float(np.linalg.norm(net_force)) <= 1e-12
    assert report.action_reaction_residual_n() <= 1e-12

    # Net moment about world origin must also vanish to <= 1e-12 N*m
    net_moment_world = np.zeros(3)
    for body_id, (f_b, m_b) in wrenches.items():
        p_body = states[body_id][:3, 3]
        net_moment_world += m_b + np.cross(p_body, f_b)
    assert float(np.linalg.norm(net_moment_world)) <= 1e-12


def test_virtual_work_synthetic_two_body_fixture(
    interface: GripInterface,
    default_topology: ContactTopology,
) -> None:
    """Virtual work on a synthetic two-body fixture matches sum(f . v_contact) to <= 1e-12."""
    model = build_topology_model(interface, default_topology, squeeze_per_hand_n=1100.0)
    # Synthetic two-body fixture: one combined 'hands' body and one 'club' body
    sites = bind_rigid_hand_sites(
        default_topology,
        lead_hand_body="hands",
        trail_hand_body="hands",
    )
    binding = HandClubBinding(
        model_id="two_body_synthetic",
        model_hash="sha256:2body",
        handedness="right",
        attachment_mode="contact",
        club_body="club",
        lead_hand_body="hands",
        trail_hand_body="hands",
        sites=sites,
        unavailable_sites={},
    )
    t_club = np.eye(4)
    t_club[:3, 3] = np.array([0.02, -0.01, 0.005])
    t_hands = np.eye(4)
    t_hands[:3, 3] = np.array([0.02, -0.01, 0.005])
    states = {"hands": t_hands, "club": t_club}

    report = evaluate_contacts(model, binding, states)
    body_wrenches = report.body_wrenches()
    assert set(body_wrenches.keys()) == {"hands", "club"}

    f_club, m_club = body_wrenches["club"]
    p_club_origin = t_club[:3, 3]
    assert float(np.linalg.norm(f_club)) > 0.0

    rng = np.random.default_rng(13323)
    for _ in range(20):
        v_club = rng.uniform(-1e-3, 1e-3, size=3)
        omega_club = rng.uniform(-1e-2, 1e-2, size=3)

        power_from_body_wrench = float(f_club @ v_club + m_club @ omega_club)
        power_from_pairs = sum(
            float(
                pw.force_b_n
                @ (v_club + np.cross(omega_club, pw.point_b_m - p_club_origin))
                + pw.moment_b_nm @ omega_club
            )
            for pw in report.pair_wrenches.values()
        )
        assert abs(power_from_body_wrench - power_from_pairs) <= 1e-12


class _SpyGripAdapter:
    """Spy implementing ModelGripAdapter / GripProvider to record apply_wrenches calls."""

    def __init__(
        self,
        binding: HandClubBinding,
        states: Mapping[str, np.ndarray],
        capabilities: GripCapabilities,
    ) -> None:
        self._binding = binding
        self._states = dict(states)
        self._capabilities = capabilities
        self.calls: list[dict[str, tuple[np.ndarray, np.ndarray]]] = []

    def capabilities(self) -> GripCapabilities:
        return self._capabilities

    def resolve_binding(self, topology: ContactTopology) -> HandClubBinding:
        return self._binding

    def sample_states(self, binding: HandClubBinding) -> BodyStates:
        return BodyStates(self._states)

    def apply_wrenches(
        self, wrenches: Mapping[str, tuple[np.ndarray, np.ndarray]]
    ) -> None:
        self.calls.append({k: (f.copy(), m.copy()) for k, (f, m) in wrenches.items()})


def test_spy_adapter_allocated_pairs_excluded_and_unavailable_nan(
    default_topology_model: TopologyContactModel,
    default_binding: HandClubBinding,
    default_capabilities: GripCapabilities,
    neutral_states: dict[str, np.ndarray],
) -> None:
    """Allocated pairs never reach apply_wrenches; unavailable channels are NaN, never 0."""
    spy = _SpyGripAdapter(default_binding, neutral_states, default_capabilities)
    assert isinstance(spy, ModelGripAdapter)
    assert isinstance(spy, GripProvider)

    # 1. Without closure wrench: allocated hand-hand pairs are NaN + reason, never 0
    report_no_closure = evaluate_contacts(
        default_topology_model, default_binding, neutral_states
    )
    hh_pairs = default_topology_model.topology.hand_hand_pairs()
    assert len(hh_pairs) > 0
    for pair in hh_pairs:
        assert report_no_closure.pair_labels[pair.name] == "allocated"
        assert np.isnan(report_no_closure.normal_force_n[pair.name])
        assert np.isnan(report_no_closure.tangential_force_n[pair.name])
        assert np.isnan(report_no_closure.spin_moment_nm[pair.name])
        assert pair.name in report_no_closure.unavailable_reasons
        assert bool(report_no_closure.unavailable_reasons[pair.name].strip())
        assert pair.name not in report_no_closure.pair_wrenches

    apply_report(spy, report_no_closure)
    assert len(spy.calls) == 1
    applied_no_closure = spy.calls[0]

    # 2. With a huge synthetic closure wrench: allocated pairs get finite values,
    #    but body_wrenches() and apply_wrenches() MUST NOT change at all!
    closure_stub = SimpleNamespace(
        available=True,
        reason="",
        wrenches=[
            SimpleNamespace(
                force_n=np.array([500.0, -300.0, 200.0]),
                torque_nm=np.array([15.0, -10.0, 5.0]),
                point_m=np.zeros(3),
            )
        ],
    )
    report_with_closure = evaluate_contacts(
        default_topology_model,
        default_binding,
        neutral_states,
        closure=closure_stub,
    )
    for pair in hh_pairs:
        assert report_with_closure.pair_labels[pair.name] == "allocated"
        assert np.isfinite(report_with_closure.normal_force_n[pair.name])
        assert np.isfinite(report_with_closure.tangential_force_n[pair.name])
        assert pair.name not in report_with_closure.pair_wrenches

    apply_report(spy, report_with_closure)
    assert len(spy.calls) == 2
    applied_with_closure = spy.calls[1]

    for body_id in applied_no_closure:
        f0, m0 = applied_no_closure[body_id]
        f1, m1 = applied_with_closure[body_id]
        np.testing.assert_allclose(f0, f1, atol=1e-15)
        np.testing.assert_allclose(m0, m1, atol=1e-15)

    # 3. Separated physical sites produce true 0.0 (not NaN), while an explicitly
    #    unavailable site produces NaN + reason (never 0.0).
    site_to_drop = "lead_heel_pad"
    partial_sites = tuple(s for s in default_binding.sites if s.site_id != site_to_drop)
    partial_binding = replace(
        default_binding,
        sites=partial_sites,
        unavailable_sites={site_to_drop: "heel pad sensor absent in simplified proxy"},
    )
    far_states = {
        "club": np.eye(4),
        "hand_l": np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.10],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        "hand_r": np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.10],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    }
    report_sep = evaluate_contacts(default_topology_model, partial_binding, far_states)
    assert np.isnan(report_sep.normal_force_n["lead_heel_pad_butt"])
    assert np.isnan(report_sep.tangential_force_n["lead_heel_pad_butt"])
    assert np.isnan(report_sep.spin_moment_nm["lead_heel_pad_butt"])
    assert (
        report_sep.unavailable_reasons["lead_heel_pad_butt"]
        == "heel pad sensor absent in simplified proxy"
    )
    # Bound physical pair at separation is a true 0.0, not NaN
    assert report_sep.normal_force_n["lead_palm_proximal_grip"] == 0.0
    assert report_sep.tangential_force_n["lead_palm_proximal_grip"] == 0.0
    assert report_sep.spin_moment_nm["lead_palm_proximal_grip"] == 0.0
    assert "lead_palm_proximal_grip" not in report_sep.unavailable_reasons


@pytest.mark.parametrize(
    "style", [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER]
)
@pytest.mark.parametrize("trigger_finger", [False, True])
def test_pinocchio_grip_adapter_reproduces_topology_side_wrench(
    interface: GripInterface,
    style: GripStyle,
    trigger_finger: bool,
) -> None:
    """PinocchioGripAdapter + evaluate_contacts reproduces topology_side_wrench to <= 1e-12 N."""
    topo = contact_topology(GripSpec(style=style, trigger_finger=trigger_finger))
    model = build_topology_model(interface, topo, squeeze_per_hand_n=1100.0)

    adapter = PinocchioGripAdapter(interface)
    assert isinstance(adapter, ModelGripAdapter)
    assert isinstance(adapter, GripProvider)

    binding = adapter.resolve_binding(topo)
    validate_binding(topo, binding, adapter.capabilities())

    # Set non-trivial club state and prescribed hand states with non-zero velocity/omega
    weld_club = RigidBodyState(
        rotation=np.eye(3),
        position_m=np.array([0.10, -0.05, 0.20]),
        velocity_m_s=np.array([0.2, -0.1, 0.05]),
        omega_rad_s=np.array([0.4, -0.3, 0.2]),
    )
    club_state = RigidBodyState(
        rotation=np.eye(3),
        position_m=np.array([0.1002, -0.0499, 0.1998]),
        velocity_m_s=np.array([0.18, -0.12, 0.04]),
        omega_rad_s=np.array([0.35, -0.25, 0.15]),
    )
    adapter.set_prescribed_states(weld_club=weld_club, club_state=club_state)
    states = adapter.sample_states(binding)

    report = evaluate_contacts(model, binding, states)

    hands = {s: weld_club.frame(interface.frame(s).matrix()) for s in ("L", "R")}
    club_frames = {s: club_state.frame(interface.frame(s).matrix()) for s in ("L", "R")}
    for side in ("L", "R"):
        expected = topology_side_wrench(model, side, hands[side], club_frames[side])
        actual = report.side_wrenches[side]
        np.testing.assert_allclose(actual.force_n, expected.force_n, atol=1e-12)
        np.testing.assert_allclose(actual.moment_nm, expected.moment_nm, atol=1e-12)
        for pair_name, pw_exp in expected.pair_wrenches.items():
            assert (
                abs(report.normal_force_n[pair_name] - pw_exp.normal_force_mag_n)
                <= 1e-12
            )
            assert (
                abs(
                    report.tangential_force_n[pair_name] - pw_exp.tangential_force_mag_n
                )
                <= 1e-12
            )
            pw_act = report.pair_wrenches[pair_name]
            np.testing.assert_allclose(pw_act.force_b_n, pw_exp.force_b_n, atol=1e-12)
            np.testing.assert_allclose(
                pw_act.moment_b_nm, pw_exp.moment_b_nm, atol=1e-12
            )


@pytest.mark.requires_pinocchio
@pytest.mark.parametrize(
    "style", [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER]
)
def test_pinocchio_grip_adapter_with_club_in_hands(
    spec_dict: dict[str, Any],
    interface: GripInterface,
    style: GripStyle,
) -> None:
    """PinocchioGripAdapter wrapping ClubInHands reproduces topology_side_wrench <= 1e-12 N."""
    try:
        import pinocchio  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"pinocchio not installed: {exc}")

    from src.engines.physics_engines.pinocchio.python.grip_contact_sim import (
        ClubInHands,
        hold_state,
    )

    topo = contact_topology(GripSpec(style=style))
    model = build_topology_model(interface, topo, squeeze_per_hand_n=1100.0)
    names = list(spec_dict["coordinate_order"])
    sim = ClubInHands(spec_dict, names, interface, topology_model=model)
    q0 = np.zeros(len(names))
    hand_source, y0 = hold_state(sim, q0)

    adapter = PinocchioGripAdapter(sim)
    binding = adapter.resolve_binding(topo)
    adapter.update_from_sim(0.0, y0, hand_source)
    states = adapter.sample_states(binding)

    report = evaluate_contacts(model, binding, states)
    _, _, law = sim.wrenches(0.0, y0, hand_source)

    for side in ("L", "R"):
        expected = law[side]
        actual = report.side_wrenches[side]
        np.testing.assert_allclose(actual.force_n, expected.force_n, atol=1e-12)
        np.testing.assert_allclose(actual.moment_nm, expected.moment_nm, atol=1e-12)
        for pair_name, pw_exp in expected.pair_wrenches.items():
            assert (
                abs(report.normal_force_n[pair_name] - pw_exp.normal_force_mag_n)
                <= 1e-12
            )
            assert (
                abs(
                    report.tangential_force_n[pair_name] - pw_exp.tangential_force_mag_n
                )
                <= 1e-12
            )


def test_phalange_representation_contacts(
    default_topology_model: TopologyContactModel,
    default_binding: HandClubBinding,
) -> None:
    """Articulated phalange site bindings on separate bodies preserve action-reaction."""
    s0 = default_binding.sites[0]
    phalange_site = SiteBinding(
        site_id=s0.site_id,
        body_id="lead_proximal_phalanx",
        body_to_site=s0.body_to_site,
        representation="phalange",
    )
    binding = replace(
        default_binding,
        sites=(phalange_site, *default_binding.sites[1:]),
    )
    t_phalange = np.eye(4)
    t_phalange[:3, 3] = np.array([0.0001, -0.0002, 0.0001])
    states = {
        "club": np.eye(4),
        "hand_l": np.eye(4),
        "hand_r": np.eye(4),
        "lead_proximal_phalanx": t_phalange,
    }
    report = evaluate_contacts(default_topology_model, binding, states)
    assert "lead_proximal_phalanx" in report.body_wrenches()
    assert report.action_reaction_residual_n() <= 1e-12


def _attribute_chain_depth(node: ast.AST) -> int:
    """Count chained attribute accesses (including through calls/subscripts)."""
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
    """No attribute chain deeper than 2 dots in binding.py and grip_binding_adapter.py;
    every function is <= 100 lines and has <= 8 parameters.
    """
    for path in (BINDING_PY, ADAPTER_PY):
        source = path.read_text()
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                depth = _attribute_chain_depth(node)
                assert depth <= 2, (
                    f"{path.name}:{node.lineno} has attribute chain of depth {depth} > 2: "
                    f"{ast.unparse(node)}"
                )

    for path in (BINDING_PY, ADAPTER_PY, PROTOCOLS_PY):
        source = path.read_text()
        lines = source.splitlines()
        assert len(lines) <= 1200, f"{path.name} has {len(lines)} lines > 1200"
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
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
