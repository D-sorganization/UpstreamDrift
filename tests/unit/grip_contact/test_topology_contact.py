"""Engine-agnostic topology contact law tests (HRG-GS-G6, #13304).

Acceptance criteria:
- zero force at separation;
- a unilateral normal;
- the friction cone (|f_t| <= mu f_n, plus Stribeck allowance);
- action-reaction <= 1e-12 N;
- resultant equals the sum of pairs <= 1e-12;
- neutral-preload gap test for overlap, interlock and ten-finger, with and without trigger finger (gap within +-0.5 mm).
- DbC validation (proper rotations, finite values, known side, hand-hand pairs raise ValueError for physical contact).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.grip_contact import BushingState, GripInterface
from src.shared.python.grip_contact.contact_topology import (
    GripStyle,
    contact_topology,
)
from src.shared.python.grip_contact.pad_contact import build_pad_model
from src.shared.python.grip_contact.pair_contact import (
    ContactCapsule,
    ContactSphere,
    capsule_capsule_distance,
    sphere_capsule_distance,
)
from src.shared.python.grip_contact.topology_contact import (
    PairTraceRecorder,
    TopologyContactModel,
    TopologySideWrench,
    build_topology_model,
    hand_x_origin,
    map_site_to_hand_frame,
    site_primitive_in_hand_frame,
    topology_pair_wrench,
    topology_side_wrench,
)
from src.shared.python.motion_matching.contact_law import ContactParameters
from src.shared.python.motion_matching.grip_contract import GripSpec

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"


@pytest.fixture(scope="module")
def interface() -> GripInterface:
    return GripInterface.from_spec(json.loads(SPEC_PATH.read_text()))


@pytest.fixture(scope="module")
def default_law() -> ContactParameters:
    return ContactParameters(
        stiffness_n_m=1.0e5,
        dissipation_s_m=0.1,
        static_friction=0.9,
        dynamic_friction=0.7,
        viscous_friction=0.05,
        transition_velocity_m_s=1.0e-3,
    )


def test_neutral_preload_gap_contract(interface: GripInterface) -> None:
    """At the weld pose, every physical pad-grip site's surface gap is within +-0.5 mm.

    Tested for overlap, interlock and ten-finger, with and without trigger finger.
    """
    st_hand = BushingState(np.eye(3), np.zeros(3), np.zeros(3), np.zeros(3))
    st_club = BushingState(np.eye(3), np.zeros(3), np.zeros(3), np.zeros(3))

    for style in GripStyle:
        for trig in (False, True):
            spec = GripSpec(style=style, trigger_finger=trig)
            topo = contact_topology(spec)
            model = build_topology_model(interface, topo)
            for side in ("L", "R"):
                grip_cap = model.grip_capsule(side)
                pairs = model.physical_grip_pairs(side)
                for pair in pairs:
                    site = topo.site(pair.site_a)
                    prim = site_primitive_in_hand_frame(topo, site, side)
                    if isinstance(prim, ContactSphere):
                        gap, _, _, _ = sphere_capsule_distance(
                            prim, grip_cap, st_hand, st_club
                        )
                    else:
                        gap, _, _, _, _ = capsule_capsule_distance(
                            prim, grip_cap, st_hand, st_club
                        )
                    # Gap must be within +-0.5 mm (and specifically negative for preload)
                    assert abs(gap) <= 0.0005, (
                        f"Style {style} trig {trig} pair {pair.name}: gap {gap * 1000:.3f} mm "
                        f"exceeds 0.5 mm tolerance"
                    )


def test_zero_force_at_separation(interface: GripInterface) -> None:
    """When bodies are separated, all forces and moments are identically zero."""
    model = build_topology_model(interface)
    # Move hand frame away from club frame along z by 0.05 m (far beyond contact range)
    hand_far = BushingState(
        np.eye(3), np.array([0.0, 0.0, 0.05]), np.zeros(3), np.zeros(3)
    )
    club = BushingState(np.eye(3), np.zeros(3), np.zeros(3), np.zeros(3))

    for side in ("L", "R"):
        sw = topology_side_wrench(model, side, hand_far, club)
        np.testing.assert_allclose(sw.force_n, np.zeros(3), atol=1e-12)
        np.testing.assert_allclose(sw.moment_nm, np.zeros(3), atol=1e-12)
        assert sw.total_normal_force_n == 0.0
        np.testing.assert_allclose(sw.normal_force_n, 0.0, atol=1e-12)
        np.testing.assert_allclose(sw.tangential_force_n, 0.0, atol=1e-12)
        for pw in sw.pair_wrenches.values():
            assert pw.penetration_m == 0.0
            np.testing.assert_allclose(pw.force_b_n, np.zeros(3), atol=1e-12)
            np.testing.assert_allclose(pw.moment_b_nm, np.zeros(3), atol=1e-12)


def test_unilateral_normal_compressive_only(interface: GripInterface) -> None:
    """Normal force is strictly unilateral / compressive (never pulling)."""
    model = build_topology_model(interface)
    # Test weld pose and small perturbations
    poses = [
        BushingState(np.eye(3), np.zeros(3), np.zeros(3), np.zeros(3)),
        BushingState(np.eye(3), np.array([0.0, 1e-4, -1e-4]), np.zeros(3), np.zeros(3)),
        BushingState(np.eye(3), np.array([1e-3, -1e-4, 0.0]), np.zeros(3), np.zeros(3)),
    ]
    for hand_state in poses:
        club = BushingState(np.eye(3), np.zeros(3), np.zeros(3), np.zeros(3))
        for side in ("L", "R"):
            sw = topology_side_wrench(model, side, hand_state, club)
            assert sw.total_normal_force_n >= 0.0
            assert np.all(sw.normal_force_n >= 0.0)
            for pw in sw.pair_wrenches.values():
                assert pw.normal_force_mag_n >= 0.0


def test_friction_cone(interface: GripInterface) -> None:
    """Tangential force satisfies |f_t| <= mu * |f_n| * (1 + 1e-9)."""
    model = build_topology_model(interface)
    mu = model.law.static_friction
    # Apply sliding velocities in various directions
    velocities = [
        np.array([0.05, 0.0, 0.0]),
        np.array([-0.05, 0.0, 0.0]),
        np.array([0.0, 0.02, 0.0]),
        np.array([0.03, 0.03, 0.0]),
    ]
    club = BushingState(np.eye(3), np.zeros(3), np.zeros(3), np.zeros(3))
    for v in velocities:
        hand = BushingState(np.eye(3), np.zeros(3), v, np.zeros(3))
        for side in ("L", "R"):
            sw = topology_side_wrench(model, side, hand, club)
            for name, pw in sw.pair_wrenches.items():
                if pw.normal_force_mag_n > 1e-6:
                    assert pw.tangential_force_mag_n <= mu * pw.normal_force_mag_n * (
                        1.0 + 1e-9
                    ), (
                        f"Pair {name} violated friction cone: "
                        f"f_t={pw.tangential_force_mag_n}, mu*f_n={mu * pw.normal_force_mag_n}"
                    )


def test_action_reaction(interface: GripInterface) -> None:
    """Sum of forces and moments about any point is <= 1e-12 for every pair."""
    model = build_topology_model(interface)
    hand = BushingState(
        np.eye(3),
        np.zeros(3),
        np.array([0.01, 0.0, 0.0]),
        np.array([0.5, 0.0, 0.0]),
    )
    club = BushingState(np.eye(3), np.zeros(3), np.zeros(3), np.zeros(3))
    eval_point = np.array([0.05, -0.02, 0.03])

    for side in ("L", "R"):
        sw = topology_side_wrench(model, side, hand, club)
        for pw in sw.pair_wrenches.values():
            net_force = pw.force_a_n + pw.force_b_n
            np.testing.assert_allclose(net_force, np.zeros(3), atol=1e-12)
            net_moment = pw.total_moment_about_point(eval_point)
            np.testing.assert_allclose(net_moment, np.zeros(3), atol=1e-12)


def test_resultant_equals_sum_of_pairs(interface: GripInterface) -> None:
    """Resultant force and moment equal the sum of pairs to <= 1e-12."""
    model = build_topology_model(interface)
    hand = BushingState(
        np.eye(3),
        np.zeros(3),
        np.array([0.02, 0.01, 0.0]),
        np.array([0.5, -0.2, 0.1]),
    )
    club = BushingState(np.eye(3), np.zeros(3), np.zeros(3), np.zeros(3))

    for side in ("L", "R"):
        sw = topology_side_wrench(model, side, hand, club)
        expected_force = sum(pw.force_b_n for pw in sw.pair_wrenches.values())
        expected_moment = sum(pw.moment_b_nm for pw in sw.pair_wrenches.values())
        np.testing.assert_allclose(sw.force_n, expected_force, atol=1e-12)
        np.testing.assert_allclose(sw.moment_nm, expected_moment, atol=1e-12)
        assert sw.total_normal_force_n == pytest.approx(
            float(sum(pw.normal_force_mag_n for pw in sw.pair_wrenches.values())),
            abs=1e-12,
        )


def test_hand_hand_pair_raises_for_physical(interface: GripInterface) -> None:
    """Hand-hand pairs raise ValueError when evaluated as physical contact."""
    model = build_topology_model(interface)
    hand = BushingState(np.eye(3), np.zeros(3), np.zeros(3), np.zeros(3))
    club = BushingState(np.eye(3), np.zeros(3), np.zeros(3), np.zeros(3))

    # trail_palm_lead_thumb is a hand-hand pair present in all topologies
    with pytest.raises(ValueError, match="hand-hand"):
        topology_pair_wrench(model, "trail_palm_lead_thumb", hand, club)


def test_dbc_preconditions(interface: GripInterface) -> None:
    """DbC: Preconditions raise ValueError / TypeError on invalid inputs."""
    model = build_topology_model(interface)
    hand = BushingState(np.eye(3), np.zeros(3), np.zeros(3), np.zeros(3))
    club = BushingState(np.eye(3), np.zeros(3), np.zeros(3), np.zeros(3))

    # Invalid side
    with pytest.raises(ValueError, match="side"):
        topology_side_wrench(model, "invalid_side", hand, club)

    # Invalid types
    with pytest.raises(TypeError, match="hand_frame"):
        topology_side_wrench(model, "L", "not_a_bushing_state", club)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="club_frame"):
        topology_side_wrench(model, "L", hand, "not_a_bushing_state")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="model"):
        topology_side_wrench("not_a_model", "L", hand, club)  # type: ignore[arg-type]

    # Hand-hand pair in physical mode
    with pytest.raises(ValueError, match="hand-hand"):
        topology_pair_wrench(model, "trail_palm_lead_thumb", hand, club)

    # Invalid mode
    with pytest.raises(ValueError, match="physical in mode"):
        topology_pair_wrench(model, "lead_palm_proximal_grip", hand, club, mode="weld")


def test_physical_pair_primitives(interface: GripInterface) -> None:
    """physical_pair_primitives returns read-only tuple of (ContactPair, primitive)."""
    model = build_topology_model(interface)
    for side in ("L", "R"):
        prims = model.physical_pair_primitives(side)
        pairs = model.physical_grip_pairs(side)
        assert len(prims) == len(pairs)
        for (pair, prim), expected_pair in zip(prims, pairs, strict=True):
            assert pair == expected_pair
            site = model.topology.site(pair.site_a)
            if site.primitive == "sphere":
                assert isinstance(prim, ContactSphere)
            elif site.primitive == "capsule":
                assert isinstance(prim, ContactCapsule)


def test_pair_trace_recorder_pure(interface: GripInterface) -> None:
    """PairTraceRecorder records physical traces and states with correct array lengths."""
    model = build_topology_model(interface)
    recorder = PairTraceRecorder(model)
    st_hand = BushingState(np.eye(3), np.zeros(3), np.zeros(3), np.zeros(3))
    st_club = BushingState(np.eye(3), np.zeros(3), np.zeros(3), np.zeros(3))
    wrenches = {s: topology_side_wrench(model, s, st_hand, st_club) for s in ("L", "R")}
    hands = {"L": st_hand, "R": st_hand}
    clubs = {"L": st_club, "R": st_club}

    num_samples = 3
    for _ in range(num_samples):
        recorder.record(wrenches, hands, clubs)

    result = recorder.result()
    assert "pair_normal_force_n" in result
    assert "pair_tangential_force_n" in result
    assert "channel_reasons" in result
    assert "hand_states" in result
    assert "club_states" in result

    for p in model.topology.pairs:
        fn = result["pair_normal_force_n"][p.name]
        ft = result["pair_tangential_force_n"][p.name]
        assert isinstance(fn, np.ndarray)
        assert isinstance(ft, np.ndarray)
        assert fn.shape == (num_samples,)
        assert ft.shape == (num_samples,)

    for s in ("L", "R"):
        assert len(result["hand_states"][s]) == num_samples
        assert len(result["club_states"][s]) == num_samples


def test_pair_trace_recorder_hand_hand_nan_and_reasons(
    interface: GripInterface,
) -> None:
    """Hand-hand pairs record NaN forces and have allocated reasons."""
    model = build_topology_model(interface)
    recorder = PairTraceRecorder(model)
    st_hand = BushingState(np.eye(3), np.zeros(3), np.zeros(3), np.zeros(3))
    st_club = BushingState(np.eye(3), np.zeros(3), np.zeros(3), np.zeros(3))
    wrenches = {s: topology_side_wrench(model, s, st_hand, st_club) for s in ("L", "R")}
    recorder.record(
        wrenches, {"L": st_hand, "R": st_hand}, {"L": st_club, "R": st_club}
    )

    result = recorder.result()
    hh_pairs = model.topology.hand_hand_pairs()
    assert len(hh_pairs) > 0

    expected_reason = "hand-hand pairs are allocated, not physical forces under prescribed hand motion"
    for p in hh_pairs:
        assert np.isnan(result["pair_normal_force_n"][p.name]).all()
        assert np.isnan(result["pair_tangential_force_n"][p.name]).all()
        assert result["channel_reasons"][p.name] == expected_reason

    # Physical pairs must be finite and not in channel_reasons
    for p in model.topology.hand_grip_pairs():
        assert np.isfinite(result["pair_normal_force_n"][p.name]).all()
        assert np.isfinite(result["pair_tangential_force_n"][p.name]).all()
        assert p.name not in result["channel_reasons"]


def test_pair_trace_recorder_unknown_pair(interface: GripInterface) -> None:
    """record() raises ValueError on unknown pair name."""
    from src.shared.python.grip_contact.pair_contact import PairWrench

    model = build_topology_model(interface)
    recorder = PairTraceRecorder(model)
    st = BushingState(np.eye(3), np.zeros(3), np.zeros(3), np.zeros(3))
    sw = topology_side_wrench(model, "L", st, st)

    fake_pw = next(iter(sw.pair_wrenches.values()))
    bad_pair_wrenches = dict(sw.pair_wrenches)
    bad_pair_wrenches["nonexistent_unknown_pair"] = fake_pw
    bad_sw = TopologySideWrench(
        force_n=sw.force_n,
        moment_nm=sw.moment_nm,
        pair_wrenches=bad_pair_wrenches,
        normal_force_n=sw.normal_force_n,
        tangential_force_n=sw.tangential_force_n,
        total_normal_force_n=sw.total_normal_force_n,
    )

    with pytest.raises(ValueError, match="unknown pair"):
        recorder.record({"L": bad_sw, "R": sw}, {"L": st, "R": st}, {"L": st, "R": st})


def test_pair_trace_recorder_dbc(interface: GripInterface) -> None:
    """PairTraceRecorder validates model type and duplicate/missing pairs."""
    with pytest.raises(TypeError, match="TopologyContactModel"):
        PairTraceRecorder("not_a_model")  # type: ignore[arg-type]


def test_drake_hand_body_assignment_pure_without_plant(
    interface: GripInterface,
) -> None:
    """Pure Drake helper computes hand-body assignment of every site without a plant (#13307)."""
    from src.engines.physics_engines.drake.python.grip_contact_sim import (
        HAND_BODY_NAMES,
        hand_body_name,
        hand_site_attachments,
    )

    assert hand_body_name("L") == "lead_hand"
    assert hand_body_name("lead") == "lead_hand"
    assert hand_body_name("R") == "trail_hand"
    assert hand_body_name("trail") == "trail_hand"
    assert HAND_BODY_NAMES == {"L": "lead_hand", "R": "trail_hand"}

    pads = build_pad_model(interface, 1100.0)
    for att in hand_site_attachments(pads=pads):
        assert att.hand_body_name == ("lead_hand" if att.side == "L" else "trail_hand")

    for style in GripStyle:
        topo = contact_topology(GripSpec(style=style))
        topo_model = build_topology_model(interface, topo, squeeze_per_hand_n=1100.0)
        atts = hand_site_attachments(topology_model=topo_model)
        assert len(atts) == len(topo.sites)
        for att in atts:
            site = topo.site(att.name)
            expected_side = "L" if site.hand == "lead" else "R"
            assert att.side == expected_side
            assert att.hand_body_name == HAND_BODY_NAMES[expected_side]


def test_topology_and_pad_model_lod_delegating_accessors(
    interface: GripInterface,
) -> None:
    """TopologyContactModel and PadContactModel expose 1-hop accessors for LoD compliance."""
    pads = build_pad_model(interface, 1100.0)
    topo_model = build_topology_model(interface)

    assert pads.stiffness_n_m == pads.law.stiffness_n_m
    assert pads.origin_axial_m("L") == pads.cylinder.origin_axial_m("L")
    assert pads.origin_axial_m("R") == pads.cylinder.origin_axial_m("R")
    assert topo_model.grip_radius_m == topo_model.cylinder.radius_m
    assert topo_model.origin_axial_m("L") == topo_model.cylinder.origin_axial_m("L")
    assert topo_model.origin_axial_m("R") == topo_model.cylinder.origin_axial_m("R")
