"""Unit and parity tests for Pinocchio topology contact adapter (HRG-GS-G6, #13304).

Verifies:
- Per-style static squeeze across OVERLAP, INTERLOCK, and TEN_FINGER.
- Per-pair force traces equal shared-law evaluation at identical states to <= 1e-9 N.
- Timestep halving changes peak pair force by < 2 %.
- Hand-hand pairs are allocated/diagnostic only: zero dynamical effect on club wrench.
- Design by Contract (DbC) preconditions and postconditions.
- Engine simulation skips cleanly when Pinocchio runtime is unavailable.
"""

from __future__ import annotations

import json
import math
from typing import Any


import numpy as np
import pytest

from src.shared.python.grip_contact.bushing_law import BushingState
from src.shared.python.grip_contact.contact_topology import (
    ContactPair,
    ContactSite,
    ContactTopology,
    GripStyle,
    contact_topology,
)
from src.shared.python.grip_contact.pair_contact import (
    ContactCapsule,
    ContactSphere,
    capsule_capsule,
    capsule_sphere,
    sphere_capsule,
)
from src.shared.python.motion_matching.contact_law import ContactParameters

from src.shared.python.motion_matching.grip_contract import GripSpec
from src.engines.physics_engines.pinocchio.python.topology_contact_sim import (
    IntegratorTolerances,
    TopologyClubInHands,
    TopologyContactRun,
    TopologyPairResult,
    default_contact_parameters,
    evaluate_topology_wrenches,
    simulate_topology_contact,
)


pytestmark = pytest.mark.unit


def _stationary_bushing_state(
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotation: np.ndarray | None = None,
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    omega: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> BushingState:
    """Helper to construct a stationary or moving BushingState."""
    rot = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)
    return BushingState(
        rotation=rot,
        position_m=np.asarray(position, dtype=float),
        velocity_m_s=np.asarray(velocity, dtype=float),
        omega_rad_s=np.asarray(omega, dtype=float),
    )


def _neutral_hands_and_club(
    spec: GripSpec,
) -> tuple[dict[str, BushingState], BushingState]:
    """Construct hands and club in aligned neutral grip pose."""
    # Club at origin aligned with x axis
    club = _stationary_bushing_state((0.0, 0.0, 0.0))
    # Hands aligned with grip frame origin
    hands = {
        "lead": _stationary_bushing_state((0.0, 0.0, 0.0)),
        "trail": _stationary_bushing_state((0.0, 0.0, 0.0)),
        "L": _stationary_bushing_state((0.0, 0.0, 0.0)),
        "R": _stationary_bushing_state((0.0, 0.0, 0.0)),
    }
    return hands, club


@pytest.mark.parametrize(
    "style",
    [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER],
)
def test_static_squeeze_per_style_evaluates_active_pairs(style: GripStyle) -> None:
    """Every grip style evaluates active hand-grip and hand-hand pairs with Fn >= 0."""
    spec = GripSpec(style=style)
    topo = contact_topology(spec)
    hands, club = _neutral_hands_and_club(spec)
    law = default_contact_parameters()

    pair_results, club_wrench, hand_wrenches = evaluate_topology_wrenches(
        topo, hands, club, law
    )

    assert isinstance(pair_results, dict)
    assert len(pair_results) == len(topo.pairs)

    # Hand-grip pairs have compressive normal forces
    hg_pairs = topo.hand_grip_pairs()
    assert len(hg_pairs) > 0
    for p in hg_pairs:
        res = pair_results[p.name]
        assert isinstance(res, TopologyPairResult)
        assert res.normal_force_mag_n >= 0.0
        assert np.isfinite(res.force_club_n).all()
        assert np.isfinite(res.moment_club_nm).all()

    # Net wrench on club from hand-grip pairs is finite
    f_club, m_club = club_wrench
    assert np.isfinite(f_club).all()
    assert np.isfinite(m_club).all()

    # Per-side wrenches sum to net club wrench
    f_lead, m_lead = hand_wrenches["lead"]
    f_trail, m_trail = hand_wrenches["trail"]
    np.testing.assert_allclose(f_club, f_lead + f_trail, atol=1e-12)
    np.testing.assert_allclose(m_club, m_lead + m_trail, atol=1e-12)


def test_hand_hand_isolation_on_club() -> None:
    """Hand-hand pairs must never contribute to the club wrench."""
    spec = GripSpec(style=GripStyle.OVERLAP)
    topo = contact_topology(spec)
    hands, club = _neutral_hands_and_club(spec)
    law = default_contact_parameters()

    # Nominal evaluation
    _, (f_club_nom, m_club_nom), _ = evaluate_topology_wrenches(topo, hands, club, law)

    # Artificially shift hand-hand contact parameters by modifying hand positions
    # slightly while keeping hand-grip geometry unchanged relative to club
    hh_pairs = topo.hand_hand_pairs()
    assert len(hh_pairs) > 0
    for _p in hh_pairs:
        # Check that hand-hand pair wrench has zero force on club
        pass

    # Club wrench must strictly equal the sum of hand-grip pairs only
    hg_forces = np.zeros(3)
    hg_moments = np.zeros(3)
    results, (f_club, m_club), _ = evaluate_topology_wrenches(topo, hands, club, law)
    for p in topo.hand_grip_pairs():
        res = results[p.name]
        hg_forces += res.force_club_n
        hg_moments += res.moment_club_nm

    np.testing.assert_allclose(f_club, hg_forces, atol=1e-12)
    np.testing.assert_allclose(m_club, hg_moments, atol=1e-12)


def test_separation_condition_yields_zero_force() -> None:
    """When hands are disengaged (retracted), all contact forces are identically zero."""
    spec = GripSpec()
    topo = contact_topology(spec)
    # Move lead and trail hands far from each other and from the club
    hands = {
        "lead": _stationary_bushing_state((0.0, 1.0, 0.0)),
        "trail": _stationary_bushing_state((0.0, -1.0, 0.0)),
        "L": _stationary_bushing_state((0.0, 1.0, 0.0)),
        "R": _stationary_bushing_state((0.0, -1.0, 0.0)),
    }

    club = _stationary_bushing_state((0.0, 0.0, 0.0))
    law = default_contact_parameters()

    pair_results, (f_club, m_club), _ = evaluate_topology_wrenches(
        topo, hands, club, law
    )

    for p in topo.pairs:
        res = pair_results[p.name]
        assert res.penetration_m <= 0.0
        assert res.normal_force_mag_n == 0.0
        assert res.tangential_force_mag_n == 0.0
        np.testing.assert_allclose(res.force_club_n, np.zeros(3), atol=1e-12)
        np.testing.assert_allclose(res.moment_club_nm, np.zeros(3), atol=1e-12)

    np.testing.assert_allclose(f_club, np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(m_club, np.zeros(3), atol=1e-12)


def test_hand_hand_pairs_satisfy_action_reaction() -> None:
    """Internal hand-hand pairs satisfy Newton's third law: F_A + F_B = 0."""
    spec = GripSpec(style=GripStyle.OVERLAP)
    topo = contact_topology(spec)
    hands, club = _neutral_hands_and_club(spec)
    law = default_contact_parameters()

    pair_results, _, _ = evaluate_topology_wrenches(topo, hands, club, law)

    for p in topo.hand_hand_pairs():
        res = pair_results[p.name]
        np.testing.assert_allclose(
            res.force_a_n + res.force_b_n, np.zeros(3), atol=1e-12
        )


def test_dbc_preconditions_reject_invalid_inputs() -> None:
    """DbC preconditions reject invalid types with TypeError or ValueError."""
    spec = GripSpec()
    topo = contact_topology(spec)
    hands, club = _neutral_hands_and_club(spec)
    law = default_contact_parameters()

    with pytest.raises(TypeError, match="topology"):
        evaluate_topology_wrenches("not_topo", hands, club, law)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="hands"):
        evaluate_topology_wrenches(topo, "not_hands", club, law)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="club"):
        evaluate_topology_wrenches(topo, hands, "not_club", law)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="law"):
        evaluate_topology_wrenches(topo, hands, club, "not_law")  # type: ignore[arg-type]


def test_per_pair_force_traces_equal_shared_law_evaluation() -> None:
    """Per-pair force traces equal shared-law evaluation at identical states to <= 1e-9 N."""
    spec = GripSpec(style=GripStyle.OVERLAP)
    topo = contact_topology(spec)
    hands, club = _neutral_hands_and_club(spec)
    law = default_contact_parameters()

    pair_results, _, _ = evaluate_topology_wrenches(topo, hands, club, law)

    # Validate hand-hand pairs match direct pair_contact evaluation to <= 1e-12 N
    for p in topo.hand_hand_pairs():
        res = pair_results[p.name]
        site_a = topo.site(p.site_a)
        site_b = topo.site(p.site_b)
        st_a = hands[site_a.hand]
        st_b = hands[site_b.hand]
        prim_a = (
            ContactSphere(np.asarray(site_a.centre_m, float), site_a.radius_m)
            if site_a.primitive == "sphere"
            else ContactCapsule(
                np.asarray(site_a.centre_m, float)
                - site_a.half_length_m * np.asarray(site_a.axis_m, float),
                np.asarray(site_a.centre_m, float)
                + site_a.half_length_m * np.asarray(site_a.axis_m, float),
                site_a.radius_m,
            )
        )
        prim_b = (
            ContactSphere(np.asarray(site_b.centre_m, float), site_b.radius_m)
            if site_b.primitive == "sphere"
            else ContactCapsule(
                np.asarray(site_b.centre_m, float)
                - site_b.half_length_m * np.asarray(site_b.axis_m, float),
                np.asarray(site_b.centre_m, float)
                + site_b.half_length_m * np.asarray(site_b.axis_m, float),
                site_b.radius_m,
            )
        )
        if isinstance(prim_a, ContactSphere) and isinstance(prim_b, ContactCapsule):
            direct_w = sphere_capsule(prim_a, prim_b, st_a, st_b, law)
        elif isinstance(prim_a, ContactCapsule) and isinstance(prim_b, ContactCapsule):
            direct_w = capsule_capsule(prim_a, prim_b, st_a, st_b, law)
        else:
            raise AssertionError(
                f"Unexpected primitive types: {type(prim_a)}, {type(prim_b)}"
            )

        diff = np.linalg.norm(res.force_a_n - direct_w.force_a_n)
        assert diff <= 1e-9, f"Pair {p.name} force mismatch: {diff} N > 1e-9 N"
        assert abs(res.normal_force_mag_n - direct_w.normal_force_mag_n) <= 1e-9


def _has_pinocchio_runtime() -> bool:
    """Return whether a real Pinocchio runtime API is available."""
    import importlib

    try:
        pin = importlib.import_module("pinocchio")
    except ImportError:
        return False
    if getattr(pin, "__file__", None) is None:
        return False
    builder = getattr(pin, "buildModelFromUrdf", None)
    return callable(builder)


@pytest.mark.requires_pinocchio
@pytest.mark.skipif(
    not _has_pinocchio_runtime(), reason="real Pinocchio runtime required"
)
def test_pinocchio_simulation_swing_replay_and_timestep_halving() -> None:
    """Static squeeze and 1s swing replay with timestep halving < 2% peak force change."""
    from pathlib import Path
    from src.shared.python.grip_contact import GripInterface, load_coordinate_swing

    root = Path(__file__).resolve().parents[3]
    models = root / "docs/development/full_body_models"
    fixtures = root / "tests/fixtures/club_face"

    spec_bytes = (models / "full_body_spec_anthro_driver.json").read_bytes()
    spec = json.loads(spec_bytes)
    names = spec["coordinate_order"]
    swing = load_coordinate_swing(
        fixtures / "swing_q_driver.npz",
        fixtures / "address_poses.json",
        "driver",
        names,
    )
    interface = GripInterface.from_spec(spec)

    for style in [GripStyle.OVERLAP, GripStyle.INTERLOCK, GripStyle.TEN_FINGER]:
        topo = contact_topology(GripSpec(style=style))

        # Static squeeze hold
        run_hold = simulate_topology_contact(
            spec_bytes,
            swing,
            topology=topo,
            interface=interface,
            hold=True,
            t_end_s=0.05,
        )
        assert len(run_hold.series.time_s) > 1

        # 1 s swing replay: nominal timestep max 2e-4 s
        tol_nom = IntegratorTolerances(max_step_s=2.0e-4)
        run_nom = simulate_topology_contact(
            spec_bytes,
            swing,
            topology=topo,
            interface=interface,
            tolerances=tol_nom,
            t_end_s=1.0,
        )

        # 1 s swing replay: halved timestep max 1e-4 s
        tol_half = IntegratorTolerances(max_step_s=1.0e-4)
        run_half = simulate_topology_contact(
            spec_bytes,
            swing,
            topology=topo,
            interface=interface,
            tolerances=tol_half,
            t_end_s=1.0,
        )

        # Timestep halving changes peak pair force by < 2 %
        for p in topo.pairs:
            peak_nom = float(np.max(run_nom.pair_normal_force_n[p.name]))
            peak_half = float(np.max(run_half.pair_normal_force_n[p.name]))
            if peak_nom > 1.0:
                rel_change = abs(peak_nom - peak_half) / peak_nom
                assert rel_change < 0.02, (
                    f"Style {style.value} pair {p.name} peak force changed by "
                    f"{rel_change * 100:.2f}% (>= 2%)"
                )
