"""NSX-60: two-foot ground contact as a second unilateral null space (#12585).

The spatial chain with a floating root and rigid point feet is the oracle.  The
dimension counts and the nullity reduction of a ground-reaction observation are
checked against the exact NSX-00 goldens ``spatial_counts`` and
``lower_body_observation`` (sympy, no repository code).
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from src.shared.python.loop_nullspace import registry
from src.shared.python.loop_nullspace.contracts import (
    ClosedLoopSnapshot,
    FootContact,
)
from src.shared.python.loop_nullspace.decomposition import (
    BALANCE_TOL,
    Decomposition,
    decompose,
)
from src.shared.python.loop_nullspace.feasible_set import FeasibleSet
from src.shared.python.loop_nullspace.fixtures import (
    load_golden,
    spatial_two_arm_club_snapshot,
)
from src.shared.python.loop_nullspace.ground_contact import (
    GRF_ID,
    ContactCandidate,
    GroundObservation,
    ObservationInconsistentError,
    contact_frame,
    foot_friction_cones,
    observation_accounting,
    observation_rows,
    restrict_to_observation,
    select_active_contacts,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_pinocchio]

VARIANTS = {
    "two_point_feet": {"foot_points": 1},
    "four_point_feet": {"foot_points": 2},
}
SPATIAL_VARIANT = {
    "two_point_feet": "floating_root_two_point_feet",
    "four_point_feet": "floating_root_four_point_feet",
}


def _consistent(snap: ClosedLoopSnapshot) -> ClosedLoopSnapshot:
    """Move the root acceleration so the contact balance has a solution.

    Two point feet are a rank-5 contact map, so a random ``a`` violates the one
    consistency condition (NSX-00 table).  Adding ``alpha M_uu^-1 l`` to the root
    acceleration, with ``l`` the left null vector, restores it and does not
    touch the closure (``J_c`` has zero root columns).
    """
    u = list(snap.topology.unactuated_indices)
    g_u = snap.J_g[:, u].T
    left, sv, _ = np.linalg.svd(g_u)
    keep = int(np.count_nonzero(sv > 1e-10 * sv[0]))
    if keep == g_u.shape[0]:
        return snap
    ell = left[:, keep]
    b_u = snap.generalized_load()[u]
    alpha = -float(ell @ b_u) / float(ell @ ell)
    a = np.array(snap.a)
    a[u] += alpha * np.linalg.solve(snap.M[np.ix_(u, u)], ell)
    return dataclasses.replace(snap, a=a)


@pytest.fixture(scope="module", params=sorted(VARIANTS))
def case(request):
    snap, golden = spatial_two_arm_club_snapshot(
        floating_root=True, feet=True, zero_root_pose=True, **VARIANTS[request.param]
    )
    snap = _consistent(snap)
    dec = decompose(snap, weights=np.eye(snap.topology.n_actuated))
    obs_golden = load_golden("lower_body_observation", request.param)
    return request.param, snap, dec, golden, obs_golden


def _selector_from_golden(entry: dict) -> np.ndarray:
    from fractions import Fraction

    return np.array(
        [[float(Fraction(c)) for c in row] for row in entry["selector"]], dtype=float
    )


def _true_observation(snap, dec, entry, sigma=5.0, shift=0.0) -> GroundObservation:
    sel = _selector_from_golden(entry)
    n_lead = dec.n_actuated + dec.m
    f_true = dec.x_p[n_lead:]
    return GroundObservation(
        selector=sel,
        values=sel @ f_true + shift,
        sigma=np.full(sel.shape[0], sigma),
        labels=tuple(f"row{i}" for i in range(sel.shape[0])),
        source="test: true loads of the particular solution",
    )


# ------------------------------------------------------------------ frames --
@pytest.mark.parametrize(
    "normal", [(0, 0, 1), (0, 1, 0), (1, 0, 0), (0.6, 0.0, 0.8), (-0.3, 0.4, 0.8660254)]
)
def test_contact_frame_is_right_handed_with_normal_last(normal) -> None:
    n = np.array(normal, dtype=float)
    n /= np.linalg.norm(n)
    frame = contact_frame(n)
    assert np.allclose(frame.T @ frame, np.eye(3), atol=1e-12)
    assert np.allclose(frame[:, 2], n, atol=1e-12)
    assert np.isclose(np.linalg.det(frame), 1.0)


def test_contact_frame_rejects_non_unit_normal() -> None:
    with pytest.raises(ValueError, match="unit"):
        contact_frame(np.array([0.0, 0.0, 2.0]))
    with pytest.raises(ValueError, match="shape"):
        contact_frame(np.array([0.0, 1.0]))


# ------------------------------------------------------------ active set ----
def _candidates(gaps):
    return tuple(
        ContactCandidate(
            FootContact(f"c{k}", np.array([k, 0.0, 0.0]), np.array([0, 0, 1.0]), 0.9),
            gap,
        )
        for k, gap in enumerate(gaps)
    )


def test_select_active_contacts_keeps_rows_in_candidate_order() -> None:
    jac = np.arange(4 * 3 * 5, dtype=float).reshape(12, 5)
    contacts, rows = select_active_contacts(
        _candidates([-0.001, 0.05, 0.0, 0.0004]), jac, gap_tol_m=0.0005
    )
    assert [c.name for c in contacts] == ["c0", "c2", "c3"]
    assert np.array_equal(rows, np.vstack([jac[0:3], jac[6:9], jac[9:12]]))
    assert not rows.flags.writeable


def test_select_active_contacts_none_in_contact_gives_no_rows() -> None:
    contacts, rows = select_active_contacts(
        _candidates([0.1, 0.2]), np.zeros((6, 4)), gap_tol_m=0.0
    )
    assert contacts == () and rows is None


@pytest.mark.parametrize("tol", [-1e-3, float("nan"), float("inf")])
def test_select_active_contacts_rejects_bad_tolerance(tol) -> None:
    with pytest.raises(ValueError, match="gap_tol_m"):
        select_active_contacts(_candidates([0.0]), np.zeros((3, 4)), gap_tol_m=tol)


def test_select_active_contacts_rejects_jacobian_row_mismatch() -> None:
    with pytest.raises(ValueError, match="rows"):
        select_active_contacts(_candidates([0.0, 0.0]), np.zeros((3, 4)), gap_tol_m=0.0)


# ------------------------------------------------------------------ cones ---
def test_foot_friction_cones_index_the_contact_loads(case) -> None:
    _, snap, dec, _, _ = case
    cones = foot_friction_cones(snap)
    lead = snap.topology.n_actuated + snap.topology.closure_kind.rows
    assert len(cones) == len(snap.foot_contacts)
    for k, (cc, contact) in enumerate(zip(cones, snap.foot_contacts, strict=True)):
        assert cc.name == contact.name
        assert cc.cone.indices == (lead + 3 * k, lead + 3 * k + 1, lead + 3 * k + 2)
        assert cc.cone.mu == contact.mu
        assert np.allclose(cc.cone.frame[:, 2], contact.normal)
        assert max(cc.cone.indices) < dec.dim_x


def test_foot_friction_cones_need_ground_contacts() -> None:
    snap, _ = spatial_two_arm_club_snapshot(floating_root=True)
    with pytest.raises(ValueError, match="foot contacts"):
        foot_friction_cones(snap)


# ------------------------------------------- two-foot oracle: counts ---------
def test_two_foot_decomposition_counts_match_the_spatial_golden(case) -> None:
    name, snap, dec, golden, _ = case
    payload = golden.payload
    assert golden.variant == SPATIAL_VARIANT[name]
    assert dec.dim_x == payload["dim_x"]
    assert dec.rank_A == payload["rank_A"]
    assert dec.nullity == payload["equality_nullity"]
    assert dec.multiplier_gauge == payload["multiplier_gauge"]
    assert dec.counts.n_g == payload["n_g"]
    assert dec.equality_residual <= BALANCE_TOL


def test_foot_cones_enter_the_feasible_set_with_their_hypothesis(case) -> None:
    from src.shared.python.loop_nullspace.registry import (
        Hypothesis,
        HypothesisSet,
        Source,
    )

    _, snap, dec, _, _ = case
    hset = HypothesisSet(
        "t",
        "Cone Only",
        (Hypothesis("H-FRICTION-CONE", "cone", Source(None, "t"), "NOT_APPLICABLE"),),
        (),
    )
    cones = foot_friction_cones(snap)
    fs = FeasibleSet.from_hypotheses(dec, hset, cones=cones)
    assert fs.n_z == dec.nullity
    assert len(fs.cones) == len(cones)
    assert {c.hypothesis_id for c in fs.cones} == {"H-FRICTION-CONE"}
    # Every cone row is unilateral in the foot normal: the normal-load rows exist.
    kinds = [r.kind for r in fs.rows]
    assert kinds.count("cone_normal") == len(cones)


# -------------------------------------- GRF observation: construction -------
def test_ground_observation_rejects_undeclared_uncertainty() -> None:
    base = dict(
        selector=np.eye(2),
        values=np.zeros(2),
        labels=("a", "b"),
        source="test",
    )
    with pytest.raises(ValueError, match="sigma"):
        GroundObservation(sigma=np.array([1.0, 0.0]), **base)
    with pytest.raises(ValueError, match="sigma"):
        GroundObservation(sigma=np.array([1.0, np.nan]), **base)
    with pytest.raises(ValueError, match="sigma"):
        GroundObservation(sigma=np.array([1.0]), **base)
    with pytest.raises(ValueError, match="labels"):
        GroundObservation(sigma=np.ones(2), **{**base, "labels": ("a", "a")})
    with pytest.raises(ValueError, match="source"):
        GroundObservation(sigma=np.ones(2), **{**base, "source": ""})
    with pytest.raises(ValueError, match="coverage"):
        GroundObservation(sigma=np.ones(2), coverage=0.0, **base)


def test_ground_observation_arrays_are_read_only() -> None:
    obs = GroundObservation(
        selector=np.eye(2),
        values=np.zeros(2),
        sigma=np.ones(2),
        labels=("a", "b"),
        source="test",
    )
    for arr in (obs.selector, obs.values, obs.sigma):
        assert not arr.flags.writeable


def test_from_point_loads_needs_observed_loads(case) -> None:
    _, snap, _, _, _ = case
    with pytest.raises(ValueError, match="grf_observed"):
        GroundObservation.from_point_loads(snap, sigma_n=5.0)


def test_from_point_loads_is_the_identity_on_the_contact_loads(case) -> None:
    _, snap, dec, _, _ = case
    f_true = dec.x_p[dec.n_actuated + dec.m :]
    observed = dataclasses.replace(snap, grf_observed=f_true)
    obs = GroundObservation.from_point_loads(observed, sigma_n=5.0)
    assert np.array_equal(obs.selector, np.eye(f_true.size))
    assert np.allclose(obs.values, f_true)
    assert np.all(obs.sigma == 5.0)
    assert len(obs.labels) == f_true.size


def test_from_contact_groups_sums_member_loads(case) -> None:
    _, snap, _, _, _ = case
    names = [c.name for c in snap.foot_contacts]
    groups = {"all": names}
    obs = GroundObservation.from_contact_groups(
        snap, groups, {"all": [1.0, 2.0, 3.0]}, sigma_n=5.0
    )
    assert obs.selector.shape == (3, 3 * len(names))
    for comp in range(3):
        assert np.array_equal(obs.selector[comp, comp::3], np.ones(len(names)))
    assert np.array_equal(obs.values, [1.0, 2.0, 3.0])


def test_from_contact_groups_rejects_unknown_contacts(case) -> None:
    _, snap, _, _, _ = case
    with pytest.raises(ValueError, match="unknown contact"):
        GroundObservation.from_contact_groups(
            snap, {"g": ["not_a_contact"]}, {"g": [0.0, 0.0, 0.0]}, sigma_n=5.0
        )


def test_normal_load_and_cop_rows_vanish_for_a_load_at_the_cop(case) -> None:
    _, snap, _, _, _ = case
    pts = np.array([c.position for c in snap.foot_contacts])
    n = pts.shape[0]
    weights = np.arange(1.0, n + 1.0)
    load = weights.sum() * 100.0
    cop = (weights[:, None] * pts).sum(axis=0) / weights.sum()
    obs = GroundObservation.from_normal_load_and_cop(
        snap,
        normal_load_n=load,
        cop_m=cop,
        sigma_force_n=5.0,
        sigma_cop_m=0.01,
    )
    f = np.zeros(3 * n)
    f[2::3] = 100.0 * weights
    assert obs.selector.shape == (3, 3 * n)
    assert np.allclose(obs.selector @ f, obs.values, atol=1e-9)
    assert np.isclose(obs.sigma[1], 0.01 * load)


# ------------------------------------- GRF observation: nullity counts ------
OBSERVATIONS = (
    "net_wrench",
    "net_force",
    "normal_load_per_foot",
    "force_per_foot",
    "all_point_forces",
)


@pytest.mark.parametrize("obs_name", OBSERVATIONS)
def test_observation_reduces_nullity_by_the_exact_rank(case, obs_name) -> None:
    name, snap, dec, _, obs_golden = case
    entry = obs_golden.payload["observations"][obs_name]
    assert dec.nullity == obs_golden.payload["nullity_free"]
    obs = _true_observation(snap, dec, entry)
    acc = observation_accounting(dec, obs)
    assert acc.nullity_free == dec.nullity
    assert acc.rank_visible == entry["reduction"], (name, obs_name)
    assert acc.nullity_observed == entry["nullity_observed"]
    assert acc.n_rows == entry["rows"]
    assert acc.rank_selector == entry["rank_selector"]


@pytest.mark.parametrize("obs_name", OBSERVATIONS)
def test_true_loads_are_consistent_with_the_balance(case, obs_name) -> None:
    _, snap, dec, _, obs_golden = case
    obs = _true_observation(snap, dec, obs_golden.payload["observations"][obs_name])
    acc = observation_accounting(dec, obs)
    assert acc.max_normalised_residual < 1e-9
    assert acc.chi2 < 1e-12


def test_restricted_family_has_the_golden_nullity_and_satisfies_both(case) -> None:
    _, snap, dec, _, obs_golden = case
    entry = obs_golden.payload["observations"]["force_per_foot"]
    obs = _true_observation(snap, dec, entry)
    restricted = restrict_to_observation(dec, obs)
    assert isinstance(restricted, Decomposition)
    assert restricted.nullity == entry["nullity_observed"]
    assert restricted.N.shape == (dec.dim_x, entry["nullity_observed"])
    # Still a solution of A x = b for every z.
    a_mat = np.hstack(
        [snap.S.T, snap.J_c.T, snap.J_g.T],
    )
    rng = np.random.default_rng(12585)
    for _ in range(3):
        x = restricted.x(rng.normal(size=restricted.nullity))
        assert np.allclose(a_mat @ x, snap.generalized_load(), atol=1e-7)
        f = x[dec.n_actuated + dec.m :]
        assert np.allclose(obs.selector @ f, obs.values, atol=1e-7)
    # Orthonormal in the scaled coordinates of the parent family.
    z = np.linalg.lstsq(dec.N, restricted.N, rcond=None)[0]
    assert np.allclose(z.T @ z, np.eye(restricted.nullity), atol=1e-9)


def test_restrict_refuses_an_inconsistent_observation(case) -> None:
    _, snap, dec, _, obs_golden = case
    entry = obs_golden.payload["observations"]["force_per_foot"]
    obs = _true_observation(snap, dec, entry, sigma=5.0, shift=500.0)
    with pytest.raises(ObservationInconsistentError) as err:
        restrict_to_observation(dec, obs)
    assert err.value.max_normalised_residual > obs.coverage


def test_restrict_with_a_redundant_observation_changes_nothing(case) -> None:
    _, snap, dec, _, obs_golden = case
    obs = _true_observation(snap, dec, obs_golden.payload["observations"]["net_wrench"])
    restricted = restrict_to_observation(dec, obs)
    assert restricted.nullity == dec.nullity
    assert np.allclose(restricted.x_p, dec.x_p, atol=1e-8)


def test_accounting_reports_singular_values_over_sigma(case) -> None:
    _, snap, dec, _, obs_golden = case
    entry = obs_golden.payload["observations"]["all_point_forces"]
    tight = _true_observation(snap, dec, entry, sigma=1.0)
    loose = _true_observation(snap, dec, entry, sigma=10.0)
    s_tight = observation_accounting(dec, tight).singular_values_over_sigma
    s_loose = observation_accounting(dec, loose).singular_values_over_sigma
    assert s_tight.size == min(entry["rows"], dec.nullity)
    assert np.allclose(s_tight, 10.0 * s_loose, rtol=1e-9)
    assert np.all(np.diff(s_tight) <= 1e-12)


def test_accounting_rejects_an_observation_of_the_wrong_size(case) -> None:
    _, _, dec, _, _ = case
    bad = GroundObservation(
        selector=np.ones((1, dec.counts.n_g + 3)),
        values=np.zeros(1),
        sigma=np.ones(1),
        labels=("x",),
        source="test",
    )
    with pytest.raises(ValueError, match="columns"):
        observation_accounting(dec, bad)


# ------------------------------------- GRF observation: feasible set --------
def _grf_hypotheses(hypothesis_id: str = GRF_ID):
    from src.shared.python.loop_nullspace.registry import (
        Hypothesis,
        HypothesisSet,
        Source,
    )

    return HypothesisSet(
        "t",
        "Grf",
        (Hypothesis(hypothesis_id, "t", Source(None, "t"), "NOT_APPLICABLE"),),
        (),
    )


def test_observation_rows_are_two_sided_bands_in_z(case) -> None:
    _, snap, dec, _, obs_golden = case
    obs = _true_observation(
        snap, dec, obs_golden.payload["observations"]["force_per_foot"], sigma=5.0
    )
    rows = observation_rows(dec, obs)
    assert len(rows) == 2 * obs.selector.shape[0]
    assert {r.hypothesis_id for r in rows} == {GRF_ID}
    assert {r.kind for r in rows} == {"grf"}
    # z = 0 (the particular solution) matches the data exactly: slack = k sigma.
    for r in rows:
        assert np.isclose(r.h, obs.coverage * 5.0, atol=1e-6)
    # A step along any null direction that the observation sees is bounded.
    e_n = obs.selector @ dec.N[dec.n_actuated + dec.m :]
    for i in range(e_n.shape[0]):
        up, lo = rows[2 * i], rows[2 * i + 1]
        assert np.allclose(up.g, e_n[i]) and np.allclose(lo.g, -e_n[i])


def test_feasible_set_requires_the_grf_hypothesis_to_be_declared(case) -> None:
    _, snap, dec, _, obs_golden = case
    obs = _true_observation(
        snap, dec, obs_golden.payload["observations"]["force_per_foot"]
    )
    with pytest.raises(ValueError, match="H-GRF-OBSERVATION"):
        FeasibleSet.from_hypotheses(
            dec, _grf_hypotheses("H-FRICTION-CONE"), observation=obs
        )


def test_feasible_set_with_observation_is_narrower_than_without(case) -> None:
    from src.shared.python.loop_nullspace.envelopes import (
        EnvelopeQuantity,
        static_envelopes,
    )

    pytest.importorskip("highspy")
    _, snap, dec, _, obs_golden = case
    entry = obs_golden.payload["observations"]["all_point_forces"]
    obs = _true_observation(snap, dec, entry, sigma=5.0)
    hset = _grf_hypotheses()
    free = FeasibleSet.from_hypotheses(dec, hset)
    seen = FeasibleSet.from_hypotheses(dec, hset, observation=obs)
    lead = dec.n_actuated + dec.m
    q = EnvelopeQuantity("f0", dec.N[lead], float(dec.x_p[lead]), "N", 100.0)
    row_free = static_envelopes([free], [q])
    row_seen = static_envelopes([seen], [q])
    width_free = float(row_free.width()[0, 0])
    width_seen = float(row_seen.width()[0, 0])
    assert width_seen < width_free or np.isnan(width_free)
    assert width_seen <= 2 * obs.coverage * 5.0 + 1e-6
    assert GRF_ID in seen.hypothesis_ids


def test_golden_files_cover_every_tested_observation() -> None:
    for variant in VARIANTS:
        payload = load_golden("lower_body_observation", variant).payload
        assert set(payload["observations"]) == set(OBSERVATIONS)
        json.dumps(payload)  # plain JSON data
