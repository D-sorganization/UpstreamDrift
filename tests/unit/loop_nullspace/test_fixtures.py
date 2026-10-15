"""NSX-12 fixtures: snapshots reproduce the NSX-00 goldens (acceptance tests)."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from src.shared.python.loop_nullspace.contracts import ClosedLoopSnapshot, ClosureKind
from src.shared.python.loop_nullspace.fixtures import (
    AnalyticEnvelopeCase,
    GoldenValues,
    PlanarGolferConfig,
    analytic_envelope_case,
    generic_planar_config,
    load_golden,
    planar_golfer_snapshot,
    spatial_two_arm_club_snapshot,
    straight_arm_snapshot,
)

pytestmark = pytest.mark.unit

RTOL, ATOL = 1e-12, 1e-12


def _rank(arr: np.ndarray) -> int:
    if arr.size == 0:
        return 0
    s = np.linalg.svd(arr, compute_uv=False)
    return int((s > 1e-10 * max(float(s[0]), 1.0)).sum())


def _counts(snap: ClosedLoopSnapshot) -> dict[str, int]:
    topo = snap.topology
    unact = list(topo.unactuated_indices)
    jg = snap.J_g if snap.J_g is not None else np.zeros((0, topo.n))
    a_mat = np.hstack([snap.S.T, snap.J_c.T, jg.T])
    rank_a = _rank(a_mat)
    return {
        "n": topo.n,
        "n_actuated": topo.n_actuated,
        "n_unactuated": len(unact),
        "m": snap.J_c.shape[0],
        "rank_J_c": _rank(snap.J_c),
        "rank_J_cu_T": _rank(snap.J_c[:, unact].T) if unact else 0,
        "rank_J_g": _rank(jg),
        "rank_A": rank_a,
        "equality_nullity": a_mat.shape[1] - rank_a,
    }


# ----------------------------------------------------------------- planar --


@pytest.mark.parametrize(
    ("actuation", "golden_name"),
    [
        ("full", "planar_two_arm_full"),
        ("club_unactuated", "planar_two_arm_club_unactuated"),
    ],
)
def test_planar_snapshot_reproduces_golden_matrices(
    actuation: str, golden_name: str
) -> None:
    snap, golden = planar_golfer_snapshot(actuation=actuation)
    assert isinstance(snap, ClosedLoopSnapshot)
    assert isinstance(golden, GoldenValues)
    assert golden.name == golden_name
    for field, key in (
        ("M", "M"),
        ("h", "h"),
        ("S", "S"),
        ("J_c", "J_c"),
        ("Jdot_v", "Jdot_v"),
    ):
        np.testing.assert_allclose(
            getattr(snap, field),
            golden.array("matrices", key),
            rtol=RTOL,
            atol=ATOL,
            err_msg=field,
        )
    for field in ("q", "v", "a"):
        np.testing.assert_allclose(
            getattr(snap, field), golden.array("state", field), rtol=RTOL, atol=ATOL
        )
    np.testing.assert_allclose(
        snap.generalized_load(), golden.array("matrices", "b"), rtol=RTOL, atol=ATOL
    )


@pytest.mark.parametrize("actuation", ["full", "club_unactuated"])
def test_planar_snapshot_dimension_counts_match_golden(actuation: str) -> None:
    snap, golden = planar_golfer_snapshot(actuation=actuation)
    expected = golden.payload["dimensions"]
    got = _counts(snap)
    for key, value in got.items():
        if key in expected:
            assert value == expected[key], key


def test_planar_snapshot_is_closed_and_admissible() -> None:
    snap, _ = planar_golfer_snapshot()
    assert snap.topology.closure_kind is ClosureKind.PLANAR_POINT2
    assert np.max(np.abs(snap.J_c @ snap.v)) < 1e-12
    assert np.max(np.abs(snap.J_c @ snap.a + snap.Jdot_v)) < 1e-12


def test_planar_club_unactuated_topology() -> None:
    snap, _ = planar_golfer_snapshot(actuation="club_unactuated")
    assert snap.topology.unactuated_indices == (2,)
    assert snap.topology.actuated_indices == (0, 1, 3, 4)


def test_planar_closure_kind_other_than_point2_is_rejected() -> None:
    with pytest.raises(ValueError, match="PLANAR_POINT2"):
        planar_golfer_snapshot(closure=ClosureKind.PLANAR_WELD3)


def test_planar_unknown_actuation_is_rejected() -> None:
    with pytest.raises(ValueError, match="actuation"):
        planar_golfer_snapshot(actuation="bogus")  # type: ignore[arg-type]


def test_planar_open_loop_config_is_rejected() -> None:
    base = generic_planar_config()
    cfg = PlanarGolferConfig(
        np.array([0.1, 1.4, -0.3, 0.9, 0.2]), base.v, base.a, name="open"
    )
    with pytest.raises(ValueError, match="closure"):
        planar_golfer_snapshot(cfg)


# --------------------------------------------------------------- straight --


@pytest.mark.parametrize("actuation", ["full", "club_unactuated"])
def test_straight_arm_snapshot_reproduces_golden(actuation: str) -> None:
    snap, golden = straight_arm_snapshot(actuation=actuation)
    np.testing.assert_allclose(
        snap.J_c, golden.array("matrices", "J_c"), rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        snap.M, golden.array("matrices", "M"), rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        snap.generalized_load(), golden.array("matrices", "b"), rtol=RTOL, atol=ATOL
    )
    assert np.max(np.abs(snap.v)) == 0.0


def test_straight_arm_is_rank_deficient_with_exact_golden_ranks() -> None:
    snap, golden = straight_arm_snapshot()
    got = _counts(snap)
    exact = golden.payload["exact"]
    assert got["rank_J_c"] == exact["rank_J_c"] == 1
    assert got["rank_J_c"] < snap.J_c.shape[0]
    assert got["rank_A"] == exact["rank_A"]


def test_golden_exact_entries_are_fractions() -> None:
    _, golden = straight_arm_snapshot()
    exact = golden.fractions("exact", "J_c")
    assert exact[1][0] == Fraction(-39, 50)


# --------------------------------------------------------------- envelope --


def test_envelope_case_matches_golden_and_independent_lp() -> None:
    linprog = pytest.importorskip("scipy.optimize").linprog
    case = analytic_envelope_case()
    assert isinstance(case, AnalyticEnvelopeCase)
    free = [(None, None)] * 2
    for name, bound in case.bounds.items():
        coeff, const = case.affine_functional(name)
        lo = linprog(coeff, A_ub=case.polytope_g, b_ub=case.polytope_h, bounds=free)
        hi = linprog(-coeff, A_ub=case.polytope_g, b_ub=case.polytope_h, bounds=free)
        assert lo.status == 0 and hi.status == 0
        assert lo.fun + const == pytest.approx(float(bound.lower), abs=1e-9)
        assert -hi.fun + const == pytest.approx(float(bound.upper), abs=1e-9)


def test_envelope_case_closed_form_bounds() -> None:
    case = analytic_envelope_case()
    assert case.bounds["lam_1"].lower == Fraction(0)
    assert case.bounds["lam_1"].upper == Fraction(1)
    assert case.bounds["tau_1"].lower == Fraction(-1, 2)
    assert case.bounds["tau_1"].upper == Fraction(1, 2)
    assert case.vertices == [
        (Fraction(0), Fraction(1, 2)),
        (Fraction(1), Fraction(-1, 2)),
        (Fraction(1), Fraction(1, 2)),
    ]
    assert case.golden.name == "envelope_analytic_lp"


# ---------------------------------------------------------------- spatial --


@pytest.mark.requires_pinocchio
@pytest.mark.parametrize(
    ("variant", "kwargs"),
    [
        ("fixed_base_arms_only", {}),
        ("floating_root_no_feet", {"floating_root": True}),
        ("floating_root_two_point_feet", {"floating_root": True, "feet": True}),
        (
            "floating_root_four_point_feet",
            {"floating_root": True, "feet": True, "foot_points": 2},
        ),
    ],
)
def test_spatial_snapshot_counts_match_golden(variant: str, kwargs: dict) -> None:
    pytest.importorskip("pinocchio")
    snap, golden = spatial_two_arm_club_snapshot(**kwargs)
    assert golden.variant == variant
    expected = golden.payload
    got = _counts(snap)
    for key, value in got.items():
        assert value == expected[key], key
    assert snap.topology.closure_kind is ClosureKind.WELD6


@pytest.mark.requires_pinocchio
def test_spatial_snapshot_is_closed_admissible_and_consistent() -> None:
    pytest.importorskip("pinocchio")
    snap, _ = spatial_two_arm_club_snapshot(floating_root=True, feet=True)
    assert np.max(np.abs(snap.J_c @ snap.v)) < 1e-10
    assert np.max(np.abs(snap.J_c @ snap.a + snap.Jdot_v)) < 1e-9
    assert snap.J_g is not None and snap.J_g.shape == (6, snap.topology.n)
    assert len(snap.foot_contacts) == 2
    assert np.max(np.abs(snap.J_c[:, :6])) < 1e-12  # root cancels in the weld
    sym = 0.5 * (snap.M + snap.M.T)
    assert np.linalg.eigvalsh(sym).min() > 0.0


@pytest.mark.requires_pinocchio
def test_spatial_jdot_v_matches_finite_difference() -> None:
    from src.shared.python.loop_nullspace import spatial_fixture as sf

    pytest.importorskip("pinocchio")
    err = sf.jdot_v_finite_difference_error(floating_root=False)
    assert err < 1e-6


@pytest.mark.requires_pinocchio
def test_spatial_feet_without_floating_root_is_rejected() -> None:
    pytest.importorskip("pinocchio")
    with pytest.raises(ValueError, match="floating"):
        spatial_two_arm_club_snapshot(feet=True)


@pytest.mark.requires_pinocchio
@pytest.mark.parametrize("floating_root", [False, True])
def test_spatial_closure_jacobian_matches_log_map_finite_difference(
    floating_root: bool,
) -> None:
    from src.shared.python.loop_nullspace import spatial_fixture as sf

    pytest.importorskip("pinocchio")
    err = sf.closure_jacobian_finite_difference_error(floating_root=floating_root)
    assert err < 1e-6
