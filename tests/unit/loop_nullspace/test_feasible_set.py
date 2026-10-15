"""Tests for the declared feasible set in null-space coordinates (NSX-21, #12564).

The set is conditional on the declared hypotheses; every row names one, the
conflict set is inclusion-minimal, and unverified bases are flagged.
"""

from __future__ import annotations

import dataclasses
from fractions import Fraction

import numpy as np
import pytest

pytest.importorskip("highspy")
pytest.importorskip("clarabel")

from src.shared.python.loop_nullspace import registry  # noqa: E402
from src.shared.python.loop_nullspace.capacity import (  # noqa: E402
    CapacitySubject,
    CapacityVector,
    capacity_for,
)
from src.shared.python.loop_nullspace.decomposition import (  # noqa: E402
    CharacteristicScales,
    Decomposition,
    DimensionAccounting,
    decompose,
)
from src.shared.python.loop_nullspace.feasible_set import (  # noqa: E402
    ContactCone,
    FeasibleRow,
    FeasibleSet,
    GripPreload,
    cone_error_bound,
)
from src.shared.python.loop_nullspace.fixtures import (  # noqa: E402
    analytic_envelope_case,
    generic_planar_config,
    load_golden,
    planar_golfer_snapshot,
)
from src.shared.python.loop_nullspace.grip_split import closing_hand_map  # noqa: E402
from src.shared.python.loop_nullspace.registry import (  # noqa: E402
    Hypothesis,
    HypothesisSet,
    Source,
)
from src.shared.python.motion_matching.force_nullspace.constraints import (  # noqa: E402
    FrictionCone,
)

pytestmark = pytest.mark.unit

TORQUE = "H-TORQUE-CAPACITY"
CONE = "H-FRICTION-CONE"
GRIP = "H-GRIP-PRELOAD"


def _hset(*ids: str, status: str = "NOT_APPLICABLE") -> HypothesisSet:
    hyps = tuple(
        Hypothesis(i, f"statement {i}", Source(None, "test"), status) for i in ids
    )
    return HypothesisSet("test", "Test", hyps, ())


def _decomp(x_p, n_matrix, n_a: int) -> Decomposition:
    x_p = np.asarray(x_p, float)
    n_matrix = np.asarray(n_matrix, float)
    dim_x, nullity = n_matrix.shape
    counts = DimensionAccounting(
        n=n_a,
        n_a=n_a,
        n_u=0,
        m=dim_x - n_a,
        n_g=0,
        dim_x=dim_x,
        rank_A=dim_x - nullity,
        nullity=nullity,
        rank_J_c=0,
        rank_J_cu_T=0,
        multiplier_gauge=0,
        attainable_torque_variation=0,
        rank_stacked=0,
        rank_stacked_u=0,
        multiplier_gauge_fg_free=0,
        attainable_torque_variation_fg_free=0,
    )
    return Decomposition(
        x_p=x_p,
        N=n_matrix,
        counts=counts,
        singular_values_scaled=np.ones(dim_x - nullity),
        condition_number=1.0,
        equality_residual=0.0,
        metric_label="test",
        scales=CharacteristicScales(),
    )


def _capacity(neg, pos, assumed=()) -> CapacityVector:
    names = tuple(f"j{i}" for i in range(len(neg)))
    return CapacityVector(
        coordinate_order=names,
        actuated_indices=tuple(range(len(names))),
        actuated_names=names,
        tau_max_negative_nm=np.asarray(neg, float),
        tau_max_positive_nm=np.asarray(pos, float),
        assumed_slots=tuple(assumed),
        verified_slots=tuple(n for n in names if n not in assumed),
        warnings=(),
        force_velocity={},
    )


def _cone_decomp() -> Decomposition:
    """dim_x = 5: tau (2), tangential forces (2) free, normal force fixed to 1."""
    nmat = np.zeros((5, 4))
    nmat[:4] = np.eye(4)
    return _decomp([0, 0, 0, 0, 1.0], nmat, 2)


def _cone(mu: float = 0.5) -> ContactCone:
    return ContactCone("foot", FrictionCone((2, 3, 4), np.eye(3), mu))


# ------------------------------------------------------------- torque rows
def test_torque_rows_are_analytic_and_traceable() -> None:
    d = _decomp([0.5, 0, 0], np.array([[1.0, 0], [0, 1.0], [1.0, 1.0]]), 2)
    cap = _capacity([2.0, 3.0], [1.0, 4.0])
    fs = FeasibleSet.from_hypotheses(d, _hset(TORQUE), capacity=cap)
    assert fs.n_z == 2
    by_label = {r.label: r for r in fs.rows}
    assert set(by_label) == {
        f"{TORQUE}:j0:upper",
        f"{TORQUE}:j0:lower",
        f"{TORQUE}:j1:upper",
        f"{TORQUE}:j1:lower",
    }
    up = by_label[f"{TORQUE}:j0:upper"]
    assert up.g.tolist() == [1.0, 0.0] and up.h == pytest.approx(1.0 - 0.5)
    lo = by_label[f"{TORQUE}:j0:lower"]
    assert lo.g.tolist() == [-1.0, 0.0] and lo.h == pytest.approx(2.0 + 0.5)
    assert by_label[f"{TORQUE}:j1:upper"].h == pytest.approx(4.0)
    assert all(r.hypothesis_id == TORQUE for r in fs.rows)
    assert fs.hypothesis_ids == (TORQUE,)
    assert not fs.is_empty()
    assert fs.conflict_set() == () and fs.conflict_rows() == ()


def test_every_row_traces_to_a_declared_hypothesis() -> None:
    d = _cone_decomp()
    fs = FeasibleSet.from_hypotheses(
        d,
        _hset(TORQUE, CONE),
        capacity=_capacity([1, 1], [1, 1]),
        cones=[_cone()],
        cone_facets=6,
    )
    declared = set(fs.hypothesis_ids)
    assert {r.hypothesis_id for r in fs.rows} <= declared
    assert {c.hypothesis_id for c in fs.cones} <= declared
    assert all(r.label.startswith(r.hypothesis_id + ":") for r in fs.rows)
    assert len({r.label for r in fs.rows}) == len(fs.rows)


def test_undeclared_or_missing_data_is_rejected() -> None:
    d = _cone_decomp()
    cap = _capacity([1, 1], [1, 1])
    with pytest.raises(ValueError, match="not declared"):
        FeasibleSet.from_hypotheses(d, _hset(CONE), capacity=cap, cones=[_cone()])
    with pytest.raises(ValueError, match="not declared"):
        FeasibleSet.from_hypotheses(d, _hset(TORQUE), capacity=cap, cones=[_cone()])
    with pytest.raises(ValueError, match="capacity"):
        FeasibleSet.from_hypotheses(d, _hset(TORQUE))
    with pytest.raises(ValueError, match="actuated"):
        FeasibleSet.from_hypotheses(
            d, _hset(TORQUE), capacity=_capacity([1, 1, 1], [1, 1, 1])
        )
    with pytest.raises(TypeError):
        FeasibleSet.from_hypotheses("nope", _hset(TORQUE), capacity=cap)
    with pytest.raises(TypeError):
        FeasibleSet.from_hypotheses(d, ["H-TORQUE-CAPACITY"], capacity=cap)
    with pytest.raises(ValueError, match="cone_facets"):
        FeasibleSet.from_hypotheses(d, _hset(CONE), cones=[_cone()], cone_facets=2)
    with pytest.raises(ValueError, match="linearization"):
        FeasibleSet.from_hypotheses(
            d, _hset(CONE), cones=[_cone()], linearization="middle"
        )
    with pytest.raises(ValueError, match="exceeds"):
        bad = ContactCone("bad", FrictionCone((3, 4, 9), np.eye(3), 0.5))
        FeasibleSet.from_hypotheses(d, _hset(CONE), cones=[bad])


# ------------------------------------------------------- analytic envelope
def test_analytic_envelope_rows_reproduce_golden_bounds() -> None:
    case = analytic_envelope_case()
    rows = tuple(
        FeasibleRow(g, float(h), label, label.split("_")[0] or "H", "torque")
        for g, h, label in zip(
            case.polytope_g, case.polytope_h, case.row_labels, strict=True
        )
    )
    fs = FeasibleSet.from_rows(2, rows)
    assert not fs.is_empty()
    lp = fs.to_lp()
    for name in ("lam_1", "lam_2"):
        coeff, _ = case.affine_functional(name)
        bound = case.bounds[name]
        lo = lp.solve(coeff)
        hi = lp.solve(-coeff)
        assert lo.objective == pytest.approx(float(bound.lower), abs=1e-9)
        assert -hi.objective == pytest.approx(float(bound.upper), abs=1e-9)


def test_analytic_infeasible_conflict_set_matches_golden() -> None:
    gold = load_golden("envelope_analytic_lp", None).get("infeasible")
    rows = tuple(
        FeasibleRow(
            np.array([float(Fraction(x)) for x in r["g"]]),
            float(Fraction(r["h"])),
            r["label"],
            r["label"].split(":")[0],
            "torque",
        )
        for r in gold["rows"]
    )
    fs = FeasibleSet.from_rows(2, rows)
    assert fs.is_empty()  # reported, not raised
    expected_rows = set(gold["inclusion_minimal_conflict_sets"][0])
    assert set(fs.conflict_rows()) == expected_rows
    assert set(fs.conflict_set()) == {r.split(":")[0] for r in expected_rows}
    # Independent inclusion-minimality: dropping any one conflicting row is feasible.
    by_label = {r.label: r for r in rows}
    for drop in fs.conflict_rows():
        kept = tuple(by_label[x] for x in fs.conflict_rows() if x != drop)
        assert not FeasibleSet.from_rows(2, kept).is_empty()
        assert FeasibleSet.from_rows(2, kept).conflict_set() == ()


def test_conflict_set_is_minimal_with_bystander_rows() -> None:
    e = np.eye(2)
    rows = (
        FeasibleRow(e[0], 1.0, "A:x:upper", "A", "torque"),
        FeasibleRow(-e[0], -2.0, "B:x:lower", "B", "torque"),  # x >= 2: conflicts A
        FeasibleRow(e[1], 5.0, "C:y:upper", "C", "torque"),
        FeasibleRow(-e[1], 5.0, "C:y:lower", "C", "torque"),
        FeasibleRow(e[0] + e[1], 100.0, "D:sum", "D", "torque"),
        FeasibleRow(e[0], 50.0, "B:x:upper", "B", "torque"),  # redundant, same id
    )
    fs = FeasibleSet.from_rows(2, rows)
    assert fs.is_empty()
    assert set(fs.conflict_set()) == {"A", "B"}
    assert set(fs.conflict_rows()) == {"A:x:upper", "B:x:lower"}


def test_two_independent_conflicts_returns_one_minimal_set() -> None:
    e = np.eye(2)
    rows = (
        FeasibleRow(e[0], 0.0, "A:u", "A", "torque"),
        FeasibleRow(-e[0], -1.0, "B:l", "B", "torque"),
        FeasibleRow(e[1], 0.0, "C:u", "C", "torque"),
        FeasibleRow(-e[1], -1.0, "D:l", "D", "torque"),
    )
    fs = FeasibleSet.from_rows(2, rows)
    ids = set(fs.conflict_set())
    assert ids in ({"A", "B"}, {"C", "D"})
    sub = tuple(r for r in rows if r.hypothesis_id in ids)
    assert FeasibleSet.from_rows(2, sub).is_empty()
    for drop in ids:
        part = tuple(r for r in sub if r.hypothesis_id != drop)
        assert not FeasibleSet.from_rows(2, part).is_empty()


def test_contradictory_hypotheses_reported_with_ids_not_exceptions() -> None:
    d = _decomp([0, 0, 0], np.array([[1.0], [0.0], [1.0]]), 2)
    cap = _capacity([1.0, 1.0], [1.0, 1.0])
    # tau0 = z; squeeze q = x2 = z must be >= 5 but |tau0| <= 1.
    grip = GripPreload("hand_L", np.array([0.0, 0.0, 1.0]), 0.0, 5.0)
    fs = FeasibleSet.from_hypotheses(
        d, _hset(TORQUE, GRIP), capacity=cap, grip_preloads=[grip]
    )
    assert fs.is_empty()
    assert set(fs.conflict_set()) == {TORQUE, GRIP}
    assert f"{GRIP}:hand_L:squeeze" in fs.conflict_rows()


def test_empty_nullspace_checks_constants() -> None:
    d = _decomp([0.5, 0.0], np.zeros((2, 0)), 2)
    ok = FeasibleSet.from_hypotheses(
        d, _hset(TORQUE), capacity=_capacity([1, 1], [1, 1])
    )
    assert ok.n_z == 0 and not ok.is_empty()
    bad = FeasibleSet.from_hypotheses(
        d, _hset(TORQUE), capacity=_capacity([1, 1], [0.25, 1])
    )
    assert bad.is_empty()
    assert bad.conflict_set() == (TORQUE,)
    assert bad.conflict_rows() == (f"{TORQUE}:j0:upper",)


# ------------------------------------------------------------------ cones
@pytest.mark.parametrize("k", [4, 8, 16])
def test_outer_linearization_bound_holds(k: int) -> None:
    mu = 0.5
    bound = cone_error_bound(k, "outer")
    assert bound == pytest.approx(1.0 / np.cos(np.pi / k) - 1.0)
    d = _cone_decomp()
    fs = FeasibleSet.from_hypotheses(
        d, _hset(CONE), cones=[_cone(mu)], cone_facets=k, linearization="outer"
    )
    assert fs.cone_facets == k and fs.cone_error_bound == pytest.approx(bound)
    lin = [r for r in fs.rows if r.kind == "cone_linear"]
    assert len(lin) == k
    worst = 0.0
    for phi in np.linspace(0, 2 * np.pi, 721):
        direction = np.array([0.0, 0.0, np.cos(phi), np.sin(phi)])
        # radius along direction: largest r with all linear rows satisfied.
        radius = np.inf
        for r in lin:
            slope = r.g @ direction
            if slope > 1e-12:
                radius = min(radius, r.h / slope)
        exact_radius = mu * 1.0
        assert radius >= exact_radius - 1e-12  # outer contains the exact cone
        worst = max(worst, radius / exact_radius - 1.0)
    assert worst <= bound + 1e-9
    assert worst == pytest.approx(bound, rel=1e-3)  # and the bound is attained


@pytest.mark.parametrize("k", [4, 8])
def test_inner_linearization_is_inscribed(k: int) -> None:
    mu = 0.5
    fs = FeasibleSet.from_hypotheses(
        _cone_decomp(),
        _hset(CONE),
        cones=[_cone(mu)],
        cone_facets=k,
        linearization="inner",
    )
    assert fs.cone_error_bound == pytest.approx(1.0 - np.cos(np.pi / k))
    lin = [r for r in fs.rows if r.kind == "cone_linear"]
    for phi in np.linspace(0, 2 * np.pi, 361):
        direction = np.array([0.0, 0.0, np.cos(phi), np.sin(phi)])
        radius = min(r.h / (r.g @ direction) for r in lin if r.g @ direction > 1e-12)
        assert radius <= mu + 1e-12
        assert radius >= mu * np.cos(np.pi / k) - 1e-12


def test_exact_socp_versus_linearized_lp_objective() -> None:
    mu, k = 0.5, 8
    d = _cone_decomp()
    fs = FeasibleSet.from_hypotheses(
        d, _hset(CONE), cones=[_cone(mu)], cone_facets=k, linearization="outer"
    )
    assert len(fs.cones) == 1
    obj = np.array([0.0, 0.0, -np.cos(np.pi / k), -np.sin(np.pi / k)])
    soc = fs.to_socp().solve(obj)
    lp = fs.to_lp().solve(obj)
    assert soc.objective == pytest.approx(-mu, abs=1e-6)  # exact radius mu
    assert lp.objective == pytest.approx(-mu / np.cos(np.pi / k), abs=1e-7)
    assert -lp.objective <= -soc.objective * (1 + fs.cone_error_bound) + 1e-7
    assert not fs.is_empty()


def test_socp_form_omits_linearized_rows_lp_form_omits_cones() -> None:
    fs = FeasibleSet.from_hypotheses(
        _cone_decomp(), _hset(CONE), cones=[_cone()], cone_facets=6
    )
    assert fs.to_socp().A_ub.shape[0] == 1  # unilateral normal row only
    assert fs.to_lp().A_ub.shape[0] == 1 + 6
    assert len(fs.to_socp().cones) == 1


def test_exact_cone_conflict_with_grip_is_found() -> None:
    d = _cone_decomp()
    # tangential demand x2 >= 0.9 contradicts ||t|| <= mu * n = 0.5.
    grip = GripPreload("hand_R", np.array([0.0, 0.0, 1.0, 0.0, 0.0]), 0.0, 0.9)
    fs = FeasibleSet.from_hypotheses(
        d, _hset(CONE, GRIP), cones=[_cone(0.5)], grip_preloads=[grip]
    )
    assert fs.is_empty()
    assert set(fs.conflict_set()) == {CONE, GRIP}
    assert f"{CONE}:foot:soc" in fs.conflict_rows()


def test_cone_validation() -> None:
    with pytest.raises(TypeError):
        ContactCone("x", "not a cone")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="name"):
        ContactCone("", FrictionCone((0, 1, 2), np.eye(3), 0.5))
    with pytest.raises(ValueError):
        cone_error_bound(2, "outer")
    with pytest.raises(ValueError):
        cone_error_bound(8, "sideways")


# -------------------------------------------------------------------- grip
def test_grip_preload_from_closing_map_matches_closing_hand_map() -> None:
    d = _decomp(np.zeros(8), np.eye(8), 2)
    dvec, net = np.array([0.0, 0.05, 0.0]), np.array([10, 5, 1, 0.5, 0.2, 0.1])
    for hand in ("L", "R"):
        gp = GripPreload.from_closing_map(
            hand, d.dim_x, lambda_start=2, d=dvec, net=net, minimum_squeeze_n=3.0
        )
        a_map, s0 = closing_hand_map(hand, dvec, net)
        lam = np.arange(1, 7, dtype=float)
        x = np.concatenate([[0, 0], lam])
        s_axial = float(a_map[0] @ lam + s0[0])
        assert gp.squeeze(x) == pytest.approx(-s_axial)
    assert gp.minimum_squeeze_n == 3.0


def test_grip_preload_rows() -> None:
    d = _decomp([0, 0, 2.0], np.array([[1.0], [0], [1.0]]), 2)
    gp = GripPreload("hand_L", np.array([0.0, 0.0, 1.0]), 1.0, 2.5)
    fs = FeasibleSet.from_hypotheses(d, _hset(GRIP), grip_preloads=[gp])
    (row,) = fs.rows
    # squeeze = x2 + 1 = (2 + z) + 1 >= 2.5   ->  -z <= 0.5
    assert row.g.tolist() == [-1.0] and row.h == pytest.approx(0.5)
    assert row.kind == "grip"
    with pytest.raises(ValueError, match="selector"):
        FeasibleSet.from_hypotheses(
            d,
            _hset(GRIP),
            grip_preloads=[GripPreload("g", np.zeros(7), 0.0, 1.0)],
        )


# ------------------------------------------------------ assumed / unverified
def test_assumed_capacity_slots_are_flagged_per_row() -> None:
    d = _decomp([0, 0], np.eye(2), 2)
    cap = _capacity([1, 1], [1, 1], assumed=("j1",))
    fs = FeasibleSet.from_hypotheses(d, _hset(TORQUE), capacity=cap)
    flagged = {r.label for r in fs.flagged_rows()}
    assert flagged == {f"{TORQUE}:j1:upper", f"{TORQUE}:j1:lower"}
    assert fs.assumed_slots == ("j1",)
    assert fs.rests_on_unverified
    clean = FeasibleSet.from_hypotheses(
        d, _hset(TORQUE), capacity=_capacity([1, 1], [1, 1])
    )
    assert not clean.rests_on_unverified and clean.flagged_rows() == ()


def test_unverified_hypothesis_flags_its_rows() -> None:
    d = _decomp([0, 0], np.eye(2), 2)
    fs = FeasibleSet.from_hypotheses(
        d, _hset(TORQUE, status="UNVERIFIED"), capacity=_capacity([1, 1], [1, 1])
    )
    assert fs.unverified_hypothesis_ids == (TORQUE,)
    assert len(fs.flagged_rows()) == 4 and fs.rests_on_unverified


def test_shipped_preset_with_real_capacity_flags_assumed_slots() -> None:
    names = ("HipInputX", "HipInputY")
    cap = capacity_for(names, CapacitySubject(78.0, 1.71), actuated=names)
    d = _decomp([0, 0], np.eye(2), 2)
    fs = FeasibleSet.from_hypotheses(
        d, registry.preset("envelope_only"), capacity=cap, cones=[], grip_preloads=[]
    )
    assert set(fs.hypothesis_ids) == set(
        registry.preset("envelope_only").hypothesis_ids()
    )
    assert fs.rests_on_unverified
    assert set(fs.assumed_slots) == set(cap.assumed_slots)
    assert fs.unverified_hypothesis_ids == ()  # the hypotheses themselves are sound


# ------------------------------------------------- real decomposition, misc
def test_real_decomposition_zero_nullcoordinate_is_feasible_with_wide_capacity() -> (
    None
):
    case = planar_golfer_snapshot(generic_planar_config(), actuation="full")
    snap = case.snapshot
    dec = decompose(snap, weights=np.eye(snap.topology.n_actuated))
    n_a = dec.n_actuated
    big = np.full(n_a, 1.0e6)
    fs = FeasibleSet.from_hypotheses(
        dec, _hset(TORQUE), capacity=_capacity(big, big)
    )
    assert fs.n_z == dec.nullity
    assert not fs.is_empty()
    tight = np.full(n_a, 1.0e-6)
    assert FeasibleSet.from_hypotheses(
        dec, _hset(TORQUE), capacity=_capacity(tight, tight)
    ).is_empty()


def test_feasible_set_is_immutable_and_rows_read_only() -> None:
    fs = FeasibleSet.from_hypotheses(
        _decomp([0, 0], np.eye(2), 2),
        _hset(TORQUE),
        capacity=_capacity([1, 1], [1, 1]),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        fs.n_z = 3  # type: ignore[misc]
    with pytest.raises(ValueError):
        fs.rows[0].g[0] = 7.0


def test_from_rows_validation() -> None:
    row = FeasibleRow(np.array([1.0, 0.0]), 1.0, "A:r", "A", "torque")
    with pytest.raises(ValueError, match="n_z"):
        FeasibleSet.from_rows(3, (row,))
    with pytest.raises(ValueError, match="duplicate"):
        FeasibleSet.from_rows(2, (row, row))
    with pytest.raises(ValueError, match="finite"):
        FeasibleRow(np.array([np.nan]), 1.0, "A:r", "A", "torque")
    with pytest.raises(ValueError, match="kind"):
        FeasibleRow(np.array([1.0]), 1.0, "A:r", "A", "mystery")
