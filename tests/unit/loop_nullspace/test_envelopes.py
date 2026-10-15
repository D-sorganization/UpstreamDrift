"""Tests for static torque and hand-split envelopes (NSX-22, #12565).

Every bound is a pair of LPs over the declared feasible set.  An unbounded
direction is NaN with status UNBOUNDED, the sign class is decided with an
explicit scaled tolerance, and each bound names the rows that are active at it.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from fractions import Fraction

import numpy as np
import pytest

pytest.importorskip("highspy")
pytest.importorskip("clarabel")

from src.shared.python.loop_nullspace.decomposition import (  # noqa: E402
    decompose,
)
from src.shared.python.loop_nullspace.envelopes import (  # noqa: E402
    SIGN_ATOL_RTOL,
    BoundStatus,
    EnvelopeQuantity,
    SignClass,
    chebyshev_margin,
    classify_sign,
    joint_torque_quantities,
    static_envelopes,
)
from src.shared.python.loop_nullspace.evidence import (  # noqa: E402
    AllocationDependence,
    InputProvenance,
    ModelProvenance,
)
from src.shared.python.loop_nullspace.feasible_rows import (  # noqa: E402
    FLAG_ASSUMED,
    FeasibleRow,
)
from src.shared.python.loop_nullspace.feasible_set import FeasibleSet  # noqa: E402
from src.shared.python.loop_nullspace.fixtures import (  # noqa: E402
    analytic_envelope_case,
    generic_planar_config,
    load_golden,
    planar_golfer_snapshot,
)

pytestmark = pytest.mark.unit

TOL = 1e-9
NAMES = ("tau_1", "tau_2", "tau_3", "lam_1", "lam_2")


def _fs(rows_g, rows_h, labels, flags=()) -> FeasibleSet:
    rows = tuple(
        FeasibleRow(
            np.asarray(g, float),
            float(h),
            lab,
            lab.split(":")[0],
            "torque",
            tuple(flags),
        )
        for g, h, lab in zip(rows_g, rows_h, labels, strict=True)
    )
    return FeasibleSet.from_rows(len(rows_g[0]), rows)


def _analytic_set() -> FeasibleSet:
    case = analytic_envelope_case()
    labels = [f"H{i}:{lab}" for i, lab in enumerate(case.row_labels)]
    return _fs(case.polytope_g, case.polytope_h, labels)


def _analytic_quantities(units: str = "N m") -> list[EnvelopeQuantity]:
    case = analytic_envelope_case()
    out = []
    for name in NAMES:
        coeff, const = case.affine_functional(name)
        out.append(EnvelopeQuantity(name, coeff, const, units, scale=5.0))
    return out


# --------------------------------------------------------------- analytic
def test_matches_analytic_envelope_to_1e9() -> None:
    case = analytic_envelope_case()
    series = static_envelopes([_analytic_set()], _analytic_quantities())
    assert series.label == "STATIC"
    assert series.names == NAMES
    for j, name in enumerate(NAMES):
        b = case.bounds[name]
        assert series.lower[0, j] == pytest.approx(float(b.lower), abs=TOL)
        assert series.upper[0, j] == pytest.approx(float(b.upper), abs=TOL)
        assert series.status_lower[0][j] is BoundStatus.BOUNDED
        assert series.status_upper[0][j] is BoundStatus.BOUNDED
        assert series.sign_class[0][j].name == b.sign_class


@pytest.mark.parametrize(
    ("key", "weights"),
    [("euclidean", None), ("weighted_diag_1_4", [1.0, 4.0])],
)
def test_chebyshev_margin_matches_golden_in_declared_metric(key, weights) -> None:
    gold = load_golden("envelope_analytic_lp", None).get("static")["chebyshev"][key]
    res = chebyshev_margin(_analytic_set(), metric_weights=weights)
    assert res.status is BoundStatus.BOUNDED
    assert res.radius == pytest.approx(gold["radius_float"], abs=TOL)
    assert res.center == pytest.approx(gold["center_float"], abs=1e-7)
    assert res.metric_label == ("euclidean" if weights is None else "diag(1,4)")


def test_series_reports_margin_per_frame() -> None:
    series = static_envelopes([_analytic_set()], _analytic_quantities())
    gold = load_golden("envelope_analytic_lp", None).get("static")["chebyshev"]
    assert series.chebyshev_margin[0] == pytest.approx(
        gold["euclidean"]["radius_float"], abs=TOL
    )
    assert series.chebyshev_metric == "euclidean"


def test_vertex_attaining_bounds_are_active_rows_with_hypothesis_ids() -> None:
    series = static_envelopes([_analytic_set()], _analytic_quantities())
    j = NAMES.index("lam_1")
    # lam_1 lower = 0 is attained on the edge z1 = 0, i.e. row cap_1_upper.
    assert "H0:cap_1_upper" in series.binding_lower[0][j]
    assert "H1:cap_1_lower" in series.binding_upper[0][j]
    assert "H0" in series.binding_hypotheses_lower[0][j]
    assert "H1" in series.binding_hypotheses_upper[0][j]


def test_sign_classes_match_golden_cases_with_tolerance() -> None:
    cases = load_golden("envelope_analytic_lp", None).get("static")["sign_class_cases"]
    for c in cases:
        lo, hi = float(Fraction(c["lower"])), float(Fraction(c["upper"]))
        got = classify_sign(lo, hi, scale=5.0)
        assert got.name == c["sign_class"], c


def test_sign_tolerance_is_scaled_not_bare_zero() -> None:
    scale = 5.0
    atol = SIGN_ATOL_RTOL * scale
    eps = 0.5 * atol
    assert classify_sign(-eps, eps, scale=scale) is SignClass.ZERO
    assert classify_sign(eps, 1.0, scale=scale) is SignClass.INDETERMINATE
    assert classify_sign(-1.0, -eps, scale=scale) is SignClass.INDETERMINATE
    assert classify_sign(2 * atol, 1.0, scale=scale) is SignClass.POSITIVE
    assert classify_sign(-1.0, -2 * atol, scale=scale) is SignClass.NEGATIVE
    # the same absolute value is decisive at a smaller characteristic scale
    assert classify_sign(2 * atol, 1.0, scale=0.1) is SignClass.POSITIVE
    assert classify_sign(eps, 1.0, scale=100.0) is SignClass.INDETERMINATE


def test_sign_class_one_sided_bound_still_decisive() -> None:
    assert classify_sign(1.0, np.nan, scale=5.0) is SignClass.POSITIVE
    assert classify_sign(np.nan, -1.0, scale=5.0) is SignClass.NEGATIVE
    assert classify_sign(np.nan, 1.0, scale=5.0) is SignClass.INDETERMINATE
    assert classify_sign(np.nan, np.nan, scale=5.0) is SignClass.INDETERMINATE


def test_classify_sign_validates() -> None:
    with pytest.raises(ValueError, match="scale"):
        classify_sign(0.0, 1.0, scale=0.0)
    with pytest.raises(ValueError, match="lower"):
        classify_sign(2.0, 1.0, scale=1.0)


# ------------------------------------------------------ unbounded/infeasible
def _unbounded_set() -> FeasibleSet:
    gold = load_golden("envelope_analytic_lp", None).get("unbounded")
    g = [[float(Fraction(x)) for x in r["g"]] for r in gold["rows"]]
    h = [float(Fraction(r["h"])) for r in gold["rows"]]
    return _fs(g, h, ["H0:lam1_lower", "H1:lam1_upper"])


def test_unbounded_directions_are_nan_with_status() -> None:
    gold = load_golden("envelope_analytic_lp", None).get("unbounded")["statuses"]
    quantities = [
        EnvelopeQuantity("lam_1", [1.0, 0.0], 0.0, "N", scale=100.0),
        EnvelopeQuantity("lam_2", [0.0, 1.0], 0.0, "N", scale=100.0),
        EnvelopeQuantity("tau_3", [0.0, -1.0], 0.5, "N m", scale=5.0),
    ]
    series = static_envelopes([_unbounded_set()], quantities)
    assert series.lower[0, 0] == pytest.approx(0.0, abs=TOL)
    assert series.upper[0, 0] == pytest.approx(1.0, abs=TOL)
    for j, name in ((1, "lam_2"), (2, "tau_3")):
        assert gold[name]["status_lower"] == "UNBOUNDED"
        assert np.isnan(series.lower[0, j]) and np.isnan(series.upper[0, j])
        assert series.status_lower[0][j] is BoundStatus.UNBOUNDED
        assert series.status_upper[0][j] is BoundStatus.UNBOUNDED
        assert series.sign_class[0][j] is SignClass.INDETERMINATE
        assert series.binding_lower[0][j] == ()


def test_unbounded_set_chebyshev_is_unbounded_not_a_big_number() -> None:
    res = chebyshev_margin(_unbounded_set())
    assert res.status is BoundStatus.UNBOUNDED
    assert np.isnan(res.radius)


def test_infeasible_frame_is_reported_not_raised() -> None:
    gold = load_golden("envelope_analytic_lp", None).get("infeasible")
    g = [[float(Fraction(x)) for x in r["g"]] for r in gold["rows"]]
    h = [float(Fraction(r["h"])) for r in gold["rows"]]
    labs = [r["label"] for r in gold["rows"]]
    series = static_envelopes(
        [_fs(g, h, labs), _analytic_set()],
        [EnvelopeQuantity("lam_1", [1.0, 0.0], 0.0, "N", scale=100.0)],
    )
    assert series.status_lower[0][0] is BoundStatus.INFEASIBLE
    assert series.status_upper[0][0] is BoundStatus.INFEASIBLE
    assert np.isnan(series.lower[0, 0]) and np.isnan(series.upper[0, 0])
    assert series.sign_class[0][0] is None
    assert series.chebyshev_status[0] is BoundStatus.INFEASIBLE
    assert np.isnan(series.chebyshev_margin[0])
    assert series.status_lower[1][0] is BoundStatus.BOUNDED  # next frame unaffected


# -------------------------------------------------------------- warm start
def _frame_sets(n: int) -> list[FeasibleSet]:
    rng = np.random.default_rng(7)
    sets = []
    for k in range(n):
        ang = np.linspace(0, 2 * np.pi, 9)[:-1] + 0.05 * k
        g = np.c_[np.cos(ang), np.sin(ang), 0.3 * np.cos(2 * ang)]
        h = 1.0 + 0.1 * np.sin(0.3 * k) + 0.05 * rng.random(8)
        sets.append(_fs(g, h, [f"H{i}:r{i}" for i in range(8)]))
    return sets


def test_warm_start_equals_cold_start() -> None:
    sets = _frame_sets(12)
    quantities = [
        EnvelopeQuantity(f"q{i}", e, 0.1 * i, "N m", scale=5.0)
        for i, e in enumerate(np.eye(3))
    ]
    warm = static_envelopes(sets, quantities, warm_start=True)
    cold = static_envelopes(sets, quantities, warm_start=False)
    assert np.allclose(warm.lower, cold.lower, atol=TOL, rtol=0)
    assert np.allclose(warm.upper, cold.upper, atol=TOL, rtol=0)
    assert warm.status_lower == cold.status_lower
    assert warm.sign_class == cold.sign_class
    assert warm.simplex_iterations <= cold.simplex_iterations
    assert warm.warm_started and not cold.warm_started


# ---------------------------------------------------------- evidence/flags
def test_flags_propagate_from_binding_rows_only() -> None:
    clean = FeasibleRow(np.array([1.0]), 1.0, "A:up", "A", "torque")
    flagged = FeasibleRow(
        np.array([-1.0]), 2.0, "B:lo", "B", "torque", (FLAG_ASSUMED,)
    )
    series = static_envelopes(
        [FeasibleSet.from_rows(1, (clean, flagged))],
        [EnvelopeQuantity("x", [1.0], 0.0, "N", scale=100.0)],
    )
    assert series.upper[0, 0] == pytest.approx(1.0, abs=TOL)
    assert series.lower[0, 0] == pytest.approx(-2.0, abs=TOL)
    assert series.upper_rests_on_unverified[0][0] is False
    assert series.lower_rests_on_unverified[0][0] is True
    assert series.rests_on_unverified[0] is False or series.rests_on_unverified[0]
    assert series.frame_rests_on_unverified == (True,)


def test_records_are_family_dependent_with_hypotheses() -> None:
    series = static_envelopes([_analytic_set()], _analytic_quantities())
    rec = series.records(
        ModelProvenance("m", "1"), InputProvenance("fixture", "analytic")
    )
    r = rec["tau_1"]
    assert r.allocation_dependence is AllocationDependence.FAMILY_DEPENDENT
    assert r.values.shape == (1, 2)
    assert r.units == "N m"
    assert "STATIC" in " ".join(r.conventions)
    assert "H0" in r.hypotheses


def test_records_unbounded_are_unavailable_with_reason() -> None:
    series = static_envelopes(
        [_unbounded_set()],
        [EnvelopeQuantity("lam_2", [0.0, 1.0], 0.0, "N", scale=100.0)],
    )
    rec = series.records(ModelProvenance("m", "1"), InputProvenance("f", "s"))["lam_2"]
    assert rec.allocation_dependence is AllocationDependence.UNAVAILABLE
    assert "UNBOUNDED" in rec.reason


# ------------------------------------------------------------------- DbC
def test_validation_errors() -> None:
    q = [EnvelopeQuantity("a", [1.0, 0.0], 0.0, "N", scale=1.0)]
    with pytest.raises(ValueError, match="at least one"):
        static_envelopes([], q)
    with pytest.raises(ValueError, match="quantit"):
        static_envelopes([_analytic_set()], [])
    with pytest.raises(TypeError, match="FeasibleSet"):
        static_envelopes([object()], q)
    with pytest.raises(ValueError, match="columns|length|n_z"):
        static_envelopes(
            [_analytic_set()], [EnvelopeQuantity("a", [1.0], 0.0, "N", scale=1.0)]
        )
    dup = q + q
    with pytest.raises(ValueError, match="duplicate"):
        static_envelopes([_analytic_set()], dup)
    with pytest.raises(ValueError, match="scale"):
        EnvelopeQuantity("a", [1.0], 0.0, "N", scale=-1.0)
    with pytest.raises(ValueError, match="finite"):
        EnvelopeQuantity("a", [np.nan], 0.0, "N", scale=1.0)
    with pytest.raises(ValueError, match="positive"):
        chebyshev_margin(_analytic_set(), metric_weights=[1.0, 0.0])
    with pytest.raises(ValueError, match="length"):
        chebyshev_margin(_analytic_set(), metric_weights=[1.0])


def test_zero_dimensional_set() -> None:
    ok = FeasibleSet.from_rows(0, ())
    series = static_envelopes(
        [ok], [EnvelopeQuantity("c", np.zeros(0), 2.0, "N", scale=100.0)]
    )
    assert series.lower[0, 0] == series.upper[0, 0] == 2.0
    assert series.sign_class[0][0] is SignClass.POSITIVE
    assert series.chebyshev_margin[0] == 0.0


# ------------------------------------------------- decomposition integration
def _planar_decomposition():
    case = planar_golfer_snapshot(generic_planar_config(), actuation="full")
    snap = case.snapshot
    return decompose(snap, weights=np.eye(snap.topology.n_actuated))


def test_joint_torque_quantities_from_decomposition() -> None:
    dec = _planar_decomposition()
    n_a = dec.n_actuated
    big = 1.0e3
    rows = []
    for i in range(n_a):
        rows.append(
            FeasibleRow(dec.N[i], big - dec.x_p[i], f"T:{i}:up", "T", "torque")
        )
        rows.append(
            FeasibleRow(-dec.N[i], big + dec.x_p[i], f"T:{i}:lo", "T", "torque")
        )
    fs = dataclasses.replace(FeasibleSet.from_rows(dec.nullity, rows), decomposition=dec)
    quantities = joint_torque_quantities()(fs)
    assert len(quantities) == n_a
    series = static_envelopes([fs], quantities)
    z = np.zeros(dec.nullity)
    for j in range(n_a):
        assert series.lower[0, j] <= dec.torque(z)[j] + TOL
        assert series.upper[0, j] >= dec.torque(z)[j] - TOL
        assert quantities[j].units == "N m"
    with pytest.raises(ValueError, match="decomposition"):
        joint_torque_quantities()(FeasibleSet.from_rows(1, ()))


# ------------------------------------------------------------- benchmark
def test_200_frame_envelopes_time_budget_recorded(caplog) -> None:
    """Benchmark, recorded not gated: 200 frames of the planar anthropometric proxy."""
    dec = _planar_decomposition()
    n_a = dec.n_actuated
    frames = []
    for k in range(200):
        xp = dec.x_p * (1.0 + 0.5 * np.sin(2 * np.pi * k / 200))
        d = dataclasses.replace(dec, x_p=xp)
        cap = 60.0 + 10.0 * np.cos(2 * np.pi * k / 200)
        rows = []
        for i in range(n_a):
            rows.append(FeasibleRow(d.N[i], cap - xp[i], f"T:{i}:up", "T", "torque"))
            rows.append(FeasibleRow(-d.N[i], cap + xp[i], f"T:{i}:lo", "T", "torque"))
        frames.append(
            dataclasses.replace(FeasibleSet.from_rows(d.nullity, rows), decomposition=d)
        )
    t0 = time.perf_counter()
    series = static_envelopes(frames, joint_torque_quantities())
    elapsed = time.perf_counter() - t0
    with caplog.at_level(logging.INFO):
        logging.getLogger(__name__).info(
            "NSX-22 benchmark: %d frames x %d quantities in %.3f s (%d simplex iters)",
            len(frames),
            len(series.names),
            elapsed,
            series.simplex_iterations,
        )
    assert series.lower.shape == (200, n_a)
    assert np.isfinite(series.lower).all() or np.isnan(series.lower).any()
    print(f"\nNSX-22 benchmark: 200 frames in {elapsed:.3f} s")
