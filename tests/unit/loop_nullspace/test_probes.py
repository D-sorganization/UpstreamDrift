"""Tests for certified affine probes and trajectory-coupled envelopes (NSX-25, #12568).

Oracles are the NSX-00 ``envelope_analytic_lp`` golden (exact rationals: the
static polygon, the infeasible and unbounded systems and the two-frame rate
coupling).  The verifier is exercised with plain numpy on the golden rows, not
through the solver, and tampered certificates must be rejected.
"""

from __future__ import annotations

import dataclasses
import json
from fractions import Fraction

import numpy as np
import pytest

pytest.importorskip("highspy")
pytest.importorskip("clarabel")

from src.shared.python.loop_nullspace.envelopes import (  # noqa: E402
    BoundStatus,
    EnvelopeQuantity,
    static_envelopes,
)
from src.shared.python.loop_nullspace.feasible_rows import (  # noqa: E402
    ConeBlock,
    FeasibleRow,
)
from src.shared.python.loop_nullspace.feasible_set import FeasibleSet  # noqa: E402
from src.shared.python.loop_nullspace.fixtures import load_golden  # noqa: E402
from src.shared.python.loop_nullspace.probes import (  # noqa: E402
    AffinePredicate,
    Certificate,
    NonAffinePredicateError,
    RateBound,
    Verdict,
    certify,
    trajectory_envelope,
    verify_certificate,
)
from src.shared.python.loop_nullspace.trajectory_system import (  # noqa: E402
    stack_system,
)

pytestmark = pytest.mark.unit

TOL = 1e-9
JCT = [[1, 0], [0, 1], [1, 1]]
CAP = [Fraction(1, 2), Fraction(3, 4), Fraction(1, 2)]
TAU1 = [Fraction(1, 2), Fraction(-1, 4), Fraction(1)]
TAU2 = [Fraction(0), Fraction(0), Fraction(1, 2)]


def _fs(g, h, labels) -> FeasibleSet:
    rows = tuple(
        FeasibleRow(np.asarray(gi, float), float(hi), lab, lab.split(":")[0], "torque")
        for gi, hi, lab in zip(g, h, labels, strict=True)
    )
    return FeasibleSet.from_rows(len(g[0]), rows)


def _polygon(tau) -> FeasibleSet:
    """``-c_i <= tau_i - (J^T lam)_i <= c_i`` as ``G lam <= h`` (golden layout)."""
    g, h, lab = [], [], []
    for i in range(3):
        g.append([-v for v in JCT[i]])
        h.append(CAP[i] - tau[i])
        lab.append(f"HCAP:cap_{i + 1}_upper")
        g.append(list(JCT[i]))
        h.append(CAP[i] + tau[i])
        lab.append(f"HCAP:cap_{i + 1}_lower")
    return _fs(g, h, lab)


def _golden_rows(key: str) -> FeasibleSet:
    rows = load_golden("envelope_analytic_lp", None).get(key)["rows"]
    g = [[float(Fraction(x)) for x in r["g"]] for r in rows]
    h = [float(Fraction(r["h"])) for r in rows]
    labels = [r.get("label", f"H{i}:row{i}") for i, r in enumerate(rows)]
    return _fs(g, h, labels)


def _lam(k: int, name: str | None = None, scale: float = 1.0) -> EnvelopeQuantity:
    e = np.zeros(2)
    e[k] = 1.0
    return EnvelopeQuantity(name or f"lam_{k + 1}", e, 0.0, "N", scale)


def _pred(k, sense, thr, **kw) -> AffinePredicate:
    return AffinePredicate(_lam(k), sense, thr, **kw)


# ------------------------------------------------------------- non-affine
def test_non_affine_predicates_are_rejected_at_construction() -> None:
    with pytest.raises(NonAffinePredicateError):
        AffinePredicate.from_function(
            lambda z: float(z @ z), 2, "ge", 0.0, units="N", scale=1.0
        )
    with pytest.raises(NonAffinePredicateError):
        AffinePredicate.from_function(
            lambda z: float(np.linalg.norm(z)), 2, "le", 1.0, units="N", scale=1.0
        )
    with pytest.raises(NonAffinePredicateError):
        AffinePredicate.from_function(
            lambda z: abs(float(z[0])), 2, "ge", 0.1, units="N", scale=1.0
        )
    assert issubclass(NonAffinePredicateError, ValueError)


def test_affine_function_becomes_exact_coefficients() -> None:
    pred = AffinePredicate.from_function(
        lambda z: 2.0 * z[0] - 3.0 * z[1] + 0.5, 2, "ge", 0.0, units="N", scale=1.0
    )
    q = pred.quantity_for(_polygon(TAU1))
    assert np.allclose(q.coeff, [2.0, -3.0], atol=1e-12)
    assert q.const == pytest.approx(0.5, abs=1e-12)


def test_predicate_argument_validation() -> None:
    with pytest.raises(TypeError):
        AffinePredicate("not a quantity", "ge", 0.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sense"):
        AffinePredicate(_lam(0), "gt", 0.0)
    with pytest.raises(ValueError, match="finite"):
        AffinePredicate(_lam(0), "ge", float("nan"))


def test_window_validation() -> None:
    sets = [_polygon(TAU1), _polygon(TAU2)]
    for bad in ([], [5], [0, 0], [-1], [0.5]):
        with pytest.raises((ValueError, TypeError)):
            certify(sets, _pred(0, "ge", -1.0), bad)
    with pytest.raises(ValueError):
        certify([], _pred(0, "ge", -1.0), None)


# ------------------------------------------------------- analytic verdicts
def test_holds_matches_golden_bound_and_verifies_with_numpy() -> None:
    gold = load_golden("envelope_analytic_lp", None).get("static")["envelopes"]
    cert = certify([_polygon(TAU1)], _pred(0, "ge", -0.25), [0])
    assert cert.verdict is Verdict.HOLDS
    assert cert.reason == ""
    assert cert.witness is None
    leaf = cert.leaves[0]
    assert leaf.lower_bound == pytest.approx(
        float(Fraction(gold["lam_1"]["lower"])), abs=TOL
    )
    assert verify_certificate(cert, [_polygon(TAU1)]).ok


def test_fails_returns_a_witness_that_satisfies_every_constraint() -> None:
    gold = load_golden("envelope_analytic_lp", None).get("static")["envelopes"]
    fs = _polygon(TAU1)
    cert = certify([fs], _pred(1, "ge", 0.0), [0])
    assert cert.verdict is Verdict.FAILS
    z = cert.witness.z[0]
    # independent re-check of the witness against the golden-derived rows
    g = np.array([r.g for r in fs.rows])
    h = np.array([r.h for r in fs.rows])
    assert np.all(g @ z <= h + 1e-9)
    assert z[1] == pytest.approx(float(Fraction(gold["lam_2"]["lower"])), abs=TOL)
    assert cert.witness.value == pytest.approx(z[1], abs=TOL)
    assert cert.witness.frame == 0
    assert np.allclose(
        z, [float(Fraction(v)) for v in gold["lam_2"]["argmin_vertex"]], atol=1e-7
    )
    assert verify_certificate(cert, [fs]).ok


def test_le_sense_and_boundary_is_undecided_not_holds() -> None:
    fs = [_polygon(TAU1)]
    assert certify(fs, _pred(0, "le", 1.5), [0]).verdict is Verdict.HOLDS
    assert certify(fs, _pred(0, "le", 0.5), [0]).verdict is Verdict.FAILS
    touching = certify(fs, _pred(0, "ge", 0.0), [0])  # min lam_1 == 0 exactly
    assert touching.verdict is Verdict.UNDECIDED
    assert touching.reason == "BOUNDARY"
    assert verify_certificate(touching, fs).ok


def test_window_selects_frames_and_all_frames_are_required() -> None:
    sets = [_polygon(TAU1), _polygon(TAU2)]
    pred = _pred(0, "ge", -0.25)  # frame 0 min 0, frame 1 min -1/2
    assert certify(sets, pred, [0]).verdict is Verdict.HOLDS
    both = certify(sets, pred, [0, 1])
    assert both.verdict is Verdict.FAILS
    assert both.witness.frame == 1
    assert certify(sets, pred, None).window == (0, 1)


def test_sum_aggregation_is_one_lp() -> None:
    sets = [_polygon(TAU1), _polygon(TAU2)]
    cert = certify(sets, _pred(0, "ge", -0.6), [0, 1], aggregation="SUM")
    assert len(cert.leaves) == 1
    assert cert.leaves[0].lower_bound == pytest.approx(0.0 + -0.5, abs=TOL)
    assert cert.verdict is Verdict.HOLDS
    assert verify_certificate(cert, sets).ok


# ----------------------------------------------------- trajectory coupling
def _rate(k: int, delta: float) -> RateBound:
    return RateBound(_lam(k), delta)


def test_coupling_changes_the_answer_and_matches_the_golden() -> None:
    gold = load_golden("envelope_analytic_lp", None).get("trajectory_coupled")
    delta = float(Fraction(gold["delta"]))
    sets = [_polygon(TAU1), _polygon(TAU2)]
    pred = _pred(0, "ge", -0.2)
    on_frame_2 = [1]
    assert certify(sets, pred, on_frame_2).verdict is Verdict.FAILS
    coupled = certify(
        sets, pred, on_frame_2, coupling=[_rate(0, delta), _rate(1, delta)]
    )
    assert coupled.verdict is Verdict.HOLDS
    assert coupled.leaves[0].lower_bound == pytest.approx(
        float(Fraction(gold["lam_1_frame_2"]["coupled_lower"])), abs=TOL
    )
    assert verify_certificate(
        coupled, sets, coupling=[_rate(0, delta), _rate(1, delta)]
    ).ok
    # the stated coupling is part of the hypotheses of the certificate
    assert "H-SMOOTH-TORQUE-RATE" in coupled.hypotheses


def test_trajectory_envelope_reproduces_the_coupled_golden() -> None:
    gold = load_golden("envelope_analytic_lp", None).get("trajectory_coupled")
    delta = float(Fraction(gold["delta"]))
    sets = [_polygon(TAU1), _polygon(TAU2)]
    quantities = [_lam(0), _lam(1)]
    env = trajectory_envelope(
        sets, quantities, coupling=[_rate(0, delta), _rate(1, delta)]
    )
    static = static_envelopes(sets, quantities)
    assert env.label == "TRAJECTORY_COUPLED"
    for j, name in enumerate(("lam_1", "lam_2")):
        for f in (0, 1):
            entry = gold[f"{name}_frame_{f + 1}"]
            assert env.lower[f, j] == pytest.approx(
                float(Fraction(entry["coupled_lower"])), abs=TOL
            )
            assert env.upper[f, j] == pytest.approx(
                float(Fraction(entry["coupled_upper"])), abs=TOL
            )
            assert env.lower[f, j] >= static.lower[f, j] - TOL
            assert env.upper[f, j] <= static.upper[f, j] + TOL
            assert env.status_lower[f][j] is BoundStatus.BOUNDED
    assert env.is_subset_of(static)
    assert env.strictly_narrower_somewhere(static)


def test_trajectory_envelope_without_coupling_equals_static() -> None:
    sets = [_polygon(TAU1), _polygon(TAU2)]
    qs = [_lam(0), _lam(1)]
    env = trajectory_envelope(sets, qs)
    static = static_envelopes(sets, qs)
    assert np.allclose(env.lower, static.lower, atol=TOL)
    assert np.allclose(env.upper, static.upper, atol=TOL)


def test_coupling_that_empties_the_set_is_reported() -> None:
    sets = [
        _fs([[1.0], [-1.0]], [0.0, 0.0], ["A:z0u", "A:z0l"]),
        _fs([[-1.0]], [-5.0], ["B:z1l"]),
    ]  # frame0: z=0 ; frame1: z>=5
    q = EnvelopeQuantity("z", [1.0], 0.0, "N", 1.0)
    cert = certify(
        sets, AffinePredicate(q, "ge", -1.0), [0, 1], coupling=[RateBound(q, 1.0)]
    )
    assert cert.verdict is Verdict.UNDECIDED
    assert cert.reason == "EMPTY_FEASIBLE_SET"
    assert verify_certificate(cert, sets, coupling=[RateBound(q, 1.0)]).ok


# ------------------------------------------------- infeasible / unbounded
def test_infeasible_golden_gives_a_farkas_certificate_matching_the_golden() -> None:
    gold = load_golden("envelope_analytic_lp", None).get("infeasible")
    fs = _golden_rows("infeasible")
    cert = certify([fs], _pred(0, "ge", 0.0), [0])
    assert cert.verdict is Verdict.UNDECIDED
    assert cert.reason == "EMPTY_FEASIBLE_SET"
    leaf = cert.leaves[0]
    y_gold = np.array([float(Fraction(v)) for v in gold["farkas_multipliers"]])
    mu = np.asarray(leaf.dual)
    assert np.allclose(mu / mu.sum(), y_gold / y_gold.sum(), atol=1e-7)
    h = np.array([r.h for r in fs.rows])
    assert float(h @ mu) < -1e-3
    assert verify_certificate(cert, [fs]).ok
    # plain numpy: y >= 0, y^T G = 0, y^T h < 0
    g = np.array([r.g for r in fs.rows])
    assert np.all(mu >= -1e-12) and np.max(np.abs(g.T @ mu)) < 1e-9


def test_unbounded_golden_fails_with_witness_and_bounded_direction_holds() -> None:
    fs = _golden_rows("unbounded")
    unbounded = certify([fs], _pred(1, "ge", -1.0), [0])  # lam_2 unbounded below
    assert unbounded.verdict is Verdict.FAILS
    z = unbounded.witness.z[0]
    assert z[0] >= -1e-9 and z[0] <= 1.0 + 1e-9 and z[1] < -1.0
    assert verify_certificate(unbounded, [fs]).ok
    bounded = certify([fs], _pred(0, "ge", -0.5), [0])
    assert bounded.verdict is Verdict.HOLDS


# ------------------------------------------------------------ exact cones
def _disk_set() -> FeasibleSet:
    a = np.vstack([np.zeros((1, 2)), -np.eye(2)])
    b = np.array([1.0, 0.0, 0.0])
    return FeasibleSet.from_rows(2, (), (ConeBlock(a, b, "C:disk", "H-FRICTION-CONE"),))


def test_exact_cone_certificate_has_a_conic_dual_and_verifies() -> None:
    fs = _disk_set()
    holds = certify([fs], _pred(0, "ge", -1.2), [0])
    assert holds.verdict is Verdict.HOLDS
    assert holds.leaves[0].lower_bound == pytest.approx(-1.0, abs=1e-6)
    assert verify_certificate(holds, [fs]).ok
    fails = certify([fs], _pred(0, "ge", -0.8), [0])
    assert fails.verdict is Verdict.FAILS
    z = fails.witness.z[0]
    assert float(np.linalg.norm(z)) <= 1.0 + 1e-7 and z[0] < -0.8
    assert verify_certificate(fails, [fs]).ok


# --------------------------------------------------------- verifier rigour
def _holds_cert():
    sets = [_polygon(TAU1)]
    return certify(sets, _pred(0, "ge", -0.25), [0]), sets


def test_verifier_rejects_a_tampered_dual() -> None:
    cert, sets = _holds_cert()
    leaf = cert.leaves[0]
    bad = dataclasses.replace(leaf, dual=np.asarray(leaf.dual) * 0.5)
    tampered = dataclasses.replace(cert, leaves=(bad,))
    report = verify_certificate(tampered, sets)
    assert not report.ok
    assert any("stationarity" in m or "bound" in m for m in report.messages)


def test_verifier_rejects_a_witness_outside_the_set() -> None:
    sets = [_polygon(TAU1)]
    cert = certify(sets, _pred(1, "ge", 0.0), [0])
    leaf = cert.leaves[0]
    z = np.array([5.0, -5.0])
    bad_leaf = dataclasses.replace(leaf, witness=z, witness_value=-5.0)
    bad = dataclasses.replace(
        cert,
        leaves=(bad_leaf,),
        witness=dataclasses.replace(cert.witness, z=(z,), value=-5.0),
    )
    assert not verify_certificate(bad, sets).ok


def test_verifier_rejects_a_certificate_for_different_sets() -> None:
    cert, _ = _holds_cert()
    report = verify_certificate(cert, [_polygon(TAU2)])
    assert not report.ok
    assert any("digest" in m for m in report.messages)


def test_verifier_rejects_a_negative_multiplier() -> None:
    cert, sets = _holds_cert()
    leaf = cert.leaves[0]
    dual = np.asarray(leaf.dual).copy()
    dual[int(np.argmax(dual))] *= -1.0
    bad = dataclasses.replace(cert, leaves=(dataclasses.replace(leaf, dual=dual),))
    assert not verify_certificate(bad, sets).ok


# ---------------------------------------------------------- serialisation
def test_certificate_json_round_trips_exactly() -> None:
    sets = [_polygon(TAU1), _polygon(TAU2)]
    for cert in (
        certify(sets, _pred(0, "ge", -0.25), [0]),
        certify(sets, _pred(1, "ge", 0.0), [0, 1]),
        certify([_golden_rows("infeasible")], _pred(0, "ge", 0.0), [0]),
        certify([_disk_set()], _pred(0, "ge", -0.8), [0]),
    ):
        text = cert.to_json()
        json.loads(text)  # valid JSON
        back = Certificate.from_json(text)
        assert back.to_json() == text
        assert back.verdict is cert.verdict
        for a, b in zip(cert.leaves, back.leaves, strict=True):
            for name in ("c", "witness", "dual"):
                x, y = getattr(a, name), getattr(b, name)
                assert (x is None) == (y is None)
                if x is not None:
                    assert np.array_equal(np.asarray(x), np.asarray(y))
            assert a.lower_bound == b.lower_bound
        assert back.system_digest == cert.system_digest
    reloaded = Certificate.from_json(
        certify(sets, _pred(1, "ge", 0.0), [0, 1]).to_json()
    )
    assert verify_certificate(reloaded, sets).ok


def test_from_json_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        Certificate.from_json("{}")
    with pytest.raises(ValueError):
        Certificate.from_json("not json")


# --------------------------------------------------------------- evidence
def test_certificate_is_family_dependent_and_names_its_hypotheses() -> None:
    cert, _ = _holds_cert()
    assert cert.hypotheses == ("HCAP",)
    assert "hypotheses" in cert.non_claim.lower()
    assert cert.record.allocation_dependence.name == "FAMILY_DEPENDENT"
    assert cert.record.hypotheses == ("HCAP",)


def test_system_digest_is_deterministic_and_sensitive() -> None:
    a = stack_system([_polygon(TAU1)])
    b = stack_system([_polygon(TAU1)])
    c = stack_system([_polygon(TAU2)])
    assert a.digest == b.digest != c.digest
