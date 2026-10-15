"""Tests for the trajectory conic allocation (NSX-23, #12566).

Oracles are independent of the allocator: closed-form least squares for the
quadratic terms, ``scipy`` SLSQP from perturbed starts for the active-set case,
the NSX-00 c-plus/c-minus identity for the gross-power term, and random
feasible perturbations for local (= global, by convexity) optimality of the
piecewise-linear objective.  There is deliberately no "zero sign flips"
criterion (review A1: circular).
"""

from __future__ import annotations

import dataclasses
import logging
import time

import numpy as np
import pytest
from scipy.optimize import minimize

pytest.importorskip("highspy")
pytest.importorskip("clarabel")

from src.shared.python.loop_nullspace.allocation import (  # noqa: E402
    AllocationResult,
    GrossPowerSpec,
    NotStrictlyConvexError,
    ObjectiveSpec,
    allocate_trajectory,
)
from src.shared.python.loop_nullspace.decomposition import (  # noqa: E402
    Decomposition,
    decompose,
)
from src.shared.python.loop_nullspace.evidence import (  # noqa: E402
    AllocationDependence,
)
from src.shared.python.loop_nullspace.feasible_rows import (  # noqa: E402
    ConeBlock,
    FeasibleRow,
)
from src.shared.python.loop_nullspace.feasible_set import FeasibleSet  # noqa: E402
from src.shared.python.loop_nullspace.fixtures import (  # noqa: E402
    PlanarGolferConfig,
    load_golden,
    planar_golfer_snapshot,
)
from src.shared.python.loop_nullspace.solvers import SolveStatus  # noqa: E402

pytestmark = pytest.mark.unit

SCALE = np.array([20.0, 35.0, 50.0, 65.0, 80.0])
N_A = 5


def _base():
    """Planar golfer at the NSX-00 ``c_pm_identity`` state (negative power is unavoidable)."""
    gold = load_golden("c_pm_identity", None)
    cfg = PlanarGolferConfig(gold.array("q"), gold.array("v"), gold.array("a"))
    snap = planar_golfer_snapshot(cfg, actuation="full").snapshot
    dec = decompose(snap, weights=np.eye(snap.topology.n_actuated))
    return dec, np.asarray(snap.v, dtype=float)


def _fs(dec: Decomposition, cap: float, *, cone_radius: float | None = None):
    """Capacity rows (and an optional ball ``|z| <= r`` cone) for one frame."""
    n_a = dec.n_actuated
    rows = []
    for i in range(n_a):
        rows.append(
            FeasibleRow(
                dec.N[i], cap - dec.x_p[i], f"T:{i}:up", "H-TORQUE-CAPACITY", "torque"
            )
        )
        rows.append(
            FeasibleRow(
                -dec.N[i], cap + dec.x_p[i], f"T:{i}:lo", "H-TORQUE-CAPACITY", "torque"
            )
        )
    cones = ()
    if cone_radius is not None:
        a = np.vstack([np.zeros((1, dec.nullity)), -np.eye(dec.nullity)])
        b = np.concatenate([[cone_radius], np.zeros(dec.nullity)])
        cones = (ConeBlock(a, b, "C:ball", "H-FRICTION-CONE"),)
    fs = FeasibleSet.from_rows(dec.nullity, rows, cones)
    return dataclasses.replace(fs, decomposition=dec)


def _trajectory(
    t_count: int,
    cap: float,
    *,
    cone_radius: float | None = None,
    relative: bool = False,
):
    """Smoothly varying frames on the planar decomposition and joint velocities.

    ``relative`` makes ``cap`` a fraction of the frame's largest particular torque.
    """
    dec, v = _base()
    frames, vel = [], []
    for k in range(t_count):
        phase = 2 * np.pi * k / max(t_count, 1)
        d = dataclasses.replace(dec, x_p=dec.x_p * (1.0 + 0.4 * np.sin(phase)))
        bound = cap * float(np.max(np.abs(d.x_p[:N_A]))) if relative else cap
        frames.append(_fs(d, bound, cone_radius=cone_radius))
        vel.append(v * (1.0 + 0.5 * np.cos(phase)))
    return frames, np.array(vel)


def _spec(t_count: int, **kw) -> ObjectiveSpec:
    return ObjectiveSpec(
        times_s=np.linspace(0.0, 0.1 * max(t_count - 1, 1), t_count),
        torque_scale=SCALE,
        **kw,
    )


def _frame_weights(times: np.ndarray) -> np.ndarray:
    """Trapezoid weights, written independently of the implementation."""
    if times.size == 1:
        return np.ones(1)
    w = np.zeros(times.size)
    d = np.diff(times)
    w[:-1] += d / 2
    w[1:] += d / 2
    return w


def _lstsq_oracle(frames, spec, ref=None):
    """Closed-form optimum of the quadratic terms with inactive constraints."""
    t_count = len(frames)
    w = _frame_weights(spec.times_s)
    nz = [f.n_z for f in frames]
    off = np.concatenate([[0], np.cumsum(nz)])
    rows, rhs = [], []
    for k, f in enumerate(frames):
        d = f.decomposition
        blk = np.zeros((N_A, off[-1]))
        blk[:, off[k] : off[k + 1]] = d.N[:N_A]
        coef = np.sqrt(spec.effort_weight * w[k]) / SCALE[:, None]
        rows.append(coef * blk)
        rhs.append(-coef[:, 0] * d.x_p[:N_A])
        if ref is not None:
            c2 = np.sqrt(spec.reference_weight * w[k]) / SCALE[:, None]
            rows.append(c2 * blk)
            rhs.append(c2[:, 0] * (ref[k] - d.x_p[:N_A]))
    for k in range(t_count - 1):
        dt = spec.times_s[k + 1] - spec.times_s[k]
        d0, d1 = frames[k].decomposition, frames[k + 1].decomposition
        blk = np.zeros((N_A, off[-1]))
        blk[:, off[k + 1] : off[k + 2]] += d1.N[:N_A]
        blk[:, off[k] : off[k + 1]] -= d0.N[:N_A]
        coef = np.sqrt(spec.rate_weight / dt) / SCALE[:, None]
        rows.append(coef * blk)
        rhs.append(-coef[:, 0] * (d1.x_p[:N_A] - d0.x_p[:N_A]))
    z = np.linalg.lstsq(np.vstack(rows), np.concatenate(rhs), rcond=None)[0]
    return np.array(
        [
            frames[k].decomposition.x_p[:N_A]
            + frames[k].decomposition.N[:N_A] @ z[off[k] : off[k + 1]]
            for k in range(t_count)
        ]
    )


# ----------------------------------------------------------- closed forms
def test_effort_only_matches_closed_form_per_frame() -> None:
    frames, _ = _trajectory(3, cap=1.0e4)
    spec = _spec(3)
    res = allocate_trajectory(frames, spec)
    assert res.status is SolveStatus.OPTIMAL
    assert res.torque == pytest.approx(_lstsq_oracle(frames, spec), abs=1e-6)


def test_effort_rate_and_reference_match_stacked_least_squares() -> None:
    frames, _ = _trajectory(6, cap=1.0e4)
    spec = _spec(6, effort_weight=0.7, rate_weight=2.5, reference_weight=1.3)
    ref = np.array([f.decomposition.x_p[:N_A] * 0.4 + 3.0 for f in frames])
    res = allocate_trajectory(frames, spec, reference=ref)
    assert res.status is SolveStatus.OPTIMAL
    assert res.torque == pytest.approx(_lstsq_oracle(frames, spec, ref), abs=1e-6)
    assert set(res.objective_terms) == {"effort", "rate", "reference", "gross_power"}


def test_reference_term_recovers_a_feasible_reference() -> None:
    frames, _ = _trajectory(4, cap=1.0e3)
    rng = np.random.default_rng(7)
    ref = np.array(
        [
            f.decomposition.x_p[:N_A]
            + f.decomposition.N[:N_A] @ rng.uniform(-1.0, 1.0, f.n_z)
            for f in frames
        ]
    )
    spec = _spec(4, effort_weight=0.0, reference_weight=1.0)
    res = allocate_trajectory(frames, spec, reference=ref)
    assert res.torque == pytest.approx(ref, abs=1e-6)
    assert res.objective_terms["reference"] == pytest.approx(0.0, abs=1e-8)


def test_infeasible_reference_is_projected_onto_the_set() -> None:
    frames, _ = _trajectory(2, 0.85, relative=True)
    caps = np.array([0.85 * np.max(np.abs(f.decomposition.x_p[:N_A])) for f in frames])
    ref = np.full((2, N_A), 500.0)  # far outside the capacity box
    spec = _spec(2, effort_weight=0.0, reference_weight=1.0)
    res = allocate_trajectory(frames, spec, reference=ref)
    assert res.status is SolveStatus.OPTIMAL
    assert np.all(np.abs(res.torque) <= caps[:, None] + 1e-6)
    assert res.objective_terms["reference"] > 1.0


# ----------------------------------------------------------- uniqueness
def _scipy_optimum(frames, spec, start):
    """Reference solution with a generic NLP from an arbitrary start."""
    w = _frame_weights(spec.times_s)
    nz = [f.n_z for f in frames]
    off = np.concatenate([[0], np.cumsum(nz)])

    def tau(zv, k):
        d = frames[k].decomposition
        return d.x_p[:N_A] + d.N[:N_A] @ zv[off[k] : off[k + 1]]

    def cost(zv):
        c = sum(
            spec.effort_weight * w[k] * np.sum((tau(zv, k) / SCALE) ** 2)
            for k in range(len(frames))
        )
        for k in range(len(frames) - 1):
            dt = spec.times_s[k + 1] - spec.times_s[k]
            c += (
                spec.rate_weight
                / dt
                * np.sum(((tau(zv, k + 1) - tau(zv, k)) / SCALE) ** 2)
            )
        return c

    cons = []
    for k, f in enumerate(frames):
        g = np.array([r.g for r in f.rows])
        h = np.array([r.h for r in f.rows])
        cons.append(
            {
                "type": "ineq",
                "fun": lambda zv, g=g, h=h, k=k: h - g @ zv[off[k] : off[k + 1]],
            }
        )
    out = minimize(
        cost,
        start,
        constraints=cons,
        method="SLSQP",
        options={"ftol": 1e-14, "maxiter": 500},
    )
    return np.array([tau(out.x, k) for k in range(len(frames))])


def test_perturbed_starts_reach_the_same_optimum_as_the_allocation() -> None:
    frames, _ = _trajectory(4, 0.85, relative=True)  # capacity rows are active
    spec = _spec(4, effort_weight=1.0, rate_weight=0.5)
    res = allocate_trajectory(frames, spec)
    assert res.status is SolveStatus.OPTIMAL
    assert any(d > 1e-6 for d in res.row_duals.values()), (
        "test needs active capacity rows"
    )
    total = sum(f.n_z for f in frames)
    rng = np.random.default_rng(3)
    for _ in range(3):
        other = _scipy_optimum(frames, spec, rng.normal(0.0, 5.0, total))
        assert other == pytest.approx(res.torque, abs=1e-4)


def test_optimum_is_invariant_to_null_basis_and_particular_solution() -> None:
    frames, _ = _trajectory(3, 0.85, relative=True)
    spec = _spec(3, rate_weight=0.5)
    base = allocate_trajectory(frames, spec)
    rng = np.random.default_rng(11)
    moved = []
    for f in frames:
        d = f.decomposition
        q, _ = np.linalg.qr(rng.normal(size=(d.nullity, d.nullity)))
        shift = d.N @ rng.normal(size=d.nullity)
        bound = 0.85 * float(np.max(np.abs(d.x_p[:N_A])))
        # x_p + N c + (N Q) z' describes the same physical set as x_p + N z.
        moved.append(_fs(dataclasses.replace(d, N=d.N @ q, x_p=d.x_p + shift), bound))
    res = allocate_trajectory(moved, spec)
    assert res.torque == pytest.approx(base.torque, abs=1e-6)


# ----------------------------------------------------- KKT and duals
def test_kkt_residuals_and_duals_are_recorded() -> None:
    frames, vel = _trajectory(5, 0.9, cone_radius=1.0, relative=True)
    spec = _spec(
        5,
        rate_weight=0.5,
        gross_power=GrossPowerSpec(
            weight=0.01,
            c_plus=np.full(N_A, 1.0),
            c_minus=np.full(N_A, 0.4),
            joint_velocity=vel,
        ),
    )
    res = allocate_trajectory(frames, spec)
    assert res.status is SolveStatus.OPTIMAL
    assert res.kkt_residuals is not None
    assert res.kkt_residuals.max() < 1e-6
    assert res.solve_time_s > 0.0
    assert res.iterations > 0
    assert all(d >= -1e-9 for d in res.row_duals.values())
    assert set(res.cone_duals) == {f"frame{k}:C:ball" for k in range(5)}
    assert "frame0:T:0:up" in res.row_duals


# --------------------------------------------------------- cone path
def test_second_order_cone_matches_generic_nlp() -> None:
    frames, _ = _trajectory(2, cap=1.0e4, cone_radius=0.2)
    spec = _spec(2)
    res = allocate_trajectory(frames, spec)
    assert res.status is SolveStatus.OPTIMAL
    for z in res.z:
        assert np.linalg.norm(z) <= 0.2 + 1e-7
    assert max(np.linalg.norm(z) for z in res.z) == pytest.approx(0.2, abs=1e-5)
    # The effort objective restricted to the ball: compare with SLSQP.
    for k, f in enumerate(frames):
        d = f.decomposition
        out = minimize(
            lambda z, d=d: np.sum(((d.x_p[:N_A] + d.N[:N_A] @ z) / SCALE) ** 2),
            np.zeros(d.nullity),
            constraints=[{"type": "ineq", "fun": lambda z: 0.04 - z @ z}],
            method="SLSQP",
            options={"ftol": 1e-14},
        )
        assert res.z[k] == pytest.approx(out.x, abs=1e-5)


# ----------------------------------------------------- gross power c+/-
def _gross(tau, vel, cp, cm):
    p = tau * vel
    return float(np.sum(cp * np.maximum(p, 0.0) + cm * np.maximum(-p, 0.0)))


def _power_spec(vel, cp, cm, *, reduce=True, weight=1.0):
    return _spec(
        len(vel),
        gross_power=GrossPowerSpec(
            weight=weight,
            c_plus=np.full(N_A, cp) if np.isscalar(cp) else np.asarray(cp, float),
            c_minus=np.full(N_A, cm) if np.isscalar(cm) else np.asarray(cm, float),
            joint_velocity=vel,
            allow_reduction=reduce,
        ),
    )


def test_uniform_ratio_is_a_dead_parameter_at_equal_mean() -> None:
    frames, vel = _trajectory(4, cap=1.0e3)
    results = [
        allocate_trajectory(frames, _power_spec(vel, cp, cm))
        for cp, cm in ((1.0, 1.0), (1.5, 0.5), (1.8, 0.2))
    ]
    for r in results[1:]:
        assert r.torque == pytest.approx(results[0].torque, abs=1e-6)
    kinds = [r.power_reduction.kind for r in results]
    assert kinds == ["ABS_POWER_ONLY", "UNIFORM_NET_PLUS_ABS", "UNIFORM_NET_PLUS_ABS"]
    assert all(r.power_reduction.ratio_is_dead_parameter for r in results[1:])
    assert all(r.power_reduction.net_power_invariant for r in results)


def test_reduction_equals_unreduced_epigraph_numerically() -> None:
    frames, vel = _trajectory(4, cap=1.0e3)
    for cp, cm in ((1.0, 1.0), (1.6, 0.4)):
        red = allocate_trajectory(frames, _power_spec(vel, cp, cm))
        full = allocate_trajectory(frames, _power_spec(vel, cp, cm, reduce=False))
        assert full.power_reduction.kind == red.power_reduction.kind != "NONE"
        assert not full.power_reduction.reduced_form_used
        assert red.power_reduction.reduced_form_used
        assert red.torque == pytest.approx(full.torque, abs=1e-6)
        assert red.objective_terms["gross_power"] == pytest.approx(
            full.objective_terms["gross_power"], rel=1e-9, abs=1e-9
        )


def test_gross_power_identity_holds_at_the_solution() -> None:
    frames, vel = _trajectory(4, cap=1.0e3)
    cp, cm = 1.7, 0.3
    res = allocate_trajectory(frames, _power_spec(vel, cp, cm, weight=1.0))
    w = _frame_weights(res.times_s)
    lhs = sum(w[k] * 1.0 * _gross(res.torque[k], vel[k], cp, cm) for k in range(4))
    p = res.torque * vel
    rhs = sum(
        w[k]
        * 1.0
        * (0.5 * (cp + cm) * np.abs(p[k]).sum() + 0.5 * (cp - cm) * p[k].sum())
        for k in range(4)
    )
    assert lhs == pytest.approx(rhs, rel=1e-12)
    assert res.objective_terms["gross_power"] == pytest.approx(lhs, rel=1e-9)


def test_heterogeneous_weights_change_the_optimum_and_are_not_reduced() -> None:
    frames, vel = _trajectory(4, cap=1.0e3)
    uniform = allocate_trajectory(frames, _power_spec(vel, 1.0, 1.0))
    cm_h = np.array([0.2, 1.0, 0.2, 1.0, 0.2])
    hetero = allocate_trajectory(frames, _power_spec(vel, 1.0, cm_h))
    assert hetero.power_reduction.kind == "NONE"
    assert not hetero.power_reduction.ratio_is_dead_parameter
    assert np.max(np.abs(hetero.torque - uniform.torque)) > 1e-3


def test_gross_power_optimum_is_locally_optimal_under_feasible_perturbations() -> None:
    """Convex => local is global; the objective is piecewise linear + quadratic."""
    frames, vel = _trajectory(3, cap=1.0e3)
    cm_h = np.array([0.2, 1.0, 0.2, 1.0, 0.2])
    spec = _power_spec(vel, 1.0, cm_h, weight=0.2)
    res = allocate_trajectory(frames, spec)
    w = _frame_weights(spec.times_s)

    def total(zs):
        value = 0.0
        for k, f in enumerate(frames):
            d = f.decomposition
            tau = d.x_p[:N_A] + d.N[:N_A] @ zs[k]
            value += w[k] * (
                np.sum((tau / SCALE) ** 2)
                + 0.2 * _gross(tau, vel[k], np.ones(N_A), cm_h)
            )
        return value

    best = total(res.z)
    rng = np.random.default_rng(5)
    for _ in range(200):
        zs = [z + rng.normal(0.0, 0.5, z.size) for z in res.z]
        feasible = all(
            np.all(
                np.array([r.g for r in f.rows]) @ zk
                <= np.array([r.h for r in f.rows]) + 1e-9
            )
            for f, zk in zip(frames, zs, strict=True)
        )
        if feasible:
            assert total(zs) >= best - 1e-9


# ----------------------------------------------- strict convexity (DbC)
def test_gross_power_alone_is_rejected_naming_the_missing_terms() -> None:
    frames, vel = _trajectory(3, cap=1.0e3)
    spec = _power_spec(vel, 1.0, 0.5)
    spec = dataclasses.replace(spec, effort_weight=0.0)
    with pytest.raises(NotStrictlyConvexError, match="effort") as err:
        allocate_trajectory(frames, spec)
    assert "gross power" in str(err.value).lower()
    assert isinstance(err.value, ValueError)


def test_no_terms_at_all_is_rejected() -> None:
    frames, _ = _trajectory(2, cap=1.0e3)
    with pytest.raises(NotStrictlyConvexError, match="no quadratic term"):
        allocate_trajectory(frames, _spec(2, effort_weight=0.0))


@pytest.mark.parametrize("t_count", [1, 4])
def test_rate_term_alone_lacks_curvature_on_the_constant_mode(t_count) -> None:
    frames, _ = _trajectory(t_count, cap=1.0e3)
    spec = _spec(t_count, effort_weight=0.0, rate_weight=1.0)
    with pytest.raises(NotStrictlyConvexError) as err:
        allocate_trajectory(frames, spec)
    msg = str(err.value)
    assert "effort" in msg and "reference" in msg


def test_rate_plus_effort_is_accepted() -> None:
    frames, _ = _trajectory(4, cap=1.0e3)
    res = allocate_trajectory(frames, _spec(4, effort_weight=0.1, rate_weight=1.0))
    assert res.convexity.min_curvature_ratio > 0.0
    assert res.convexity.rank == res.convexity.required_rank


def test_torque_gauge_directions_do_not_break_strict_convexity() -> None:
    """A z direction that changes no torque is a gauge, not a missing term."""
    dec, _ = _base()
    n_z = dec.nullity
    n_new = np.hstack([dec.N, np.zeros((dec.dim_x, 1))])  # one gauge column
    d2 = dataclasses.replace(
        dec, N=n_new, counts=dataclasses.replace(dec.counts, nullity=n_z + 1)
    )
    fs = _fs(d2, cap=1.0e3)
    res = allocate_trajectory([fs], _spec(1))
    assert res.status is SolveStatus.OPTIMAL
    assert res.convexity.gauge_dimension == 1


# ----------------------------------------------------------- the record
def test_result_is_family_dependent_with_hypotheses() -> None:
    frames, vel = _trajectory(3, cap=22.0)
    spec = _spec(
        3,
        rate_weight=0.2,
        gross_power=GrossPowerSpec(0.01, np.ones(N_A), np.ones(N_A), vel),
    )
    res = allocate_trajectory(frames, spec)
    rec = res.record
    assert rec is not None
    assert rec.allocation_dependence is AllocationDependence.FAMILY_DEPENDENT
    for hid in (
        "H-TORQUE-CAPACITY",
        "H-EFFORT-CAPACITY",
        "H-SMOOTH-TORQUE-RATE",
        "H-GROSS-POWER-COST",
    ):
        assert hid in rec.hypotheses
    assert "C-METRIC-CAPACITY" in rec.conventions
    assert rec.values.shape == (3, N_A)
    assert rec.values == pytest.approx(res.torque)
    assert "not an estimate" in res.non_claim.lower()


def test_reference_is_declared_on_the_result() -> None:
    frames, _ = _trajectory(2, cap=1.0e3)
    ref = np.zeros((2, N_A))
    spec = _spec(2, reference_weight=1.0)
    res = allocate_trajectory(frames, spec, reference=ref)
    assert res.reference_used


# ---------------------------------------------------------- preconditions
def test_preconditions() -> None:
    frames, vel = _trajectory(3, cap=1.0e3)
    spec = _spec(3)
    with pytest.raises(ValueError, match="at least one"):
        allocate_trajectory([], spec)
    with pytest.raises(TypeError, match="ObjectiveSpec"):
        allocate_trajectory(frames, object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="times_s"):
        allocate_trajectory(frames, _spec(2))
    no_dec = dataclasses.replace(frames[0], decomposition=None)
    with pytest.raises(ValueError, match="decomposition"):
        allocate_trajectory([no_dec, *frames[1:]], spec)
    with pytest.raises(ValueError, match="reference"):
        allocate_trajectory(frames, _spec(3, reference_weight=1.0))
    with pytest.raises(ValueError, match="reference_weight"):
        allocate_trajectory(frames, spec, reference=np.zeros((3, N_A)))
    with pytest.raises(ValueError, match="shape"):
        allocate_trajectory(
            frames, _spec(3, reference_weight=1.0), reference=np.zeros((3, N_A + 1))
        )
    short = GrossPowerSpec(1.0, np.ones(N_A), np.ones(N_A), vel[:2])
    with pytest.raises(ValueError, match="joint_velocity"):
        allocate_trajectory(frames, _spec(3, gross_power=short))


def test_objective_spec_validation() -> None:
    t = np.array([0.0, 0.1])
    with pytest.raises(ValueError, match="strictly increasing"):
        ObjectiveSpec(np.array([0.1, 0.1]), np.ones(2))
    with pytest.raises(ValueError, match="torque_scale"):
        ObjectiveSpec(t, np.array([1.0, 0.0]))
    with pytest.raises(ValueError, match="non-negative"):
        ObjectiveSpec(t, np.ones(2), rate_weight=-1.0)
    with pytest.raises(ValueError, match="positive"):
        GrossPowerSpec(0.0, np.ones(2), np.ones(2), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="c_minus"):
        GrossPowerSpec(1.0, np.ones(2), -np.ones(2), np.zeros((2, 2)))


def test_infeasible_set_reports_status_without_a_record() -> None:
    dec, _ = _base()
    bad = FeasibleSet.from_rows(
        dec.nullity,
        [
            FeasibleRow(
                np.eye(dec.nullity)[0], -1.0, "R:up", "H-TORQUE-CAPACITY", "torque"
            ),
            FeasibleRow(
                -np.eye(dec.nullity)[0], -1.0, "R:lo", "H-TORQUE-CAPACITY", "torque"
            ),
        ],
    )
    bad = dataclasses.replace(bad, decomposition=dec)
    res = allocate_trajectory([bad], _spec(1))
    assert isinstance(res, AllocationResult)
    assert res.status is SolveStatus.INFEASIBLE
    assert res.record is None and res.torque is None


# ------------------------------------------------------------- benchmark
def test_planar_oracle_trajectory_benchmark_recorded(caplog) -> None:
    """Recorded, not gated on wall time: 200 planar frames, all four terms."""
    t_count = 200
    frames, vel = _trajectory(t_count, 0.9, relative=True)
    ref = np.array([f.decomposition.x_p[:N_A] * 0.5 for f in frames])
    spec = _spec(
        t_count,
        rate_weight=0.5,
        reference_weight=0.1,
        gross_power=GrossPowerSpec(
            0.01, np.full(N_A, 1.0), np.array([0.2, 1.0, 0.2, 1.0, 0.2]), vel
        ),
    )
    t0 = time.perf_counter()
    res = allocate_trajectory(frames, spec, reference=ref)
    wall = time.perf_counter() - t0
    with caplog.at_level(logging.INFO):
        logging.getLogger(__name__).info(
            "NSX-23 planar benchmark: %d frames, %.3f s wall, %.3f s solve, "
            "%d iterations, KKT %s",
            t_count,
            wall,
            res.solve_time_s,
            res.iterations,
            res.kkt_residuals,
        )
    assert res.status is SolveStatus.OPTIMAL
    assert res.kkt_residuals is not None and res.kkt_residuals.max() < 1e-6
    print(
        f"\nNSX-23 planar benchmark: {t_count} frames, wall {wall:.3f} s, "
        f"solve {res.solve_time_s:.3f} s, iterations {res.iterations}, "
        f"KKT primal/dual/comp = {res.kkt_residuals.primal:.2e}/"
        f"{res.kkt_residuals.dual:.2e}/{res.kkt_residuals.complementarity:.2e}"
    )
