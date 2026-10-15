"""Tests for the convex solver backends (NSX-20, #12563).

Certificates are verified independently of the solver: a returned ray is only
accepted when it demonstrably certifies infeasibility or unboundedness.
"""

from __future__ import annotations

import builtins

import numpy as np
import pytest
import scipy.sparse as sp

pytest.importorskip("highspy")
pytest.importorskip("clarabel")

from src.shared.python.loop_nullspace.solvers import (  # noqa: E402
    Basis,
    ConicResult,
    KKTResiduals,
    LPResult,
    LPSession,
    SolveStatus,
    SOCBlock,
    available_backends,
    solve_lp,
    solve_socp,
)

pytestmark = pytest.mark.unit

INF = np.inf


def _farkas_gap(a, row_lo, row_hi, lo, hi, ray):
    """Positive iff ``ray`` separates ``{A x}`` from the row-bound box."""
    a = sp.csr_matrix(a)
    d = a.T @ ray  # ray^T A x = d^T x
    with np.errstate(invalid="ignore"):
        box = np.where(d == 0.0, 0.0, np.minimum(d * lo, d * hi))
        box_hi = np.where(d == 0.0, 0.0, np.maximum(d * lo, d * hi))
        rlo = np.where(ray == 0.0, 0.0, np.minimum(ray * row_lo, ray * row_hi))
        rhi = np.where(ray == 0.0, 0.0, np.maximum(ray * row_lo, ray * row_hi))
    return max(box.sum() - rhi.sum(), rlo.sum() - box_hi.sum())


# ---------------------------------------------------------------- LP optima
def test_lp_analytic_optimum_dense() -> None:
    # min -x - 2y  s.t. x + y <= 4, x <= 3, y <= 2.5, x,y >= 0 -> (1.5, 2.5)
    res = solve_lp(
        c=[-1.0, -2.0],
        A_eq=None,
        b_eq=None,
        A_ub=np.array([[1.0, 1.0]]),
        b_ub=np.array([4.0]),
        bounds=[(0.0, 3.0), (0.0, 2.5)],
    )
    assert isinstance(res, LPResult)
    assert res.status is SolveStatus.OPTIMAL
    np.testing.assert_allclose(res.x, [1.5, 2.5], atol=1e-9)
    assert res.objective == pytest.approx(-6.5)
    assert res.dual_ray is None and res.primal_ray is None


def test_lp_with_equalities_and_sparse_inputs() -> None:
    # min x + y + z  s.t. x + 2y = 4, y + z = 3, all >= 0 -> (0, 2, 1)
    a_eq = sp.csr_matrix(np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 1.0]]))
    res = solve_lp([1.0, 1.0, 1.0], a_eq, [4.0, 3.0], None, None, [(0.0, None)] * 3)
    assert res.status is SolveStatus.OPTIMAL
    np.testing.assert_allclose(res.x, [0.0, 2.0, 1.0], atol=1e-9)


def test_lp_free_variables_and_none_bounds() -> None:
    # min x s.t. x >= -3 (as -x <= 3), x free.
    res = solve_lp(
        [1.0], None, None, np.array([[-1.0]]), np.array([3.0]), [(None, None)]
    )
    assert res.status is SolveStatus.OPTIMAL
    assert res.x[0] == pytest.approx(-3.0)


def test_kkt_residuals_small_and_scale_invariant() -> None:
    rng = np.random.default_rng(3)
    a = rng.normal(size=(6, 4))
    x0 = rng.uniform(0.2, 1.0, 4)
    b = a @ x0 + rng.uniform(0.1, 1.0, 6)
    c = rng.normal(size=4)
    bounds = [(0.0, 5.0)] * 4
    base = solve_lp(c, None, None, a, b, bounds)
    assert base.status is SolveStatus.OPTIMAL
    assert isinstance(base.kkt_residuals, KKTResiduals)
    assert base.kkt_residuals.max() < 1e-7
    # Scaling data by 1e6 (same optimum x) must keep the *relative* residuals
    # small; absolute residuals would grow with the data.
    s = 1e6
    big = solve_lp(s * c, None, None, s * a, s * b, bounds)
    assert big.status is SolveStatus.OPTIMAL
    np.testing.assert_allclose(big.x, base.x, atol=1e-6)
    assert big.kkt_residuals.max() < 1e-6


def test_kkt_residuals_detect_a_wrong_point() -> None:
    from src.shared.python.loop_nullspace.solvers import lp_kkt_residuals

    c = np.array([-1.0, -2.0])
    a_ub = np.array([[1.0, 1.0]])
    good = solve_lp(c, None, None, a_ub, [4.0], [(0.0, 3.0), (0.0, 2.5)])
    bad = lp_kkt_residuals(
        c,
        None,
        None,
        a_ub,
        np.array([4.0]),
        [(0.0, 3.0), (0.0, 2.5)],
        # x violates x + y <= 4
        primal_dual=(np.array([2.0, 2.5]), good.row_dual, good.col_dual),
    )
    assert bad.primal > 0.05


# ------------------------------------------------------------ certificates
def test_infeasible_lp_returns_valid_farkas_certificate() -> None:
    # x + y <= -1 with x, y >= 0.
    a = np.array([[1.0, 1.0]])
    res = solve_lp([1.0, 1.0], None, None, a, [-1.0], [(0.0, None)] * 2)
    assert res.status is SolveStatus.INFEASIBLE
    assert res.dual_ray is not None
    gap = _farkas_gap(
        a,
        np.array([-INF]),
        np.array([-1.0]),
        np.zeros(2),
        np.full(2, INF),
        res.dual_ray,
    )
    assert gap > 1e-9
    assert res.primal_ray is None


def test_infeasible_with_equalities_certificate_row_order() -> None:
    # Rows are stacked [eq; ub]: x = 2 (eq) and x <= 1 (ub).
    a = np.array([[1.0], [1.0]])
    res = solve_lp(
        [0.0], np.array([[1.0]]), [2.0], np.array([[1.0]]), [1.0], [(None, None)]
    )
    assert res.status is SolveStatus.INFEASIBLE
    assert res.dual_ray is not None and res.dual_ray.shape == (2,)
    gap = _farkas_gap(
        a,
        np.array([2.0, -INF]),
        np.array([2.0, 1.0]),
        np.array([-INF]),
        np.array([INF]),
        res.dual_ray,
    )
    assert gap > 1e-9


def test_unbounded_lp_returns_valid_primal_ray() -> None:
    # min -x s.t. x - y <= 1, x, y >= 0.
    a = np.array([[1.0, -1.0]])
    c = np.array([-1.0, 0.0])
    res = solve_lp(c, None, None, a, [1.0], [(0.0, None)] * 2)
    assert res.status is SolveStatus.UNBOUNDED
    d = res.primal_ray
    assert d is not None
    assert np.all(d >= -1e-9)  # stays inside x >= 0 recession cone
    assert (a @ d)[0] <= 1e-9  # A d <= 0 (recession of x - y <= 1)
    assert c @ d < -1e-9  # strictly improving
    assert res.dual_ray is None


def test_unbounded_ray_respects_equalities() -> None:
    # min -y s.t. x - y = 0 (x=y), free x -> ray along (1, 1).
    a_eq = np.array([[1.0, -1.0]])
    c = np.array([0.0, -1.0])
    res = solve_lp(c, a_eq, [0.0], None, None, [(None, None), (None, None)])
    assert res.status is SolveStatus.UNBOUNDED
    d = res.primal_ray
    assert d is not None
    assert abs((a_eq @ d)[0]) < 1e-9 and c @ d < -1e-9


# ----------------------------------------------------------- warm starts
def _random_lp(rng, m=40, n=60):
    a = rng.normal(size=(m, n))
    x0 = rng.uniform(0.0, 1.0, n)
    b = a @ x0 + rng.uniform(0.0, 0.5, m)
    return a, b


def test_warm_start_reduces_iterations_on_perturbed_objectives() -> None:
    rng = np.random.default_rng(11)
    a, b = _random_lp(rng)
    bounds = [(0.0, 2.0)] * a.shape[1]
    c0 = rng.normal(size=a.shape[1])
    first = solve_lp(c0, None, None, a, b, bounds)
    assert first.status is SolveStatus.OPTIMAL
    cold_total = warm_total = 0
    for _ in range(8):
        c = c0 + 0.02 * rng.normal(size=c0.shape)
        cold = solve_lp(c, None, None, a, b, bounds)
        warm = solve_lp(c, None, None, a, b, bounds, warm=first.basis)
        assert warm.status is cold.status is SolveStatus.OPTIMAL
        assert warm.objective == pytest.approx(cold.objective, abs=1e-8)
        cold_total += cold.iterations
        warm_total += warm.iterations
    assert warm_total < 0.5 * cold_total


def test_session_reuses_basis_across_objective_swaps() -> None:
    rng = np.random.default_rng(5)
    a, b = _random_lp(rng)
    n = a.shape[1]
    bounds = [(0.0, 2.0)] * n
    c0 = rng.normal(size=n)
    with LPSession(None, None, a, b, bounds) as session:
        r0 = session.solve(c0)
        assert r0.status is SolveStatus.OPTIMAL and r0.iterations > 0
        # Same objective again: the retained optimal basis needs 0 iterations.
        again = session.solve(c0)
        assert again.iterations == 0
        np.testing.assert_allclose(again.x, r0.x, atol=1e-9)
        # min and max of one quantity are two LPs on one constraint set.
        lo = session.solve(c0)
        hi = session.solve(-c0)
        assert lo.status is hi.status is SolveStatus.OPTIMAL
        assert lo.objective <= -hi.objective + 1e-9
        # Explicit cold start is available for benchmarking.
        session.reset()
        cold = session.solve(c0)
        assert cold.iterations >= r0.iterations - 1 and cold.iterations > 0


def test_session_matches_one_shot_solve() -> None:
    rng = np.random.default_rng(8)
    a, b = _random_lp(rng, 10, 12)
    bounds = [(0.0, 1.0)] * 12
    c = rng.normal(size=12)
    one = solve_lp(c, None, None, a, b, bounds)
    with LPSession(None, None, a, b, bounds) as s:
        two = s.solve(c)
    assert one.objective == pytest.approx(two.objective, abs=1e-10)


def test_basis_is_immutable_and_validated() -> None:
    res = solve_lp([1.0], None, None, np.array([[-1.0]]), [-1.0], [(0.0, None)])
    assert isinstance(res.basis, Basis)
    with pytest.raises(ValueError):
        res.basis.col_status[0] = 2
    wrong = Basis(
        col_status=np.zeros(3, dtype=np.int8), row_status=np.zeros(1, np.int8)
    )
    with pytest.raises(ValueError, match="basis"):
        solve_lp(
            [1.0], None, None, np.array([[-1.0]]), [-1.0], [(0.0, None)], warm=wrong
        )


# ------------------------------------------------------------ status mapping
def test_status_enum_has_exactly_the_contracted_members() -> None:
    assert {s.name for s in SolveStatus} == {
        "OPTIMAL",
        "INFEASIBLE",
        "UNBOUNDED",
        "NUMERICAL",
    }


def test_iteration_limit_maps_to_numerical_with_reason() -> None:
    rng = np.random.default_rng(1)
    a, b = _random_lp(rng, 60, 90)
    res = solve_lp(
        rng.normal(size=90),
        None,
        None,
        a,
        b,
        [(0.0, 2.0)] * 90,
        max_iterations=1,
    )
    assert res.status is SolveStatus.NUMERICAL
    assert res.reason


# ----------------------------------------------------------------- DbC
@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"c": [1.0, np.nan]}, "finite"),
        ({"c": [1.0, 2.0, 3.0]}, "bounds"),
        ({"A_ub": np.ones((2, 3)), "b_ub": [1.0, 1.0]}, "columns"),
        ({"A_ub": np.ones((2, 2)), "b_ub": [1.0]}, "rows"),
        ({"A_ub": np.ones((1, 2)), "b_ub": None}, "b_ub"),
        ({"bounds": [(1.0, 0.0), (0.0, 1.0)]}, "lower"),
        ({"A_eq": np.array([[np.inf, 0.0]]), "b_eq": [1.0]}, "finite"),
    ],
)
def test_lp_preconditions(kwargs, match) -> None:
    base = {
        "c": [1.0, 1.0],
        "A_eq": None,
        "b_eq": None,
        "A_ub": None,
        "b_ub": None,
        "bounds": [(0.0, 1.0), (0.0, 1.0)],
    }
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        solve_lp(**base)


# ----------------------------------------------------------------- SOCP
def test_socp_analytic_optimum() -> None:
    # min t s.t. ||(x-3, y-4)|| <= t, (x,y) free; optimum t=0 at (3,4)? add a
    # linear constraint x + y <= 1 -> nearest point of half-plane to (3,4):
    # projection (0,1)+..., distance = |3+4-1|/sqrt(2).
    # variables (x, y, t); cone rows: s = [t; x-3; y-4] = b - A v.
    cone = SOCBlock(
        A=-np.array([[0, 0, 1.0], [1, 0, 0], [0, 1, 0]]),
        b=np.array([0.0, -3.0, -4.0]),
    )
    res = solve_socp(
        c=[0.0, 0.0, 1.0],
        A_eq=None,
        b_eq=None,
        A_ub=np.array([[1.0, 1.0, 0.0]]),
        b_ub=np.array([1.0]),
        cones=[cone],
    )
    assert isinstance(res, ConicResult)
    assert res.status is SolveStatus.OPTIMAL
    expected_t = 6.0 / np.sqrt(2.0)
    assert res.x[2] == pytest.approx(expected_t, rel=1e-6)
    # nearest point = (3,4) - 3*(1,1) = (0, 1)
    np.testing.assert_allclose(res.x[:2], [0.0, 1.0], atol=1e-5)
    assert res.kkt_residuals.max() < 1e-6
    assert res.dual_ray is None and res.primal_ray is None


def test_socp_with_equality_sparse_and_quadratic_free_objective() -> None:
    # min x s.t. ||(y, z)|| <= x, y = 1, z = 1 (equalities) -> x = sqrt(2).
    cone = SOCBlock(
        A=-sp.csr_matrix(np.array([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]])),
        b=np.zeros(3),
    )
    a_eq = sp.csr_matrix(np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
    res = solve_socp([1.0, 0.0, 0.0], a_eq, [1.0, 1.0], None, None, [cone])
    assert res.status is SolveStatus.OPTIMAL
    assert res.objective == pytest.approx(np.sqrt(2.0), rel=1e-6)


def test_socp_bounds_are_honoured() -> None:
    # min -x s.t. ||(y)|| <= x (i.e. y<=x), x<=2, y free -> x = 2.
    cone = SOCBlock(A=-np.array([[1.0, 0.0], [0.0, 1.0]]), b=np.zeros(2))
    res = solve_socp(
        [-1.0, 0.0], None, None, None, None, [cone], bounds=[(None, 2.0), (None, None)]
    )
    assert res.status is SolveStatus.OPTIMAL
    assert res.x[0] == pytest.approx(2.0, abs=1e-6)


def _in_soc(v, tol):
    return v[0] >= np.linalg.norm(v[1:]) - tol


def test_socp_infeasible_returns_farkas_certificate() -> None:
    # ||(y)|| <= x together with x <= -1 is infeasible. Variables (x, y).
    cone = SOCBlock(A=-np.eye(2), b=np.zeros(2))
    a_ub = np.array([[1.0, 0.0]])
    res = solve_socp([0.0, 0.0], None, None, a_ub, [-1.0], [cone])
    assert res.status is SolveStatus.INFEASIBLE
    z = res.dual_ray
    assert z is not None
    # Stacked rows: [ub (1, nonneg); cone (2, SOC)]; A^T z = 0, b^T z < 0, z in K*.
    big_a = np.vstack([a_ub, -np.eye(2)])  # cone rows carry SOCBlock.A
    big_b = np.array([-1.0, 0.0, 0.0])
    assert np.linalg.norm(big_a.T @ z) < 1e-6 * max(1.0, np.linalg.norm(z))
    assert big_b @ z < -1e-9
    assert z[0] >= -1e-9
    assert _in_soc(z[1:], 1e-8)
    assert res.primal_ray is None


def test_socp_unbounded_returns_primal_ray() -> None:
    # min -x s.t. ||(y)|| <= x : unbounded along (1, 0).
    cone = SOCBlock(A=-np.eye(2), b=np.zeros(2))
    c = np.array([-1.0, 0.0])
    res = solve_socp(c, None, None, None, None, [cone])
    assert res.status is SolveStatus.UNBOUNDED
    d = res.primal_ray
    assert d is not None
    assert c @ d < -1e-9
    assert _in_soc(d, 1e-6)  # recession direction stays in the cone
    assert res.dual_ray is None


def test_socp_preconditions() -> None:
    with pytest.raises(ValueError, match="columns"):
        solve_socp(
            [1.0, 1.0],
            None,
            None,
            None,
            None,
            [SOCBlock(A=-np.eye(3)[:, :3], b=np.zeros(3))],
        )
    with pytest.raises(ValueError, match="dimension"):
        SOCBlock(A=np.zeros((1, 2)), b=np.zeros(1))


# --------------------------------------------------------- lazy / missing
def test_available_backends_reports_both() -> None:
    assert available_backends() == {"highspy": True, "clarabel": True}


@pytest.mark.parametrize("missing", ["highspy", "clarabel"])
def test_missing_extra_raises_actionable_importerror(monkeypatch, missing) -> None:
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == missing or name.startswith(missing + "."):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="loop-nullspace") as info:
        if missing == "highspy":
            solve_lp([1.0], None, None, None, None, [(0.0, 1.0)])
        else:
            cone = SOCBlock(A=-np.eye(2), b=np.zeros(2))
            solve_socp([1.0, 0.0], None, None, None, None, [cone])
    assert missing in str(info.value)
    assert available_backends()[missing] is False


def test_importing_the_module_does_not_import_solvers(monkeypatch) -> None:
    import subprocess
    import sys

    code = (
        "import sys; import src.shared.python.loop_nullspace.solvers; "
        "assert 'highspy' not in sys.modules and 'clarabel' not in sys.modules"
    )
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(p) for p in sys.path if p] + [env.get("PYTHONPATH", "")]
    )
    done = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr


# --------------------------------------------------- quadratic objective (P)
def test_socp_quadratic_objective_matches_closed_form() -> None:
    """min 1/2 x'Px + c'x s.t. x1 + x2 = 1 has a closed-form optimum (NSX-23)."""
    p = np.diag([2.0, 4.0])
    c = np.array([1.0, 0.0])
    a_eq = np.array([[1.0, 1.0]])
    res = solve_socp(c, a_eq, [1.0], None, None, [], P=p)
    assert res.status is SolveStatus.OPTIMAL
    # Stationarity: Px + c = nu * 1, x1 + x2 = 1 -> x = (1/6 .. ) solved exactly.
    kkt = np.block([[p, -a_eq.T], [a_eq, np.zeros((1, 1))]])
    sol = np.linalg.solve(kkt, np.array([-1.0, 0.0, 1.0]))
    assert res.x == pytest.approx(sol[:2], abs=1e-7)
    assert res.objective == pytest.approx(0.5 * sol[:2] @ p @ sol[:2] + c @ sol[:2])
    assert res.kkt_residuals is not None
    assert res.kkt_residuals.max() < 1e-7


def test_socp_quadratic_objective_preconditions() -> None:
    with pytest.raises(ValueError, match="P must be"):
        solve_socp([0.0, 0.0], None, None, None, None, [], P=np.eye(3))
    with pytest.raises(ValueError, match="symmetric"):
        solve_socp(
            [0.0, 0.0], None, None, None, None, [], P=np.array([[1.0, 2.0], [0.0, 1.0]])
        )
