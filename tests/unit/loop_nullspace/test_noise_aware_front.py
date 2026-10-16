"""Tests of the variance fronts (NSX-92, #12721).

Oracles: a dense one-dimensional grid over the pendulum family (for a given
variance target, the minimum effort over the grid points that meet it), and the
NSX-25 dominance rule applied by hand to the CHS family.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("highspy")
pytest.importorskip("clarabel")

from src.shared.python.loop_nullspace.allocation_spec import (  # noqa: E402
    ObjectiveSpec,
    VarianceTerm,
)
from src.shared.python.loop_nullspace.noise_aware_front import (  # noqa: E402
    ChsCase,
    chs_variance_front,
    variance_effort_front,
)
from src.shared.python.loop_nullspace.noise_aware_spec import (  # noqa: E402
    NoiseAwareStatus,
)
from src.shared.python.loop_nullspace.pareto import pareto_front  # noqa: E402
from tests.unit.loop_nullspace import _noise_aware_fixtures as fx  # noqa: E402

pytestmark = pytest.mark.unit

ZS = np.linspace(-20.0, 20.0, 400001)
TAUS = fx.U_P[None, :] - ZS[:, None] * fx.Q[None, :]
EFFORT = np.sum((TAUS / fx.SCALE) ** 2, axis=1)  # one frame, trapezoid weight 1


def _grid_variance(model) -> np.ndarray:
    """Vectorised M1 variance on the grid: constant + sum coeff * tau^2."""
    coeff = model.signal_coefficients(np.zeros(1))[0]
    return model.invariant_variance(np.zeros(1)) + (TAUS**2) @ coeff


def _min_effort_at(variance: np.ndarray, target: float) -> float:
    return float(EFFORT[variance <= target + 1e-15].min())


def _convex_setup():
    sp = fx.golden_spec()
    return fx.m1_model(sp), fx.declaration(True, False, sp)


def test_convex_front_matches_the_dense_grid_oracle_and_is_pareto() -> None:
    model, decl = _convex_setup()
    front = variance_effort_front([fx.frame()], fx.times(), fx.SCALE, model, decl, 7)
    var_grid = _grid_variance(model)
    assert front.convex and front.pareto is not None
    assert front.is_monotone()
    assert all(not p.dominated for p in front.solved)
    assert len(front.solved) == 7
    for p in front.solved:
        assert p.certificate == "global-convex"
        assert p.variance <= p.target_variance * (1 + 1e-7)
        assert p.effort == pytest.approx(
            _min_effort_at(var_grid, p.target_variance), rel=2e-5
        )
        assert (
            front.variance_floor - 1e-9 <= p.variance <= front.baseline_variance + 1e-9
        )
    # the loosest point is the unconstrained minimum-effort allocation
    assert front.solved[-1].effort == pytest.approx(front.baseline_effort, rel=1e-6)
    assert front.solved[-1].variance == pytest.approx(front.baseline_variance, rel=1e-6)


def test_convex_front_slope_is_the_derivative_of_the_dense_front() -> None:
    model, decl = _convex_setup()
    front = variance_effort_front([fx.frame()], fx.times(), fx.SCALE, model, decl, 9)
    var_grid = _grid_variance(model)
    pts = front.solved
    for p in pts[1:-1]:
        h = 1e-3 * (front.baseline_variance - front.variance_floor)
        de = (
            _min_effort_at(var_grid, p.target_variance + h)
            - _min_effort_at(var_grid, p.target_variance - h)
        ) / (2 * h)
        assert p.slope == pytest.approx(de, rel=5e-3)
        assert p.slope < 0.0  # more variance allowed, less effort


def test_front_uses_the_nsx25_machinery_not_a_second_engine() -> None:
    model, decl = _convex_setup()
    front = variance_effort_front([fx.frame()], fx.times(), fx.SCALE, model, decl, 4)
    coeff = model.signal_coefficients(np.zeros(1))
    term = VarianceTerm(1.0, coeff)
    eps = [float(front.solved[1].variance - model.invariant_variance(np.zeros(1)))]
    direct = pareto_front(
        [fx.frame()],
        (
            ObjectiveSpec(times_s=fx.times(), torque_scale=fx.SCALE),
            ObjectiveSpec(
                times_s=fx.times(),
                torque_scale=fx.SCALE,
                effort_weight=0.0,
                variance=term,
            ),
        ),
        eps,
    )
    assert direct.points[0].primary == pytest.approx(front.solved[1].effort, rel=1e-6)


def test_nonconvex_front_is_local_and_matches_the_grid_in_one_dimension() -> None:
    sp = fx.golden_spec()
    model, decl = fx.m2_model(sp), fx.declaration(True, True, sp)
    front = variance_effort_front([fx.frame()], fx.times(), fx.SCALE, model, decl, 5)
    assert not front.convex and front.pareto is None
    assert front.solved, "at least one point must solve"
    sub = slice(None, None, 200)
    var_grid = np.array(
        [
            model.variance(np.array([z]), t[None])
            for z, t in zip(ZS[sub], TAUS[sub], strict=True)
        ]
    )
    eff_grid = EFFORT[sub]
    for p in front.solved:
        assert p.certificate == "local"
        assert p.status is NoiseAwareStatus.LOCAL_STATIONARY
        assert p.variance >= front.variance_floor - 1e-9
        # in one dimension a certified local optimum of the cap problem is the
        # grid optimum: nothing on the grid beats it within the cap
        ok = eff_grid <= p.effort * (1 + 1e-6)
        assert p.variance <= var_grid[ok].min() + 1e-6 * p.variance
    assert var_grid.min() >= front.variance_floor - 1e-6


def test_flat_objective_has_no_front() -> None:
    sp = fx.spec(None)
    model, decl = fx.m1_model(sp), fx.declaration(False, False, sp)
    with pytest.raises(ValueError, match="V1"):
        variance_effort_front([fx.frame()], fx.times(), fx.SCALE, model, decl)
    with pytest.raises(ValueError, match="n_points"):
        variance_effort_front(
            [fx.frame()], fx.times(), fx.SCALE, *_convex_setup(), n_points=1
        )


def _case(label: str, chs: float, k: tuple[float, float]) -> ChsCase:
    sp = fx.spec(k)
    return ChsCase(label, chs, (fx.frame(),), fx.m1_model(sp))


def test_chs_front_marks_dominated_cases_and_keeps_the_tradeoff() -> None:
    cases = [
        _case("slow-quiet", 40.0, (0.1, 0.25)),
        _case("fast-quiet", 45.0, (0.1, 0.25)),  # same variance, faster: dominates slow
        _case("fastest-noisy", 50.0, (0.6, 1.2)),  # faster but noisier: trade-off
    ]
    decl = fx.declaration(True, False, fx.spec((0.1, 0.25)))
    front = chs_variance_front(cases, fx.times(), fx.SCALE, decl, effort_cap=0.4)
    by = {p.label: p for p in front.points}
    assert all(p.status is NoiseAwareStatus.OPTIMAL for p in front.points)
    assert all(p.chs_allocation_dependence == "INVARIANT" for p in front.points)
    assert all(p.effort <= 0.4 * (1 + 1e-6) for p in front.points)
    assert by["slow-quiet"].variance == pytest.approx(
        by["fast-quiet"].variance, rel=1e-9
    )
    assert by["slow-quiet"].dominated and not by["fast-quiet"].dominated
    assert by["fastest-noisy"].variance > by["fast-quiet"].variance
    assert not by["fastest-noisy"].dominated
    assert {p.label for p in front.nondominated} == {"fast-quiet", "fastest-noisy"}
    assert "H-NOISE-SIGNAL-DEPENDENT" in front.hypotheses


def test_chs_front_reports_an_unreachable_cap_as_infeasible_not_relaxed() -> None:
    cases = [_case("a", 40.0, (0.1, 0.25)), _case("b", 42.0, (0.2, 0.3))]
    decl = fx.declaration(True, False, fx.spec((0.1, 0.25)))
    front = chs_variance_front(cases, fx.times(), fx.SCALE, decl, effort_cap=1e-3)
    assert all(p.status is NoiseAwareStatus.INFEASIBLE for p in front.points)
    assert all(p.variance is None for p in front.points)
    assert front.nondominated == ()


def test_chs_front_preconditions() -> None:
    decl = fx.declaration(True, False, fx.spec((0.1, 0.25)))
    one = [_case("a", 40.0, (0.1, 0.25))]
    with pytest.raises(ValueError, match="at least two"):
        chs_variance_front(one, fx.times(), fx.SCALE, decl, 0.4)
    dup = [_case("a", 40.0, (0.1, 0.25)), _case("a", 41.0, (0.1, 0.25))]
    with pytest.raises(ValueError, match="unique"):
        chs_variance_front(dup, fx.times(), fx.SCALE, decl, 0.4)
    two = [_case("a", 40.0, (0.1, 0.25)), _case("b", 41.0, (0.1, 0.25))]
    with pytest.raises(ValueError, match="effort_cap"):
        chs_variance_front(two, fx.times(), fx.SCALE, decl, 0.0)
    with pytest.raises(ValueError, match="chs_mps"):
        ChsCase("x", -1.0, (fx.frame(),), fx.m1_model())
    with pytest.raises(ValueError, match="label"):
        ChsCase("", 40.0, (fx.frame(),), fx.m1_model())
