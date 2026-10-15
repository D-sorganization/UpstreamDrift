"""Tests for epsilon-constraint Pareto fronts (NSX-25, #12568).

The oracle is a two-frame, one-torque problem with a closed-form front:
``tau_k = z_k``, ``z_1 >= 1``, ``|z_2| <= 3``.  Effort ``J_a`` and torque rate
``J_b <= eps`` give, with ``d = s * sqrt(eps * dt)``,
``J_a*(eps) = (dt / 2)(1 + (1 - d)^2) / s^2`` for ``d < 1`` and the constant
``(dt / 2) / s^2`` beyond (derived by hand; independent of the implementation).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("highspy")
pytest.importorskip("clarabel")

from src.shared.python.loop_nullspace.allocation_spec import (  # noqa: E402
    GrossPowerSpec,
    NotStrictlyConvexError,
    ObjectiveSpec,
)
from src.shared.python.loop_nullspace.decomposition import (  # noqa: E402
    CharacteristicScales,
    Decomposition,
    DimensionAccounting,
)
from src.shared.python.loop_nullspace.feasible_rows import FeasibleRow  # noqa: E402
from src.shared.python.loop_nullspace.feasible_set import FeasibleSet  # noqa: E402
from src.shared.python.loop_nullspace.probes import pareto_front  # noqa: E402
from src.shared.python.loop_nullspace.solvers import SolveStatus  # noqa: E402

pytestmark = pytest.mark.unit

S = 2.0
DT = 0.5
TIMES = np.array([0.0, DT])


def _decomp() -> Decomposition:
    counts = DimensionAccounting(
        n=1,
        n_a=1,
        n_u=0,
        m=0,
        n_g=0,
        dim_x=1,
        rank_A=0,
        nullity=1,
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
        x_p=np.zeros(1),
        N=np.ones((1, 1)),
        counts=counts,
        singular_values_scaled=np.ones(0),
        condition_number=1.0,
        equality_residual=0.0,
        metric_label="test",
        scales=CharacteristicScales(),
    )


def _set(lower: float, upper: float) -> FeasibleSet:
    rows = (
        FeasibleRow(np.array([-1.0]), -lower, "T:lo", "H-TORQUE-CAPACITY", "torque"),
        FeasibleRow(np.array([1.0]), upper, "T:up", "H-TORQUE-CAPACITY", "torque"),
    )
    fs = FeasibleSet.from_rows(1, rows)
    import dataclasses

    return dataclasses.replace(fs, decomposition=_decomp())


def _frames() -> list[FeasibleSet]:
    return [_set(1.0, 5.0), _set(-3.0, 3.0)]


def _effort() -> ObjectiveSpec:
    return ObjectiveSpec(TIMES, np.array([S]), effort_weight=1.0)


def _rate() -> ObjectiveSpec:
    return ObjectiveSpec(TIMES, np.array([S]), effort_weight=0.0, rate_weight=1.0)


def _analytic(eps: float) -> float:
    d = S * np.sqrt(eps * DT)
    if d >= 1.0:
        return (DT / 2) / S**2
    return (DT / 2) * (1.0 + (1.0 - d) ** 2) / S**2


EPS = np.array([0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0, 2.0])


def test_front_matches_the_closed_form_on_the_oracle() -> None:
    front = pareto_front(_frames(), (_effort(), _rate()), EPS)
    assert [p.eps for p in front.points] == list(EPS)
    for p in front.points:
        assert p.status is SolveStatus.OPTIMAL
        assert p.primary == pytest.approx(_analytic(p.eps), rel=1e-6, abs=1e-9)
        assert p.constrained <= p.eps + 1e-7
    knee = 1.0 / (S**2 * DT)  # d = 1  <=>  eps = 1 / (s^2 dt)
    for p in front.points:
        assert p.active == (p.eps < knee - 1e-6)


def test_front_is_monotone_and_nondominated() -> None:
    front = pareto_front(_frames(), (_effort(), _rate()), EPS)
    assert front.is_monotone()
    prim = [p.primary for p in front.solved()]
    assert all(b <= a + 1e-9 for a, b in zip(prim, prim[1:], strict=False))
    assert prim[0] > prim[3] > prim[-1] - 1e-12  # strictly improving before the knee
    assert all(not p.dominated for p in front.solved())
    assert len(front.nondominated()) == len(front.solved())


def test_dominated_point_detection_flags_a_dominated_solution() -> None:
    from src.shared.python.loop_nullspace.pareto import _mark_dominated, ParetoPoint

    ok = SolveStatus.OPTIMAL
    pts = [
        ParetoPoint(1.0, ok, primary=1.0, constrained=1.0),
        ParetoPoint(2.0, ok, primary=2.0, constrained=2.0),  # dominated by the first
        ParetoPoint(3.0, ok, primary=0.5, constrained=3.0),
    ]
    flags = [p.dominated for p in _mark_dominated(pts)]
    assert flags == [False, True, False]


def test_slope_is_the_derivative_of_the_front() -> None:
    front = pareto_front(_frames(), (_effort(), _rate()), [0.1, 0.1001, 0.0999])
    mid = next(p for p in front.points if p.eps == 0.1)
    hi = next(p for p in front.points if p.eps == 0.1001)
    lo = next(p for p in front.points if p.eps == 0.0999)
    fd = (hi.primary - lo.primary) / (hi.eps - lo.eps)
    assert mid.slope == pytest.approx(fd, rel=1e-3)
    assert mid.slope < 0.0


def test_points_are_family_dependent_with_all_hypotheses() -> None:
    front = pareto_front(_frames(), (_effort(), _rate()), EPS[:3])
    assert {"H-TORQUE-CAPACITY", "H-EFFORT-CAPACITY"} <= set(front.hypotheses)
    assert "H-SMOOTH-TORQUE-RATE" in front.hypotheses
    for p in front.solved():
        assert p.allocation_dependence == "FAMILY_DEPENDENT"
        assert p.record.allocation_dependence.name == "FAMILY_DEPENDENT"
        assert set(p.record.hypotheses) == set(front.hypotheses)
        assert p.kkt_residuals is not None
    assert "not an estimate" in front.non_claim
    assert front.primary_terms == ("effort",)
    assert front.constrained_terms == ("rate",)


def test_eps_below_the_attainable_rate_is_infeasible_not_relaxed() -> None:
    # z_1 in [1, 2] and z_2 in [3, 4]: |z_2 - z_1| >= 1 so rate >= 1 / (s^2 dt) = 0.5
    frames = [_set(1.0, 2.0), _set(3.0, 4.0)]
    front = pareto_front(frames, (_effort(), _rate()), [0.4, 5.0])
    low, high = front.points
    assert low.status is SolveStatus.INFEASIBLE
    assert low.primary is None and low.allocation_dependence == "UNAVAILABLE"
    assert high.status is SolveStatus.OPTIMAL
    assert front.solved() == (high,)


def test_front_serialises_to_json() -> None:
    front = pareto_front(_frames(), (_effort(), _rate()), EPS[:3])
    data = json.loads(front.to_json())
    assert data["schema"] == "loop-nullspace-pareto/1"
    assert len(data["points"]) == 3
    assert data["points"][0]["allocation_dependence"] == "FAMILY_DEPENDENT"
    assert front.to_json() == front.to_json()


def test_precondition_errors() -> None:
    frames = _frames()
    with pytest.raises(ValueError):
        pareto_front(frames, (_effort(),), EPS)
    with pytest.raises(TypeError):
        pareto_front(frames, (_effort(), "rate"), EPS)  # type: ignore[arg-type]
    for bad in ([], [-1.0], [0.1, 0.1], [np.nan], [[0.1]]):
        with pytest.raises(ValueError):
            pareto_front(frames, (_effort(), _rate()), bad)
    with pytest.raises(ValueError, match="no enabled quadratic"):
        pareto_front(
            frames,
            (_effort(), ObjectiveSpec(TIMES, np.array([S]), effort_weight=0.0)),
            EPS,
        )
    gp = GrossPowerSpec(1.0, np.ones(1), np.ones(1), np.ones((2, 1)))
    with pytest.raises(ValueError, match="gross power"):
        pareto_front(
            frames,
            (
                _effort(),
                ObjectiveSpec(TIMES, np.array([S]), effort_weight=1.0, gross_power=gp),
            ),
            EPS,
        )
    with pytest.raises(ValueError, match="times_s"):
        pareto_front(
            frames,
            (
                _effort(),
                ObjectiveSpec(np.array([0.0, 1.0]), np.array([S]), rate_weight=1.0),
            ),
            EPS,
        )


def test_minimised_objective_must_be_strictly_convex() -> None:
    with pytest.raises(NotStrictlyConvexError):
        pareto_front(_frames(), (_rate(), _effort()), EPS)


def test_reference_term_is_honoured() -> None:
    ref = np.array([[2.0], [2.0]])
    track = ObjectiveSpec(TIMES, np.array([S]), effort_weight=0.0, reference_weight=1.0)
    front = pareto_front(
        _frames(), (track, _rate()), [0.0, 5.0], references=(ref, None)
    )
    free = front.points[-1]
    assert free.torque[:, 0] == pytest.approx([2.0, 2.0], abs=1e-6)
    assert free.primary == pytest.approx(0.0, abs=1e-9)
    # eps = 0 forces z_1 = z_2: best constant tracks the reference exactly too
    assert front.points[0].primary == pytest.approx(0.0, abs=1e-9)
