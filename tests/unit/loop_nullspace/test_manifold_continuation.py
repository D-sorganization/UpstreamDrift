"""NSE-UD-2 (#13462): local solution-manifold continuation.

Tolerances (derived):

* Circle ``F(u) = |u|^2 - 1``.  For a unit tangent ``t`` at ``u0`` on the
  circle, ``F(u0 + h t) = h^2`` exactly, so the predictor residual has slope 2
  in ``log h``; the minimum-norm Newton step from ``u`` with residual ``F``
  gives ``F_new = F^2 / (4 |u|^2) <= F^2 / 4`` (as ``|u|^2 = 1 + F >= 1``), so
  the quadratic constant is at most ``1/4``; the tests allow ``0.26``.
* Orthonormality in the declared scaling is ``N^T diag(s)^-2 N = I`` (``N`` is
  the ``ForceNullSpace`` basis, which already carries ``s``); a few ``eps`` of
  SVD error, hence ``1e-12``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.shared.python.loop_nullspace.manifold_continuation import (
    ContinuationStep,
    ManifoldChart,
    ManifoldConvergenceError,
    ManifoldSingularityError,
    local_chart,
    predictor_corrector_step,
)

pytestmark = pytest.mark.unit

TOL = 1e-12


def _circle_f(u: np.ndarray) -> np.ndarray:
    return np.array([u @ u - 1.0])


def _circle_j(u: np.ndarray) -> np.ndarray:
    return 2.0 * u.reshape(1, -1)


def _fold_f(u: np.ndarray) -> np.ndarray:
    return np.array([u[0] - u[1] ** 2])


def _fold_j(u: np.ndarray) -> np.ndarray:
    return np.array([[1.0, -2.0 * u[1]]])


def _cross_f(u: np.ndarray) -> np.ndarray:
    return np.array([u[0] * u[1]])


def _cross_j(u: np.ndarray) -> np.ndarray:
    return np.array([[u[1], u[0]]])


def _sine_f(u: np.ndarray) -> np.ndarray:
    return np.array([u[1] - np.sin(u[0])])


def _sine_j(u: np.ndarray) -> np.ndarray:
    return np.array([[-np.cos(u[0]), 1.0]])


def _chart(f=_circle_f, j=_circle_j, u=(1.0, 0.0), scale=(1.0, 1.0)) -> ManifoldChart:
    return local_chart(f, j, np.array(u), variable_scale=np.array(scale), tol=TOL)


def _tangent(chart: ManifoldChart) -> np.ndarray:
    return chart.tangent_basis[:, 0]


# ---------------------------------------------------------------- local_chart


def test_circle_chart_has_rank_one_and_a_unit_tangent() -> None:
    chart = _chart(u=(0.6, 0.8))
    assert isinstance(chart, ManifoldChart)
    assert chart.rank == 1
    assert chart.tangent_basis.shape == (2, 1)
    np.testing.assert_allclose(np.abs(_tangent(chart)), [0.8, 0.6], atol=1e-12)
    np.testing.assert_allclose(chart.singular_values, [2.0], atol=1e-12)
    assert chart.condition_number == pytest.approx(1.0)


def test_chart_is_frozen_and_arrays_are_read_only() -> None:
    chart = _chart()
    with pytest.raises(AttributeError):
        chart.rank = 2  # type: ignore[misc]
    with pytest.raises(ValueError):
        chart.tangent_basis[0, 0] = 5.0
    with pytest.raises(ValueError):
        chart.u_base[0] = 5.0


def test_tangent_basis_annihilated_by_the_jacobian_in_raw_coordinates() -> None:
    chart = _chart(_sine_f, _sine_j, u=(0.7, np.sin(0.7)), scale=(3.0, 0.25))
    residual = _sine_j(chart.u_base) @ chart.tangent_basis
    np.testing.assert_allclose(residual, 0.0, atol=1e-12)


def test_tangent_basis_is_orthonormal_in_the_declared_scaling() -> None:
    scale = np.array([3.0, 0.25])
    chart = _chart(_sine_f, _sine_j, u=(0.7, np.sin(0.7)), scale=tuple(scale))
    scaled = chart.tangent_basis / scale[:, None]
    np.testing.assert_allclose(scaled.T @ scaled, np.eye(1), atol=1e-12)
    # ... and it is NOT orthonormal in raw coordinates (scaling is not applied twice).
    assert abs(float(chart.tangent_basis[:, 0] @ chart.tangent_basis[:, 0]) - 1) > 1e-3


def test_singular_values_are_those_of_the_scaled_jacobian() -> None:
    scale = np.array([3.0, 0.25])
    u = np.array([0.7, np.sin(0.7)])
    chart = _chart(_sine_f, _sine_j, u=tuple(u), scale=tuple(scale))
    expected = np.linalg.svd(_sine_j(u) * scale[None, :], compute_uv=False)
    np.testing.assert_allclose(chart.singular_values, expected, atol=1e-12)


def test_rank_zero_at_the_crossing_is_reported_not_hidden() -> None:
    chart = _chart(_cross_f, _cross_j, u=(0.0, 0.0))
    assert chart.rank == 0
    assert chart.condition_number == float("inf")


def test_local_chart_preconditions() -> None:
    good: dict[str, Any] = {"variable_scale": np.ones(2), "tol": TOL}
    u = np.array([1.0, 0.0])
    with pytest.raises(TypeError):
        local_chart("f", _circle_j, u, **good)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        local_chart(_circle_f, None, u, **good)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="u_base"):
        local_chart(_circle_f, _circle_j, np.array([1.0, np.nan]), **good)
    with pytest.raises(ValueError, match="u_base"):
        local_chart(_circle_f, _circle_j, np.ones((2, 1)), **good)
    with pytest.raises(ValueError, match="variable_scale"):
        local_chart(_circle_f, _circle_j, u, variable_scale=np.ones(3), tol=TOL)
    with pytest.raises(ValueError, match="variable_scale"):
        local_chart(_circle_f, _circle_j, u, variable_scale=np.array([1, 0.0]), tol=TOL)
    for bad_tol in (0.0, -1e-9, float("nan"), 1.0):
        with pytest.raises(ValueError, match="tol"):
            local_chart(_circle_f, _circle_j, u, variable_scale=np.ones(2), tol=bad_tol)
    with pytest.raises(ValueError, match="on the manifold"):
        local_chart(_circle_f, _circle_j, np.array([2.0, 0.0]), **good)
    with pytest.raises(ValueError, match="jacobian"):
        local_chart(_circle_f, lambda v: np.ones((2, 2)), u, **good)
    with pytest.raises(ValueError, match="residual"):
        local_chart(lambda v: np.array([np.inf]), _circle_j, u, **good)


# ------------------------------------------------------ predictor / corrector


def test_circle_step_lands_on_the_circle_with_the_requested_orientation() -> None:
    chart = _chart()
    out = predictor_corrector_step(
        chart,
        np.array([0.0, 1.0]),
        0.3,
        _circle_f,
        _circle_j,
        tol=TOL,
        max_iter=20,
    )
    assert isinstance(out, ContinuationStep)
    new = out.chart.u_base
    assert np.linalg.norm(_circle_f(new)) <= TOL
    assert new[1] > 0.25  # moved in the +u2 direction, not backwards
    assert out.chart.rank == chart.rank
    assert out.iterations == len(out.residual_history) - 1
    assert out.residual_history[-1] <= TOL


def test_postconditions_hold_for_a_scaled_problem() -> None:
    scale = np.array([3.0, 0.25])
    chart = _chart(_sine_f, _sine_j, u=(0.7, np.sin(0.7)), scale=tuple(scale))
    out = predictor_corrector_step(
        chart, _tangent(chart), 0.2, _sine_f, _sine_j, tol=TOL, max_iter=20
    )
    new = out.chart
    assert np.linalg.norm(_sine_f(new.u_base)) <= TOL
    scaled = new.tangent_basis / scale[:, None]
    np.testing.assert_allclose(scaled.T @ scaled, np.eye(1), atol=1e-12)
    np.testing.assert_allclose(new.variable_scale, scale)
    # The step is measured in the scaled norm, so the move is about 0.2 there.
    moved = np.linalg.norm((new.u_base - chart.u_base) / scale)
    assert moved == pytest.approx(0.2, rel=0.05)


def test_predictor_residual_slope_is_two() -> None:
    chart = _chart(_sine_f, _sine_j, u=(0.7, np.sin(0.7)))
    steps = np.array([1e-1, 5e-2, 2.5e-2, 1.25e-2, 6.25e-3])
    first = []
    for h in steps:
        out = predictor_corrector_step(
            chart, _tangent(chart), h, _sine_f, _sine_j, tol=1e-14, max_iter=20
        )
        first.append(out.residual_history[0])
    slope = np.polyfit(np.log(steps), np.log(first), 1)[0]
    assert slope == pytest.approx(2.0, abs=0.05)
    # No cubic claim: the residual is O(h^2), and at this point it is not smaller.
    assert first[0] / steps[0] ** 2 > 0.05


def test_corrector_converges_quadratically() -> None:
    chart = _chart()
    out = predictor_corrector_step(
        chart, np.array([0.0, 1.0]), 0.5, _circle_f, _circle_j, tol=1e-14, max_iter=20
    )
    history = out.residual_history
    assert out.iterations >= 3
    for prev, nxt in zip(history[:-1], history[1:], strict=True):
        if prev > 1e-7:
            assert nxt <= 0.26 * prev**2
    assert history[0] == pytest.approx(0.25, rel=1e-9)  # h^2 on the unit circle


def test_the_newton_correction_is_not_the_nearest_point_projection() -> None:
    # Off-manifold point u = (3, 4)/5 * 2 -> the nearest circle point is (0.6, 0.8),
    # but with an anisotropic scaling the minimum-scaled-norm step lands elsewhere.
    scale = np.array([4.0, 1.0])
    chart = _chart(scale=tuple(scale))
    out = predictor_corrector_step(
        chart, np.array([0.0, 1.0]), 0.6, _circle_f, _circle_j, tol=1e-13, max_iter=30
    )
    predicted = chart.u_base + 0.6 * _tangent(chart) * np.sign(_tangent(chart)[1])
    nearest = predicted / np.linalg.norm(predicted)
    assert np.linalg.norm(_circle_f(out.chart.u_base)) <= 1e-13
    assert np.linalg.norm(out.chart.u_base - nearest) > 1e-3


def test_fold_is_passed_without_a_rank_change() -> None:
    # {u1 = u2^2} is a graph over u2 with a turning point in u1 at the origin;
    # arclength continuation along the tangent crosses it at constant rank 1.
    chart = _chart(_fold_f, _fold_j, u=(1.0, -1.0))
    heading = np.array([-2.0, 1.0])
    points = [chart.u_base]
    for _ in range(16):
        out = predictor_corrector_step(
            chart, heading, 0.2, _fold_f, _fold_j, tol=TOL, max_iter=20
        )
        assert out.chart.rank == 1
        heading = out.chart.u_base - chart.u_base
        chart = out.chart
        points.append(chart.u_base)
    u1 = [float(p[0]) for p in points]
    assert min(u1) < 0.05
    assert points[-1][1] > 0.5
    assert u1.index(min(u1)) not in (0, len(u1) - 1)


def test_rank_drop_counterexample_raises() -> None:
    # F(u) = u1 u2: the zero set is a cross; at (1, 0) the rank is 1, at the origin 0.
    chart = _chart(_cross_f, _cross_j, u=(1.0, 0.0))
    np.testing.assert_allclose(np.abs(_tangent(chart)), [1.0, 0.0], atol=1e-12)
    with pytest.raises(ManifoldSingularityError, match="rank"):
        predictor_corrector_step(
            chart, np.array([-1.0, 0.0]), 1.0, _cross_f, _cross_j, tol=TOL, max_iter=20
        )
    # Short of the crossing the same chain is fine.
    ok = predictor_corrector_step(
        chart, np.array([-1.0, 0.0]), 0.5, _cross_f, _cross_j, tol=TOL, max_iter=20
    )
    assert ok.chart.rank == 1


def test_singular_error_is_a_runtime_error_and_distinct_from_convergence() -> None:
    assert issubclass(ManifoldSingularityError, RuntimeError)
    assert issubclass(ManifoldConvergenceError, RuntimeError)
    assert not issubclass(ManifoldConvergenceError, ManifoldSingularityError)


def test_non_convergence_within_max_iter_raises() -> None:
    chart = _chart()
    with pytest.raises(ManifoldConvergenceError, match="max_iter"):
        predictor_corrector_step(
            chart,
            np.array([0.0, 1.0]),
            1.0,
            _circle_f,
            _circle_j,
            tol=1e-14,
            max_iter=1,
        )


def test_step_preconditions() -> None:
    chart = _chart()
    d = np.array([0.0, 1.0])
    kw: dict[str, Any] = {"tol": TOL, "max_iter": 10}
    with pytest.raises(TypeError):
        predictor_corrector_step("chart", d, 0.1, _circle_f, _circle_j, **kw)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        predictor_corrector_step(chart, d, 0.1, None, _circle_j, **kw)  # type: ignore[arg-type]
    for bad_step in (0.0, -0.1, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="step"):
            predictor_corrector_step(chart, d, bad_step, _circle_f, _circle_j, **kw)
    with pytest.raises(ValueError, match="direction"):
        predictor_corrector_step(
            chart, np.array([0.0, np.nan]), 0.1, _circle_f, _circle_j, **kw
        )
    with pytest.raises(ValueError, match="direction"):
        predictor_corrector_step(chart, np.ones(3), 0.1, _circle_f, _circle_j, **kw)
    with pytest.raises(ValueError, match="tangent"):
        # Normal to the circle at (1, 0): no tangent component.
        predictor_corrector_step(
            chart, np.array([1.0, 0.0]), 0.1, _circle_f, _circle_j, **kw
        )
    with pytest.raises(ValueError, match="tol"):
        predictor_corrector_step(
            chart, d, 0.1, _circle_f, _circle_j, tol=0.0, max_iter=10
        )
    bad_iters: tuple[Any, ...] = (0, -1, 2.5)
    for bad_iter in bad_iters:
        with pytest.raises(ValueError, match="max_iter"):
            predictor_corrector_step(
                chart, d, 0.1, _circle_f, _circle_j, tol=TOL, max_iter=bad_iter
            )
