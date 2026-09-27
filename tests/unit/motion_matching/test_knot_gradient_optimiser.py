"""Unit tests for the JAX-free knot basis, horizon mask, and Adam driver."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.shared.python.core.contracts import PreconditionError
from src.shared.python.motion_matching.knot_gradient_optimiser import (
    AdamSettings,
    KnotOptimisationResult,
    adam_minimise,
    horizon_knot_mask,
    knot_basis,
    knot_grid,
)

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# knot_grid tests
# ---------------------------------------------------------------------------


def test_knot_grid_exact_span() -> None:
    times = np.linspace(0.0, 1.60, 41)
    knots = knot_grid(times, 0.04)
    # (1.60 - 0.0) / 0.04 = 40.0 -> count is 41
    assert len(knots) == 41
    assert np.isclose(knots[0], 0.0)
    assert np.isclose(knots[-1], 1.60)
    assert np.allclose(knots, np.linspace(0.0, 1.60, 41))


def test_knot_grid_non_exact_span_and_no_untouched_trailing_knot() -> None:
    # 0 to 1.65 s with spacing 0.04
    times = np.linspace(0.0, 1.65, 166)
    knots = knot_grid(times, 0.04)
    # 1.65 / 0.04 = 41.25 -> ceil is 42 -> count is 43 (knots up to 1.68)
    assert len(knots) == 43
    assert np.isclose(knots[0], 0.0)
    assert np.isclose(knots[-2], 1.64)
    assert np.isclose(knots[-1], 1.68)

    # Every knot including the last one must be touched by the basis
    basis = knot_basis(times, knots)
    col_sums = np.sum(basis, axis=0)
    assert np.all(col_sums > 0.0)
    assert col_sums[-1] > 0.0


def test_knot_grid_contracts() -> None:
    with pytest.raises(PreconditionError):
        knot_grid(np.ones((2, 2)), 0.04)

    with pytest.raises(PreconditionError):
        knot_grid(np.array([1.0]), 0.04)

    with pytest.raises(PreconditionError):
        knot_grid(np.array([1.0, 1.0]), 0.04)

    with pytest.raises(PreconditionError):
        knot_grid(np.array([1.0, 0.5]), 0.04)

    with pytest.raises(PreconditionError):
        knot_grid(np.array([0.0, np.nan]), 0.04)

    with pytest.raises(PreconditionError):
        knot_grid(np.array([0.0, 1.0]), 0.0)

    with pytest.raises(PreconditionError):
        knot_grid(np.array([0.0, 1.0]), -0.04)


# ---------------------------------------------------------------------------
# knot_basis tests
# ---------------------------------------------------------------------------


def test_knot_basis_row_sum_and_exact_at_knots() -> None:
    knots = np.linspace(0.0, 1.0, 6)
    times = np.linspace(0.0, 1.0, 51)
    basis = knot_basis(times, knots)
    assert basis.shape == (51, 6)
    assert np.allclose(np.sum(basis, axis=1), 1.0)
    assert np.all(basis >= 0.0)

    # Exact at knots: basis at knot points is the identity matrix
    basis_exact = knot_basis(knots, knots)
    assert np.allclose(basis_exact, np.eye(6))


def test_knot_basis_linear_signal_reproduction() -> None:
    knots = np.linspace(0.0, 2.0, 11)
    times = np.linspace(0.0, 2.0, 101)
    slope, intercept = 3.5, -1.2
    y_knots = slope * knots + intercept
    basis = knot_basis(times, knots)
    y_interp = basis @ y_knots
    y_true = slope * times + intercept
    assert np.allclose(y_interp, y_true, atol=1e-12)


def test_knot_basis_contracts() -> None:
    knots = np.linspace(0.0, 1.0, 5)

    # Non-monotone times
    with pytest.raises(PreconditionError):
        knot_basis(np.array([0.0, 0.5, 0.3, 1.0]), knots)

    # Untouched knots (knots 2.0 and 3.0 untouched by times)
    untouched_knots = np.array([0.0, 1.0, 2.0, 3.0])
    times_short = np.array([0.0, 0.2, 0.5, 0.8])
    with pytest.raises(PreconditionError):
        knot_basis(times_short, untouched_knots)

    # Times outside knot span
    with pytest.raises(PreconditionError):
        knot_basis(np.array([-0.1, 0.5]), np.array([0.0, 1.0]))

    with pytest.raises(PreconditionError):
        knot_basis(np.array([0.0, 1.1]), np.array([0.0, 1.0]))


# ---------------------------------------------------------------------------
# horizon_knot_mask tests
# ---------------------------------------------------------------------------


def test_horizon_knot_mask() -> None:
    knots = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    mask = horizon_knot_mask(knots, 1.2)
    expected = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
    assert np.array_equal(mask, expected)

    mask_exact = horizon_knot_mask(knots, 1.0)
    assert np.array_equal(mask_exact, np.array([1.0, 1.0, 1.0, 0.0, 0.0]))

    with pytest.raises(PreconditionError):
        horizon_knot_mask(knots, 0.0)

    with pytest.raises(PreconditionError):
        horizon_knot_mask(knots, -1.0)


# ---------------------------------------------------------------------------
# AdamSettings tests
# ---------------------------------------------------------------------------


def test_adam_settings_contracts() -> None:
    settings = AdamSettings(learning_rate=0.01, max_iterations=10)
    assert settings.learning_rate == 0.01
    assert settings.beta1 == 0.9
    assert settings.beta2 == 0.999
    assert settings.eps == 1e-8
    assert settings.max_iterations == 10

    with pytest.raises(PreconditionError):
        AdamSettings(learning_rate=0.0, max_iterations=10)
    with pytest.raises(PreconditionError):
        AdamSettings(learning_rate=-0.01, max_iterations=10)

    with pytest.raises(PreconditionError):
        AdamSettings(learning_rate=0.01, beta1=-0.1, max_iterations=10)
    with pytest.raises(PreconditionError):
        AdamSettings(learning_rate=0.01, beta1=1.0, max_iterations=10)

    with pytest.raises(PreconditionError):
        AdamSettings(learning_rate=0.01, beta2=-0.1, max_iterations=10)
    with pytest.raises(PreconditionError):
        AdamSettings(learning_rate=0.01, beta2=1.0, max_iterations=10)

    with pytest.raises(PreconditionError):
        AdamSettings(learning_rate=0.01, eps=0.0, max_iterations=10)

    with pytest.raises(PreconditionError):
        AdamSettings(learning_rate=0.01, max_iterations=-1)


# ---------------------------------------------------------------------------
# adam_minimise tests
# ---------------------------------------------------------------------------


def test_adam_first_three_iterates_handwritten_recurrence() -> None:
    # f(x) = 1/2 * ||A x - b||^2
    mat_a = np.array([[2.0, 0.5], [0.5, 1.5]])
    vec_b = np.array([1.0, -2.0])
    x0 = np.array([0.5, -0.5])
    lr = 0.05
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8

    def val_and_grad(x: np.ndarray) -> tuple[tuple[float, float], np.ndarray]:
        residual = mat_a @ x - vec_b
        cost = 0.5 * float(np.sum(residual**2))
        grad = mat_a.T @ residual
        return (cost, cost), grad

    # Independently hand-written bias-corrected Adam recurrence in the test itself
    x_ref = x0.copy()
    m_ref = np.zeros_like(x_ref)
    v_ref = np.zeros_like(x_ref)
    iterates_ref = [x_ref.copy()]
    for k in range(1, 4):
        residual_k = mat_a @ x_ref - vec_b
        g_k = mat_a.T @ residual_k
        m_ref = b1 * m_ref + (1.0 - b1) * g_k
        v_ref = b2 * v_ref + (1.0 - b2) * (g_k**2)
        m_hat = m_ref / (1.0 - b1**k)
        v_hat = v_ref / (1.0 - b2**k)
        step = lr * m_hat / (np.sqrt(v_hat) + eps)
        x_ref = x_ref - step
        iterates_ref.append(x_ref.copy())

    iterates_opt: list[np.ndarray] = []

    def on_iter(_k: int, x: np.ndarray, _total: float, _obj: float) -> None:
        iterates_opt.append(x.copy())

    settings = AdamSettings(
        learning_rate=lr, beta1=b1, beta2=b2, eps=eps, max_iterations=3
    )
    result = adam_minimise(val_and_grad, x0, settings, on_iteration=on_iter)

    assert len(iterates_opt) == 4
    for k in range(4):
        assert np.allclose(iterates_opt[k], iterates_ref[k], atol=1e-12, rtol=1e-12)

    assert len(result.history) == 4
    assert result.stop_reason == "max_iterations"


def test_adam_converges_on_convex_quadratic() -> None:
    mat_a = np.array([[3.0, 1.0], [1.0, 2.0]])
    vec_b = np.array([2.0, -1.0])
    x_star = np.linalg.solve(mat_a, vec_b)
    x0 = np.zeros(2)

    def val_and_grad(x: np.ndarray) -> tuple[tuple[float, float], np.ndarray]:
        residual = mat_a @ x - vec_b
        cost = 0.5 * float(np.sum(residual**2))
        grad = mat_a.T @ residual
        return (cost, cost), grad

    settings = AdamSettings(learning_rate=0.1, max_iterations=600)
    result = adam_minimise(val_and_grad, x0, settings)

    assert np.allclose(result.best_x, x_star, atol=1e-6)
    assert result.best_objective < 1e-12
    assert result.stop_reason == "max_iterations"


def test_adam_best_iterate_not_last_when_objective_rises() -> None:
    objectives = [10.0, 5.0, 2.0, 0.5, 3.0, 7.0]
    call_count = 0

    def val_and_grad(x: np.ndarray) -> tuple[tuple[float, float], np.ndarray]:
        nonlocal call_count
        idx = min(call_count, len(objectives) - 1)
        obj = objectives[idx]
        call_count += 1
        return (obj + 1.0, obj), np.array([0.1])

    x0 = np.array([0.0])
    settings = AdamSettings(learning_rate=0.01, max_iterations=5)
    result = adam_minimise(val_and_grad, x0, settings)

    assert result.best_iteration == 3
    assert np.isclose(result.best_objective, 0.5)
    assert result.history[-1]["objective"] == 7.0
    assert result.history[3]["objective"] == 0.5
    assert not result.best_x.flags.writeable


def test_adam_injected_nan_gradient_stops() -> None:
    step_count = 0

    def val_and_grad(x: np.ndarray) -> tuple[tuple[float, float], np.ndarray]:
        nonlocal step_count
        step_count += 1
        if step_count == 4:  # iteration 3 evaluate after step 3
            return (1.0, 1.0), np.array([np.nan])
        return (float(step_count), float(step_count)), np.array([0.1])

    x0 = np.array([0.0])
    settings = AdamSettings(learning_rate=0.01, max_iterations=10)
    result = adam_minimise(val_and_grad, x0, settings)

    assert result.stop_reason == "non_finite_gradient"
    assert result.best_iteration <= 2
    assert len(result.history) == 4
    assert np.isfinite(result.best_objective)


def test_adam_injected_nan_cost_stops() -> None:
    step_count = 0

    def val_and_grad(x: np.ndarray) -> tuple[tuple[float, float], np.ndarray]:
        nonlocal step_count
        step_count += 1
        if step_count == 4:  # iteration 3
            return (np.nan, np.nan), np.array([0.1])
        return (float(step_count), float(step_count)), np.array([0.1])

    x0 = np.array([0.0])
    settings = AdamSettings(learning_rate=0.01, max_iterations=10)
    result = adam_minimise(val_and_grad, x0, settings)

    assert result.stop_reason == "non_finite_cost"
    assert result.best_iteration <= 2
    assert len(result.history) == 4


def test_adam_max_iterations_zero() -> None:
    x0 = np.array([1.0, 2.0, 3.0])

    def val_and_grad(x: np.ndarray) -> tuple[tuple[float, float], np.ndarray]:
        return (10.0, 5.0), np.array([0.1, 0.2, 0.3])

    settings = AdamSettings(learning_rate=0.01, max_iterations=0)
    result = adam_minimise(val_and_grad, x0, settings)

    assert len(result.history) == 1
    assert result.history[0]["iteration"] == 0
    assert result.history[0]["total"] == 10.0
    assert result.history[0]["objective"] == 5.0
    assert result.best_iteration == 0
    assert np.isclose(result.best_objective, 5.0)
    assert np.array_equal(result.best_x, x0)
    assert result.stop_reason == "max_iterations"
    assert not result.best_x.flags.writeable


def test_adam_best_selection_uses_objective_not_total() -> None:
    # At iter 0: total 10, obj 10
    # At iter 1: total 1, obj 9
    # At iter 2: total 5, obj 2 (total higher than iter 1, but obj lower!)
    evals = [(10.0, 10.0), (1.0, 9.0), (5.0, 2.0)]
    idx = 0

    def val_and_grad(x: np.ndarray) -> tuple[tuple[float, float], np.ndarray]:
        nonlocal idx
        cost_pair = evals[idx]
        idx += 1
        return cost_pair, np.array([0.1])

    x0 = np.array([0.0])
    settings = AdamSettings(learning_rate=0.01, max_iterations=2)
    result = adam_minimise(val_and_grad, x0, settings)

    assert result.best_iteration == 2
    assert np.isclose(result.best_objective, 2.0)


def test_adam_minimise_does_not_freeze_or_alias_caller_x0() -> None:
    """The caller's x0 stays writable even when x0 itself is the best iterate."""
    x0 = np.array([1.0, -2.0])

    def rising(x: np.ndarray) -> tuple[tuple[float, float], np.ndarray]:
        return (1.0, 1.0 + float(np.sum((x - x0) ** 2))), -2.0 * (x - x0) - 1.0

    result = adam_minimise(
        rising, x0, AdamSettings(learning_rate=0.1, max_iterations=3)
    )
    assert result.best_iteration == 0
    assert x0.flags["WRITEABLE"]
    assert result.best_x is not x0
    x0[0] = 5.0
    assert result.best_x[0] == 1.0
