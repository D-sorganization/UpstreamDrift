"""Tests for the batch swing optimizer (epic #8390, B5/#8400).

Runs entirely on the dependency-free CPU reference path ('ode' backend),
per the acceptance criteria: accelerators change throughput only.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.optimization.batch_swing_optimizer import (
    BatchSwingObjectiveWeights,
    BatchSwingOptimizer,
    BatchSwingResult,
    score_batch,
    terminal_clubhead_speed,
)
from src.shared.python.simulation_backends.model_params import GolfModelParams
from src.shared.python.simulation_backends.protocol import BatchTrace

pytestmark = pytest.mark.unit


def _trace(q: np.ndarray, v: np.ndarray, dt: float = 0.01) -> BatchTrace:
    t = np.arange(q.shape[1]) * dt
    return BatchTrace(t=t, q=q, v=v, dt=dt, backend="test")


def test_terminal_clubhead_speed_matches_hand_computation() -> None:
    params = GolfModelParams.default()
    l1, l2 = params.upper.length_m, params.lower.length_m
    # Straight configuration (q=0) rotating at the shoulder only.
    q = np.zeros((1, 3, 2))
    v = np.zeros((1, 3, 2))
    v[0, -1, 0] = 2.0  # rad/s at the shoulder, wrist frozen
    speed = terminal_clubhead_speed(_trace(q, v), params)
    assert speed[0] == pytest.approx((l1 + l2) * 2.0)


def test_score_batch_prefers_faster_lower_effort_swings() -> None:
    params = GolfModelParams.default()
    weights = BatchSwingObjectiveWeights()
    q = np.zeros((2, 4, 2))
    v = np.zeros((2, 4, 2))
    v[1, -1, 0] = 3.0  # env 1 ends faster
    controls = np.zeros((2, 3, 2))
    scores = score_batch(_trace(q, v), controls, params, weights)
    assert scores[1] > scores[0]


def test_optimizer_improves_over_initial_mean_on_cpu_path() -> None:
    optimizer = BatchSwingOptimizer("ode", n_candidates=16, n_iterations=3, seed=7)
    result = optimizer.optimize(horizon=30, dt=0.01)
    assert isinstance(result, BatchSwingResult)
    assert result.improved
    assert result.best_score > result.initial_score
    assert result.best_controls.shape == (30, 2)
    assert result.candidates_evaluated == 1 + 16 * 3
    assert result.backend == "cpu_batch[ode]"
    assert len(result.score_history) == 3
    # Best score is monotone across iterations by construction.
    assert list(result.score_history) == sorted(result.score_history)


def test_optimizer_is_deterministic_for_fixed_seed() -> None:
    a = BatchSwingOptimizer("ode", n_candidates=8, n_iterations=2, seed=3).optimize(
        horizon=20, dt=0.01
    )
    b = BatchSwingOptimizer("ode", n_candidates=8, n_iterations=2, seed=3).optimize(
        horizon=20, dt=0.01
    )
    assert a.best_score == b.best_score
    np.testing.assert_array_equal(a.best_controls, b.best_controls)


def test_mppi_method_also_improves() -> None:
    result = BatchSwingOptimizer(
        "ode", method="mppi", n_candidates=16, n_iterations=3, seed=5
    ).optimize(horizon=20, dt=0.01)
    assert result.method == "mppi"
    assert result.improved


def test_controls_respect_torque_bound() -> None:
    result = BatchSwingOptimizer(
        "ode", n_candidates=8, n_iterations=2, seed=1, tau_max=50.0
    ).optimize(horizon=15, dt=0.01)
    assert np.all(np.abs(result.best_controls) <= 50.0 + 1e-12)


def test_parameter_validation() -> None:
    with pytest.raises(ValueError, match="method"):
        BatchSwingOptimizer(method="annealing")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_candidates"):
        BatchSwingOptimizer(n_candidates=1)
    with pytest.raises(ValueError, match="elite_fraction"):
        BatchSwingOptimizer(elite_fraction=0.0)
    with pytest.raises(ValueError, match="tau_max"):
        BatchSwingOptimizer(tau_max=-1.0)
    with pytest.raises(ValueError, match="horizon"):
        BatchSwingOptimizer().optimize(horizon=0)
    with pytest.raises(ValueError, match="finite and non-negative"):
        BatchSwingObjectiveWeights(effort=-1.0)
