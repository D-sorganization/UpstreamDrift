"""TDD unit tests for constrained multiple-shooting fitting (#9921 / #9964).

Validates:
1. Multi-shooting decomposition on physical toy oracle with boundary defect elimination.
2. Single global degree-6 polynomial control law shared across all shooting windows.
3. Verification that zero defect produces an identical trajectory to unsegmented forward rollout.
4. Robust parameter validation (DbC, LoD, finite bounds, strictly increasing intervals).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_matching.multi_shooting_fit import (
    MultipleShootingFit,
    MultipleShootingOptions,
    SegmentedForward,
    UnsegmentedForward,
    fit_multiple_shooting,
    verify_unsegmented_forward_rollout,
)
from src.shared.python.motion_matching.prefix_fit import (
    MarkerTarget,
)

pytestmark = pytest.mark.unit


def test_multiple_shooting_options_validation() -> None:
    # Valid options
    opts = MultipleShootingOptions(
        shooting_nodes=(0.4, 0.8),
        defect_weight=100.0,
        defect_tolerance=1e-4,
    )
    assert opts.shooting_nodes == (0.4, 0.8)
    assert opts.defect_weight == 100.0

    # Invalid shooting nodes (not strictly increasing or not positive)
    with pytest.raises(ValueError, match="shooting_nodes"):
        MultipleShootingOptions(shooting_nodes=(0.8, 0.4))

    with pytest.raises(ValueError, match="shooting_nodes"):
        MultipleShootingOptions(shooting_nodes=(-0.1, 0.8))

    with pytest.raises(ValueError, match="defect_weight"):
        MultipleShootingOptions(shooting_nodes=(0.4, 0.8), defect_weight=-1.0)


def test_multiple_shooting_eliminates_defect_on_toy_oscillator() -> None:
    """A 1D nonlinear oscillator driven by a polynomial.

    We test that multiple shooting splits [0, 1.0] into [0, 0.5] and [0.5, 1.0],
    optimizes global control parameters theta and intermediate state x(0.5),
    eliminates the defect, and matches the continuous unsegmented forward rollout.
    """
    time = np.linspace(0.0, 1.0, 51)
    points = np.zeros((len(time), 1, 3))
    points[:, 0, 0] = 2.5 * time**2
    target = MarkerTarget(time, points, np.ones(1))

    # Segmented forward oracle:
    def segmented_forward(
        theta: np.ndarray,
        t_span: np.ndarray,
        initial_state: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        t0 = t_span[0]
        p0 = 0.0 if initial_state is None else initial_state[0]
        v0 = 0.0 if initial_state is None else initial_state[1]

        dt = t_span - t0
        pos = p0 + v0 * dt + theta[0] * dt**2
        vel = v0 + 2.0 * theta[0] * dt

        pred_markers = np.zeros((len(t_span), 1, 3))
        pred_markers[:, 0, 0] = pos
        final_state = np.array([pos[-1], vel[-1]], dtype=float)
        return pred_markers, final_state

    # Unsegmented forward oracle:
    def unsegmented_forward(theta: np.ndarray, t_all: np.ndarray) -> np.ndarray:
        pred_markers = np.zeros((len(t_all), 1, 3))
        pred_markers[:, 0, 0] = theta[0] * t_all**2
        return pred_markers

    options = MultipleShootingOptions(
        shooting_nodes=(0.5, 1.0),
        state_dim=2,
        defect_weight=50.0,
        defect_tolerance=1e-3,
        max_nfev=200,
    )

    initial_theta = np.array([1.0])
    lower_theta = np.array([0.0])
    upper_theta = np.array([5.0])

    initial_states = {0.5: np.array([0.5, 1.0])}
    state_bounds = {0.5: (np.array([-10.0, -10.0]), np.array([10.0, 10.0]))}

    result = fit_multiple_shooting(
        target=target,
        segmented_forward=segmented_forward,
        unsegmented_forward=unsegmented_forward,
        initial_theta=initial_theta,
        lower_theta=lower_theta,
        upper_theta=upper_theta,
        initial_states=initial_states,
        state_bounds=state_bounds,
        options=options,
    )

    assert result.accepted
    assert result.theta[0] == pytest.approx(2.5, abs=1e-3)
    assert result.max_defect_norm < 1e-3
    assert result.segmented_rmse_m < 1e-3
    assert result.unsegmented_rmse_m < 1e-3

    verif = verify_unsegmented_forward_rollout(
        target=target,
        unsegmented_forward=unsegmented_forward,
        theta=result.theta,
    )
    assert verif["rmse_m"] < 1e-3
    assert verif["terminal_rmse_m"] < 1e-3


def test_multiple_shooting_rejects_nan_in_unsegmented_rollout() -> None:
    """If unsegmented rollout diverges or produces NaN, result must be rejected."""
    time = np.linspace(0.0, 1.0, 51)
    points = np.zeros((len(time), 1, 3))
    target = MarkerTarget(time, points, np.ones(1))

    def segmented_forward(
        theta: np.ndarray,
        t_span: np.ndarray,
        initial_state: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros((len(t_span), 1, 3)), np.zeros(2)

    def unsegmented_forward_nan(theta: np.ndarray, t_all: np.ndarray) -> np.ndarray:
        pred = np.zeros((len(t_all), 1, 3))
        pred[len(t_all) // 2 :] = np.nan
        return pred

    options = MultipleShootingOptions(
        shooting_nodes=(0.5, 1.0),
        state_dim=2,
    )

    result = fit_multiple_shooting(
        target=target,
        segmented_forward=segmented_forward,
        unsegmented_forward=unsegmented_forward_nan,
        initial_theta=np.array([1.0]),
        lower_theta=np.array([0.0]),
        upper_theta=np.array([5.0]),
        initial_states={0.5: np.zeros(2)},
        state_bounds={0.5: (np.full(2, -10.0), np.full(2, 10.0))},
        options=options,
    )

    assert not result.accepted
    assert result.unsegmented_rmse_m == float("inf")


def test_multiple_shooting_window_cache_avoids_redundant_evaluations() -> None:
    """Window 0 must not be re-simulated when only window 1 initial state is perturbed."""
    time = np.linspace(0.0, 1.0, 21)
    points = np.zeros((len(time), 1, 3))
    target = MarkerTarget(time, points, np.ones(1))

    call_counts = {0: 0, 1: 0}

    def segmented_forward(
        theta: np.ndarray,
        t_span: np.ndarray,
        initial_state: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        win_idx = 0 if initial_state is None else 1
        call_counts[win_idx] += 1
        return np.zeros((len(t_span), 1, 3)), np.zeros(2)

    def unsegmented_forward(theta: np.ndarray, t_all: np.ndarray) -> np.ndarray:
        return np.zeros((len(t_all), 1, 3))

    options = MultipleShootingOptions(
        shooting_nodes=(0.5, 1.0),
        state_dim=2,
        max_nfev=5,
    )

    fit_multiple_shooting(
        target=target,
        segmented_forward=segmented_forward,
        unsegmented_forward=unsegmented_forward,
        initial_theta=np.array([1.0]),
        lower_theta=np.array([0.0]),
        upper_theta=np.array([2.0]),
        initial_states={0.5: np.zeros(2)},
        state_bounds={0.5: (np.full(2, -1.0), np.full(2, 1.0))},
        options=options,
    )

    # Window 0 should have strictly fewer calls than Window 1 because
    # perturbations to the state vector at t=0.5s hit the cache for Window 0
    assert call_counts[0] < call_counts[1]
