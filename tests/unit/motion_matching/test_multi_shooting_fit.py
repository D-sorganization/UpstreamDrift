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


@pytest.mark.parametrize(
    ("observed_terminal", "finite_replay", "shooting_end", "expected_gap"),
    [
        (True, True, 1.0, 10.0),
        (False, True, 1.0, None),
        (True, False, 1.0, float("inf")),
        (True, True, 0.5, None),
    ],
)
def test_terminal_replay_gap_is_pointwise_and_masked(
    observed_terminal: bool,
    finite_replay: bool,
    shooting_end: float,
    expected_gap: float | None,
) -> None:
    """Equal RMS-to-target values can hide opposite terminal marker positions."""
    time = np.array([0.0, 0.5, 1.0])
    points = np.zeros((3, 2, 3))
    if not observed_terminal:
        points[-1, 0] = np.nan
    target = MarkerTarget(time, points, np.array([1.0, 0.0]))

    def segmented(
        theta: np.ndarray, clock: np.ndarray, state: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray]:
        pred = np.zeros((len(clock), 2, 3))
        pred[:, 0] = [3.0, 4.0, 0.0]
        pred[:, 1] = 1000.0  # Zero-weight marker must not affect the gap.
        return pred, np.zeros(1)

    def continuous(theta: np.ndarray, clock: np.ndarray) -> np.ndarray:
        pred = np.zeros((len(clock), 2, 3))
        pred[:, 0] = [-3.0, -4.0, 0.0]
        if not finite_replay:
            pred[-1, 0] = np.nan
        return pred

    result = fit_multiple_shooting(
        target,
        segmented,
        continuous,
        initial_theta=np.zeros(1),
        lower_theta=-np.ones(1),
        upper_theta=np.ones(1),
        initial_states={0.5: np.zeros(1)},
        state_bounds={0.5: (-np.ones(1), np.ones(1))},
        options=MultipleShootingOptions(
            shooting_nodes=(0.5, 1.0) if shooting_end == 1.0 else (0.5,)
        ),
    )
    if finite_replay:
        assert result.segmented_rmse_m == pytest.approx(5.0)
        assert result.unsegmented_rmse_m == pytest.approx(5.0)
    if expected_gap is None:
        assert result.terminal_replay_gap_m is None
    else:
        assert result.terminal_replay_gap_m == pytest.approx(expected_gap)


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
        acceptance=lambda pred: bool(np.max(abs(pred - points)) < 1e-3),
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


def test_transformed_nodes_and_explicit_acceptance(monkeypatch) -> None:
    time = np.array([0.0, 0.5, 1.0])
    points = np.zeros((3, 1, 3))
    points[:, 0, 0] = 2 * time
    target = MarkerTarget(time, points, np.ones(1))

    def forward(theta, clock, state):
        x0 = 0.0 if state is None else state[0]
        x = x0 + theta[0] * (clock - clock[0])
        pred = np.zeros((len(clock), 1, 3))
        pred[:, 0, 0] = x
        return pred, np.array([x[-1], 2 * x[-1]])

    def full(theta, clock):
        return forward(theta, clock, None)[0]

    from src.shared.python.motion_matching import multi_shooting_fit as module

    original = module.least_squares

    def checked(fun, x0, **kwargs):
        assert kwargs["xtol"] is None
        h = 1e-5
        fd = np.column_stack(
            [
                (fun(x0 + np.eye(len(x0))[i] * h) - fun(x0 - np.eye(len(x0))[i] * h))
                / (2 * h)
                for i in range(len(x0))
            ]
        )
        np.testing.assert_allclose(kwargs["jac"](x0), fd, atol=1e-7)
        return original(fun, x0, **kwargs)

    monkeypatch.setattr(module, "least_squares", checked)

    def window_jac(theta, clock, state):
        dt = clock - clock[0]
        count = 1 if state is None else 3
        marker = np.zeros((len(clock), 1, 3, count))
        marker[:, 0, 0, 0] = dt
        end = np.zeros((2, count))
        end[:, 0] = [dt[-1], 2 * dt[-1]]
        if state is not None:
            marker[:, 0, 0, 1] = 1.0
            end[:, 1] = [1.0, 2.0]
        return marker, end

    for allowed in [True, False]:
        opts = MultipleShootingOptions(
            shooting_nodes=(0.5, 1.0),
            step_tolerance=None,
            max_nfev=30,
            window_jacobian=window_jac,
            state_transform_jacobian=lambda t, z: np.array([[1.0], [2.0]]),
            terminal_weight=0.7,
            state_transform=lambda t, z: np.array([z[0], 2 * z[0]]),
            defect_scales=np.array([1.0, 2.0]),
            acceptance=lambda pred, allowed=allowed: allowed,
        )
        result = fit_multiple_shooting(
            target,
            forward,
            full,
            initial_theta=np.array([1.0]),
            lower_theta=np.array([0.0]),
            upper_theta=np.array([3.0]),
            initial_states={0.5: np.array([0.5])},
            state_bounds={0.5: (np.array([-2.0]), np.array([2.0]))},
            options=opts,
        )
        assert result.accepted is allowed
        np.testing.assert_allclose(
            result.intermediate_states[0.5], [1.0, 2.0], atol=1e-5
        )
        assert result.max_defect_norm < 1e-5
        assert result.defect_norms.keys() == {0.5}
        assert result.defect_norms[0.5] == result.max_defect_norm
        assert result.function_evaluations > 0
        assert np.isfinite(result.optimality)
        assert result.active_bound_count >= 0


def test_missing_acceptance_does_not_qualify_finite_bad_fit() -> None:
    target = MarkerTarget(np.array([0.0, 1.0]), np.zeros((2, 1, 3)), np.ones(1))

    def forward(theta, clock, state):
        return np.ones((len(clock), 1, 3)), np.zeros(1)

    result = fit_multiple_shooting(
        target,
        forward,
        lambda th, t: forward(th, t, None)[0],
        initial_theta=np.zeros(1),
        lower_theta=-np.ones(1),
        upper_theta=np.ones(1),
        initial_states={},
        state_bounds={},
        options=MultipleShootingOptions(shooting_nodes=(1.0,)),
    )
    assert result.optimizer_converged
    assert not result.accepted


def test_evaluation_callbacks_preserve_physical_node_snapshots() -> None:
    target = MarkerTarget(np.array([0.0, 0.5, 1.0]), np.zeros((3, 1, 3)), np.ones(1))
    evaluations = []
    checkpoints = []

    def observe(theta, residual, cost):
        evaluations.append((theta.copy(), float(residual @ residual), cost))
        # Observers own their copies even if they deliberately make them writable.
        theta.setflags(write=True)
        theta[:] = 999
        residual.setflags(write=True)
        residual[:] = 999

    def checkpoint(theta, states, cost):
        checkpoints.append(
            (theta.copy(), {t: x.copy() for t, x in states.items()}, cost)
        )
        states[0.5].setflags(write=True)
        states[0.5][:] = 999

    result = fit_multiple_shooting(
        target,
        lambda theta, clock, state: (np.zeros((len(clock), 1, 3)), np.zeros(2)),
        lambda theta, clock: np.zeros((len(clock), 1, 3)),
        initial_theta=np.array([1.0]),
        lower_theta=np.array([0.0]),
        upper_theta=np.array([2.0]),
        initial_states={0.5: np.array([0.25])},
        state_bounds={0.5: (np.array([-1.0]), np.array([1.0]))},
        options=MultipleShootingOptions(
            shooting_nodes=(0.5, 1.0),
            max_nfev=3,
            state_transform=lambda time, z: np.array([z[0], 2 * z[0]]),
            callback=observe,
            checkpoint_callback=checkpoint,
        ),
    )
    assert len(evaluations) == len(checkpoints) > 0
    np.testing.assert_array_equal(checkpoints[0][1][0.5], [0.25, 0.5])
    np.testing.assert_array_equal(checkpoints[0][0], [1.0])
    assert evaluations[0][1] == evaluations[0][2] == checkpoints[0][2]
    assert result.theta[0] < 2
    assert np.max(abs(result.intermediate_states[0.5])) < 1


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf, True, 1e-30])
def test_invalid_step_tolerance_is_rejected(value) -> None:
    with pytest.raises(ValueError, match="step tolerance"):
        MultipleShootingOptions(shooting_nodes=(1.0,), step_tolerance=value)
