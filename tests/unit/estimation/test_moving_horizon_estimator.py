from __future__ import annotations

import numpy as np

from src.shared.python.estimation import (
    MapEstimatorOptions,
    MovingHorizonEstimator,
    MovingHorizonOptions,
    MovingHorizonProblem,
)


def _tracking_problem(
    *,
    window_size: int = 3,
    step_size: int = 1,
    callback=None,
) -> MovingHorizonProblem:
    options = MovingHorizonOptions(
        window_size=window_size,
        step_size=step_size,
        latency_budget_ms=100.0,
        solver_options=MapEstimatorOptions(max_iterations=25),
    )

    def residual(evaluation, parameters):
        scale = parameters["scale"]
        truth = scale * evaluation.times
        return scale * evaluation.q[:, 0] - truth

    def jacobian(evaluation, parameters, layout):
        jac = np.zeros((evaluation.times.size, layout.size), dtype=float)
        jac[:, : layout.trajectory_size] = (
            parameters["scale"] * evaluation.q_basis[:, 0, :]
        )
        return jac

    return MovingHorizonProblem(
        n_dof=1,
        fixed_parameters={"scale": 2.0},
        residual=residual,
        jacobian=jacobian,
        options=options,
        callback=callback,
    )


def test_window_advances_deterministically_and_retains_bounded_samples() -> None:
    estimator = MovingHorizonEstimator(_tracking_problem(window_size=3, step_size=1))

    estimator.append_samples([0.0, 0.5, 1.0], np.array([[0.0], [0.5], [1.0]]))
    first = estimator.solve_next()
    estimator.append_samples([1.5], np.array([[1.5]]))
    second = estimator.solve_next()

    assert first is not None
    assert second is not None
    assert estimator.buffered_sample_count == 3
    assert first.sample_start == 0
    assert first.sample_stop == 3
    assert second.sample_start == 1
    assert second.sample_stop == 4
    np.testing.assert_allclose(first.window_times, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(second.window_times, [0.5, 1.0, 1.5])


def test_state_carryover_warm_starts_from_previous_window_solution() -> None:
    estimator = MovingHorizonEstimator(_tracking_problem(window_size=3, step_size=1))

    estimator.append_samples([0.0, 0.5, 1.0], np.array([[0.0], [0.5], [1.0]]))
    first = estimator.solve_next()
    estimator.append_samples([1.5], np.array([[9.0]]))
    second = estimator.solve_next()

    assert first is not None
    assert second is not None
    assert not first.warm_started
    assert second.warm_started
    initial_knot_q = second.initial_coefficients[:3]
    np.testing.assert_allclose(initial_knot_q[:2], [0.5, 1.0], atol=1e-6)
    assert initial_knot_q[2] < 2.0


def test_objective_uses_fixed_parameters_and_empty_shared_block() -> None:
    seen: list[dict[str, float]] = []

    def residual(evaluation, parameters):
        seen.append(dict(parameters))
        return parameters["theta"] * evaluation.q[:, 0] - evaluation.times

    estimator = MovingHorizonEstimator(
        MovingHorizonProblem(
            n_dof=1,
            fixed_parameters={"theta": 3.0},
            residual=residual,
            options=MovingHorizonOptions(
                window_size=2,
                solver_options=MapEstimatorOptions(max_iterations=5),
            ),
        )
    )
    estimator.append_samples([0.0, 1.0], np.array([[0.0], [1.0]]))

    problem = estimator.build_current_problem()
    result = estimator.solve_next()

    assert problem.shared_parameters.size == 0
    assert problem.trajectory.n_knots == 2
    assert result is not None
    assert result.parameters == {"theta": 3.0}
    assert seen
    assert all(item == {"theta": 3.0} for item in seen)


def test_callback_receives_serialisable_latency_payload() -> None:
    callbacks = []
    estimator = MovingHorizonEstimator(
        _tracking_problem(window_size=3, step_size=1, callback=callbacks.append)
    )

    estimator.append_samples([0.0, 0.5, 1.0], np.array([[0.0], [0.5], [1.0]]))
    result = estimator.solve_next()

    assert result is not None
    assert callbacks == [result]
    payload = result.callback_payload()
    assert payload["window_index"] == 0
    assert payload["latency_budget_ms"] == 100.0
    assert payload["parameters"] == {"scale": 2.0}
    assert isinstance(payload["over_budget"], bool)
