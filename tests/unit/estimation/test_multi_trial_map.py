from __future__ import annotations

import json

import numpy as np

from src.shared.python.estimation import (
    CubicHermiteSplineTrajectory,
    MapEstimatorOptions,
    MultiTrialMapProblem,
    MultiTrialObservation,
    SharedParameterBlock,
    SharedParameterSpec,
    SplineTrajectoryEvaluation,
    shared_parameter_covariance,
    solve_multi_trial_map,
    stack_shared_parameter_jacobians,
)


def test_shared_parameter_block_indexes_locks_and_serializes() -> None:
    block = SharedParameterBlock.from_specs(
        [
            SharedParameterSpec("locked_mass_kg", 70.0, locked=True),
            SharedParameterSpec("club_length_m", 1.0, lower=0.8, upper=1.4),
        ]
    )

    payload = json.loads(json.dumps(block.to_dict()))
    restored = SharedParameterBlock.from_dict(payload)

    assert restored.index("locked_mass_kg") == 0
    assert restored.index("club_length_m") == 1
    assert restored.free_index("club_length_m") == 0
    assert restored.free_parameter_names == ("club_length_m",)
    np.testing.assert_allclose(
        restored.expand_free_vector(np.array([1.2])), [70.0, 1.2]
    )


def test_multi_trial_layout_excludes_locked_parameter_columns() -> None:
    problem = _problem_with_trials(
        trial_scales=(1.0, 1.1),
        parameter_initial=1.0,
        locked_mass=True,
    )
    seen_columns: list[int] = []

    def jacobian(observation, evaluation, parameters, layout):
        trial_slice = layout.trajectory_slice(observation.key)
        parameter_column = layout.parameter_column("club_length_m")
        seen_columns.append(parameter_column)
        jac = np.zeros((2 * evaluation.times.size, layout.size), dtype=float)
        jac[: evaluation.times.size, trial_slice] = (
            parameters["club_length_m"] * evaluation.q_basis[:, 0, :]
        )
        jac[: evaluation.times.size, parameter_column] = evaluation.q[:, 0]
        jac[evaluation.times.size :, trial_slice] = evaluation.q_basis[:, 0, :]
        return jac

    observations = tuple(
        MultiTrialObservation(
            trial_id=item.trial_id,
            view_id=item.view_id,
            trajectory=item.trajectory,
            evaluation_times=item.evaluation_times,
            initial_coefficients=item.initial_coefficients,
            residual=item.residual,
            jacobian=jacobian,
        )
        for item in problem.observations
    )

    result = solve_multi_trial_map(
        MultiTrialMapProblem(
            observations=observations,
            shared_parameters=problem.shared_parameters,
            options=MapEstimatorOptions(max_iterations=80),
        )
    )

    assert result.success
    assert len(seen_columns) >= 2
    assert set(seen_columns) == {16}
    assert result.posterior_parameter_names == ("club_length_m",)
    assert result.parameters["locked_mass_kg"] == 70.0


def test_multi_trial_shared_parameter_posterior_tightens() -> None:
    one_trial = _solve_problem(trial_scales=(1.0,))
    two_trials = _solve_problem(trial_scales=(1.0, 1.35))

    assert one_trial.success
    assert two_trials.success
    np.testing.assert_allclose(two_trials.parameters["club_length_m"], 1.2, atol=1e-6)
    assert two_trials.posterior_variance("club_length_m") < (
        0.6 * one_trial.posterior_variance("club_length_m")
    )
    assert set(two_trials.coefficients_by_trial) == {"trial-0", "trial-1"}


def test_shared_parameter_covariance_stacks_identifiable_rows() -> None:
    first = np.array([[1.0], [2.0]])
    second = np.array([[3.0], [4.0]])
    stacked = stack_shared_parameter_jacobians([first, second])
    one = shared_parameter_covariance(first)
    both = shared_parameter_covariance(stacked)

    np.testing.assert_allclose(stacked[:, 0], [1.0, 2.0, 3.0, 4.0])
    assert both[0, 0] < one[0, 0]


def _solve_problem(trial_scales: tuple[float, ...]):
    return solve_multi_trial_map(
        _problem_with_trials(
            trial_scales=trial_scales,
            parameter_initial=1.0,
            locked_mass=False,
        )
    )


def _problem_with_trials(
    trial_scales: tuple[float, ...],
    parameter_initial: float,
    locked_mass: bool,
) -> MultiTrialMapProblem:
    params = [
        SharedParameterSpec(
            "club_length_m",
            parameter_initial,
            kind="length",
            lower=0.8,
            upper=1.5,
        )
    ]
    if locked_mass:
        params.insert(0, SharedParameterSpec("locked_mass_kg", 70.0, locked=True))
    shared = SharedParameterBlock.from_specs(params)
    return MultiTrialMapProblem(
        observations=tuple(
            _observation(index, scale) for index, scale in enumerate(trial_scales)
        ),
        shared_parameters=shared,
        options=MapEstimatorOptions(max_iterations=80),
    )


def _observation(index: int, scale: float) -> MultiTrialObservation:
    times = np.linspace(0.0, 1.0, 4)
    trajectory = CubicHermiteSplineTrajectory(times, n_dof=1)
    truth_coefficients = trajectory.pack(
        knot_q=(scale * (1.0 + times**2))[:, None],
        knot_v=(scale * 2.0 * times)[:, None],
    )
    initial_coefficients = trajectory.pack(
        knot_q=(0.9 * scale * (1.0 + times**2))[:, None],
        knot_v=(0.9 * scale * 2.0 * times)[:, None],
    )
    truth = trajectory.evaluate(truth_coefficients, times)
    truth_q = truth.q[:, 0]
    observations = 1.2 * truth_q

    def residual(
        _observation: MultiTrialObservation,
        evaluation: SplineTrajectoryEvaluation,
        parameters: dict[str, float],
    ) -> np.ndarray:
        scaled_position = parameters["club_length_m"] * evaluation.q[:, 0]
        return np.concatenate(
            [scaled_position - observations, evaluation.q[:, 0] - truth_q]
        )

    def jacobian(observation, evaluation, parameters, layout):
        trial_slice = layout.trajectory_slice(observation.key)
        jac = np.zeros((2 * times.size, layout.size), dtype=float)
        jac[: times.size, trial_slice] = (
            parameters["club_length_m"] * evaluation.q_basis[:, 0, :]
        )
        jac[: times.size, layout.parameter_column("club_length_m")] = evaluation.q[:, 0]
        jac[times.size :, trial_slice] = evaluation.q_basis[:, 0, :]
        return jac

    return MultiTrialObservation(
        trial_id=f"trial-{index}",
        trajectory=trajectory,
        evaluation_times=times,
        initial_coefficients=initial_coefficients,
        residual=residual,
        jacobian=jacobian,
    )
