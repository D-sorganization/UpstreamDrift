from __future__ import annotations

import numpy as np

from src.shared.python.estimation import (
    CubicHermiteSplineTrajectory,
    MapEstimatorOptions,
    MapEstimatorProblem,
    SharedParameterBlock,
    SharedParameterSpec,
    SplineTrajectoryEvaluation,
    solve_single_trial_map,
)


def test_spline_evaluation_is_analytic_and_deterministic() -> None:
    trajectory = CubicHermiteSplineTrajectory(np.array([0.0, 0.5, 1.0]), n_dof=1)
    coefficients = trajectory.pack(
        knot_q=np.array([[0.0], [0.25], [1.0]]),
        knot_v=np.array([[0.0], [1.0], [2.0]]),
    )
    times = np.array([0.25, 0.75])

    first = trajectory.evaluate(coefficients, times)
    second = trajectory.evaluate(coefficients, times)

    np.testing.assert_allclose(first.q, second.q)
    np.testing.assert_allclose(first.v, second.v)
    np.testing.assert_allclose(first.a, second.a)
    np.testing.assert_allclose(
        first.q,
        np.einsum("tdc,c->td", first.q_basis, coefficients),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        first.v,
        np.einsum("tdc,c->td", first.v_basis, coefficients),
        atol=1e-12,
    )


def test_single_trial_map_recovers_shared_length_parameter() -> None:
    times = np.linspace(0.0, 1.0, 5)
    true_scale = 1.08
    trajectory = CubicHermiteSplineTrajectory(times, n_dof=1)
    true_coefficients = trajectory.pack(
        knot_q=(times**2)[:, None],
        knot_v=(2.0 * times)[:, None],
    )
    truth = trajectory.evaluate(true_coefficients, times)
    observations = true_scale * truth.q[:, 0]
    initial_coefficients = trajectory.pack(
        knot_q=(0.8 * times**2)[:, None],
        knot_v=(1.6 * times)[:, None],
    )
    params = SharedParameterBlock.from_specs(
        [
            SharedParameterSpec(
                name="upper_length_m",
                initial=1.0,
                kind="length",
                lower=0.5,
                upper=1.5,
            )
        ]
    )

    def residual(
        evaluation: SplineTrajectoryEvaluation,
        parameters: dict[str, float],
    ) -> np.ndarray:
        scaled_position = parameters["upper_length_m"] * evaluation.q[:, 0]
        trajectory_anchor = evaluation.q[:, 0] - truth.q[:, 0]
        return np.concatenate([scaled_position - observations, trajectory_anchor])

    def jacobian(evaluation, parameters, layout):
        jac = np.zeros((2 * times.size, layout.size), dtype=float)
        jac[: times.size, : layout.trajectory_size] = (
            parameters["upper_length_m"] * evaluation.q_basis[:, 0, :]
        )
        jac[: times.size, layout.parameter_column("upper_length_m")] = evaluation.q[
            :, 0
        ]
        jac[times.size :, : layout.trajectory_size] = evaluation.q_basis[:, 0, :]
        return jac

    result = solve_single_trial_map(
        MapEstimatorProblem(
            trajectory=trajectory,
            evaluation_times=times,
            initial_coefficients=initial_coefficients,
            shared_parameters=params,
            residual=residual,
            jacobian=jacobian,
            options=MapEstimatorOptions(max_iterations=80),
        )
    )

    fitted = trajectory.evaluate(result.coefficients, times)
    predicted = result.parameters["upper_length_m"] * fitted.q[:, 0]
    assert result.success
    assert result.objective < 1e-12
    np.testing.assert_allclose(predicted, observations, atol=1e-6)


def test_inertia_parameter_is_shared_and_bound_limited() -> None:
    times = np.linspace(0.0, 1.0, 4)
    trajectory = CubicHermiteSplineTrajectory(times, n_dof=1)
    coefficients = trajectory.pack(
        knot_q=np.zeros((times.size, 1)),
        knot_v=np.zeros((times.size, 1)),
    )
    params = SharedParameterBlock.from_specs(
        [
            SharedParameterSpec(
                name="club_inertia_kg_m2",
                initial=1.0,
                kind="inertia",
                lower=0.95,
                upper=1.05,
                prior=1.0,
                prior_scale=0.01,
            )
        ]
    )

    def residual(_evaluation, parameters):
        return np.full(times.size, parameters["club_inertia_kg_m2"] - 1.20)

    def jacobian(_evaluation, _parameters, layout):
        jac = np.zeros((times.size, layout.size), dtype=float)
        jac[:, layout.parameter_column("club_inertia_kg_m2")] = 1.0
        return jac

    result = solve_single_trial_map(
        MapEstimatorProblem(
            trajectory=trajectory,
            evaluation_times=times,
            initial_coefficients=coefficients,
            shared_parameters=params,
            residual=residual,
            jacobian=jacobian,
            options=MapEstimatorOptions(max_iterations=40),
        )
    )

    assert result.success
    assert result.parameters["club_inertia_kg_m2"] <= 1.05
    assert result.parameters["club_inertia_kg_m2"] > 1.0
