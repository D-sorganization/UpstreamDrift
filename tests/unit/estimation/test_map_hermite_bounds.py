"""The canonical MAP solve keeps adversarial targets inside full spline bounds."""

from collections.abc import Mapping

import numpy as np
import pytest

from src.shared.python.estimation import (
    CubicHermiteSplineTrajectory,
    MapEstimatorOptions,
    MapEstimatorProblem,
    SharedParameterBlock,
    SharedParameterSpec,
    SplineTrajectoryEvaluation,
    solve_single_trial_map,
)
from src.shared.python.estimation.hermite_bounds import HermiteBoundsDomain
from src.shared.python.estimation.map_estimator import MapDecisionLayout

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("analytic", [False, True])
def test_map_bounds_every_evaluation_and_saved_cubic(analytic: bool) -> None:
    knots = np.array([0.0, 0.7, 2.0])
    trajectory = CubicHermiteSplineTrajectory(knots, 1)
    domain = HermiteBoundsDomain(tuple(knots), ((-1.0, 1.0),))
    times = np.linspace(0.0, 2.0, 51)
    initial = trajectory.pack(np.full((3, 1), 0.2), np.zeros((3, 1)))
    evaluations: list[np.ndarray] = []

    def residual(
        evaluation: SplineTrajectoryEvaluation, _: Mapping[str, float]
    ) -> np.ndarray:
        assert np.all(evaluation.q >= -1.0 - 1e-12)
        assert np.all(evaluation.q <= 1.0 + 1e-12)
        evaluations.append(evaluation.q.copy())
        return evaluation.q[:, 0] - 5.0

    def jacobian(
        evaluation: SplineTrajectoryEvaluation,
        parameters: Mapping[str, float],
        layout: MapDecisionLayout,
    ) -> np.ndarray:
        assert layout.trajectory_size == trajectory.coefficient_size
        return evaluation.q_basis[:, 0, :]

    result = solve_single_trial_map(
        MapEstimatorProblem(
            trajectory,
            times,
            initial,
            SharedParameterBlock(()),
            residual,
            jacobian=jacobian if analytic else None,
            options=MapEstimatorOptions(max_iterations=100, trajectory_domain=domain),
        )
    )
    assert evaluations and result.success
    dense = trajectory.evaluate(result.coefficients, np.linspace(0, 2, 1001))
    assert np.max(dense.q) <= 1.0 + 1e-12
    assert np.min(dense.q) >= -1.0 - 1e-12
    assert result.coefficients.shape == initial.shape


@pytest.mark.parametrize("shared", [False, True])
def test_fixed_domain_handles_shared_prior_and_zero_decisions(shared: bool) -> None:
    trajectory = CubicHermiteSplineTrajectory(np.array([0.0, 1.0]), 1)
    domain = HermiteBoundsDomain((0.0, 1.0), ((0.4, 0.4),))
    parameters = (
        SharedParameterBlock.from_specs(
            [SharedParameterSpec("scale", 0.6, prior=0.8, prior_scale=0.2)]
        )
        if shared
        else SharedParameterBlock(())
    )

    def residual(
        evaluation: SplineTrajectoryEvaluation, values: Mapping[str, float]
    ) -> np.ndarray:
        return evaluation.q[:, 0] * values.get("scale", 1.0) - 0.4

    def jacobian(
        evaluation: SplineTrajectoryEvaluation,
        values: Mapping[str, float],
        layout: MapDecisionLayout,
    ) -> np.ndarray:
        physical = evaluation.q_basis[:, 0, :] * values.get("scale", 1.0)
        return np.column_stack([physical, evaluation.q[:, 0]]) if shared else physical

    result = solve_single_trial_map(
        MapEstimatorProblem(
            trajectory,
            np.array([0.0, 0.5, 1.0]),
            trajectory.pack(np.full((2, 1), 0.4), np.zeros((2, 1))),
            parameters,
            residual,
            jacobian,
            MapEstimatorOptions(trajectory_domain=domain),
        )
    )
    assert result.success
    np.testing.assert_array_equal(result.coefficients, [0.4, 0.4, 0.0, 0.0])
    if shared:
        assert result.parameters["scale"] == pytest.approx(20.48 / 25.48)
    else:
        assert result.n_iterations == 0


def test_domain_rejects_incompatible_clock_before_callbacks() -> None:
    trajectory = CubicHermiteSplineTrajectory(np.array([0.0, 1.0]), 1)
    domain = HermiteBoundsDomain((0.0, 2.0), ((-1.0, 1.0),))

    def residual(
        evaluation: SplineTrajectoryEvaluation, values: Mapping[str, float]
    ) -> np.ndarray:
        pytest.fail("Mismatched domain must fail before evaluating data")

    with pytest.raises(ValueError, match="domain|Domain"):
        solve_single_trial_map(
            MapEstimatorProblem(
                trajectory,
                np.array([0.0, 1.0]),
                np.zeros(4),
                SharedParameterBlock(()),
                residual,
                options=MapEstimatorOptions(trajectory_domain=domain),
            )
        )


def test_full_decoder_chain_and_prior_match_independent_differences() -> None:
    from src.shared.python.estimation.map_estimator import (
        _objective_jacobian,
        _objective_residual,
    )

    trajectory = CubicHermiteSplineTrajectory(np.array([0.0, 0.7, 2.0]), 2)
    domain = HermiteBoundsDomain((0.0, 0.7, 2.0), ((-1.0, 1.0), (2.0, 2.0)))
    coefficients = trajectory.pack(
        np.array([[0.1, 2.0], [0.2, 2.0], [0.3, 2.0]]),
        np.array([[0.2, 0.0], [-0.1, 0.0], [0.1, 0.0]]),
    )
    parameters = SharedParameterBlock.from_specs(
        [SharedParameterSpec("scale", 0.7, prior=0.8, prior_scale=0.2)]
    )

    def residual(
        evaluation: SplineTrajectoryEvaluation, values: Mapping[str, float]
    ) -> np.ndarray:
        return evaluation.q[:, 0] * values["scale"]

    def jacobian(
        evaluation: SplineTrajectoryEvaluation,
        values: Mapping[str, float],
        layout: MapDecisionLayout,
    ) -> np.ndarray:
        return np.column_stack(
            [
                evaluation.q_basis[:, 0, :] * values["scale"],
                evaluation.q[:, 0],
            ]
        )

    problem = MapEstimatorProblem(
        trajectory,
        np.array([0.2, 0.9, 1.6]),
        coefficients,
        parameters,
        residual,
        jacobian,
        MapEstimatorOptions(trajectory_domain=domain),
    )
    decision = np.r_[domain.encode(coefficients), 0.7]
    layout = MapDecisionLayout(trajectory.coefficient_size, ("scale",))
    analytic = _objective_jacobian(problem, layout, decision)
    numerical = np.empty_like(analytic)
    for index in range(len(decision)):
        delta = np.zeros_like(decision)
        delta[index] = 1e-6
        numerical[:, index] = (
            _objective_residual(problem, decision + delta)
            - _objective_residual(problem, decision - delta)
        ) / 2e-6
    np.testing.assert_allclose(analytic, numerical, atol=1e-8, rtol=1e-6)


def test_multi_trial_rejects_single_trial_domain_instead_of_ignoring_it() -> None:
    from src.shared.python.estimation import MultiTrialMapProblem, solve_multi_trial_map

    domain = HermiteBoundsDomain((0.0, 1.0), ((-1.0, 1.0),))
    problem = MultiTrialMapProblem(
        (),
        SharedParameterBlock(()),
        options=MapEstimatorOptions(trajectory_domain=domain),
    )
    with pytest.raises(ValueError, match="domain"):
        solve_multi_trial_map(problem)
