"""#9757 sentinel visibility and #9758 identifiability gate for the MAP solvers."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from src.shared.python.estimation.identifiability import (
    IdentifiabilityGateOptions,
    UnidentifiableParametersError,
    gate_shared_parameters,
)
from src.shared.python.estimation.map_estimator import (
    CubicHermiteSplineTrajectory,
    MapEstimatorOptions,
    MapEstimatorProblem,
    NonFiniteResidualError,
    SharedParameterBlock,
    SharedParameterSpec,
    solve_single_trial_map,
)
from src.shared.python.estimation.multi_trial import (
    MultiTrialMapProblem,
    MultiTrialObservation,
    solve_multi_trial_map,
)

pytestmark = pytest.mark.unit


def _spline() -> tuple[CubicHermiteSplineTrajectory, np.ndarray, np.ndarray]:
    times = np.linspace(0.0, 1.0, 4)
    trajectory = CubicHermiteSplineTrajectory(times, n_dof=1)
    coefficients = trajectory.pack(
        knot_q=np.zeros((times.size, 1)), knot_v=np.zeros((times.size, 1))
    )
    return trajectory, times, coefficients


def _block(*names: str) -> SharedParameterBlock:
    return SharedParameterBlock.from_specs(
        [
            SharedParameterSpec(name=name, initial=1.0, lower=0.5, upper=1.5)
            for name in names
        ]
    )


def _problem(
    residual, options: MapEstimatorOptions, *names: str
) -> MapEstimatorProblem:
    trajectory, times, coefficients = _spline()
    return MapEstimatorProblem(
        trajectory=trajectory,
        evaluation_times=times,
        initial_coefficients=coefficients,
        shared_parameters=_block(*names),
        residual=residual,
        options=options,
    )


# --- #9757 ------------------------------------------------------------------


def test_finite_residual_reports_zero_non_finite_evaluations() -> None:
    def residual(evaluation, parameters):
        return evaluation.q[:, 0] - parameters["length"] + 1.0

    result = solve_single_trial_map(_problem(residual, MapEstimatorOptions(), "length"))
    assert result.n_non_finite_evaluations == 0


def test_sentinel_policy_counts_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    def residual(evaluation, parameters):
        out = evaluation.q[:, 0] - parameters["length"]
        out[0] = np.nan
        return out

    options = MapEstimatorOptions(
        max_iterations=5, identifiability=IdentifiabilityGateOptions("off")
    )
    with caplog.at_level(logging.WARNING, logger="src.shared.python.estimation"):
        result = solve_single_trial_map(_problem(residual, options, "length"))
    assert result.n_non_finite_evaluations > 0
    assert any("non-finite" in record.message for record in caplog.records)
    assert any("residual_jacobian" in record.message for record in caplog.records)


def test_raise_policy_refuses_non_finite_residual() -> None:
    def residual(evaluation, parameters):
        return np.full(evaluation.q.shape[0], np.inf)

    options = MapEstimatorOptions(
        non_finite_policy="raise",
        identifiability=IdentifiabilityGateOptions("off"),
    )
    with pytest.raises(NonFiniteResidualError, match="non-finite"):
        solve_single_trial_map(_problem(residual, options, "length"))


def test_multi_trial_counts_non_finite_evaluations() -> None:
    trajectory, times, coefficients = _spline()

    def residual(_observation, evaluation, parameters):
        out = evaluation.q[:, 0] - parameters["length"]
        out[-1] = np.nan
        return out

    problem = MultiTrialMapProblem(
        observations=(
            MultiTrialObservation(
                trial_id="a",
                trajectory=trajectory,
                evaluation_times=times,
                initial_coefficients=coefficients,
                residual=residual,
            ),
        ),
        shared_parameters=_block("length"),
        options=MapEstimatorOptions(
            max_iterations=5, identifiability=IdentifiabilityGateOptions("off")
        ),
    )
    result = solve_multi_trial_map(problem)
    assert result.n_non_finite_evaluations > 0


# --- #9758 ------------------------------------------------------------------


def _unobservable_residual(evaluation, parameters):
    # ``ghost`` never enters the residual; ``length`` does.
    return evaluation.q[:, 0] - parameters["length"] + 1.0


def test_gate_flags_parameter_absent_from_residual() -> None:
    block = _block("length", "ghost")
    trajectory, times, coefficients = _spline()
    evaluation = trajectory.evaluate(coefficients, times)

    def residual_of_free(values):
        params = block.to_mapping(block.expand_free_vector(values))
        return _unobservable_residual(evaluation, params)

    report = gate_shared_parameters(residual_of_free, block)
    assert report.rank == 1
    assert report.flagged_parameters == ("ghost",)
    assert report.locked_parameters == ()
    assert report.to_dict()["n_free"] == 2


def test_warn_policy_solves_and_reports(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="src.shared.python.estimation"):
        result = solve_single_trial_map(
            _problem(_unobservable_residual, MapEstimatorOptions(), "length", "ghost")
        )
    assert result.identifiability is not None
    assert result.identifiability.flagged_parameters == ("ghost",)
    assert result.locked_by_gate == ()
    assert any("not identifiable" in r.message for r in caplog.records)


def test_lock_policy_freezes_flagged_parameter() -> None:
    options = MapEstimatorOptions(identifiability=IdentifiabilityGateOptions("lock"))
    result = solve_single_trial_map(
        _problem(_unobservable_residual, options, "length", "ghost")
    )
    assert result.locked_by_gate == ("ghost",)
    assert result.parameters["ghost"] == 1.0  # bit-for-bit at its initial value
    assert result.parameters["length"] == pytest.approx(1.0, abs=1e-6)


def test_raise_policy_refuses_unidentifiable_solve() -> None:
    options = MapEstimatorOptions(identifiability=IdentifiabilityGateOptions("raise"))
    with pytest.raises(UnidentifiableParametersError, match="ghost"):
        solve_single_trial_map(
            _problem(_unobservable_residual, options, "length", "ghost")
        )


def test_identifiable_problem_is_not_flagged() -> None:
    def residual(evaluation, parameters):
        return evaluation.q[:, 0] - parameters["length"] + 1.0

    result = solve_single_trial_map(_problem(residual, MapEstimatorOptions(), "length"))
    assert result.identifiability is not None
    assert result.identifiability.rank == 1
    assert result.identifiability.flagged_parameters == ()


def test_gate_off_or_no_free_parameters_skips_probe() -> None:
    trajectory, times, coefficients = _spline()
    problem = MapEstimatorProblem(
        trajectory=trajectory,
        evaluation_times=times,
        initial_coefficients=coefficients,
        shared_parameters=SharedParameterBlock.from_specs([]),
        residual=lambda evaluation, _p: evaluation.q[:, 0],
    )
    assert solve_single_trial_map(problem).identifiability is None
    options = MapEstimatorOptions(identifiability=IdentifiabilityGateOptions("off"))
    result = solve_single_trial_map(
        _problem(_unobservable_residual, options, "length", "ghost")
    )
    assert result.identifiability is None


def test_multi_trial_lock_policy() -> None:
    trajectory, times, coefficients = _spline()

    def residual(_observation, evaluation, parameters):
        return _unobservable_residual(evaluation, parameters)

    problem = MultiTrialMapProblem(
        observations=(
            MultiTrialObservation(
                trial_id="a",
                trajectory=trajectory,
                evaluation_times=times,
                initial_coefficients=coefficients,
                residual=residual,
            ),
        ),
        shared_parameters=_block("length", "ghost"),
        options=MapEstimatorOptions(identifiability=IdentifiabilityGateOptions("lock")),
    )
    result = solve_multi_trial_map(problem)
    assert result.locked_by_gate == ("ghost",)
    assert result.posterior_parameter_names == ("length",)
    assert result.parameters["ghost"] == 1.0


def test_gate_options_contracts() -> None:
    with pytest.raises(ValueError):
        IdentifiabilityGateOptions(policy="maybe")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        IdentifiabilityGateOptions(alignment=0.0)
