"""Backend counts exclude post-solve diagnostics and requested ceilings."""

import numpy as np
import pytest

from src.shared.python.estimation import (
    CubicHermiteSplineTrajectory,
    HermiteBoundsDomain,
    MapEstimatorOptions,
    MapEstimatorProblem,
    SharedParameterBlock,
    solve_single_trial_map,
)
from src.shared.python.estimation import map_estimator as module


def _problem(fixed=False):
    trajectory = CubicHermiteSplineTrajectory(np.array([0.0, 1.0]), 1)
    coefficients = np.zeros(4)
    return MapEstimatorProblem(
        trajectory,
        np.array([0.0, 1.0]),
        coefficients,
        SharedParameterBlock(()),
        lambda evaluation, _: evaluation.q.ravel(),
        options=MapEstimatorOptions(
            max_iterations=120,
            trajectory_domain=HermiteBoundsDomain((0.0, 1.0), ((0.0, 0.0),))
            if fixed
            else None,
        ),
    )


def test_analytical_early_stop_reports_actual_backend_counts_and_fake_clock(
    monkeypatch,
):
    calls = iter([10.0, 10.25])
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(calls))
    result = solve_single_trial_map(_problem())
    assert result.telemetry.nfev == result.n_iterations == 1
    assert result.telemetry.njev == 1
    assert result.telemetry.nfev < 120
    assert result.telemetry.solver_elapsed_s == 0.25
    assert result.telemetry.backend.method == "trf"
    assert result.telemetry.backend.version
    assert result.telemetry.worker_elapsed_s is None


def test_all_fixed_has_zero_backend_evaluations_and_no_fictional_solver_time(
    monkeypatch,
):
    monkeypatch.setattr(
        module, "least_squares", lambda *a, **k: pytest.fail("solver called")
    )
    result = solve_single_trial_map(_problem(True))
    assert result.n_iterations == result.telemetry.nfev == 0
    assert result.telemetry.njev is None
    assert result.telemetry.solver_elapsed_s is None
    assert result.telemetry.backend.name == "canonical_fixed_decisions"
    np.testing.assert_array_equal(result.coefficients, np.zeros(4))


@pytest.mark.parametrize("count", [True, 1.0])
def test_backend_counter_does_not_silently_coerce_malformed_values(monkeypatch, count):
    from scipy.optimize import OptimizeResult

    monkeypatch.setattr(
        module,
        "least_squares",
        lambda *a, **k: OptimizeResult(
            x=a[1], success=True, nfev=count, njev=None, message="mocked"
        ),
    )
    with pytest.raises(ValueError, match="nfev"):
        solve_single_trial_map(_problem())


def test_records_effective_bounded_method_not_requested_lm():
    from dataclasses import replace

    problem = _problem()
    options = replace(
        problem.options,
        method="lm",
        trajectory_domain=HermiteBoundsDomain((0.0, 1.0), ((-1.0, 1.0),)),
    )
    result = solve_single_trial_map(replace(problem, options=options))
    assert result.telemetry.backend.method == "trf"
