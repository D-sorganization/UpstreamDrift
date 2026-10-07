from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from src.shared.python.contracts import (
    ContractLevel,
    ContractViolationError,
    get_contract_level,
    set_contract_level,
)
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
from src.shared.python.estimation.multi_trial import (
    _build_layout,
    _marginal_shared_block,
    _objective_jacobian,
    _objective_residual,
    _pack_decision,
    _validate_observation,
    _validate_problem,
)


@pytest.fixture
def contracts_enforced():
    """Force DbC enforcement so ``require(...)`` raises (issue #6941).

    ``require`` is a no-op at ``ContractLevel.OFF``; the validation guards in
    ``multi_trial`` rely on it, so the negative-path tests must run with
    enforcement on regardless of the ambient ``DBC_LEVEL``.
    """
    previous = get_contract_level()
    set_contract_level(ContractLevel.ENFORCE)
    try:
        yield
    finally:
        set_contract_level(previous)


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


# --------------------------------------------------------------------------
# Negative-path / guard tests (issue #6941)
# --------------------------------------------------------------------------


def _single_problem() -> MultiTrialMapProblem:
    return _problem_with_trials(
        trial_scales=(1.0,), parameter_initial=1.0, locked_mass=False
    )


class TestValidateProblemGuards:
    """Negative-path coverage for ``_validate_problem`` (issue #6941)."""

    def test_empty_observations_rejected(self, contracts_enforced) -> None:
        problem = _single_problem()
        empty = MultiTrialMapProblem(
            observations=(),
            shared_parameters=problem.shared_parameters,
            options=problem.options,
        )
        with pytest.raises(ContractViolationError, match="at least one observation"):
            _validate_problem(empty)

    def test_duplicate_trial_keys_rejected(self, contracts_enforced) -> None:
        obs = _observation(0, 1.0)
        problem = _single_problem()
        dup = MultiTrialMapProblem(
            observations=(obs, obs),  # identical keys
            shared_parameters=problem.shared_parameters,
            options=problem.options,
        )
        with pytest.raises(ContractViolationError, match="keys must be unique"):
            _validate_problem(dup)

    def test_non_positive_max_iterations_rejected(self, contracts_enforced) -> None:
        problem = _single_problem()
        bad = MultiTrialMapProblem(
            observations=problem.observations,
            shared_parameters=problem.shared_parameters,
            options=MapEstimatorOptions(max_iterations=0),
        )
        with pytest.raises(ContractViolationError, match="max_iterations"):
            _validate_problem(bad)

    def test_negative_covariance_regularization_rejected(
        self, contracts_enforced
    ) -> None:
        problem = _single_problem()
        bad = MultiTrialMapProblem(
            observations=problem.observations,
            shared_parameters=problem.shared_parameters,
            options=problem.options,
            covariance_regularization=-1.0,
        )
        with pytest.raises(ContractViolationError, match="covariance_regularization"):
            _validate_problem(bad)


def _observation_with(**overrides) -> MultiTrialObservation:
    base = _observation(0, 1.0)
    fields = {
        "trial_id": base.trial_id,
        "trajectory": base.trajectory,
        "evaluation_times": base.evaluation_times,
        "initial_coefficients": base.initial_coefficients,
        "residual": base.residual,
        "jacobian": base.jacobian,
        "view_id": base.view_id,
    }
    fields.update(overrides)
    return MultiTrialObservation(**fields)


class TestValidateObservationGuards:
    """Negative-path coverage for ``_validate_observation`` (issue #6941)."""

    def test_empty_trial_id_rejected(self, contracts_enforced) -> None:
        obs = _observation_with(trial_id="   ")
        with pytest.raises(ContractViolationError, match="trial_id"):
            _validate_observation(obs)

    def test_non_1d_evaluation_times_rejected(self, contracts_enforced) -> None:
        obs = _observation_with(evaluation_times=np.zeros((2, 2)))
        with pytest.raises(ContractViolationError, match="1D array"):
            _validate_observation(obs)

    def test_non_finite_evaluation_times_rejected(self, contracts_enforced) -> None:
        times = np.array([0.0, np.inf, 1.0])
        obs = _observation_with(evaluation_times=times)
        with pytest.raises(ContractViolationError, match="must be finite"):
            _validate_observation(obs)

    def test_coefficient_shape_mismatch_rejected(self, contracts_enforced) -> None:
        obs = _observation_with(initial_coefficients=np.zeros(3))
        with pytest.raises(ContractViolationError, match="shape must match"):
            _validate_observation(obs)

    def test_non_finite_coefficients_rejected(self, contracts_enforced) -> None:
        base = _observation(0, 1.0)
        bad_coeffs = np.array(base.initial_coefficients, dtype=float)
        bad_coeffs[0] = np.nan
        obs = _observation_with(initial_coefficients=bad_coeffs)
        with pytest.raises(ContractViolationError, match="must be finite"):
            _validate_observation(obs)


class TestPosteriorVarianceAccessor:
    """``posterior_variance`` must raise ``KeyError`` for unknown names (#6941)."""

    def test_unknown_parameter_raises_key_error(self) -> None:
        result = _solve_problem(trial_scales=(1.0,))
        with pytest.raises(KeyError):
            result.posterior_variance("nonexistent")


class TestStackSharedParameterJacobians:
    """Negative-path coverage for ``stack_shared_parameter_jacobians`` (#6941)."""

    def test_empty_sequence_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one Jacobian block"):
            stack_shared_parameter_jacobians([])

    def test_mismatched_widths_rejected(self) -> None:
        first = np.array([[1.0, 2.0]])
        second = np.array([[3.0]])
        with pytest.raises(ValueError, match="matching widths"):
            stack_shared_parameter_jacobians([first, second])

    def test_non_2d_block_rejected(self) -> None:
        with pytest.raises(ValueError, match="2D"):
            stack_shared_parameter_jacobians([np.array([1.0, 2.0, 3.0])])

    def test_non_finite_block_rejected(self) -> None:
        block = np.array([[1.0, np.nan]])
        with pytest.raises(ValueError, match="must be finite"):
            stack_shared_parameter_jacobians([block])


# --------------------------------------------------------------------------
# Row layout and marginal covariance (issue #11548)
# --------------------------------------------------------------------------


def _locked_prior_problem() -> MultiTrialMapProblem:
    """Two trials; a locked parameter that carries a prior plus a free one."""
    base = _problem_with_trials(
        trial_scales=(1.0, 1.35), parameter_initial=1.0, locked_mass=False
    )
    shared = SharedParameterBlock.from_specs(
        [
            SharedParameterSpec(
                "locked_mass_kg", 70.0, prior=72.0, prior_scale=2.0, locked=True
            ),
            SharedParameterSpec(
                "club_length_m",
                1.0,
                kind="length",
                lower=0.8,
                upper=1.5,
                prior=1.1,
                prior_scale=0.5,
            ),
        ]
    )
    return replace(base, shared_parameters=shared)


def _central_difference_jacobian(fn, x: np.ndarray, step: float) -> np.ndarray:
    columns = []
    for column in range(x.size):
        delta = np.zeros_like(x)
        delta[column] = step
        columns.append((fn(x + delta) - fn(x - delta)) / (2.0 * step))
    return np.column_stack(columns)


@pytest.mark.unit
def test_locked_prior_residual_and_jacobian_share_row_layout(
    contracts_enforced,
) -> None:
    """A locked parameter's prior is excluded from BOTH residual and Jacobian.

    The fixture residuals are at most bilinear in the decision vector
    (``club_length * q`` with ``q`` linear in the spline coefficients), so the
    central-difference truncation error is exactly zero and the only error is
    rounding, ``~eps * |r| / h ~ 1e-9`` for ``h = 1e-6``; ``atol = 1e-7``
    leaves two decades of margin while catching any wrong or missing row.
    """
    problem = _locked_prior_problem()
    layout = _build_layout(problem)
    decision = _pack_decision(problem) + 0.03

    residual = _objective_residual(problem, layout, decision)
    jacobian = _objective_jacobian(problem, layout, decision)

    assert jacobian.shape == (residual.size, layout.size)
    # 2 trials x 8 data rows + one prior row for the single FREE parameter.
    assert residual.size == 2 * 8 + 1
    numeric = _central_difference_jacobian(
        lambda x: _objective_residual(problem, layout, x), decision, step=1e-6
    )
    np.testing.assert_allclose(jacobian, numeric, rtol=0.0, atol=1e-7)


@pytest.mark.unit
def test_posterior_covariance_is_schur_marginal_not_conditional() -> None:
    """Shared-parameter covariance marginalises the trajectory coefficients.

    The independent reference inverts the full regularised Fisher matrix
    densely and reads the shared block, and separately evaluates the Schur
    complement ``(F_ss - F_st F_tt^-1 F_ts)^-1``. Both are exact linear algebra
    on a 17x17 well-conditioned matrix, so ``rtol = 1e-8`` is far above the
    ``cond(F) * eps`` rounding floor.
    """
    problem = _problem_with_trials(
        trial_scales=(1.0, 1.35), parameter_initial=1.0, locked_mass=False
    )
    result = solve_multi_trial_map(problem)
    assert result.success
    layout = _build_layout(problem)
    decision = np.concatenate(
        [result.coefficients_by_trial[o.key] for o in problem.observations]
        + [np.array([result.parameters["club_length_m"]])]
    )
    jacobian = _objective_jacobian(problem, layout, decision)
    regularization = problem.covariance_regularization
    fisher = jacobian.T @ jacobian + regularization * np.eye(layout.size)
    shared = slice(layout.trajectory_size, layout.size)
    traj = slice(0, layout.trajectory_size)

    dense_marginal = np.linalg.inv(fisher)[shared, shared]
    schur = fisher[shared, shared] - fisher[shared, traj] @ np.linalg.solve(
        fisher[traj, traj], fisher[traj, shared]
    )
    schur_marginal = np.linalg.inv(schur)
    conditional = np.linalg.inv(fisher[shared, shared])

    np.testing.assert_allclose(dense_marginal, schur_marginal, rtol=1e-8)
    np.testing.assert_allclose(result.posterior_covariance, dense_marginal, rtol=1e-8)
    # Correlated example: marginalising cannot shrink, and here strictly grows.
    assert result.posterior_variance("club_length_m") > 1.5 * conditional[0, 0]


@pytest.mark.unit
def test_marginal_block_of_singular_fisher_is_unavailable_not_pinv() -> None:
    """A confounded trajectory/shared column pair has no marginal (#11548).

    Jacobian row ``[1, 1]`` (one trajectory column, one shared column) gives
    ``F = [[1, 1], [1, 1]]`` with ``lambda = 0``. The Schur complement
    ``F_ss - F_st F_tt^-1 F_ts = 1 - 1 = 0``, so the marginal variance is
    undefined (infinite). The Moore-Penrose block ``pinv(F)[1, 1] = 0.25``
    must never be reported in its place.
    """
    covariance, status = _marginal_shared_block(
        np.array([[1.0, 1.0]]), trajectory_size=1, regularization=0.0
    )

    assert status == "rank_deficient"
    assert covariance.shape == (1, 1)
    assert np.all(np.isnan(covariance))


@pytest.mark.unit
def test_marginal_block_of_full_rank_fisher_matches_schur_inverse() -> None:
    """A full-rank Fisher matrix reports the inverse Schur complement.

    ``J = [[1, 1], [0, 1]]`` gives ``F = [[1, 1], [1, 2]]``; the Schur
    complement of ``F_tt = 1`` is ``2 - 1 = 1``, so the exact marginal is
    ``1.0`` (the conditional ``1 / F_ss`` would be ``0.5``). Exact 2x2
    arithmetic, so ``rtol = 1e-12`` is far above rounding.
    """
    covariance, status = _marginal_shared_block(
        np.array([[1.0, 1.0], [0.0, 1.0]]), trajectory_size=1, regularization=0.0
    )

    assert status == "estimated"
    np.testing.assert_allclose(covariance, [[1.0]], rtol=1e-12)


@pytest.mark.unit
def test_solver_reports_unavailable_covariance_for_confounded_problem() -> None:
    """``q * club_length`` alone is scale-confounded; the result fails closed.

    Without the trajectory anchor rows, scaling the spline coefficients by
    ``c`` and ``club_length_m`` by ``1 / c`` leaves every residual unchanged,
    so the regularisation-free Fisher matrix is singular at every point.
    """
    base = _observation(0, 1.0)
    times = base.evaluation_times
    observed = np.linspace(1.0, 2.0, times.size)

    def residual(_observation, evaluation, parameters):
        return parameters["club_length_m"] * evaluation.q[:, 0] - observed

    def jacobian(observation, evaluation, parameters, layout):
        jac = np.zeros((times.size, layout.size), dtype=float)
        jac[:, layout.trajectory_slice(observation.key)] = (
            parameters["club_length_m"] * evaluation.q_basis[:, 0, :]
        )
        jac[:, layout.parameter_column("club_length_m")] = evaluation.q[:, 0]
        return jac

    observation = replace(base, residual=residual, jacobian=jacobian)
    problem = replace(
        _single_problem(),
        observations=(observation,),
        covariance_regularization=0.0,
    )

    result = solve_multi_trial_map(problem)

    assert result.covariance_status == "rank_deficient"
    assert np.isnan(result.posterior_variance("club_length_m"))


# --------------------------------------------------------------------------
# Closed-form linear benchmark (MOSAIC-13, issue #11545)
# --------------------------------------------------------------------------
#
# Two trials, each a two-knot, one-DoF Hermite spline evaluated at its knots,
# so the decision vector is exactly linear in the residual. Per trial ``i``
# with coefficients ``c_i`` (4 entries) and one shared scalar ``s``:
#
#     r_i = [ q(0), q(1), v(0), v(1),  s + q(0) + q(1) - z ]
#
# plus a prior row ``s / sigma``. The coupling row has trajectory gradient
# ``b`` with ``beta = |b|^2 = 2`` (the q-basis rows at the knots are unit
# vectors). Per trial ``F_tt = I + b b^T`` and ``F_ts = b``; by
# Sherman-Morrison ``b^T (I + b b^T)^-1 b = beta / (1 + beta)``, so
#
#     marginal    = (n / (1 + beta) + 1 / sigma^2)^-1 = (2/3 + 1)^-1 = 0.6
#     conditional = (n + 1 / sigma^2)^-1                = 1/3
#
# Profiling ``c_i`` out of the objective gives ``n (z - s)^2 / (1 + beta) +
# s^2 / sigma^2``, whose minimiser is ``s* = 0.6 * (2/3) * z = 0.4`` for
# ``z = 1``. Everything is exact linear algebra on a 9x9 matrix with
# condition number ~10, so ``rtol = 1e-9`` sits far above rounding.

_LINEAR_TRIALS = 2
_LINEAR_SIGMA = 1.0
_LINEAR_Z = 1.0
_LINEAR_MARGINAL = 0.6
_LINEAR_CONDITIONAL = 1.0 / 3.0
_LINEAR_MAP_S = 0.4


def _linear_observation(index: int) -> MultiTrialObservation:
    times = np.array([0.0, 1.0])
    trajectory = CubicHermiteSplineTrajectory(times, n_dof=1)

    def residual(_observation, evaluation, parameters) -> np.ndarray:
        q = evaluation.q[:, 0]
        v = evaluation.v[:, 0]
        coupling = parameters["s"] + q.sum() - _LINEAR_Z
        return np.concatenate([q, v, [coupling]])

    def jacobian(observation, evaluation, _parameters, layout) -> np.ndarray:
        columns = layout.trajectory_slice(observation.key)
        jac = np.zeros((5, layout.size), dtype=float)
        jac[0:2, columns] = evaluation.q_basis[:, 0, :]
        jac[2:4, columns] = evaluation.v_basis[:, 0, :]
        jac[4, columns] = evaluation.q_basis[:, 0, :].sum(axis=0)
        jac[4, layout.parameter_column("s")] = 1.0
        return jac

    return MultiTrialObservation(
        trial_id=f"linear-{index}",
        trajectory=trajectory,
        evaluation_times=times,
        initial_coefficients=np.zeros(trajectory.coefficient_size),
        residual=residual,
        jacobian=jacobian,
    )


def _linear_problem(*, locked_prior: bool) -> MultiTrialMapProblem:
    specs = [SharedParameterSpec("s", 0.0, prior=0.0, prior_scale=_LINEAR_SIGMA)]
    if locked_prior:
        specs.insert(
            0, SharedParameterSpec("m", 70.0, prior=72.0, prior_scale=2.0, locked=True)
        )
    return MultiTrialMapProblem(
        observations=tuple(_linear_observation(i) for i in range(_LINEAR_TRIALS)),
        shared_parameters=SharedParameterBlock.from_specs(specs),
        options=MapEstimatorOptions(max_iterations=200),
        covariance_regularization=0.0,
    )


@pytest.mark.unit
def test_linear_benchmark_coupling_has_beta_two() -> None:
    """Guard the closed form's premise: the knot q-basis gives ``|b|^2 = 2``."""
    observation = _linear_observation(0)
    evaluation = observation.trajectory.evaluate(
        observation.initial_coefficients, observation.evaluation_times
    )
    coupling = evaluation.q_basis[:, 0, :].sum(axis=0)
    assert float(coupling @ coupling) == pytest.approx(2.0, abs=1e-12)


@pytest.mark.unit
def test_linear_benchmark_covariance_is_closed_form_marginal() -> None:
    """Reported variance equals the analytic Schur marginal, not the conditional."""
    result = solve_multi_trial_map(_linear_problem(locked_prior=False))

    assert result.success
    variance = result.posterior_variance("s")
    np.testing.assert_allclose(variance, _LINEAR_MARGINAL, rtol=1e-9)
    # The two candidates differ by 80 %, so this cannot pass by accident.
    assert abs(variance - _LINEAR_CONDITIONAL) > 0.25
    assert result.covariance_status == "estimated"


@pytest.mark.unit
def test_linear_benchmark_locked_prior_keeps_rows_and_estimate(
    contracts_enforced,
) -> None:
    """A locked parameter with a prior changes neither rows nor the MAP answer.

    The locked prior row would be the constant ``(70 - 72) / 2``; it carries
    no decision variable, so the solve must report the same ``s*`` and the
    same marginal as the problem without it, with one prior row (for ``s``).
    """
    result = solve_multi_trial_map(_linear_problem(locked_prior=True))

    assert result.success
    assert result.residual.size == _LINEAR_TRIALS * 5 + 1
    assert result.parameters["m"] == 70.0
    np.testing.assert_allclose(result.parameters["s"], _LINEAR_MAP_S, rtol=1e-9)
    np.testing.assert_allclose(
        result.posterior_variance("s"), _LINEAR_MARGINAL, rtol=1e-9
    )


@pytest.mark.unit
def test_marginal_covariance_consistent_with_fit_uncertainty() -> None:
    """Both estimators read the same block of the full inverse Fisher matrix.

    :func:`least_squares_parameter_uncertainty` scales the unit-noise
    covariance by ``sigma^2 = 2 cost / (m - n)``; ``solve_multi_trial_map``
    reports the unit-noise MAP covariance. With ``lambda = 0`` the two must
    agree after removing that scale.
    """
    from scipy.optimize import OptimizeResult

    from src.shared.python.estimation.fit_uncertainty import (
        least_squares_parameter_uncertainty,
    )

    problem = _linear_problem(locked_prior=False)
    result = solve_multi_trial_map(problem)
    layout = _build_layout(problem)
    decision = np.concatenate(
        [result.coefficients_by_trial[o.key] for o in problem.observations]
        + [np.array([result.parameters["s"]])]
    )
    optimum = OptimizeResult(
        x=decision,
        jac=_objective_jacobian(problem, layout, decision),
        cost=result.objective,
        active_mask=np.zeros(layout.size, dtype=int),
    )

    uncertainty = least_squares_parameter_uncertainty(
        optimum, parameter_indices=[layout.parameter_column("s")]
    )

    assert uncertainty.status == "estimated"
    assert uncertainty.covariance is not None
    np.testing.assert_allclose(
        uncertainty.covariance / uncertainty.residual_variance,
        result.posterior_covariance,
        rtol=1e-9,
    )
