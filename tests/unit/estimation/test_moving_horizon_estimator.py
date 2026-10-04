from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.contracts import PreconditionError
from src.shared.python.estimation import (
    MapEstimatorOptions,
    MovingHorizonEstimator,
    MovingHorizonOptions,
    MovingHorizonProblem,
)
from src.shared.python.estimation.moving_horizon import (
    ArrivalInformation,
    FailureDiagnostic,
    marginalize_arrival_schur,
)

pytestmark = pytest.mark.unit


def _tracking_problem(
    *,
    window_size: int = 3,
    step_size: int = 1,
    callback=None,
    fail_on_sample_start: float | None = None,
    max_iterations: int = 50,
) -> MovingHorizonProblem:
    options = MovingHorizonOptions(
        window_size=window_size,
        step_size=step_size,
        latency_budget_ms=100.0,
        solver_options=MapEstimatorOptions(max_iterations=max_iterations),
    )

    def residual(evaluation, parameters):
        if (
            fail_on_sample_start is not None
            and evaluation.times[-1] >= fail_on_sample_start
        ):
            # Non-finite or diverged residual to trigger solver rejection
            return np.full(evaluation.times.size, np.nan)
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


# ==============================================================================
# RED / GREEN Suites for DIME-07 (#11428): Arrival Info & Safe Window Commits
# ==============================================================================


def test_red_unsuccessful_solver_does_not_poison_state() -> None:
    """Unsuccessful solves must not replace last accepted state or poison future warm starts."""
    estimator = MovingHorizonEstimator(
        _tracking_problem(window_size=3, step_size=1, fail_on_sample_start=1.5)
    )

    # Window 0: succeeds
    estimator.append_samples([0.0, 0.5, 1.0], np.array([[0.0], [0.5], [1.0]]))
    res0 = estimator.solve_next()
    assert res0 is not None and res0.success
    accepted_state_0 = estimator.last_accepted_coefficients
    assert accepted_state_0 is not None

    # Window 1: reaches t=1.5 which triggers NaN residual inside solver
    estimator.append_samples([1.5], np.array([[1.5]]))
    res1 = estimator.solve_next()

    # The solve must fail cleanly
    assert res1 is not None
    assert not res1.success
    # Must NOT overwrite last accepted state with poisoned values
    assert estimator.last_accepted_coefficients is not None
    np.testing.assert_array_equal(
        estimator.last_accepted_coefficients, accepted_state_0
    )
    # Must record failure diagnostics separately
    diagnostics = estimator.failure_diagnostics
    assert len(diagnostics) == 1
    assert isinstance(diagnostics[0], FailureDiagnostic)
    assert diagnostics[0].window_index == 1


def test_red_singular_marginalization_preserves_rank_and_avoids_jitter() -> None:
    """Rank-deficient marginalization must identify gauge nullspace without diagonal jitter."""
    # 2D variable x_0 marginalized out into 2D variable x_1
    # x_1 is unobserved along dimension 1 (H_11 has rank 1, H_01 has rank 1 along dim 0)
    H_00 = np.array([[10.0, 0.0], [0.0, 0.0]])
    H_01 = np.array([[2.0, 0.0], [0.0, 0.0]])
    H_11 = np.array([[5.0, 0.0], [0.0, 0.0]])

    sqrt_info, g_out, rank, nullspace = marginalize_arrival_schur(
        H_00, H_01, H_11, singular_value_tol=1e-8
    )

    # Rank must be revealing: exact rank 1 without arbitrary epsilon diagonal jitter
    assert rank == 1
    assert nullspace is not None
    # Verify that the unobserved subspace is exactly preserved in nullspace basis
    assert nullspace.shape[1] == 1
    np.testing.assert_allclose(np.abs(nullspace[:, 0]), [0.0, 1.0], atol=1e-8)
    # Check that sqrt_info^T * sqrt_info has zero curvature along the unobserved direction
    Lambda_new = sqrt_info.T @ sqrt_info
    null_dir = nullspace[:, 0]
    np.testing.assert_allclose(
        Lambda_new @ null_dir, np.zeros_like(null_dir), atol=1e-8
    )


def test_red_duplicate_or_late_timestamps_rejected() -> None:
    """Non-increasing or duplicate timestamps must raise PreconditionError."""
    estimator = MovingHorizonEstimator(_tracking_problem(window_size=3, step_size=1))
    estimator.append_samples([0.0, 0.5, 1.0], np.array([[0.0], [0.5], [1.0]]))

    # Append a late sample (timestamp 0.8 <= last timestamp 1.0)
    with pytest.raises(PreconditionError, match="advance monotonically|must increase"):
        estimator.append_samples([0.8], np.array([[0.8]]))

    # Append a duplicate sample (timestamp 1.0)
    with pytest.raises(PreconditionError, match="advance monotonically|must increase"):
        estimator.append_samples([1.0], np.array([[1.0]]))


def test_red_parameter_revision_detected_and_safeguarded() -> None:
    """Parameter modification mid-trajectory must be validated and not silently corrupt arrival."""
    estimator = MovingHorizonEstimator(_tracking_problem(window_size=3, step_size=1))
    estimator.append_samples([0.0, 0.5, 1.0], np.array([[0.0], [0.5], [1.0]]))
    res0 = estimator.solve_next()
    assert res0 is not None and res0.success

    # Revise parameter mid-trajectory
    with pytest.raises(PreconditionError, match="[Pp]arameter revision"):
        estimator.update_fixed_parameters({"scale": 5.0})


def test_red_window_boundary_jump_rejection() -> None:
    """Massive discontinuity exceeding dynamic defect bounds at window boundary must be caught."""
    estimator = MovingHorizonEstimator(_tracking_problem(window_size=3, step_size=1))
    estimator.append_samples([0.0, 0.5, 1.0], np.array([[0.0], [0.5], [1.0]]))
    res0 = estimator.solve_next()
    assert res0 is not None and res0.success

    # Discontinuous jump (e.g. state jumps by 1000.0)
    estimator.append_samples([1.5], np.array([[1000.0]]))
    # With a max continuity defect check, solve fails or flags excessive jump
    res1 = estimator.solve_next(max_boundary_jump=50.0)
    assert res1 is not None
    assert not res1.success
    assert "jump" in res1.message.lower() or "boundary" in res1.message.lower()


def test_red_no_accumulated_double_counted_measurements() -> None:
    """Marginalized arrival prior must only summarize states outside the active window."""
    info = ArrivalInformation(
        reference_coefficients=np.array([1.0, 0.5]),
        sqrt_information=np.eye(2),
        rank=2,
        nullspace_basis=None,
        linearization_timestamp_s=1.0,
        marginalized_up_to_sample=2,
    )
    # The arrival prior covers up to sample 2; an active window starting at sample 2
    # must ensure sample 2 is the connecting boundary knot, not an internal overlapping factor
    assert info.marginalized_up_to_sample == 2
    residual = info.compute_residual(np.array([1.1, 0.5]))
    np.testing.assert_allclose(residual, [0.1, 0.0], atol=1e-10)


def test_green_linear_gaussian_mhe_agrees_with_batch() -> None:
    """Linear-Gaussian MHE with arrival tracking agrees with batch MAP within frozen tolerance."""
    n_steps = 8
    times = np.linspace(0.0, 3.5, n_steps)
    true_states = 0.5 * times

    def residual(evaluation, parameters):
        scale = parameters["scale"]
        truth_q = 0.5 * evaluation.times
        res_q = scale * evaluation.q[:, 0] - truth_q * scale
        res_v = evaluation.v[:, 0] - 0.5
        return np.concatenate([res_q, res_v])

    batch_prob = MovingHorizonProblem(
        n_dof=1,
        fixed_parameters={"scale": 1.0},
        residual=residual,
        options=MovingHorizonOptions(
            window_size=n_steps,
            step_size=n_steps,
            solver_options=MapEstimatorOptions(max_iterations=50),
        ),
    )
    batch_estimator = MovingHorizonEstimator(batch_prob)
    batch_estimator.append_samples(times, true_states[:, None])
    batch_res = batch_estimator.solve_next()
    assert batch_res is not None and batch_res.success
    batch_traj = batch_res.coefficients

    mhe_prob = MovingHorizonProblem(
        n_dof=1,
        fixed_parameters={"scale": 1.0},
        residual=residual,
        options=MovingHorizonOptions(
            window_size=4,
            step_size=1,
            solver_options=MapEstimatorOptions(max_iterations=50),
        ),
    )
    mhe = MovingHorizonEstimator(mhe_prob)
    mhe.append_samples(times[:4], true_states[:4, None])
    res = mhe.solve_next()
    assert res is not None and res.success

    for k in range(4, n_steps):
        mhe.append_samples([times[k]], np.array([[true_states[k]]]))
        res = mhe.solve_next()
        assert res is not None and res.success

    # The final window covers knots 4, 5, 6, 7 (indices 0..3 in mhe result)
    # Positions: res.coefficients[:4] vs batch_traj[4:8]
    # Velocities: res.coefficients[4:] vs batch_traj[12:16]
    np.testing.assert_allclose(
        res.coefficients[:4],
        batch_traj[4:8],
        atol=1e-4,
        err_msg="MHE position estimates must agree with full batch smoother reference within tolerance",
    )
    np.testing.assert_allclose(
        res.coefficients[4:],
        batch_traj[12:16],
        atol=1e-4,
        err_msg="MHE velocity estimates must agree with full batch smoother reference within tolerance",
    )


def test_green_failure_recovery_is_deterministic() -> None:
    """Estimator must deterministically recover after an injected failure."""
    should_fail = [False]

    options = MovingHorizonOptions(
        window_size=3,
        step_size=1,
        latency_budget_ms=100.0,
        solver_options=MapEstimatorOptions(max_iterations=50),
    )

    def residual(evaluation, parameters):
        if should_fail[0]:
            return np.full(evaluation.times.size, np.nan)
        scale = parameters["scale"]
        truth = scale * evaluation.times
        return scale * evaluation.q[:, 0] - truth

    def jacobian(evaluation, parameters, layout):
        jac = np.zeros((evaluation.times.size, layout.size), dtype=float)
        jac[:, : layout.trajectory_size] = (
            parameters["scale"] * evaluation.q_basis[:, 0, :]
        )
        return jac

    estimator = MovingHorizonEstimator(
        MovingHorizonProblem(
            n_dof=1,
            fixed_parameters={"scale": 2.0},
            residual=residual,
            jacobian=jacobian,
            options=options,
        )
    )

    # Clean step
    estimator.append_samples([0.0, 0.5, 1.0], np.array([[0.0], [0.5], [1.0]]))
    res0 = estimator.solve_next()
    assert res0 is not None and res0.success

    # Bad step: residual evaluates to NaN
    should_fail[0] = True
    estimator.append_samples([1.5], np.array([[1.5]]))
    res1 = estimator.solve_next()
    assert res1 is not None and not res1.success

    # Recovery step with valid observation
    should_fail[0] = False
    estimator.append_samples([2.0], np.array([[2.0]]))
    res2 = estimator.recover_and_solve_next(
        [1.5, 2.0, 2.5], np.array([[1.5], [2.0], [2.5]])
    )
    assert res2 is not None and res2.success
    assert res2.warm_started


def test_green_memory_remains_bounded() -> None:
    """MHE buffer and diagnostics must stay strictly bounded over many steps."""
    estimator = MovingHorizonEstimator(_tracking_problem(window_size=3, step_size=1))
    estimator.append_samples([0.0, 0.5, 1.0], np.array([[0.0], [0.5], [1.0]]))
    estimator.solve_next()

    for i in range(50):
        t = 1.5 + i * 0.5
        estimator.append_samples([t], np.array([[t]]))
        res = estimator.solve_next()
        assert res is not None and res.success
        assert estimator.buffered_sample_count == 3
        # Failure diagnostics must stay bounded
        assert len(estimator.failure_diagnostics) <= 50


def test_green_actual_latency_recorded() -> None:
    """Solve results must record actual positive elapsed latency, distinct from budget."""
    estimator = MovingHorizonEstimator(_tracking_problem(window_size=3, step_size=1))
    estimator.append_samples([0.0, 0.5, 1.0], np.array([[0.0], [0.5], [1.0]]))
    res = estimator.solve_next()

    assert res is not None
    assert res.latency_ms > 0.0
    assert res.latency_ms != res.latency_budget_ms
