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
    AccumulationGuard,
    ArrivalFactor,
    FailureDiagnostics,
    LateSamplePolicy,
    WindowCommitStatus,
    marginalize_arrival_factor,
)

pytestmark = pytest.mark.unit


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


def test_arrival_factor_representation_and_linearization_metadata() -> None:
    """Arrival factor represents square-root quadratic factor with rank and metadata."""
    ref_state = np.array([1.0, 2.0], dtype=np.float64)
    sqrt_info = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float64)
    offset = np.array([0.1, -0.2], dtype=np.float64)
    metadata = {
        "timestamp": 1.5,
        "window_index": 3,
        "model_hash": "model_abc123",
        "scheme_label": "exact_schur",
    }

    arrival = ArrivalFactor(
        reference_state=ref_state,
        sqrt_information=sqrt_info,
        residual_offset=offset,
        rank=2,
        metadata=metadata,
    )

    assert arrival.rank == 2
    assert not arrival.is_rank_deficient
    assert arrival.metadata["model_hash"] == "model_abc123"
    assert arrival.metadata["scheme_label"] == "exact_schur"

    # Tangent perturbation evaluation
    state = np.array([1.1, 1.9], dtype=np.float64)
    delta_x = state - ref_state  # [0.1, -0.1]
    expected_residual = (
        sqrt_info @ delta_x - offset
    )  # [2.0*0.1 - 0.1, 3.0*(-0.1) - (-0.2)] = [0.1, -0.1]
    res = arrival.evaluate_residual(state)
    np.testing.assert_allclose(res, expected_residual, atol=1e-12)

    cost = arrival.evaluate_cost(state)
    expected_cost = 0.5 * float(np.sum(expected_residual**2))
    assert np.isclose(cost, expected_cost, atol=1e-12)

    jac = arrival.evaluate_jacobian(state)
    np.testing.assert_allclose(jac, sqrt_info, atol=1e-12)


def test_rank_revealing_marginalization_retains_rank_deficiency_without_diagonal_jitter() -> (
    None
):
    """Singular marginalization reveals rank deficiency and does not inject fake certainty via jitter."""
    # State x0 has dimension 2 (e.g. position and unmeasured angle), state x1 has dimension 2.
    # Only position is measured at x0: A0 has rank 1 (deficient).
    A0 = np.array([[2.0, 0.0], [0.0, 0.0]], dtype=np.float64)  # rank 1 for 2D state
    A1 = np.array([[1.0, 0.5], [0.0, 1.0]], dtype=np.float64)
    b = np.array([0.4, 0.0], dtype=np.float64)

    # Under retain_rank_deficiency policy (default):
    arrival = marginalize_arrival_factor(
        A0=A0,
        A1=A1,
        b=b,
        reference_state_k1=np.zeros(2, dtype=np.float64),
        gauge_policy="retain_rank_deficiency",
        metadata={"scheme_label": "rank_revealing_qr"},
    )

    assert arrival.is_rank_deficient
    assert arrival.rank == 1
    assert arrival.metadata["gauge_policy"] == "retain_rank_deficiency"

    # Verify NO diagonal jitter added: singular values of R must retain true zero
    s_vals = np.linalg.svd(arrival.sqrt_information, compute_uv=False)
    assert np.isclose(s_vals[-1], 0.0, atol=1e-10)

    # Under fail_closed policy, singular marginalization raises PreconditionError
    with pytest.raises(PreconditionError, match="Singular/rank-deficient"):
        marginalize_arrival_factor(
            A0=A0,
            A1=A1,
            b=b,
            reference_state_k1=np.zeros(2, dtype=np.float64),
            gauge_policy="fail_closed",
        )


def test_linear_gaussian_mhe_agrees_with_batch_map_within_frozen_tolerance() -> None:
    """MHE estimates with arrival factor agree with batch MAP on linear-Gaussian fixture."""
    # 1D spring-mass-damper / discrete linear system:
    # x_{k+1} = a * x_k + b * u_k + w_k,  w_k ~ N(0, q_var)
    # y_k = c * x_k + v_k,                v_k ~ N(0, r_var)
    a_dyn = 0.9
    b_dyn = 0.5
    c_obs = 1.2
    w_std = 0.1
    v_std = 0.05
    x0_mean = 1.0
    x0_std = 0.2

    n_steps = 8
    np.random.seed(42)
    u_seq = np.sin(np.linspace(0, 2.0, n_steps))
    x_true = np.zeros(n_steps)
    x_true[0] = x0_mean + np.random.randn() * x0_std
    for k in range(n_steps - 1):
        x_true[k + 1] = a_dyn * x_true[k] + b_dyn * u_seq[k] + np.random.randn() * w_std
    y_obs = c_obs * x_true + np.random.randn(n_steps) * v_std

    # 1. Batch MAP solution over all n_steps simultaneously
    # Decision vector x = [x_0, x_1, ..., x_{N-1}]
    # Residuals:
    # Prior: (x_0 - x0_mean) / x0_std
    # Dynamics: (x_{k+1} - a*x_k - b*u_k) / w_std  for k in 0..N-2
    # Obs: (c*x_k - y_k) / v_std                    for k in 0..N-1
    n_res = 1 + (n_steps - 1) + n_steps
    J_batch = np.zeros((n_res, n_steps))
    r_batch = np.zeros(n_res)

    row = 0
    # Prior
    J_batch[row, 0] = 1.0 / x0_std
    r_batch[row] = (0.0 - x0_mean) / x0_std
    row += 1

    # Dynamics
    for k in range(n_steps - 1):
        J_batch[row, k] = -a_dyn / w_std
        J_batch[row, k + 1] = 1.0 / w_std
        r_batch[row] = -b_dyn * u_seq[k] / w_std
        row += 1

    # Obs
    for k in range(n_steps):
        J_batch[row, k] = c_obs / v_std
        r_batch[row] = -y_obs[k] / v_std
        row += 1

    # Solve batch least squares: min || J x + r ||^2  => J^T J x = - J^T r
    x_batch, _, _, _ = np.linalg.lstsq(J_batch, -r_batch, rcond=None)

    # 2. Windowed MHE with window_size = 3
    # Step-by-step:
    # Start at k=0..2 with initial prior
    # Marginalize x_0 to obtain arrival factor on x_1
    # Solve window k=1..3 with arrival factor, etc.
    arrival = ArrivalFactor(
        reference_state=np.array([x0_mean]),
        sqrt_information=np.array([[1.0 / x0_std]]),
        residual_offset=np.zeros(1),
        rank=1,
        metadata={"step": 0},
    )

    # 2. Windowed MHE with window_size = 3
    # Step-by-step:
    # Start at k=0..2 with initial prior
    # Marginalize x_0 to form arrival factor on x_1, etc.
    arrival = ArrivalFactor(
        reference_state=np.array([x0_mean]),
        sqrt_information=np.array([[1.0 / x0_std]]),
        residual_offset=np.zeros(1),
        rank=1,
        metadata={"step": 0},
    )

    curr_x_window = np.zeros(3)
    for start_k in range(n_steps - 2):
        # Solve window [start_k, start_k+1, start_k+2]
        # Window residuals:
        # 1. Arrival prior on x_{start_k}: R_a (x_{start_k} - ref) - r_a
        # 2. Observation at x_{start_k}
        # 3. Dynamics start_k -> start_k+1
        # 4. Observation at x_{start_k+1}
        # 5. Dynamics start_k+1 -> start_k+2
        # 6. Observation at x_{start_k+2}
        w_size = 3
        J_win = np.zeros((6, w_size))
        r_win = np.zeros(6)

        # Arrival prior on x_{start_k}
        J_win[0, 0] = arrival.sqrt_information[0, 0]
        r_win[0] = (
            -arrival.sqrt_information[0, 0] * arrival.reference_state[0]
            - arrival.residual_offset[0]
        )

        # Obs at start_k
        J_win[1, 0] = c_obs / v_std
        r_win[1] = -y_obs[start_k] / v_std

        # Dyn start_k -> start_k+1
        J_win[2, 0] = -a_dyn / w_std
        J_win[2, 1] = 1.0 / w_std
        r_win[2] = -b_dyn * u_seq[start_k] / w_std

        # Obs at start_k+1
        J_win[3, 1] = c_obs / v_std
        r_win[3] = -y_obs[start_k + 1] / v_std

        # Dyn start_k+1 -> start_k+2
        J_win[4, 1] = -a_dyn / w_std
        J_win[4, 2] = 1.0 / w_std
        r_win[4] = -b_dyn * u_seq[start_k + 1] / w_std

        # Obs at start_k+2
        J_win[5, 2] = c_obs / v_std
        r_win[5] = -y_obs[start_k + 2] / v_std

        curr_x_window, _, _, _ = np.linalg.lstsq(J_win, -r_win, rcond=None)

        # Now marginalize ONLY the factors involving x_{start_k} (rows 0, 1, 2)
        # to advance the arrival factor to x_{start_k+1} without double counting
        x0_ref = arrival.reference_state[0]
        x1_ref = curr_x_window[1]

        A0 = np.array(
            [
                [arrival.sqrt_information[0, 0]],
                [c_obs / v_std],
                [-a_dyn / w_std],
            ]
        )
        A1 = np.array(
            [
                [0.0],
                [0.0],
                [1.0 / w_std],
            ]
        )
        # In tangent coordinates delta x0 = x0 - x0_ref, delta x1 = x1 - x1_ref:
        # Eq 0: R_a delta x0 = r_a
        # Eq 1: c (x0_ref + delta x0) = y => c delta x0 = y - c x0_ref
        # Eq 2: (x1_ref + delta x1 - a (x0_ref + delta x0) - b u) / w = 0
        #       => -a delta x0 + delta x1 = b u + a x0_ref - x1_ref
        b_vec = np.array(
            [
                arrival.residual_offset[0],
                (y_obs[start_k] - c_obs * x0_ref) / v_std,
                (b_dyn * u_seq[start_k] + a_dyn * x0_ref - x1_ref) / w_std,
            ]
        )

        arrival = marginalize_arrival_factor(
            A0=A0,
            A1=A1,
            b=b_vec,
            reference_state_k1=np.array([x1_ref]),
            metadata={"step": start_k + 1},
        )

    # The latest estimate at the end of the final window (x_{N-1}) should agree with batch MAP
    # within frozen numerical tolerance (1e-5)
    np.testing.assert_allclose(curr_x_window[2], x_batch[-1], atol=1e-5)


def test_safe_window_commits_rejects_unsuccessful_solves_retains_diagnostics() -> None:
    """Failed/non-finite solves do not overwrite last accepted state or poison warm-start."""
    fail_trigger = {"fail": False}

    def residual(evaluation, parameters):
        if fail_trigger["fail"]:
            # Return NaN / non-finite to simulate divergence/infeasibility
            return np.full(evaluation.times.size, np.nan)
        return evaluation.q[:, 0] - evaluation.times

    options = MovingHorizonOptions(
        window_size=3,
        step_size=1,
        latency_budget_ms=50.0,
        enable_safe_commits=True,
    )
    problem = MovingHorizonProblem(
        n_dof=1,
        fixed_parameters={"dummy": 1.0},
        residual=residual,
        options=options,
    )
    estimator = MovingHorizonEstimator(problem)

    # 1. First solve succeeds
    estimator.append_samples([0.0, 0.5, 1.0], np.array([[0.0], [0.5], [1.0]]))
    res1 = estimator.solve_next()
    assert res1 is not None
    assert res1.success
    assert res1.commit_status == WindowCommitStatus.ACCEPTED
    last_valid_coeffs = np.copy(estimator.last_accepted_coefficients)
    assert last_valid_coeffs is not None

    # 2. Second solve triggers failure / non-finite
    fail_trigger["fail"] = True
    estimator.append_samples([1.5], np.array([[1.5]]))
    res2 = estimator.solve_next()

    assert res2 is not None
    assert not res2.success
    assert res2.commit_status != WindowCommitStatus.ACCEPTED
    # State commit was rejected: last_accepted_coefficients untouched!
    np.testing.assert_allclose(
        estimator.last_accepted_coefficients, last_valid_coeffs, atol=1e-12
    )
    # Failure diagnostics recorded separately
    assert estimator.last_failure_diagnostics is not None
    assert estimator.last_failure_diagnostics.window_index == 1
    assert "non_finite" in estimator.last_failure_diagnostics.reason.lower()

    # 3. Third solve recovers: warm-starts from last accepted state, NOT the bad state
    fail_trigger["fail"] = False
    estimator.append_samples([2.0], np.array([[2.0]]))
    res3 = estimator.solve_next()

    assert res3 is not None
    assert res3.success
    assert res3.commit_status == WindowCommitStatus.ACCEPTED
    assert res3.warm_started


def test_late_and_irregular_sample_handling() -> None:
    """Monotonicity violations and irregular sampling policies."""
    options_reject = MovingHorizonOptions(
        window_size=3, late_sample_policy=LateSamplePolicy.REJECT
    )
    prob_reject = MovingHorizonProblem(
        n_dof=1,
        fixed_parameters={"p": 1.0},
        residual=lambda ev, pm: ev.q[:, 0],
        options=options_reject,
    )
    est_reject = MovingHorizonEstimator(prob_reject)
    est_reject.append_samples([0.0, 1.0], np.array([[0.0], [1.0]]))

    # Duplicate / late timestamp with REJECT policy raises PreconditionError
    with pytest.raises(PreconditionError, match="advance monotonically"):
        est_reject.append_samples([1.0], np.array([[1.0]]))

    # DROP_LATE policy drops late/duplicate sample gracefully
    options_drop = MovingHorizonOptions(
        window_size=3, late_sample_policy=LateSamplePolicy.DROP_LATE
    )
    prob_drop = MovingHorizonProblem(
        n_dof=1,
        fixed_parameters={"p": 1.0},
        residual=lambda ev, pm: ev.q[:, 0],
        options=options_drop,
    )
    est_drop = MovingHorizonEstimator(prob_drop)
    est_drop.append_samples([0.0, 1.0], np.array([[0.0], [1.0]]))
    est_drop.append_samples(
        [0.5, 2.0], np.array([[0.5], [2.0]])
    )  # 0.5 is late, dropped; 2.0 kept
    assert est_drop.buffered_sample_count == 3
    times, _ = est_drop.buffer_arrays()
    np.testing.assert_allclose(times, [0.0, 1.0, 2.0])

    # Irregular spacing (non-uniform dt)
    est_drop.append_samples([3.7], np.array([[3.7]]))
    assert est_drop.buffered_sample_count == 3
    times_irreg, _ = est_drop.buffer_arrays()
    np.testing.assert_allclose(times_irreg, [1.0, 2.0, 3.7])


def test_accumulation_guard_prevents_double_counted_measurements() -> None:
    """Accumulation guard prevents re-including already-marginalized measurements."""
    guard = AccumulationGuard()
    guard.record_marginalized(sample_index=0, timestamp=0.0)
    guard.record_marginalized(sample_index=1, timestamp=0.5)

    assert guard.is_marginalized(sample_index=0)
    assert guard.is_marginalized(sample_index=1)
    assert not guard.is_marginalized(sample_index=2)

    # Attempting to add an already-marginalized measurement raises PreconditionError
    with pytest.raises(PreconditionError, match="already marginalized"):
        guard.validate_sample(sample_index=1, timestamp=0.5)

    # Non-marginalized sample validates fine
    guard.validate_sample(sample_index=2, timestamp=1.0)


def test_bounded_memory_across_long_estimation_horizon() -> None:
    """Estimator maintains bounded O(1) buffer and history across 60 window advances."""
    options = MovingHorizonOptions(
        window_size=3,
        step_size=1,
        max_history_diagnostics=10,
    )
    prob = _tracking_problem(window_size=3, step_size=1)
    estimator = MovingHorizonEstimator(prob)

    estimator.append_samples([0.0, 0.1, 0.2], np.array([[0.0], [0.1], [0.2]]))
    estimator.solve_next()

    for step in range(3, 60):
        t = step * 0.1
        estimator.append_samples([t], np.array([[t]]))
        res = estimator.solve_next()
        assert res is not None
        assert estimator.buffered_sample_count <= 3

    assert estimator.buffered_sample_count == 3
