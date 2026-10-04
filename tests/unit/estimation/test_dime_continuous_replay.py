"""Tests for DIME Offline Smoothing and Independent Continuous Replay (#11421, #11430).

Validates:
1. Fail-closed rejection of replay with changed model hash.
2. Fail-closed rejection of missing or truncated control sequences.
3. Fail-closed rejection of non-finite inputs.
4. Detection and rejection of undeclared root wrenches.
5. Detection and rejection of hidden target-force feedback.
6. Rejection of per-frame state resets / interrupted replay.
7. Backward inference uses adjoint/information propagation without reverse-time contact integration.
8. Uninterrupted continuous replay reproducing motion within frozen tolerance manifest.
9. Offline fixed-interval smoothing with zero-phase latency.
10. Explicit separation of optimization cost from recomputed replay metrics.
11. Fresh engine state independence without cross-run memory contamination.
12. Full receipt serialization roundtrip.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.contracts import PreconditionError
from src.shared.python.estimation.dime_contracts import (
    ContactPolicy,
    DimeCompleteState,
)
from src.shared.python.estimation.dime_manifest import NumericAcceptanceThresholds
from src.shared.python.estimation.dime_offline_smoothing import (
    ContinuousReplayConfig,
    ContinuousReplayReceipt,
    OfflineSmoothingProblem,
    OfflineSmoothingResult,
    execute_continuous_replay,
    run_offline_smoothing,
)
from src.shared.python.estimation.synthetic_fixtures import (
    make_fixed_base_pendulum_fixture,
)


class MockDynamicsProvider:
    """Mock dynamics provider tracking step directions and resets."""

    def __init__(
        self,
        *,
        model_hash: str = "mock-model-hash-v1",
        step_dt: float = 0.01,
        omega_sq: float = 9.81 / 1.0,
    ) -> None:
        self.model_hash = model_hash
        self.step_dt = step_dt
        self.omega_sq = omega_sq
        self.step_calls: list[float] = []
        self.reset_count = 0
        self.current_q = 0.1
        self.current_v = 0.0

    def reset(self, q0: float = 0.1, v0: float = 0.0) -> None:
        self.reset_count += 1
        self.current_q = q0
        self.current_v = v0

    def step(self, dt: float, tau: float = 0.0) -> tuple[float, float]:
        self.step_calls.append(dt)
        # Simple symplectic Euler integration: qddot = -omega_sq * q + tau
        qddot = -self.omega_sq * self.current_q + tau
        self.current_v += qddot * dt
        self.current_q += self.current_v * dt
        return self.current_q, self.current_v


def _create_test_initial_state(
    model_hash: str = "test-model-hash-v1",
    q0: float = 0.1,
    v0: float = 0.0,
) -> DimeCompleteState:
    return DimeCompleteState(
        t=0.0,
        q=np.array([q0], dtype=np.float64),
        v=np.array([v0], dtype=np.float64),
        model_hash=model_hash,
    )


# ==============================================================================
# RED Tests: Rejection & Safety Contracts
# ==============================================================================


def test_red_replay_with_changed_model_hash_rejected() -> None:
    """Replay with changed model hash must fail before simulation."""
    state = _create_test_initial_state(model_hash="model-hash-A")
    config = ContinuousReplayConfig(model_hash="model-hash-B")
    time_grid = np.linspace(0.0, 1.0, 11)
    controls = np.zeros((10, 1), dtype=np.float64)
    provider = MockDynamicsProvider(model_hash="model-hash-B")

    with pytest.raises(PreconditionError, match="model hash"):
        execute_continuous_replay(state, controls, time_grid, provider, config)


def test_red_missing_or_truncated_controls_rejected() -> None:
    """Controls with incorrect row length must be rejected fail-closed."""
    state = _create_test_initial_state(model_hash="model-hash-v1")
    config = ContinuousReplayConfig(model_hash="model-hash-v1")
    time_grid = np.linspace(0.0, 1.0, 11)
    controls = np.zeros((5, 1), dtype=np.float64)  # Truncated: 5 instead of 10
    provider = MockDynamicsProvider(model_hash="model-hash-v1")

    with pytest.raises(PreconditionError, match="controls"):
        execute_continuous_replay(state, controls, time_grid, provider, config)


def test_red_non_finite_inputs_rejected() -> None:
    """Non-finite numbers in initial state or controls must be rejected."""
    state = _create_test_initial_state(model_hash="model-hash-v1")
    config = ContinuousReplayConfig(model_hash="model-hash-v1")
    time_grid = np.linspace(0.0, 1.0, 11)
    controls = np.zeros((10, 1), dtype=np.float64)
    controls[2, 0] = np.nan
    provider = MockDynamicsProvider(model_hash="model-hash-v1")

    with pytest.raises(PreconditionError, match="finite"):
        execute_continuous_replay(state, controls, time_grid, provider, config)


def test_red_undeclared_root_wrench_rejected() -> None:
    """Undeclared root wrench in unactuated coordinates must fail physical acceptance."""
    state = _create_test_initial_state(model_hash="model-hash-v1")
    config = ContinuousReplayConfig(
        model_hash="model-hash-v1",
        allow_assistance_wrench=False,
        has_floating_base=True,
    )
    time_grid = np.linspace(0.0, 1.0, 11)
    # Control with 7 coordinates where first 6 are root DOFs and root force is non-zero
    controls = np.zeros((10, 7), dtype=np.float64)
    controls[:, 0] = 5.0  # Undeclared root wrench!
    provider = MockDynamicsProvider(model_hash="model-hash-v1")

    receipt = execute_continuous_replay(
        state,
        controls,
        time_grid,
        provider,
        config,
    )
    assert receipt.has_undeclared_root_forces is True
    assert receipt.is_physically_accepted is False


def test_red_hidden_target_force_feedback_rejected() -> None:
    """Injecting feedback assistance forces fails physical qualification."""
    state = _create_test_initial_state(model_hash="model-hash-v1")
    config = ContinuousReplayConfig(model_hash="model-hash-v1")
    time_grid = np.linspace(0.0, 1.0, 11)
    controls = np.zeros((10, 1), dtype=np.float64)
    provider = MockDynamicsProvider(model_hash="model-hash-v1")

    # Inject assistance forces
    assistance = np.full((10, 1), 2.5, dtype=np.float64)
    receipt = execute_continuous_replay(
        state,
        controls,
        time_grid,
        provider,
        config,
        assistance_wrench=assistance,
    )
    assert receipt.has_hidden_target_force_feedback is True
    assert receipt.is_physically_accepted is False


def test_red_per_frame_state_resets_rejected() -> None:
    """Mid-run state resets must mark replay as interrupted and physically unaccepted."""
    state = _create_test_initial_state(model_hash="model-hash-v1")
    config = ContinuousReplayConfig(
        model_hash="model-hash-v1",
        simulated_reset_count=5,  # 5 resets!
    )
    time_grid = np.linspace(0.0, 1.0, 11)
    controls = np.zeros((10, 1), dtype=np.float64)
    provider = MockDynamicsProvider(model_hash="model-hash-v1")

    # Simulate an interrupted replay with mid-run resets
    receipt = execute_continuous_replay(
        state,
        controls,
        time_grid,
        provider,
        config,
    )
    assert receipt.is_uninterrupted is False
    assert receipt.reset_count == 5
    assert receipt.is_physically_accepted is False


def test_red_backward_inference_never_integrates_reverse_contact() -> None:
    """Backward inference must NOT call reverse-time contact integration."""
    provider = MockDynamicsProvider(model_hash="model-hash-v1")
    state = _create_test_initial_state(model_hash="model-hash-v1")
    time_grid = np.linspace(0.0, 1.0, 11)
    observations = np.sin(time_grid)

    problem = OfflineSmoothingProblem(
        initial_state=state,
        time_grid=time_grid,
        observations=observations,
        model_hash="model-hash-v1",
        contact_policy=ContactPolicy.NATIVE_ELIMINATED,
    )

    result = run_offline_smoothing(problem, provider=provider)
    assert result.success is True
    # Verify that all step calls in the provider had positive dt (never negative dt)
    assert all(dt > 0 for dt in provider.step_calls), (
        "Backward inference illegally performed reverse-time integration (dt < 0)"
    )


# ==============================================================================
# GREEN Tests: Validated Behavior & Tolerance Contracts
# ==============================================================================


def test_green_uninterrupted_synthetic_replay_reproduces_within_tolerance() -> None:
    """Uninterrupted forward simulation reproduces analytic motion within frozen tolerance."""
    fixture = make_fixed_base_pendulum_fixture(n_frames=20, fps=100.0)
    theta_0, theta_dot_0 = fixture.initial_state
    state = DimeCompleteState(
        t=0.0,
        q=np.array([theta_0], dtype=np.float64),
        v=np.array([theta_dot_0], dtype=np.float64),
        model_hash="pendulum-v1",
    )
    config = ContinuousReplayConfig(
        model_hash="pendulum-v1",
        max_drift_m=NumericAcceptanceThresholds().max_drift_m,
        max_angular_drift_rad=NumericAcceptanceThresholds().max_angular_drift_rad,
        reference_trajectory_q=np.array([f.q[0] for f in fixture.frames]),
    )
    time_grid = np.array([f.timestamp for f in fixture.frames], dtype=np.float64)
    controls = np.zeros((len(time_grid) - 1, 1), dtype=np.float64)

    omega_sq = fixture.gravity_m_s2 / fixture.length_m
    dt = time_grid[1] - time_grid[0]
    provider = MockDynamicsProvider(
        model_hash="pendulum-v1",
        step_dt=dt,
        omega_sq=omega_sq,
    )

    receipt = execute_continuous_replay(
        state,
        controls,
        time_grid,
        provider,
        config,
    )

    assert receipt.is_uninterrupted is True
    assert receipt.reset_count == 1
    assert receipt.has_undeclared_root_forces is False
    assert receipt.has_hidden_target_force_feedback is False
    assert receipt.tracking_rmse_q < config.max_drift_m
    assert receipt.is_physically_accepted is True


def test_green_offline_smoothing_zero_phase_latency() -> None:
    """Offline smoothing uses past and future observations to achieve zero phase lag."""
    time_grid = np.linspace(0.0, 2.0, 101)
    true_signal = np.sin(2.0 * np.pi * 1.0 * time_grid)
    noisy_obs = true_signal + 0.05 * np.cos(2.0 * np.pi * 5.0 * time_grid)

    state = DimeCompleteState(
        t=0.0,
        q=np.array([true_signal[0]], dtype=np.float64),
        v=np.array([2.0 * np.pi], dtype=np.float64),
        model_hash="signal-v1",
    )
    problem = OfflineSmoothingProblem(
        initial_state=state,
        time_grid=time_grid,
        observations=noisy_obs,
        model_hash="signal-v1",
    )
    result = run_offline_smoothing(problem)

    assert result.success is True
    assert result.zero_phase_latency_s == 0.0
    # Smoothed trajectory should closely match true_signal with minimal phase distortion
    smoothed_q = np.array([s.q[0] for s in result.smoothed_states])
    correlation = np.corrcoef(smoothed_q, true_signal)[0, 1]
    assert correlation > 0.99


def test_green_optimization_cost_separated_from_replay_metrics() -> None:
    """Optimization objective cost must be strictly partitioned from recomputed replay metrics."""
    state = _create_test_initial_state(model_hash="model-hash-v1")
    config = ContinuousReplayConfig(model_hash="model-hash-v1")
    time_grid = np.linspace(0.0, 1.0, 11)
    controls = np.zeros((10, 1), dtype=np.float64)
    provider = MockDynamicsProvider(model_hash="model-hash-v1")

    opt_cost = 42.123
    receipt = execute_continuous_replay(
        state,
        controls,
        time_grid,
        provider,
        config,
        optimization_cost=opt_cost,
    )
    assert receipt.optimization_cost == opt_cost
    assert "tracking_rmse_q" in receipt.recomputed_replay_metrics
    assert receipt.recomputed_replay_metrics["tracking_rmse_q"] != opt_cost


def test_green_independent_fresh_engine_reset() -> None:
    """Sequential replays on the same provider must produce deterministic reset count and state."""
    state = _create_test_initial_state(model_hash="model-hash-v1")
    config = ContinuousReplayConfig(model_hash="model-hash-v1")
    time_grid = np.linspace(0.0, 0.5, 6)
    controls = np.zeros((5, 1), dtype=np.float64)
    provider = MockDynamicsProvider(model_hash="model-hash-v1")

    r1 = execute_continuous_replay(state, controls, time_grid, provider, config)
    assert r1.reset_count == 1
    assert provider.reset_count == 1

    r2 = execute_continuous_replay(state, controls, time_grid, provider, config)
    assert r2.reset_count == 1
    assert provider.reset_count == 2
    np.testing.assert_allclose(r1.trajectory_q, r2.trajectory_q)


def test_green_receipt_serialization_roundtrip() -> None:
    """Receipt must roundtrip through to_dict and from_dict without data loss."""
    time_grid = np.linspace(0.0, 1.0, 5)
    traj_q = np.array([[0.1], [0.2], [0.3], [0.4], [0.5]])
    traj_v = np.array([[0.0], [0.1], [0.2], [0.3], [0.4]])

    receipt = ContinuousReplayReceipt(
        schema_version="dime-continuous-replay-receipt-v1",
        is_physically_accepted=True,
        is_uninterrupted=True,
        reset_count=1,
        has_undeclared_root_forces=False,
        has_hidden_target_force_feedback=False,
        model_hash="model-hash-v1",
        coverage_start_s=0.0,
        coverage_end_s=1.0,
        tracking_rmse_q=0.005,
        tracking_rmse_v=0.01,
        max_drift_m=0.015,
        max_angular_drift_rad=0.05,
        recomputed_replay_metrics={"tracking_rmse_q": 0.005, "tracking_rmse_v": 0.01},
        optimization_cost=12.5,
        trajectory_q=traj_q,
        trajectory_v=traj_v,
        time_grid=time_grid,
    )

    data = receipt.to_dict()
    assert data["schema_version"] == "dime-continuous-replay-receipt-v1"
    assert data["is_physically_accepted"] is True

    recovered = ContinuousReplayReceipt.from_dict(data)
    assert recovered.is_physically_accepted == receipt.is_physically_accepted
    assert recovered.reset_count == receipt.reset_count
    assert recovered.model_hash == receipt.model_hash
    np.testing.assert_allclose(recovered.trajectory_q, receipt.trajectory_q)
    np.testing.assert_allclose(recovered.time_grid, receipt.time_grid)
