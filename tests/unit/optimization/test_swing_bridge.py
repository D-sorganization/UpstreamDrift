"""Unit tests for the SwingOptimizationBridge module.

Tests cover:
- SwingOptimizationConfig validation (types, ranges, edge cases)
- SwingOptimizationResult dataclass
- SwingOptimizationBridge initialisation
- Cost matrix construction (symmetry, PSD, dimensions)
- Trajectory evaluation (double-integrator dynamics)
- Full optimization with and without mock engine
- Initial state validation (dimensions, finiteness)
- Convergence and iteration behaviour
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.shared.python.optimization.swing_bridge import (
    SwingOptimizationBridge,
    SwingOptimizationConfig,
    SwingOptimizationResult,
)


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def default_config() -> SwingOptimizationConfig:
    """Default configuration with standard parameters."""
    return SwingOptimizationConfig()


@pytest.fixture
def small_config() -> SwingOptimizationConfig:
    """Small configuration for fast tests."""
    return SwingOptimizationConfig(
        n_joints=2,
        horizon_steps=10,
        dt=0.01,
        max_iterations=5,
        convergence_tol=1e-4,
        target_clubhead_velocity=30.0,
        control_cost_weight=0.01,
        terminal_cost_weight=50.0,
    )


@pytest.fixture
def bridge(default_config: SwingOptimizationConfig) -> SwingOptimizationBridge:
    """Bridge with default config and no engine."""
    return SwingOptimizationBridge(default_config)


@pytest.fixture
def small_bridge(
    small_config: SwingOptimizationConfig,
) -> SwingOptimizationBridge:
    """Bridge with small config for fast optimisation tests."""
    return SwingOptimizationBridge(small_config)


# =========================================================================
# SwingOptimizationConfig — defaults and valid construction
# =========================================================================


class TestSwingOptimizationConfigDefaults:
    """Verify default configuration values."""

    def test_default_n_joints(self, default_config: SwingOptimizationConfig) -> None:
        assert default_config.n_joints == 7

    def test_default_horizon_steps(
        self, default_config: SwingOptimizationConfig
    ) -> None:
        assert default_config.horizon_steps == 100

    def test_default_dt(self, default_config: SwingOptimizationConfig) -> None:
        assert default_config.dt == 0.01

    def test_default_max_iterations(
        self, default_config: SwingOptimizationConfig
    ) -> None:
        assert default_config.max_iterations == 50

    def test_default_convergence_tol(
        self, default_config: SwingOptimizationConfig
    ) -> None:
        assert default_config.convergence_tol == 1e-6

    def test_default_target_clubhead_velocity(
        self, default_config: SwingOptimizationConfig
    ) -> None:
        assert default_config.target_clubhead_velocity == 50.0

    def test_default_control_cost_weight(
        self, default_config: SwingOptimizationConfig
    ) -> None:
        assert default_config.control_cost_weight == 0.01

    def test_default_terminal_cost_weight(
        self, default_config: SwingOptimizationConfig
    ) -> None:
        assert default_config.terminal_cost_weight == 100.0


# =========================================================================
# SwingOptimizationConfig — validation errors
# =========================================================================


class TestSwingOptimizationConfigValidation:
    """Validate that bad inputs are rejected with proper exceptions."""

    def test_n_joints_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="n_joints"):
            SwingOptimizationConfig(n_joints=0)

    def test_n_joints_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="n_joints"):
            SwingOptimizationConfig(n_joints=-1)

    def test_n_joints_too_large_raises(self) -> None:
        with pytest.raises(ValueError, match="n_joints"):
            SwingOptimizationConfig(n_joints=51)

    def test_n_joints_float_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="n_joints must be int"):
            SwingOptimizationConfig(n_joints=7.0)  # type: ignore[arg-type]

    def test_horizon_steps_one_raises(self) -> None:
        with pytest.raises(ValueError, match="horizon_steps"):
            SwingOptimizationConfig(horizon_steps=1)

    def test_horizon_steps_too_large_raises(self) -> None:
        with pytest.raises(ValueError, match="horizon_steps"):
            SwingOptimizationConfig(horizon_steps=10_001)

    def test_dt_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="dt"):
            SwingOptimizationConfig(dt=0.0)

    def test_dt_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="dt"):
            SwingOptimizationConfig(dt=-0.01)

    def test_dt_too_large_raises(self) -> None:
        with pytest.raises(ValueError, match="dt"):
            SwingOptimizationConfig(dt=1.1)

    def test_max_iterations_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="max_iterations"):
            SwingOptimizationConfig(max_iterations=0)

    def test_convergence_tol_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="convergence_tol"):
            SwingOptimizationConfig(convergence_tol=0.0)

    def test_convergence_tol_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="convergence_tol"):
            SwingOptimizationConfig(convergence_tol=-1e-6)

    def test_target_clubhead_velocity_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="target_clubhead_velocity"):
            SwingOptimizationConfig(target_clubhead_velocity=0.0)

    def test_control_cost_weight_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="control_cost_weight"):
            SwingOptimizationConfig(control_cost_weight=-0.01)

    def test_terminal_cost_weight_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="terminal_cost_weight"):
            SwingOptimizationConfig(terminal_cost_weight=-1.0)


# =========================================================================
# SwingOptimizationConfig — edge cases that should succeed
# =========================================================================


class TestSwingOptimizationConfigEdgeCases:
    """Edge cases that are valid and should NOT raise."""

    def test_n_joints_one(self) -> None:
        cfg = SwingOptimizationConfig(n_joints=1)
        assert cfg.n_joints == 1

    def test_n_joints_max(self) -> None:
        cfg = SwingOptimizationConfig(n_joints=50)
        assert cfg.n_joints == 50

    def test_horizon_steps_min(self) -> None:
        cfg = SwingOptimizationConfig(horizon_steps=2)
        assert cfg.horizon_steps == 2

    def test_dt_minimum(self) -> None:
        cfg = SwingOptimizationConfig(dt=1e-6)
        assert cfg.dt == pytest.approx(1e-6)

    def test_control_cost_weight_zero(self) -> None:
        """Zero control cost is valid (no regularisation)."""
        cfg = SwingOptimizationConfig(control_cost_weight=0.0)
        assert cfg.control_cost_weight == 0.0

    def test_terminal_cost_weight_zero(self) -> None:
        """Zero terminal cost is valid (pure tracking)."""
        cfg = SwingOptimizationConfig(terminal_cost_weight=0.0)
        assert cfg.terminal_cost_weight == 0.0


# =========================================================================
# SwingOptimizationResult
# =========================================================================


class TestSwingOptimizationResult:
    """Verify the result dataclass stores and exposes fields correctly."""

    def test_result_fields(self) -> None:
        result = SwingOptimizationResult(
            optimal_torques=[np.zeros(3)],
            trajectory=[np.zeros(6), np.zeros(6)],
            clubhead_velocity=45.0,
            total_cost=12.3,
            converged=True,
            iterations=10,
            computation_time_s=0.05,
        )
        assert result.clubhead_velocity == 45.0
        assert result.converged is True
        assert result.iterations == 10
        assert result.total_cost == pytest.approx(12.3)
        assert result.computation_time_s >= 0
        assert len(result.optimal_torques) == 1
        assert len(result.trajectory) == 2


# =========================================================================
# SwingOptimizationBridge — initialisation
# =========================================================================


class TestSwingOptimizationBridgeInit:
    """Test bridge construction and property access."""

    def test_init_with_default_config(
        self, bridge: SwingOptimizationBridge
    ) -> None:
        assert bridge.config.n_joints == 7
        assert bridge.engine is None

    def test_state_dim(self, bridge: SwingOptimizationBridge) -> None:
        assert bridge.state_dim == 14  # 2 * 7

    def test_control_dim(self, bridge: SwingOptimizationBridge) -> None:
        assert bridge.control_dim == 7

    def test_init_with_engine(
        self, default_config: SwingOptimizationConfig
    ) -> None:
        engine = MagicMock()
        b = SwingOptimizationBridge(default_config, engine=engine)
        assert b.engine is engine

    def test_init_bad_config_type_raises(self) -> None:
        with pytest.raises(TypeError, match="config must be"):
            SwingOptimizationBridge(config={"n_joints": 7})  # type: ignore[arg-type]

    def test_init_none_config_raises(self) -> None:
        with pytest.raises(TypeError, match="config must be"):
            SwingOptimizationBridge(config=None)  # type: ignore[arg-type]


# =========================================================================
# Cost matrix construction
# =========================================================================


class TestCostMatrices:
    """Verify Q and R cost matrices have correct properties."""

    def test_q_shape(self, bridge: SwingOptimizationBridge) -> None:
        Q, _ = bridge._build_cost_matrices(7)
        assert Q.shape == (14, 14)

    def test_r_shape(self, bridge: SwingOptimizationBridge) -> None:
        _, R = bridge._build_cost_matrices(7)
        assert R.shape == (7, 7)

    def test_q_symmetric(self, bridge: SwingOptimizationBridge) -> None:
        Q, _ = bridge._build_cost_matrices(7)
        np.testing.assert_array_equal(Q, Q.T)

    def test_r_symmetric(self, bridge: SwingOptimizationBridge) -> None:
        _, R = bridge._build_cost_matrices(7)
        np.testing.assert_array_equal(R, R.T)

    def test_q_positive_semi_definite(
        self, bridge: SwingOptimizationBridge
    ) -> None:
        Q, _ = bridge._build_cost_matrices(7)
        eigenvalues = np.linalg.eigvalsh(Q)
        assert np.all(eigenvalues >= -1e-12), (
            f"Q is not PSD: min eigenvalue = {eigenvalues.min()}"
        )

    def test_r_positive_semi_definite(
        self, bridge: SwingOptimizationBridge
    ) -> None:
        _, R = bridge._build_cost_matrices(7)
        eigenvalues = np.linalg.eigvalsh(R)
        assert np.all(eigenvalues >= -1e-12), (
            f"R is not PSD: min eigenvalue = {eigenvalues.min()}"
        )

    def test_r_diagonal_values(
        self, bridge: SwingOptimizationBridge
    ) -> None:
        _, R = bridge._build_cost_matrices(7)
        expected = 0.01 * np.eye(7)
        np.testing.assert_allclose(R, expected)

    def test_q_velocity_block_identity(
        self, bridge: SwingOptimizationBridge
    ) -> None:
        Q, _ = bridge._build_cost_matrices(7)
        np.testing.assert_array_equal(Q[7:, 7:], np.eye(7))

    def test_q_position_block_zero(
        self, bridge: SwingOptimizationBridge
    ) -> None:
        Q, _ = bridge._build_cost_matrices(7)
        np.testing.assert_array_equal(Q[:7, :7], np.zeros((7, 7)))

    def test_build_cost_matrices_n_zero_raises(
        self, bridge: SwingOptimizationBridge
    ) -> None:
        with pytest.raises(ValueError, match="n must be >= 1"):
            bridge._build_cost_matrices(0)

    def test_build_cost_matrices_n_one(
        self, bridge: SwingOptimizationBridge
    ) -> None:
        Q, R = bridge._build_cost_matrices(1)
        assert Q.shape == (2, 2)
        assert R.shape == (1, 1)


# =========================================================================
# Initial state validation
# =========================================================================


class TestInitialStateValidation:
    """Verify that optimize_swing rejects invalid initial states."""

    def test_non_array_raises(
        self, small_bridge: SwingOptimizationBridge
    ) -> None:
        with pytest.raises(TypeError, match="initial_state must be np.ndarray"):
            small_bridge.optimize_swing([0.0] * 4)  # type: ignore[arg-type]

    def test_wrong_length_raises(
        self, small_bridge: SwingOptimizationBridge
    ) -> None:
        x0 = np.zeros(10)  # config has n_joints=2 -> expects 4
        with pytest.raises(ValueError, match="initial_state length must be 4"):
            small_bridge.optimize_swing(x0)

    def test_2d_array_raises(
        self, small_bridge: SwingOptimizationBridge
    ) -> None:
        x0 = np.zeros((4, 1))
        with pytest.raises(ValueError, match="must be 1-D"):
            small_bridge.optimize_swing(x0)

    def test_nan_raises(
        self, small_bridge: SwingOptimizationBridge
    ) -> None:
        x0 = np.array([0.0, 0.0, float("nan"), 0.0])
        with pytest.raises(ValueError, match="finite values"):
            small_bridge.optimize_swing(x0)

    def test_inf_raises(
        self, small_bridge: SwingOptimizationBridge
    ) -> None:
        x0 = np.array([0.0, 0.0, float("inf"), 0.0])
        with pytest.raises(ValueError, match="finite values"):
            small_bridge.optimize_swing(x0)


# =========================================================================
# Trajectory evaluation (double-integrator)
# =========================================================================


class TestTrajectoryEvaluation:
    """Test the internal _evaluate_trajectory with double-integrator."""

    def test_trajectory_length(
        self, small_bridge: SwingOptimizationBridge
    ) -> None:
        n = small_bridge.config.n_joints
        controls = [np.zeros(n) for _ in range(small_bridge.config.horizon_steps)]
        x0 = np.zeros(small_bridge.state_dim)
        traj, _ = small_bridge._evaluate_trajectory(controls, x0)
        # trajectory has horizon_steps + 1 entries (initial + each step)
        assert len(traj) == small_bridge.config.horizon_steps + 1

    def test_zero_controls_zero_initial_state(
        self, small_bridge: SwingOptimizationBridge
    ) -> None:
        """Zero controls from rest should keep state at zero."""
        n = small_bridge.config.n_joints
        controls = [np.zeros(n) for _ in range(small_bridge.config.horizon_steps)]
        x0 = np.zeros(small_bridge.state_dim)
        traj, vel = small_bridge._evaluate_trajectory(controls, x0)
        np.testing.assert_allclose(traj[-1], np.zeros(small_bridge.state_dim))
        assert vel == pytest.approx(0.0)

    def test_constant_torque_increases_velocity(
        self, small_bridge: SwingOptimizationBridge
    ) -> None:
        """Constant positive torque should increase velocity."""
        n = small_bridge.config.n_joints
        controls = [np.ones(n) for _ in range(small_bridge.config.horizon_steps)]
        x0 = np.zeros(small_bridge.state_dim)
        _, vel = small_bridge._evaluate_trajectory(controls, x0)
        assert vel > 0.0

    def test_trajectory_first_entry_is_initial_state(
        self, small_bridge: SwingOptimizationBridge
    ) -> None:
        n = small_bridge.config.n_joints
        controls = [np.zeros(n) for _ in range(small_bridge.config.horizon_steps)]
        x0 = np.array([1.0, 2.0, 0.5, -0.5])
        traj, _ = small_bridge._evaluate_trajectory(controls, x0)
        np.testing.assert_array_equal(traj[0], x0)


# =========================================================================
# Full optimisation
# =========================================================================


class TestOptimizeSwing:
    """End-to-end optimisation tests."""

    def test_returns_result_type(
        self, small_bridge: SwingOptimizationBridge
    ) -> None:
        x0 = np.zeros(small_bridge.state_dim)
        result = small_bridge.optimize_swing(x0)
        assert isinstance(result, SwingOptimizationResult)

    def test_result_torques_length(
        self, small_bridge: SwingOptimizationBridge
    ) -> None:
        x0 = np.zeros(small_bridge.state_dim)
        result = small_bridge.optimize_swing(x0)
        assert len(result.optimal_torques) == small_bridge.config.horizon_steps

    def test_result_trajectory_length(
        self, small_bridge: SwingOptimizationBridge
    ) -> None:
        x0 = np.zeros(small_bridge.state_dim)
        result = small_bridge.optimize_swing(x0)
        assert len(result.trajectory) == small_bridge.config.horizon_steps + 1

    def test_computation_time_positive(
        self, small_bridge: SwingOptimizationBridge
    ) -> None:
        x0 = np.zeros(small_bridge.state_dim)
        result = small_bridge.optimize_swing(x0)
        assert result.computation_time_s > 0

    def test_iterations_at_least_one(
        self, small_bridge: SwingOptimizationBridge
    ) -> None:
        x0 = np.zeros(small_bridge.state_dim)
        result = small_bridge.optimize_swing(x0)
        assert result.iterations >= 1

    def test_clubhead_velocity_non_negative(
        self, small_bridge: SwingOptimizationBridge
    ) -> None:
        x0 = np.zeros(small_bridge.state_dim)
        result = small_bridge.optimize_swing(x0)
        assert result.clubhead_velocity >= 0.0

    def test_total_cost_finite(
        self, small_bridge: SwingOptimizationBridge
    ) -> None:
        x0 = np.zeros(small_bridge.state_dim)
        result = small_bridge.optimize_swing(x0)
        assert np.isfinite(result.total_cost)

    def test_optimisation_reduces_cost_vs_zero_control(
        self, small_bridge: SwingOptimizationBridge
    ) -> None:
        """Optimised controls should yield lower cost than zero controls."""
        x0 = np.zeros(small_bridge.state_dim)
        result = small_bridge.optimize_swing(x0)

        # Zero-control cost = terminal_cost_weight * target_vel^2
        zero_cost = (
            small_bridge.config.terminal_cost_weight
            * small_bridge.config.target_clubhead_velocity ** 2
        )
        assert result.total_cost < zero_cost


# =========================================================================
# Optimisation with mock engine
# =========================================================================


class TestOptimizeSwingWithMockEngine:
    """Test that a mock engine's .step() is called during optimisation."""

    def test_engine_step_called(self) -> None:
        config = SwingOptimizationConfig(
            n_joints=2, horizon_steps=5, max_iterations=2
        )
        engine = MagicMock()
        # Engine returns a plausible next state
        engine.step.side_effect = lambda state, u, dt: state + np.concatenate(
            [u * dt, u * dt]
        )

        b = SwingOptimizationBridge(config, engine=engine)
        x0 = np.zeros(4)
        result = b.optimize_swing(x0)

        assert engine.step.call_count > 0
        assert isinstance(result, SwingOptimizationResult)

    def test_engine_result_uses_engine_dynamics(self) -> None:
        """Verify that when an engine is provided, its dynamics are used."""
        config = SwingOptimizationConfig(
            n_joints=1, horizon_steps=3, max_iterations=1
        )

        # Engine that always returns a fixed state
        fixed_state = np.array([1.0, 99.0])
        engine = MagicMock()
        engine.step.return_value = fixed_state.copy()

        b = SwingOptimizationBridge(config, engine=engine)
        x0 = np.zeros(2)
        result = b.optimize_swing(x0)

        # Terminal velocity is the velocity part of fixed_state
        assert result.clubhead_velocity == pytest.approx(99.0)
