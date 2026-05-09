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


# =========================================================================
# SwingOptimizationConfig — validation errors
# =========================================================================


# =========================================================================
# SwingOptimizationConfig — edge cases that should succeed
# =========================================================================


# =========================================================================
# SwingOptimizationResult
# =========================================================================


# =========================================================================
# SwingOptimizationBridge — initialisation
# =========================================================================


# =========================================================================
# Cost matrix construction
# =========================================================================


# =========================================================================
# Initial state validation
# =========================================================================


# =========================================================================
# Trajectory evaluation (double-integrator)
# =========================================================================


class TestTrajectoryEvaluation:
    """Test the internal _evaluate_trajectory with double-integrator."""

    def test_trajectory_length(self, small_bridge: SwingOptimizationBridge) -> None:
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


# =========================================================================
# Optimisation with mock engine
# =========================================================================
