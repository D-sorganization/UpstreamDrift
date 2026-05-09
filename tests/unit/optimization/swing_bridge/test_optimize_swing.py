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


# =========================================================================
# Full optimisation
# =========================================================================


class TestOptimizeSwing:
    """End-to-end optimisation tests."""

    def test_returns_result_type(self, small_bridge: SwingOptimizationBridge) -> None:
        x0 = np.zeros(small_bridge.state_dim)
        result = small_bridge.optimize_swing(x0)
        assert isinstance(result, SwingOptimizationResult)

    def test_result_torques_length(self, small_bridge: SwingOptimizationBridge) -> None:
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

    def test_total_cost_finite(self, small_bridge: SwingOptimizationBridge) -> None:
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
            * small_bridge.config.target_clubhead_velocity**2
        )
        assert result.total_cost < zero_cost


# =========================================================================
# Optimisation with mock engine
# =========================================================================
