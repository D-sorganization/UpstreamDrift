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


# =========================================================================
# Optimisation with mock engine
# =========================================================================


class TestOptimizeSwingWithMockEngine:
    """Test that a mock engine's .step() is called during optimisation."""

    def test_engine_step_called(self) -> None:
        config = SwingOptimizationConfig(n_joints=2, horizon_steps=5, max_iterations=2)
        engine = MagicMock()
        # Engine returns a plausible next state
        engine.step.side_effect = lambda state, u, dt: (
            state + np.concatenate([u * dt, u * dt])
        )

        b = SwingOptimizationBridge(config, engine=engine)
        x0 = np.zeros(4)
        result = b.optimize_swing(x0)

        assert engine.step.call_count > 0
        assert isinstance(result, SwingOptimizationResult)

    def test_engine_result_uses_engine_dynamics(self) -> None:
        """Verify that when an engine is provided, its dynamics are used."""
        config = SwingOptimizationConfig(n_joints=1, horizon_steps=3, max_iterations=1)

        # Engine that always returns a fixed state
        fixed_state = np.array([1.0, 99.0])
        engine = MagicMock()
        engine.step.return_value = fixed_state.copy()

        b = SwingOptimizationBridge(config, engine=engine)
        x0 = np.zeros(2)
        result = b.optimize_swing(x0)

        # Terminal velocity is the velocity part of fixed_state
        assert result.clubhead_velocity == pytest.approx(99.0)
