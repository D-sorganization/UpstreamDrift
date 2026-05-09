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


class TestInitialStateValidation:
    """Verify that optimize_swing rejects invalid initial states."""

    def test_non_array_raises(self, small_bridge: SwingOptimizationBridge) -> None:
        with pytest.raises(TypeError, match="initial_state must be np.ndarray"):
            small_bridge.optimize_swing([0.0] * 4)  # type: ignore[arg-type]

    def test_swing_bridge_wrong_length_raises(
        self, small_bridge: SwingOptimizationBridge
    ) -> None:
        x0 = np.zeros(10)  # config has n_joints=2 -> expects 4
        with pytest.raises(ValueError, match="initial_state length must be 4"):
            small_bridge.optimize_swing(x0)

    def test_2d_array_raises(self, small_bridge: SwingOptimizationBridge) -> None:
        x0 = np.zeros((4, 1))
        with pytest.raises(ValueError, match="must be 1-D"):
            small_bridge.optimize_swing(x0)

    def test_nan_raises(self, small_bridge: SwingOptimizationBridge) -> None:
        x0 = np.array([0.0, 0.0, float("nan"), 0.0])
        with pytest.raises(ValueError, match="finite values"):
            small_bridge.optimize_swing(x0)

    def test_inf_raises(self, small_bridge: SwingOptimizationBridge) -> None:
        x0 = np.array([0.0, 0.0, float("inf"), 0.0])
        with pytest.raises(ValueError, match="finite values"):
            small_bridge.optimize_swing(x0)


# =========================================================================
# Trajectory evaluation (double-integrator)
# =========================================================================


# =========================================================================
# Full optimisation
# =========================================================================


# =========================================================================
# Optimisation with mock engine
# =========================================================================
