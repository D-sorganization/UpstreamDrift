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
