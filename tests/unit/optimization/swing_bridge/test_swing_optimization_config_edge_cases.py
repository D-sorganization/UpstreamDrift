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
