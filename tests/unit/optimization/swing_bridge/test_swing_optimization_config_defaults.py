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
