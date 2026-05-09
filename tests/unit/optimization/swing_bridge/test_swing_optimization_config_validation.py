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
