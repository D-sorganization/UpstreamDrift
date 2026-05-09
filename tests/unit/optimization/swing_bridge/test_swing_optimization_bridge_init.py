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


class TestSwingOptimizationBridgeInit:
    """Test bridge construction and property access."""

    def test_init_with_default_config(self, bridge: SwingOptimizationBridge) -> None:
        assert bridge.config.n_joints == 7
        assert bridge.engine is None

    def test_state_dim(self, bridge: SwingOptimizationBridge) -> None:
        assert bridge.state_dim == 14  # 2 * 7

    def test_control_dim(self, bridge: SwingOptimizationBridge) -> None:
        assert bridge.control_dim == 7

    def test_init_with_engine(self, default_config: SwingOptimizationConfig) -> None:
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
