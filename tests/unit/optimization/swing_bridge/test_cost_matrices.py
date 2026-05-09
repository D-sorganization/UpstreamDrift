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

    def test_q_positive_semi_definite(self, bridge: SwingOptimizationBridge) -> None:
        Q, _ = bridge._build_cost_matrices(7)
        eigenvalues = np.linalg.eigvalsh(Q)
        assert np.all(eigenvalues >= -1e-12), (
            f"Q is not PSD: min eigenvalue = {eigenvalues.min()}"
        )

    def test_r_positive_semi_definite(self, bridge: SwingOptimizationBridge) -> None:
        _, R = bridge._build_cost_matrices(7)
        eigenvalues = np.linalg.eigvalsh(R)
        assert np.all(eigenvalues >= -1e-12), (
            f"R is not PSD: min eigenvalue = {eigenvalues.min()}"
        )

    def test_r_diagonal_values(self, bridge: SwingOptimizationBridge) -> None:
        _, R = bridge._build_cost_matrices(7)
        expected = 0.01 * np.eye(7)
        np.testing.assert_allclose(R, expected)

    def test_q_velocity_block_identity(self, bridge: SwingOptimizationBridge) -> None:
        Q, _ = bridge._build_cost_matrices(7)
        np.testing.assert_array_equal(Q[7:, 7:], np.eye(7))

    def test_q_position_block_zero(self, bridge: SwingOptimizationBridge) -> None:
        Q, _ = bridge._build_cost_matrices(7)
        np.testing.assert_array_equal(Q[:7, :7], np.zeros((7, 7)))

    def test_build_cost_matrices_n_zero_raises(
        self, bridge: SwingOptimizationBridge
    ) -> None:
        with pytest.raises(ValueError, match="n must be >= 1"):
            bridge._build_cost_matrices(0)

    def test_build_cost_matrices_n_one(self, bridge: SwingOptimizationBridge) -> None:
        Q, R = bridge._build_cost_matrices(1)
        assert Q.shape == (2, 2)
        assert R.shape == (1, 1)


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
