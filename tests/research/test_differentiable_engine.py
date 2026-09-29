"""Smoke tests for research differentiable engine module."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from src.research.differentiable.engine import (
    AutodiffBackend,
    ContactDifferentiableEngine,
    DifferentiableEngine,
    OptimizationResult,
)


@pytest.fixture()
def mock_engine() -> MagicMock:
    """Create a minimal mock physics engine."""
    engine = MagicMock()
    engine.n_q = 3
    engine.n_v = 3
    engine.get_joint_positions.return_value = np.zeros(3)
    engine.get_joint_velocities.return_value = np.zeros(3)
    return engine


class TestAutodiffBackend:
    """Smoke tests for AutodiffBackend enum."""

    def test_numpy_backend(self) -> None:
        assert AutodiffBackend("numpy") == AutodiffBackend.NUMPY

    def test_invalid_backend(self) -> None:
        with pytest.raises(ValueError):
            AutodiffBackend("invalid")


class TestDifferentiableEngine:
    """Smoke tests for DifferentiableEngine."""

    def test_differentiable_engine_construction(self, mock_engine: MagicMock) -> None:
        de = DifferentiableEngine(mock_engine, backend="numpy")
        assert de._n_q == 3
        assert de._n_v == 3
        assert de._n_x == 6

    def test_simulate_trajectory(self, mock_engine: MagicMock) -> None:
        de = DifferentiableEngine(mock_engine, backend="numpy")
        initial = np.zeros(6)
        controls = np.zeros((5, 3))
        traj = de.simulate_trajectory(initial, controls, dt=0.01)
        assert traj.shape == (6, 6)

    def test_differentiable_engine_compute_jacobian(
        self, mock_engine: MagicMock
    ) -> None:
        de = DifferentiableEngine(mock_engine, backend="numpy")
        state = np.zeros(6)
        control = np.zeros(3)
        A, B = de.compute_jacobian(state, control, dt=0.01)
        assert A.shape == (6, 6)
        assert B.shape == (6, 3)


class TestContactDifferentiableEngine:
    """Smoke tests for ContactDifferentiableEngine."""

    def test_differentiable_engine_construction(self, mock_engine: MagicMock) -> None:
        cde = ContactDifferentiableEngine(
            mock_engine, contact_method="smoothed", smoothing_factor=0.01
        )
        assert cde.contact_method == "smoothed"
        assert cde.smoothing_factor == 0.01


class TestOptimizationResult:
    """Smoke tests for OptimizationResult dataclass."""

    def test_differentiable_engine_construction(self) -> None:
        result = OptimizationResult(
            success=True,
            optimal_states=np.zeros((5, 6)),
            optimal_controls=np.zeros((4, 3)),
            final_cost=0.1,
            iterations=10,
            gradient_norm=1e-7,
        )
        assert result.solver_status == "success"
        assert result.iterations == 10
