"""Tests for muscle equilibrium solver.

Tests the muscle-tendon equilibrium computation that solves for fiber length
and velocity given muscle-tendon unit kinematics.
"""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.biomechanics.hill_muscle import HillMuscleModel, MuscleParameters
from src.shared.python.biomechanics.muscle_equilibrium import (
    EquilibriumSolver,
    compute_equilibrium_state,
)
from src.shared.python.core.contracts import PostconditionError


@pytest.fixture
def standard_muscle() -> HillMuscleModel:
    """Create a standard muscle for testing."""
    params = MuscleParameters(
        F_max=1000.0,  # N
        l_opt=0.12,  # m (12 cm)
        l_slack=0.25,  # m (25 cm)
        v_max=10.0,  # l_opt/s
        pennation_angle=0.0,  # rad (parallel fibers)
    )
    return HillMuscleModel(params)


@pytest.fixture
def pennated_muscle() -> HillMuscleModel:
    """Create a pennated muscle for testing."""
    params = MuscleParameters(
        F_max=1500.0,
        l_opt=0.10,
        l_slack=0.20,
        v_max=10.0,
        pennation_angle=np.deg2rad(15),  # 15 degrees
    )
    return HillMuscleModel(params)


class TestComputeEquilibriumState:
    """Test compute_equilibrium_state convenience function."""

    def test_static_equilibrium(self, standard_muscle) -> None:
        """Test computing equilibrium state with zero velocity."""
        l_MT = 0.37
        v_MT = 0.0
        activation = 0.5

        l_CE, v_CE = compute_equilibrium_state(standard_muscle, l_MT, v_MT, activation)

        # Should return valid fiber length and zero velocity
        assert 0.05 < l_CE < 0.20, f"l_CE out of range: {l_CE:.4f}m"
        assert abs(v_CE) < 1e-6, f"v_CE should be zero for static case, got {v_CE}"

    def test_dynamic_equilibrium(self, standard_muscle) -> None:
        """Test computing equilibrium state with non-zero velocity."""
        l_MT = 0.37
        v_MT = 0.1  # m/s
        activation = 0.5

        l_CE, v_CE = compute_equilibrium_state(standard_muscle, l_MT, v_MT, activation)

        # Should return valid values
        assert 0.05 < l_CE < 0.20, f"l_CE out of range: {l_CE:.4f}m"
        assert np.isfinite(v_CE), "v_CE should be finite"

    def test_custom_initial_guess(self, standard_muscle) -> None:
        """Test with custom initial fiber length guess."""
        l_MT = 0.37
        v_MT = 0.0
        activation = 0.5
        initial_l_CE = 0.10

        l_CE, v_CE = compute_equilibrium_state(
            standard_muscle, l_MT, v_MT, activation, initial_l_CE=initial_l_CE
        )

        # Should converge to same solution
        assert 0.05 < l_CE < 0.20, f"l_CE out of range: {l_CE:.4f}m"

    def test_muscle_equilibrium_returns_tuple(self, standard_muscle) -> None:
        """Test that function returns a tuple of two values."""
        l_MT = 0.37
        v_MT = 0.0
        activation = 0.5

        result = compute_equilibrium_state(standard_muscle, l_MT, v_MT, activation)

        assert isinstance(result, tuple), "Should return a tuple"
        assert len(result) == 2, "Should return (l_CE, v_CE)"

        l_CE, v_CE = result
        assert isinstance(l_CE, float), "l_CE should be float"
        assert isinstance(v_CE, float | int), "v_CE should be numeric"

    def test_different_muscle_parameters(self, pennated_muscle) -> None:
        """Test with different muscle (pennated)."""
        l_MT = 0.30
        v_MT = 0.0
        activation = 0.3

        l_CE, v_CE = compute_equilibrium_state(pennated_muscle, l_MT, v_MT, activation)

        # Should work with pennated muscle
        assert 0.05 < l_CE < 0.20, f"l_CE out of range: {l_CE:.4f}m"
        assert abs(v_CE) < 1e-6, "v_CE should be zero for static"
