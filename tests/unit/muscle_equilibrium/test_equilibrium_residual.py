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


class TestEquilibriumResidual:
    """Test _equilibrium_residual method."""

    def test_residual_at_equilibrium_is_zero(self, standard_muscle) -> None:
        """Test that residual is zero when fiber and tendon forces balance."""
        solver = EquilibriumSolver(standard_muscle)

        # First solve for equilibrium length
        l_MT = 0.37  # Total length
        activation = 0.5
        l_CE = solver.solve_fiber_length(l_MT, activation)

        # Now check residual at this solution
        residual = solver._equilibrium_residual(l_CE, l_MT, activation, v_CE=0.0)

        # Residual should be very close to zero
        assert (
            abs(residual) < 1.0
        ), f"Residual should be near zero, got {residual:.6f} N"

    def test_residual_changes_sign_across_equilibrium(self, standard_muscle) -> None:
        """Test that residual changes sign across equilibrium point."""
        solver = EquilibriumSolver(standard_muscle)
        l_MT = 0.37
        activation = 0.5

        # Find equilibrium
        l_CE_eq = solver.solve_fiber_length(l_MT, activation)

        # Test slightly shorter fiber (more tendon force)
        residual_short = solver._equilibrium_residual(
            l_CE_eq * 0.95, l_MT, activation, v_CE=0.0
        )

        # Test slightly longer fiber (less tendon force)
        residual_long = solver._equilibrium_residual(
            l_CE_eq * 1.05, l_MT, activation, v_CE=0.0
        )

        # Residuals should have opposite signs
        # Note: exact sign depends on force-length curves, but they should differ
        assert (
            residual_short * residual_long < 0
        ), "Residuals should have opposite signs across equilibrium"

    def test_residual_with_pennation(self, pennated_muscle) -> None:
        """Test residual calculation with pennated muscle."""
        solver = EquilibriumSolver(pennated_muscle)
        l_MT = 0.30
        activation = 0.3

        # Should solve without errors
        l_CE = solver.solve_fiber_length(l_MT, activation)
        residual = solver._equilibrium_residual(l_CE, l_MT, activation, v_CE=0.0)

        assert abs(residual) < 1.0, "Residual should be small for pennated muscle"

    def test_residual_with_velocity(self, standard_muscle) -> None:
        """Test residual with non-zero fiber velocity."""
        solver = EquilibriumSolver(standard_muscle)
        l_MT = 0.37
        activation = 0.5
        v_CE = 0.1  # m/s

        # Solve assuming static
        l_CE_static = solver.solve_fiber_length(l_MT, activation, v_CE=0.0)

        # Compute residual with velocity
        residual = solver._equilibrium_residual(
            l_CE_static, l_MT, activation, v_CE=v_CE
        )

        # With velocity, residual won't be exactly zero
        # (force-velocity relationship changes fiber force)
        assert np.isfinite(residual), "Residual should be finite with velocity"
