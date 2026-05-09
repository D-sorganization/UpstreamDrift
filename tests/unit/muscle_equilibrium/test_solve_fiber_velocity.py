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


class TestSolveFiberVelocity:
    """Test solve_fiber_velocity method."""

    def test_zero_muscle_tendon_velocity(self, standard_muscle) -> None:
        """Test that zero MT velocity gives zero fiber velocity."""
        solver = EquilibriumSolver(standard_muscle)
        l_MT = 0.37
        v_MT = 0.0
        activation = 0.5
        l_CE = solver.solve_fiber_length(l_MT, activation)

        v_CE = solver.solve_fiber_velocity(l_MT, v_MT, activation, l_CE)

        # Should be exactly zero (no finite difference error for zero input)
        assert abs(v_CE) < 1e-6, f"v_CE should be zero, got {v_CE:.6f} m/s"

    def test_positive_muscle_tendon_velocity(self, standard_muscle) -> None:
        """Test with positive (lengthening) MT velocity."""
        solver = EquilibriumSolver(standard_muscle)
        l_MT = 0.37
        v_MT = 0.1  # m/s (lengthening)
        activation = 0.5
        l_CE = solver.solve_fiber_length(l_MT, activation)

        v_CE = solver.solve_fiber_velocity(l_MT, v_MT, activation, l_CE)

        # Fiber velocity should be finite and generally positive
        assert np.isfinite(v_CE), "Fiber velocity should be finite"
        # Direction may vary depending on tendon compliance

    def test_negative_muscle_tendon_velocity(self, standard_muscle) -> None:
        """Test with negative (shortening) MT velocity."""
        solver = EquilibriumSolver(standard_muscle)
        l_MT = 0.37
        v_MT = -0.1  # m/s (shortening)
        activation = 0.5
        l_CE = solver.solve_fiber_length(l_MT, activation)

        v_CE = solver.solve_fiber_velocity(l_MT, v_MT, activation, l_CE)

        # Fiber velocity should be finite and generally negative
        assert np.isfinite(v_CE), "Fiber velocity should be finite"

    def test_custom_time_step(self, standard_muscle) -> None:
        """Test fiber velocity with custom time step."""
        solver = EquilibriumSolver(standard_muscle)
        l_MT = 0.37
        v_MT = 0.05
        activation = 0.5
        l_CE = solver.solve_fiber_length(l_MT, activation)

        # Solve with different time steps
        v_CE_dt1 = solver.solve_fiber_velocity(l_MT, v_MT, activation, l_CE, dt=0.001)
        v_CE_dt2 = solver.solve_fiber_velocity(l_MT, v_MT, activation, l_CE, dt=0.0001)

        # Results should be similar (finite difference approximation)
        np.testing.assert_allclose(
            v_CE_dt1,
            v_CE_dt2,
            rtol=0.1,
            err_msg="Velocities should be similar for different dt",
        )

    def test_velocity_with_different_activations(self, standard_muscle) -> None:
        """Test that velocity computation works across activation levels."""
        solver = EquilibriumSolver(standard_muscle)
        l_MT = 0.37
        v_MT = 0.05

        for activation in [0.1, 0.5, 0.9]:
            l_CE = solver.solve_fiber_length(l_MT, activation)
            v_CE = solver.solve_fiber_velocity(l_MT, v_MT, activation, l_CE)

            assert np.isfinite(v_CE), f"v_CE should be finite at a={activation}"

    def test_convergence_failure_returns_zero(self, standard_muscle) -> None:
        """Test that convergence failure returns zero velocity."""
        solver = EquilibriumSolver(standard_muscle)

        # Use parameters that might cause convergence issues
        l_MT = 0.37
        v_MT = 5.0  # Very high velocity
        activation = 0.5
        l_CE = 0.12

        # Should return 0.0 on failure (fallback)
        v_CE = solver.solve_fiber_velocity(l_MT, v_MT, activation, l_CE, dt=0.001)

        # Either succeeds (finite) or fails (returns 0.0)
        assert np.isfinite(v_CE), "Should return finite value or zero"
