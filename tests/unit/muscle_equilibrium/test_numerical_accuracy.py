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
)


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


class TestNumericalAccuracy:
    """Test numerical accuracy and convergence properties."""

    def test_residual_below_tolerance(self, standard_muscle) -> None:
        """Test that converged solution has residual below tolerance."""
        solver = EquilibriumSolver(standard_muscle)
        l_MT = 0.37
        activation = 0.5

        l_CE = solver.solve_fiber_length(l_MT, activation)
        residual = solver._equilibrium_residual(l_CE, l_MT, activation, v_CE=0.0)

        # Residual should be much smaller than typical forces
        tolerance_N = 1.0  # 1 N tolerance
        assert abs(residual) < tolerance_N, (
            f"Residual {residual:.6f} N exceeds tolerance {tolerance_N} N"
        )

    def test_repeated_solves_give_consistent_results(self, standard_muscle) -> None:
        """Test that solving the same problem multiple times gives consistent results."""
        solver = EquilibriumSolver(standard_muscle)
        l_MT = 0.37
        activation = 0.5

        solutions = []
        for _ in range(5):
            l_CE = solver.solve_fiber_length(l_MT, activation)
            solutions.append(l_CE)

        # All solutions should be identical (deterministic solver)
        for sol in solutions[1:]:
            np.testing.assert_allclose(
                sol,
                solutions[0],
                rtol=1e-10,
                err_msg="Repeated solves should give identical results",
            )

    def test_solver_convergence_with_good_guess(self, standard_muscle) -> None:
        """Test that solver converges quickly with good initial guess."""
        solver = EquilibriumSolver(standard_muscle)
        l_MT = 0.37
        activation = 0.5

        # First solve to get answer
        l_CE_answer = solver.solve_fiber_length(l_MT, activation)

        # Use answer as initial guess
        l_CE_with_guess = solver.solve_fiber_length(
            l_MT, activation, initial_guess=l_CE_answer
        )

        # Should converge to exact same answer
        np.testing.assert_allclose(
            l_CE_with_guess,
            l_CE_answer,
            rtol=1e-10,
            err_msg="Perfect initial guess should give exact answer",
        )

    @pytest.mark.parametrize("activation", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_solver_robustness_across_activations(
        self, standard_muscle, activation
    ) -> None:
        """Test solver robustness across full activation range."""
        solver = EquilibriumSolver(standard_muscle)
        l_MT = 0.37

        l_CE = solver.solve_fiber_length(l_MT, activation)

        # Should converge for all activation levels
        assert np.isfinite(l_CE), f"Failed to converge at activation={activation}"
        assert 0.05 < l_CE < 0.20, f"Solution out of range at activation={activation}"

        # Check residual
        residual = solver._equilibrium_residual(l_CE, l_MT, activation, v_CE=0.0)
        assert abs(residual) < 1.0, f"Large residual at activation={activation}"
