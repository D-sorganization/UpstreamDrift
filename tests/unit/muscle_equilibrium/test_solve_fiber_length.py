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


class TestSolveFiberLength:
    """Test solve_fiber_length method."""

    def test_convergence_at_optimal_length(self, standard_muscle) -> None:
        """Test solver converges at optimal muscle-tendon length."""
        solver = EquilibriumSolver(standard_muscle)

        # l_MT = l_opt + l_slack (fiber at optimal, tendon at slack)
        l_MT = standard_muscle.params.l_opt + standard_muscle.params.l_slack
        activation = 0.5

        l_CE = solver.solve_fiber_length(l_MT, activation)

        # Should be close to optimal length (relaxed bounds)
        assert 0.05 < l_CE < 0.20, f"l_CE should be near l_opt (0.12m), got {l_CE:.4f}m"
        assert np.isfinite(l_CE), "Solution should be finite"

    def test_convergence_at_different_activations(self, standard_muscle) -> None:
        """Test solver converges across different activation levels."""
        solver = EquilibriumSolver(standard_muscle)
        l_MT = 0.37

        activations = [0.1, 0.3, 0.5, 0.7, 0.9]
        solutions = []

        for activation in activations:
            l_CE = solver.solve_fiber_length(l_MT, activation)
            solutions.append(l_CE)

            # All should converge to valid lengths
            assert 0.05 < l_CE < 0.20, f"l_CE out of range at a={activation}: {l_CE}"

        # Higher activation -> shorter fiber (more force, more tendon stretch)
        # This is a general trend
        assert solutions[0] > solutions[-1], (
            "Higher activation should generally lead to shorter fiber "
            f"(low a: {solutions[0]:.4f}m, high a: {solutions[-1]:.4f}m)"
        )

    def test_convergence_at_different_lengths(self, standard_muscle) -> None:
        """Test solver converges at different muscle-tendon lengths."""
        solver = EquilibriumSolver(standard_muscle)
        activation = 0.5

        # Test at various lengths
        l_MT_values = [0.35, 0.37, 0.40, 0.43]
        solutions = []

        for l_MT in l_MT_values:
            l_CE = solver.solve_fiber_length(l_MT, activation)
            solutions.append(l_CE)

            assert np.isfinite(l_CE), f"Solution should be finite at l_MT={l_MT}"

        # Longer muscle-tendon -> longer fiber (generally)
        assert solutions[0] < solutions[-1], (
            "Longer muscle-tendon should have longer fiber"
        )

    def test_custom_initial_guess(self, standard_muscle) -> None:
        """Test solver with custom initial guess."""
        solver = EquilibriumSolver(standard_muscle)
        l_MT = 0.37
        activation = 0.5

        # Solve with default guess
        l_CE_default = solver.solve_fiber_length(l_MT, activation)

        # Solve with custom guess
        l_CE_custom = solver.solve_fiber_length(l_MT, activation, initial_guess=0.10)

        # Should converge to same solution regardless of initial guess
        np.testing.assert_allclose(
            l_CE_default,
            l_CE_custom,
            rtol=1e-4,
            err_msg="Solution should be independent of initial guess",
        )

    def test_zero_activation_uses_passive_force(self, standard_muscle) -> None:
        """Test that solver works with zero activation (passive only)."""
        solver = EquilibriumSolver(standard_muscle)
        l_MT = 0.37
        activation = 0.0

        l_CE = solver.solve_fiber_length(l_MT, activation)

        # Should find valid solution using passive forces
        assert np.isfinite(l_CE), "Should converge with zero activation"
        assert 0.05 < l_CE < 0.20, f"Solution out of range: {l_CE:.4f}m"

    def test_full_activation(self, standard_muscle) -> None:
        """Test solver with full activation."""
        solver = EquilibriumSolver(standard_muscle)
        l_MT = 0.37
        activation = 1.0

        l_CE = solver.solve_fiber_length(l_MT, activation)

        assert np.isfinite(l_CE), "Should converge with full activation"
        assert 0.05 < l_CE < 0.20, f"Solution out of range: {l_CE:.4f}m"

    def test_convergence_failure_raises_error(self, standard_muscle) -> None:
        """Test that convergence failure raises RuntimeError or PostconditionError."""
        solver = EquilibriumSolver(standard_muscle)

        # Use unrealistic parameters that may cause convergence issues
        # Very short muscle-tendon length
        l_MT = 0.05  # Extremely short (shorter than l_opt alone)
        activation = 0.5

        # May or may not converge depending on solver robustness
        # If it fails, should raise RuntimeError or PostconditionError
        try:
            l_CE = solver.solve_fiber_length(l_MT, activation)
            # If it succeeds, that's also okay
            assert np.isfinite(l_CE), "Assertion failed: np.isfinite(l_CE)"
        except (RuntimeError, PostconditionError):
            # Convergence failure or postcondition violation are acceptable
            pass

    def test_solution_satisfies_bounds(self, standard_muscle) -> None:
        """Test that solution is physically reasonable."""
        solver = EquilibriumSolver(standard_muscle)
        l_MT = 0.37
        activation = 0.5

        l_CE = solver.solve_fiber_length(l_MT, activation)

        # Fiber length should be positive
        assert l_CE > 0, "Fiber length must be positive"

        # Fiber length should be less than total muscle-tendon length
        assert l_CE < l_MT, "Fiber length cannot exceed total muscle-tendon length"

        # Should be within reasonable range of optimal length
        assert (
            0.5 * standard_muscle.params.l_opt
            < l_CE
            < 2.0 * standard_muscle.params.l_opt
        ), f"Fiber length should be within 0.5-2.0x optimal length, got {l_CE:.4f}m"
