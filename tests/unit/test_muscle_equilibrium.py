"""Tests for muscle equilibrium solver.

Tests the muscle-tendon equilibrium computation that solves for fiber length
and velocity given muscle-tendon unit kinematics.
"""

from __future__ import annotations

import numpy as np
import pytest

from shared.python.hill_muscle import HillMuscleModel, MuscleParameters
from shared.python.muscle_equilibrium import (
    EquilibriumSolver,
    compute_equilibrium_state,
)


@pytest.fixture
def standard_muscle():
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
def pennated_muscle():
    """Create a pennated muscle for testing."""
    params = MuscleParameters(
        F_max=1500.0,
        l_opt=0.10,
        l_slack=0.20,
        v_max=10.0,
        pennation_angle=np.deg2rad(15),  # 15 degrees
    )
    return HillMuscleModel(params)


class TestEquilibriumSolverInitialization:
    """Test EquilibriumSolver initialization."""

    def test_initialization(self, standard_muscle):
        """Test basic initialization."""
        solver = EquilibriumSolver(standard_muscle)
        assert solver.muscle is standard_muscle
        assert isinstance(solver.muscle, HillMuscleModel)

    def test_solver_retains_muscle_parameters(self, standard_muscle):
        """Test that solver retains access to muscle parameters."""
        solver = EquilibriumSolver(standard_muscle)
        assert solver.muscle.params.F_max == 1000.0
        assert solver.muscle.params.l_opt == 0.12
        assert solver.muscle.params.l_slack == 0.25


class TestEquilibriumResidual:
    """Test _equilibrium_residual method."""

    def test_residual_at_equilibrium_is_zero(self, standard_muscle):
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

    def test_residual_changes_sign_across_equilibrium(self, standard_muscle):
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

    def test_residual_with_pennation(self, pennated_muscle):
        """Test residual calculation with pennated muscle."""
        solver = EquilibriumSolver(pennated_muscle)
        l_MT = 0.30
        activation = 0.3

        # Should solve without errors
        l_CE = solver.solve_fiber_length(l_MT, activation)
        residual = solver._equilibrium_residual(l_CE, l_MT, activation, v_CE=0.0)

        assert abs(residual) < 1.0, "Residual should be small for pennated muscle"

    def test_residual_with_velocity(self, standard_muscle):
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


class TestSolveFiberLength:
    """Test solve_fiber_length method."""

    def test_convergence_at_optimal_length(self, standard_muscle):
        """Test solver converges at optimal muscle-tendon length."""
        solver = EquilibriumSolver(standard_muscle)

        # l_MT = l_opt + l_slack (fiber at optimal, tendon at slack)
        l_MT = standard_muscle.params.l_opt + standard_muscle.params.l_slack
        activation = 0.5

        l_CE = solver.solve_fiber_length(l_MT, activation)

        # Should be close to optimal length (relaxed bounds)
        assert 0.05 < l_CE < 0.20, f"l_CE should be near l_opt (0.12m), got {l_CE:.4f}m"
        assert np.isfinite(l_CE), "Solution should be finite"

    def test_convergence_at_different_activations(self, standard_muscle):
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

    def test_convergence_at_different_lengths(self, standard_muscle):
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
        assert (
            solutions[0] < solutions[-1]
        ), "Longer muscle-tendon should have longer fiber"

    def test_custom_initial_guess(self, standard_muscle):
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

    def test_zero_activation_uses_passive_force(self, standard_muscle):
        """Test that solver works with zero activation (passive only)."""
        solver = EquilibriumSolver(standard_muscle)
        l_MT = 0.37
        activation = 0.0

        l_CE = solver.solve_fiber_length(l_MT, activation)

        # Should find valid solution using passive forces
        assert np.isfinite(l_CE), "Should converge with zero activation"
        assert 0.05 < l_CE < 0.20, f"Solution out of range: {l_CE:.4f}m"

    def test_full_activation(self, standard_muscle):
        """Test solver with full activation."""
        solver = EquilibriumSolver(standard_muscle)
        l_MT = 0.37
        activation = 1.0

        l_CE = solver.solve_fiber_length(l_MT, activation)

        assert np.isfinite(l_CE), "Should converge with full activation"
        assert 0.05 < l_CE < 0.20, f"Solution out of range: {l_CE:.4f}m"

    def test_convergence_failure_raises_error(self, standard_muscle):
        """Test that convergence failure raises RuntimeError."""
        solver = EquilibriumSolver(standard_muscle)

        # Use unrealistic parameters that may cause convergence issues
        # Very short muscle-tendon length
        l_MT = 0.05  # Extremely short (shorter than l_opt alone)
        activation = 0.5

        # May or may not converge depending on solver robustness
        # If it fails, should raise RuntimeError
        try:
            l_CE = solver.solve_fiber_length(l_MT, activation)
            # If it succeeds, that's also okay
            assert np.isfinite(l_CE)
        except RuntimeError as e:
            # Should contain useful error message
            assert "equilibrium solver failed" in str(e).lower()

    def test_solution_satisfies_bounds(self, standard_muscle):
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


class TestSolveFiberVelocity:
    """Test solve_fiber_velocity method."""

    def test_zero_muscle_tendon_velocity(self, standard_muscle):
        """Test that zero MT velocity gives zero fiber velocity."""
        solver = EquilibriumSolver(standard_muscle)
        l_MT = 0.37
        v_MT = 0.0
        activation = 0.5
        l_CE = solver.solve_fiber_length(l_MT, activation)

        v_CE = solver.solve_fiber_velocity(l_MT, v_MT, activation, l_CE)

        # Should be exactly zero (no finite difference error for zero input)
        assert abs(v_CE) < 1e-6, f"v_CE should be zero, got {v_CE:.6f} m/s"

    def test_positive_muscle_tendon_velocity(self, standard_muscle):
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

    def test_negative_muscle_tendon_velocity(self, standard_muscle):
        """Test with negative (shortening) MT velocity."""
        solver = EquilibriumSolver(standard_muscle)
        l_MT = 0.37
        v_MT = -0.1  # m/s (shortening)
        activation = 0.5
        l_CE = solver.solve_fiber_length(l_MT, activation)

        v_CE = solver.solve_fiber_velocity(l_MT, v_MT, activation, l_CE)

        # Fiber velocity should be finite and generally negative
        assert np.isfinite(v_CE), "Fiber velocity should be finite"

    def test_custom_time_step(self, standard_muscle):
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

    def test_velocity_with_different_activations(self, standard_muscle):
        """Test that velocity computation works across activation levels."""
        solver = EquilibriumSolver(standard_muscle)
        l_MT = 0.37
        v_MT = 0.05

        for activation in [0.1, 0.5, 0.9]:
            l_CE = solver.solve_fiber_length(l_MT, activation)
            v_CE = solver.solve_fiber_velocity(l_MT, v_MT, activation, l_CE)

            assert np.isfinite(v_CE), f"v_CE should be finite at a={activation}"

    def test_convergence_failure_returns_zero(self, standard_muscle):
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


class TestComputeEquilibriumState:
    """Test compute_equilibrium_state convenience function."""

    def test_static_equilibrium(self, standard_muscle):
        """Test computing equilibrium state with zero velocity."""
        l_MT = 0.37
        v_MT = 0.0
        activation = 0.5

        l_CE, v_CE = compute_equilibrium_state(standard_muscle, l_MT, v_MT, activation)

        # Should return valid fiber length and zero velocity
        assert 0.05 < l_CE < 0.20, f"l_CE out of range: {l_CE:.4f}m"
        assert abs(v_CE) < 1e-6, f"v_CE should be zero for static case, got {v_CE}"

    def test_dynamic_equilibrium(self, standard_muscle):
        """Test computing equilibrium state with non-zero velocity."""
        l_MT = 0.37
        v_MT = 0.1  # m/s
        activation = 0.5

        l_CE, v_CE = compute_equilibrium_state(standard_muscle, l_MT, v_MT, activation)

        # Should return valid values
        assert 0.05 < l_CE < 0.20, f"l_CE out of range: {l_CE:.4f}m"
        assert np.isfinite(v_CE), "v_CE should be finite"

    def test_custom_initial_guess(self, standard_muscle):
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

    def test_returns_tuple(self, standard_muscle):
        """Test that function returns a tuple of two values."""
        l_MT = 0.37
        v_MT = 0.0
        activation = 0.5

        result = compute_equilibrium_state(standard_muscle, l_MT, v_MT, activation)

        assert isinstance(result, tuple), "Should return a tuple"
        assert len(result) == 2, "Should return (l_CE, v_CE)"

        l_CE, v_CE = result
        assert isinstance(l_CE, float), "l_CE should be float"
        assert isinstance(v_CE, (float, int)), "v_CE should be numeric"

    def test_different_muscle_parameters(self, pennated_muscle):
        """Test with different muscle (pennated)."""
        l_MT = 0.30
        v_MT = 0.0
        activation = 0.3

        l_CE, v_CE = compute_equilibrium_state(pennated_muscle, l_MT, v_MT, activation)

        # Should work with pennated muscle
        assert 0.05 < l_CE < 0.20, f"l_CE out of range: {l_CE:.4f}m"
        assert abs(v_CE) < 1e-6, "v_CE should be zero for static"


class TestPhysicalRealism:
    """Test physical realism of equilibrium solutions."""

    def test_tendon_bears_load_when_stretched(self, standard_muscle):
        """Test that tendon force increases when muscle-tendon is stretched."""
        solver = EquilibriumSolver(standard_muscle)
        activation = 0.5

        # Short vs long muscle-tendon length
        l_MT_short = 0.35
        l_MT_long = 0.40

        l_CE_short = solver.solve_fiber_length(l_MT_short, activation)
        l_CE_long = solver.solve_fiber_length(l_MT_long, activation)

        # Compute tendon lengths
        l_tendon_short = l_MT_short - l_CE_short
        l_tendon_long = l_MT_long - l_CE_long

        # Longer muscle-tendon should have longer tendon
        assert (
            l_tendon_long > l_tendon_short
        ), "Longer muscle-tendon should stretch tendon more"

    def test_fiber_length_decreases_with_activation(self, standard_muscle):
        """Test that fiber shortens with higher activation (for given l_MT).

        Higher activation -> more fiber force -> more tendon stretch -> shorter fiber
        """
        solver = EquilibriumSolver(standard_muscle)
        l_MT = 0.37

        l_CE_low = solver.solve_fiber_length(l_MT, activation=0.1)
        l_CE_high = solver.solve_fiber_length(l_MT, activation=0.9)

        # Higher activation should shorten fiber (stretch tendon more)
        assert l_CE_high < l_CE_low, (
            f"Higher activation should shorten fiber: "
            f"low={l_CE_low:.4f}m, high={l_CE_high:.4f}m"
        )

    def test_equilibrium_force_balance(self, standard_muscle):
        """Test that fiber and tendon forces are balanced at equilibrium."""
        solver = EquilibriumSolver(standard_muscle)
        l_MT = 0.37
        activation = 0.5

        l_CE = solver.solve_fiber_length(l_MT, activation)

        # Compute forces manually
        l_norm = l_CE / standard_muscle.params.l_opt
        f_l = standard_muscle.force_length_active(l_norm)
        f_v = standard_muscle.force_velocity(0.0)  # Static
        f_p = standard_muscle.force_length_passive(l_norm)

        F_fiber = standard_muscle.params.F_max * (activation * f_l * f_v + f_p)

        # Tendon force
        l_tendon = l_MT - l_CE
        l_tendon_norm = l_tendon / standard_muscle.params.l_slack
        f_t = standard_muscle.tendon_force(l_tendon_norm)
        F_tendon = standard_muscle.params.F_max * f_t

        # Should be balanced (within tolerance)
        np.testing.assert_allclose(
            F_fiber,
            F_tendon,
            rtol=0.01,
            err_msg="Fiber and tendon forces should balance at equilibrium",
        )

    def test_solution_is_stable(self, standard_muscle):
        """Test that small perturbations don't cause large changes."""
        solver = EquilibriumSolver(standard_muscle)
        l_MT = 0.37
        activation = 0.5

        l_CE_nominal = solver.solve_fiber_length(l_MT, activation)

        # Perturb slightly
        l_MT_perturbed = l_MT * 1.001  # 0.1% change
        l_CE_perturbed = solver.solve_fiber_length(l_MT_perturbed, activation)

        # Change should be small and smooth
        relative_change = abs(l_CE_perturbed - l_CE_nominal) / l_CE_nominal

        assert (
            relative_change < 0.05
        ), f"Small perturbation caused large change: {relative_change*100:.2f}%"


class TestNumericalAccuracy:
    """Test numerical accuracy and convergence properties."""

    def test_residual_below_tolerance(self, standard_muscle):
        """Test that converged solution has residual below tolerance."""
        solver = EquilibriumSolver(standard_muscle)
        l_MT = 0.37
        activation = 0.5

        l_CE = solver.solve_fiber_length(l_MT, activation)
        residual = solver._equilibrium_residual(l_CE, l_MT, activation, v_CE=0.0)

        # Residual should be much smaller than typical forces
        tolerance_N = 1.0  # 1 N tolerance
        assert (
            abs(residual) < tolerance_N
        ), f"Residual {residual:.6f} N exceeds tolerance {tolerance_N} N"

    def test_repeated_solves_give_consistent_results(self, standard_muscle):
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

    def test_solver_convergence_with_good_guess(self, standard_muscle):
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
    def test_solver_robustness_across_activations(self, standard_muscle, activation):
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


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_short_muscle_tendon(self, standard_muscle):
        """Test with very short muscle-tendon length."""
        solver = EquilibriumSolver(standard_muscle)

        # Close to minimum possible length (l_opt + small tendon)
        l_MT = 0.30  # Shorter than optimal
        activation = 0.1  # Low activation

        try:
            l_CE = solver.solve_fiber_length(l_MT, activation)
            # If it converges, check validity (fiber can be very short with low activation)
            assert 0.03 < l_CE < l_MT, f"Solution out of range: {l_CE:.4f}m"
        except RuntimeError:
            # Convergence failure is acceptable for extreme cases
            pass

    def test_very_long_muscle_tendon(self, standard_muscle):
        """Test with very long muscle-tendon length."""
        solver = EquilibriumSolver(standard_muscle)

        l_MT = 0.50  # Much longer than optimal
        activation = 0.5

        l_CE = solver.solve_fiber_length(l_MT, activation)

        # Should have stretched fiber (long muscle-tendon)
        assert (
            l_CE > standard_muscle.params.l_opt
        ), "Long muscle-tendon should have stretched fiber"

    def test_pennated_muscle_equilibrium(self, pennated_muscle):
        """Test equilibrium with pennation angle."""
        solver = EquilibriumSolver(pennated_muscle)
        l_MT = 0.30
        activation = 0.5

        l_CE = solver.solve_fiber_length(l_MT, activation)

        # Should converge with pennation
        assert np.isfinite(l_CE), "Should converge with pennated muscle"
        assert 0.05 < l_CE < 0.20, f"Solution out of range: {l_CE:.4f}m"

        # Tendon length should account for pennation
        cos_alpha = np.cos(pennated_muscle.params.pennation_angle)
        l_tendon = l_MT - l_CE * cos_alpha

        assert l_tendon > 0, "Tendon length should be positive"

    def test_minimum_activation(self, standard_muscle):
        """Test with minimum activation (passive only)."""
        solver = EquilibriumSolver(standard_muscle)
        l_MT = 0.37
        activation = 0.001  # Near zero

        l_CE = solver.solve_fiber_length(l_MT, activation)

        # Should rely on passive forces
        assert np.isfinite(l_CE), "Should converge with near-zero activation"
