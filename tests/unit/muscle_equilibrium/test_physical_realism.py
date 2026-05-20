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


class TestPhysicalRealism:
    """Test physical realism of equilibrium solutions."""

    def test_tendon_bears_load_when_stretched(self, standard_muscle) -> None:
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
        assert l_tendon_long > l_tendon_short, (
            "Longer muscle-tendon should stretch tendon more"
        )

    def test_fiber_length_decreases_with_activation(self, standard_muscle) -> None:
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

    def test_equilibrium_force_balance(self, standard_muscle) -> None:
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

    def test_solution_is_stable(self, standard_muscle) -> None:
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

        assert relative_change < 0.05, (
            f"Small perturbation caused large change: {relative_change * 100:.2f}%"
        )
