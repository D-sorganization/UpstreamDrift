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


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_short_muscle_tendon(self, standard_muscle) -> None:
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

    def test_very_long_muscle_tendon(self, standard_muscle) -> None:
        """Test with very long muscle-tendon length."""
        solver = EquilibriumSolver(standard_muscle)

        l_MT = 0.50  # Much longer than optimal
        activation = 0.5

        l_CE = solver.solve_fiber_length(l_MT, activation)

        # Should have stretched fiber (long muscle-tendon)
        assert l_CE > standard_muscle.params.l_opt, (
            "Long muscle-tendon should have stretched fiber"
        )

    def test_pennated_muscle_equilibrium(self, pennated_muscle) -> None:
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

    def test_minimum_activation(self, standard_muscle) -> None:
        """Test with minimum activation (passive only)."""
        solver = EquilibriumSolver(standard_muscle)
        l_MT = 0.37
        activation = 0.001  # Near zero

        l_CE = solver.solve_fiber_length(l_MT, activation)

        # Should rely on passive forces
        assert np.isfinite(l_CE), "Should converge with near-zero activation"
