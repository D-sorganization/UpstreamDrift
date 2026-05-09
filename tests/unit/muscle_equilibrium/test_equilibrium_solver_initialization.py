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


class TestEquilibriumSolverInitialization:
    """Test EquilibriumSolver initialization."""

    def test_muscle_equilibrium_initialization(self, standard_muscle) -> None:
        """Test basic initialization."""
        solver = EquilibriumSolver(standard_muscle)
        assert solver.muscle is standard_muscle, (
            "Assertion failed: solver.muscle is standard_muscle"
        )
        assert isinstance(solver.muscle, HillMuscleModel), (
            "Assertion failed: isinstance(solver.muscle, HillMuscleModel)"
        )

    def test_solver_retains_muscle_parameters(self, standard_muscle) -> None:
        """Test that solver retains access to muscle parameters."""
        solver = EquilibriumSolver(standard_muscle)
        assert solver.muscle.params.F_max == 1000.0, (
            "Assertion failed: solver.muscle.params.F_max == 1000.0"
        )
        assert solver.muscle.params.l_opt == 0.12, (
            "Assertion failed: solver.muscle.params.l_opt == 0.12"
        )
        assert solver.muscle.params.l_slack == 0.25, (
            "Assertion failed: solver.muscle.params.l_slack == 0.25"
        )
