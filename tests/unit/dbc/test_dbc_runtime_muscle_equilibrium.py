"""Runtime DbC tests for muscle equilibrium solver contracts.

Tests the require()/ensure() contracts added to:
- EquilibriumSolver.solve_fiber_length (preconditions: l_MT>0, activation in [0,1];
  postconditions: result>0, finite)
- EquilibriumSolver.solve_fiber_velocity (preconditions: dt>0, l_CE>0;
  postcondition: finite)
- compute_equilibrium_state (preconditions + postconditions)
"""

from __future__ import annotations

import unittest

import numpy as np


def _make_muscle() -> object:
    """Create a standard HillMuscleModel for testing."""
    from src.shared.python.biomechanics.hill_muscle import (
        HillMuscleModel,
        MuscleParameters,
    )

    params = MuscleParameters(
        F_max=1000.0,
        l_opt=0.12,
        l_slack=0.25,
        v_max=1.2,
        pennation_angle=0.0,
    )
    return HillMuscleModel(params)


class TestSolveFiberLengthPreconditions(unittest.TestCase):
    """Verify require() preconditions on solve_fiber_length."""

    def test_valid_solve(self) -> None:
        from src.shared.python.biomechanics.muscle_equilibrium import (
            EquilibriumSolver,
        )

        muscle = _make_muscle()
        solver = EquilibriumSolver(muscle)
        l_CE = solver.solve_fiber_length(l_MT=0.37, activation=0.5)
        self.assertGreater(l_CE, 0)
        self.assertTrue(np.isfinite(l_CE))

    def test_negative_l_MT_raises(self) -> None:
        from src.shared.python.biomechanics.muscle_equilibrium import (
            EquilibriumSolver,
        )

        muscle = _make_muscle()
        solver = EquilibriumSolver(muscle)
        with self.assertRaises((ValueError, Exception)):
            solver.solve_fiber_length(l_MT=-0.1, activation=0.5)

    def test_zero_l_MT_raises(self) -> None:
        from src.shared.python.biomechanics.muscle_equilibrium import (
            EquilibriumSolver,
        )

        muscle = _make_muscle()
        solver = EquilibriumSolver(muscle)
        with self.assertRaises((ValueError, Exception)):
            solver.solve_fiber_length(l_MT=0.0, activation=0.5)

    def test_activation_above_one_raises(self) -> None:
        from src.shared.python.biomechanics.muscle_equilibrium import (
            EquilibriumSolver,
        )

        muscle = _make_muscle()
        solver = EquilibriumSolver(muscle)
        with self.assertRaises((ValueError, Exception)):
            solver.solve_fiber_length(l_MT=0.37, activation=1.5)

    def test_activation_below_zero_raises(self) -> None:
        from src.shared.python.biomechanics.muscle_equilibrium import (
            EquilibriumSolver,
        )

        muscle = _make_muscle()
        solver = EquilibriumSolver(muscle)
        with self.assertRaises((ValueError, Exception)):
            solver.solve_fiber_length(l_MT=0.37, activation=-0.1)

    def test_activation_at_boundaries(self) -> None:
        from src.shared.python.biomechanics.muscle_equilibrium import (
            EquilibriumSolver,
        )

        muscle = _make_muscle()
        solver = EquilibriumSolver(muscle)
        # activation=0.0 and activation=1.0 should both be valid
        l_CE_0 = solver.solve_fiber_length(l_MT=0.37, activation=0.0)
        self.assertGreater(l_CE_0, 0)
        l_CE_1 = solver.solve_fiber_length(l_MT=0.37, activation=1.0)
        self.assertGreater(l_CE_1, 0)


class TestSolveFiberVelocityPreconditions(unittest.TestCase):
    """Verify require() preconditions on solve_fiber_velocity."""

    def test_valid_solve_velocity(self) -> None:
        from src.shared.python.biomechanics.muscle_equilibrium import (
            EquilibriumSolver,
        )

        muscle = _make_muscle()
        solver = EquilibriumSolver(muscle)
        l_CE = solver.solve_fiber_length(l_MT=0.37, activation=0.5)
        v_CE = solver.solve_fiber_velocity(
            l_MT=0.37, v_MT=0.01, activation=0.5, l_CE=l_CE, dt=0.001
        )
        self.assertTrue(np.isfinite(v_CE))

    def test_zero_dt_raises(self) -> None:
        from src.shared.python.biomechanics.muscle_equilibrium import (
            EquilibriumSolver,
        )

        muscle = _make_muscle()
        solver = EquilibriumSolver(muscle)
        with self.assertRaises((ValueError, Exception)):
            solver.solve_fiber_velocity(
                l_MT=0.37, v_MT=0.01, activation=0.5, l_CE=0.1, dt=0.0
            )

    def test_negative_dt_raises(self) -> None:
        from src.shared.python.biomechanics.muscle_equilibrium import (
            EquilibriumSolver,
        )

        muscle = _make_muscle()
        solver = EquilibriumSolver(muscle)
        with self.assertRaises((ValueError, Exception)):
            solver.solve_fiber_velocity(
                l_MT=0.37, v_MT=0.01, activation=0.5, l_CE=0.1, dt=-0.001
            )

    def test_zero_l_CE_raises(self) -> None:
        from src.shared.python.biomechanics.muscle_equilibrium import (
            EquilibriumSolver,
        )

        muscle = _make_muscle()
        solver = EquilibriumSolver(muscle)
        with self.assertRaises((ValueError, Exception)):
            solver.solve_fiber_velocity(
                l_MT=0.37, v_MT=0.01, activation=0.5, l_CE=0.0, dt=0.001
            )

    def test_negative_l_CE_raises(self) -> None:
        from src.shared.python.biomechanics.muscle_equilibrium import (
            EquilibriumSolver,
        )

        muscle = _make_muscle()
        solver = EquilibriumSolver(muscle)
        with self.assertRaises((ValueError, Exception)):
            solver.solve_fiber_velocity(
                l_MT=0.37, v_MT=0.01, activation=0.5, l_CE=-0.1, dt=0.001
            )


class TestComputeEquilibriumStatePreconditions(unittest.TestCase):
    """Verify require()/ensure() on compute_equilibrium_state."""

    def test_valid_equilibrium(self) -> None:
        from src.shared.python.biomechanics.muscle_equilibrium import (
            compute_equilibrium_state,
        )

        muscle = _make_muscle()
        l_CE, v_CE = compute_equilibrium_state(
            muscle, l_MT=0.37, v_MT=0.0, activation=0.5
        )
        self.assertGreater(l_CE, 0)
        self.assertTrue(np.isfinite(l_CE))
        self.assertTrue(np.isfinite(v_CE))
        self.assertEqual(v_CE, 0.0)  # static case

    def test_negative_l_MT_raises(self) -> None:
        from src.shared.python.biomechanics.muscle_equilibrium import (
            compute_equilibrium_state,
        )

        muscle = _make_muscle()
        with self.assertRaises((ValueError, Exception)):
            compute_equilibrium_state(muscle, l_MT=-0.1, v_MT=0.0, activation=0.5)

    def test_activation_out_of_range_raises(self) -> None:
        from src.shared.python.biomechanics.muscle_equilibrium import (
            compute_equilibrium_state,
        )

        muscle = _make_muscle()
        with self.assertRaises((ValueError, Exception)):
            compute_equilibrium_state(muscle, l_MT=0.37, v_MT=0.0, activation=2.0)

    def test_with_nonzero_velocity(self) -> None:
        from src.shared.python.biomechanics.muscle_equilibrium import (
            compute_equilibrium_state,
        )

        muscle = _make_muscle()
        l_CE, v_CE = compute_equilibrium_state(
            muscle, l_MT=0.37, v_MT=0.01, activation=0.5
        )
        self.assertGreater(l_CE, 0)
        self.assertTrue(np.isfinite(l_CE))
        self.assertTrue(np.isfinite(v_CE))


if __name__ == "__main__":
    unittest.main()
