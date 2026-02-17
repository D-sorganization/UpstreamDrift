"""Runtime DbC tests for multi_muscle.py.

Tests that require()/ensure() contracts fire correctly at runtime for
MuscleGroup.add_muscle, MuscleGroup.compute_net_torque, and
AntagonistPair.compute_net_torque.
"""

from __future__ import annotations

import unittest

import numpy as np


class TestMuscleGroupAddMuscleContracts(unittest.TestCase):
    """Verify add_muscle preconditions."""

    def _make_group(self) -> object:
        from src.shared.python.biomechanics.multi_muscle import MuscleGroup

        return MuscleGroup("Test Flexors")

    def _make_muscle(self) -> object:
        from src.shared.python.biomechanics.hill_muscle import (
            HillMuscleModel,
            MuscleParameters,
        )

        return HillMuscleModel(MuscleParameters(F_max=1000.0, l_opt=0.15, l_slack=0.20))

    def test_valid_add_muscle(self) -> None:
        group = self._make_group()
        muscle = self._make_muscle()
        group.add_muscle("biceps", muscle, moment_arm=0.04)  # type: ignore[attr-defined]
        self.assertIn("biceps", group.muscles)  # type: ignore[attr-defined]

    def test_empty_name_raises(self) -> None:
        group = self._make_group()
        muscle = self._make_muscle()
        with self.assertRaises((ValueError, Exception)):
            group.add_muscle("", muscle, moment_arm=0.04)  # type: ignore[attr-defined]

    def test_zero_moment_arm_raises(self) -> None:
        group = self._make_group()
        muscle = self._make_muscle()
        with self.assertRaises((ValueError, Exception)):
            group.add_muscle("biceps", muscle, moment_arm=0.0)  # type: ignore[attr-defined]


class TestMuscleGroupTorqueContracts(unittest.TestCase):
    """Verify compute_net_torque pre/postconditions."""

    def _make_group_with_muscle(self) -> object:
        from src.shared.python.biomechanics.hill_muscle import (
            HillMuscleModel,
            MuscleParameters,
        )
        from src.shared.python.biomechanics.multi_muscle import MuscleGroup

        group = MuscleGroup("Test Flexors")
        params = MuscleParameters(F_max=1000.0, l_opt=0.15, l_slack=0.20)
        group.add_muscle("biceps", HillMuscleModel(params), moment_arm=0.04)
        return group

    def test_valid_torque_computation(self) -> None:
        group = self._make_group_with_muscle()
        activations = {"biceps": 0.5}
        states = {"biceps": (0.15, 0.0)}
        torque = group.compute_net_torque(activations, states)  # type: ignore[attr-defined]
        self.assertIsInstance(torque, float)
        self.assertTrue(np.isfinite(torque))

    def test_activation_above_one_raises(self) -> None:
        group = self._make_group_with_muscle()
        activations = {"biceps": 1.5}
        states = {"biceps": (0.15, 0.0)}
        with self.assertRaises((ValueError, Exception)):
            group.compute_net_torque(activations, states)  # type: ignore[attr-defined]

    def test_negative_activation_raises(self) -> None:
        group = self._make_group_with_muscle()
        activations = {"biceps": -0.1}
        states = {"biceps": (0.15, 0.0)}
        with self.assertRaises((ValueError, Exception)):
            group.compute_net_torque(activations, states)  # type: ignore[attr-defined]

    def test_zero_activation_ok(self) -> None:
        group = self._make_group_with_muscle()
        activations = {"biceps": 0.0}
        states = {"biceps": (0.15, 0.0)}
        torque = group.compute_net_torque(activations, states)  # type: ignore[attr-defined]
        self.assertIsInstance(torque, float)

    def test_full_activation_ok(self) -> None:
        group = self._make_group_with_muscle()
        activations = {"biceps": 1.0}
        states = {"biceps": (0.15, 0.0)}
        torque = group.compute_net_torque(activations, states)  # type: ignore[attr-defined]
        self.assertIsInstance(torque, float)

    def test_torque_is_finite(self) -> None:
        group = self._make_group_with_muscle()
        activations = {"biceps": 0.7}
        states = {"biceps": (0.15, 0.0)}
        torque = group.compute_net_torque(activations, states)  # type: ignore[attr-defined]
        self.assertTrue(np.isfinite(torque))


class TestAntagonistPairContracts(unittest.TestCase):
    """Verify AntagonistPair contracts."""

    def _make_pair(self) -> object:
        from src.shared.python.biomechanics.multi_muscle import (
            create_elbow_muscle_system,
        )

        return create_elbow_muscle_system()

    def test_valid_antagonist_torque(self) -> None:
        pair = self._make_pair()
        flex = {"biceps": 0.5, "brachialis": 0.5}
        ext = {"triceps": 0.2}
        states = {
            "biceps": (0.15, 0.0),
            "brachialis": (0.12, 0.0),
            "triceps": (0.18, 0.0),
        }
        torque = pair.compute_net_torque(flex, ext, states)  # type: ignore[attr-defined]
        self.assertIsInstance(torque, float)
        self.assertTrue(np.isfinite(torque))

    def test_invalid_agonist_activation_raises(self) -> None:
        pair = self._make_pair()
        flex = {"biceps": 2.0, "brachialis": 0.5}
        ext = {"triceps": 0.2}
        states = {
            "biceps": (0.15, 0.0),
            "brachialis": (0.12, 0.0),
            "triceps": (0.18, 0.0),
        }
        with self.assertRaises((ValueError, Exception)):
            pair.compute_net_torque(flex, ext, states)  # type: ignore[attr-defined]

    def test_co_contraction_reduces_net_torque(self) -> None:
        """Higher antagonist activation should reduce net (positive) torque."""
        pair = self._make_pair()
        states = {
            "biceps": (0.15, 0.0),
            "brachialis": (0.12, 0.0),
            "triceps": (0.18, 0.0),
        }
        flex = {"biceps": 0.8, "brachialis": 0.8}

        tau_low_antag = pair.compute_net_torque(  # type: ignore[attr-defined]
            flex, {"triceps": 0.1}, states
        )
        tau_high_antag = pair.compute_net_torque(  # type: ignore[attr-defined]
            flex, {"triceps": 0.8}, states
        )
        # Higher triceps co-contraction (negative moment arm) reduces net
        # (Triceps moment arm is negative so it opposes flexion)
        self.assertGreater(tau_low_antag, tau_high_antag)


if __name__ == "__main__":
    unittest.main()
