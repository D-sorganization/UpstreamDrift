"""Unit tests for shared/python/multi_muscle.py."""

import unittest
from unittest.mock import MagicMock

from shared.python.hill_muscle import HillMuscleModel, MuscleParameters
from shared.python.multi_muscle import AntagonistPair, MuscleGroup


class TestMultiMuscle(unittest.TestCase):
    """Test suite for multi-muscle coordination."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.mock_muscle = MagicMock(spec=HillMuscleModel)
        self.mock_muscle.params = MuscleParameters(F_max=100.0, l_opt=0.1, l_slack=0.2)
        # Setup compute_force to return a value based on input or fixed
        self.mock_muscle.compute_force.return_value = (
            50.0  # Constant force for simplicity
        )

    def test_muscle_group_add_muscle(self) -> None:
        """Test adding muscles to a group."""
        group = MuscleGroup("flexors")
        group.add_muscle("biceps", self.mock_muscle, moment_arm=0.05)

        self.assertIn("biceps", group.muscles)
        self.assertIn("biceps", group.attachments)
        self.assertEqual(group.attachments["biceps"].moment_arm, 0.05)

    def test_muscle_group_compute_net_torque(self) -> None:
        """Test computing net torque for a group."""
        group = MuscleGroup("flexors")
        group.add_muscle("muscle1", self.mock_muscle, moment_arm=0.05)
        group.add_muscle("muscle2", self.mock_muscle, moment_arm=0.03)

        activations = {"muscle1": 0.5, "muscle2": 0.8}
        states = {
            "muscle1": (0.1, 0.0),
            "muscle2": (0.1, 0.0),
        }

        # Mock compute_force logic if needed, but we set return_value=50.0
        # Torque = sum(r * F)
        # T1 = 0.05 * 50 = 2.5
        # T2 = 0.03 * 50 = 1.5
        # Total = 4.0

        torque = group.compute_net_torque(activations, states)
        self.assertAlmostEqual(torque, 4.0)

        # Verify compute_force called with correct activations
        calls = self.mock_muscle.compute_force.call_args_list
        self.assertEqual(len(calls), 2)
        # Check first call args (order depends on dict iteration, which is insertion order in py3.7+)
        # muscle1 was added first
        call1_state = calls[0][0][0]
        self.assertEqual(call1_state.activation, 0.5)
        call2_state = calls[1][0][0]
        self.assertEqual(call2_state.activation, 0.8)

    def test_antagonist_pair_compute_net_torque(self) -> None:
        """Test torque computation for antagonist pair."""
        # Agonist group
        agonist = MuscleGroup("agonist")
        agonist.add_muscle("ago1", self.mock_muscle, moment_arm=0.05)  # +2.5 Nm

        # Antagonist group
        antagonist = MuscleGroup("antagonist")
        antagonist.add_muscle("ant1", self.mock_muscle, moment_arm=-0.04)  # -2.0 Nm

        pair = AntagonistPair(agonist, antagonist)

        ago_act = {"ago1": 1.0}
        ant_act = {"ant1": 1.0}
        states = {"ago1": (0.1, 0.0), "ant1": (0.1, 0.0)}

        net_torque = pair.compute_net_torque(ago_act, ant_act, states)
        # Net = 2.5 + (-2.0) = 0.5
        self.assertAlmostEqual(net_torque, 0.5)

    def test_factory_function(self) -> None:
        """Test the elbow muscle system factory."""
        from shared.python.multi_muscle import create_elbow_muscle_system

        elbow = create_elbow_muscle_system()
        self.assertIsInstance(elbow, AntagonistPair)
        self.assertEqual(elbow.agonist.name, "Elbow Flexors")
        self.assertEqual(elbow.antagonist.name, "Elbow Extensors")
        self.assertIn("biceps", elbow.agonist.muscles)
        self.assertIn("triceps", elbow.antagonist.muscles)


if __name__ == "__main__":
    unittest.main()
