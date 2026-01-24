"""Tests for MyoSuite adapter module.

Tests the MyoSuite muscle-driven environment adapter for integration
with the Golf Modeling Suite.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.shared.python.multi_muscle import MuscleGroup
from src.shared.python.myosuite_adapter import MuscleDrivenEnv, train_muscle_policy


@pytest.fixture
def mock_muscle_system():
    """Create a mock muscle system."""
    system = MagicMock(spec=MuscleGroup)
    system.muscles = {"muscle1": MagicMock(), "muscle2": MagicMock()}
    system.compute_net_torque.return_value = 10.0
    return system


class TestMuscleDrivenEnv:
    """Tests for MuscleDrivenEnv class."""

    def test_initialization(self, mock_muscle_system):
        """Test environment initialization."""
        env = MuscleDrivenEnv(mock_muscle_system, task="tracking", dt=0.01)
        assert env.task == "tracking"
        assert env.dt == 0.01
        assert env.muscle_system == mock_muscle_system

    def test_reset(self, mock_muscle_system):
        """Test environment reset."""
        env = MuscleDrivenEnv(mock_muscle_system)
        obs = env.reset()

        assert isinstance(obs, np.ndarray)
        assert len(obs) == 2 + 2  # q, v, + 2 muscles
        assert env.step_count == 0

    @patch("src.shared.python.activation_dynamics.ActivationDynamics")
    def test_step(self, mock_dynamics_cls, mock_muscle_system):
        """Test environment step."""
        # Mock ActivationDynamics
        mock_dynamics = mock_dynamics_cls.return_value
        mock_dynamics.update.return_value = 0.5

        env = MuscleDrivenEnv(mock_muscle_system)
        env.reset()

        action = np.array([0.5, 0.5])
        obs, reward, done, info = env.step(action)

        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool | np.bool_)
        assert isinstance(info, dict)

        assert "tau_muscle" in info
        assert "q_error" in info

        # Check if compute_net_torque was called
        mock_muscle_system.compute_net_torque.assert_called()

    def test_action_to_excitations(self, mock_muscle_system):
        """Test action conversion."""
        env = MuscleDrivenEnv(mock_muscle_system)
        action = np.array([0.8, 0.2])
        excitations = env._action_to_excitations(action)

        assert excitations["muscle1"] == 0.8
        assert excitations["muscle2"] == 0.2

    def test_compute_reward_tracking(self, mock_muscle_system):
        """Test reward computation for tracking task."""
        env = MuscleDrivenEnv(mock_muscle_system, task="tracking")
        env.reset()
        env.q = 1.0
        env.target_q = 1.0

        reward = env._compute_reward()
        assert reward == 0.0  # Perfect match

        env.q = 2.0
        reward = env._compute_reward()
        assert reward == -1.0  # Error of 1.0

    def test_compute_reward_reach(self, mock_muscle_system):
        """Test reward computation for reach task."""
        env = MuscleDrivenEnv(mock_muscle_system, task="reach")
        env.reset()
        env.q = 1.0
        env.target_q = 2.0
        env.step_count = 10

        reward = env._compute_reward()
        # -dist - 0.01 * step_count
        expected = -1.0 - 0.01 * 10
        assert reward == pytest.approx(expected)


class TestTrainMusclePolicy:
    """Tests for train_muscle_policy function."""

    @patch("src.shared.python.myosuite_adapter.MYOSUITE_AVAILABLE", False)
    def test_train_policy_unavailable(self, mock_muscle_system):
        """Test training when MyoSuite is not available."""
        env = MuscleDrivenEnv(mock_muscle_system)
        model = train_muscle_policy(env)
        assert model is None

    @patch("src.shared.python.myosuite_adapter.MYOSUITE_AVAILABLE", True)
    def test_train_policy_available(self, mock_muscle_system):
        """Test training when MyoSuite is available."""
        # We need to mock stable_baselines3 import
        mock_sb3 = MagicMock()
        mock_sac = MagicMock()
        mock_sb3.SAC = mock_sac
        mock_model = mock_sac.return_value

        with patch.dict(sys.modules, {"stable_baselines3": mock_sb3}):
            env = MuscleDrivenEnv(mock_muscle_system)
            model = train_muscle_policy(env, total_timesteps=100)

            assert model == mock_model
            mock_sac.assert_called_once()
            mock_model.learn.assert_called_with(total_timesteps=100)
