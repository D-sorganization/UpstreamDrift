"""Smoke tests for learning.rl.base_env module."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from src.learning.rl.base_env import RoboticsGymEnv  # noqa: E402
from src.learning.rl.configs import (  # noqa: E402
    ActionConfig,
    ObservationConfig,
)


@pytest.fixture()
def mock_engine() -> MagicMock:
    """Create a minimal mock physics engine."""
    engine = MagicMock()
    engine.n_q = 7
    engine.n_v = 7
    return engine


class ConcreteEnv(RoboticsGymEnv):
    """Minimal concrete subclass for testing the base class."""

    def _apply_action(self, action):
        pass

    def _step_simulation(self):
        pass

    def _get_observation(self):
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def _compute_reward(self, action):
        return 0.0

    def _check_termination(self):
        return False

    def _reset_simulation(self, options):
        pass


class TestRoboticsGymEnvConstruction:
    """Test that the base environment can be constructed."""

    def test_rl_base_env_basic_construction(self, mock_engine: MagicMock) -> None:
        env = ConcreteEnv(engine=mock_engine)
        assert env.observation_space is not None
        assert env.action_space is not None

    def test_custom_configs(self, mock_engine: MagicMock) -> None:
        obs = ObservationConfig(include_joint_pos=True, include_joint_vel=False)
        act = ActionConfig(action_clip=2.0)
        env = ConcreteEnv(engine=mock_engine, obs_config=obs, action_config=act)
        assert env.action_space.high[0] == pytest.approx(2.0)

    def test_reset_returns_obs_and_info(self, mock_engine: MagicMock) -> None:
        env = ConcreteEnv(engine=mock_engine)
        obs, info = env.reset(seed=42)
        assert obs is not None
        assert isinstance(info, dict)

    def test_step_returns_tuple(self, mock_engine: MagicMock) -> None:
        env = ConcreteEnv(engine=mock_engine)
        env.reset(seed=42)
        action = np.zeros(7, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_close(self, mock_engine: MagicMock) -> None:
        env = ConcreteEnv(engine=mock_engine)
        env.close()
