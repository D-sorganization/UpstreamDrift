"""Smoke tests for learning.rl.humanoid_envs module."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from src.learning.rl.humanoid_envs import (  # noqa: E402
    HumanoidStandEnv,
    HumanoidWalkEnv,
)


@pytest.fixture()
def mock_engine() -> MagicMock:
    """Create a minimal mock physics engine for humanoid tests."""
    engine = MagicMock()
    engine.n_q = 7
    engine.n_v = 7
    engine.get_joint_positions.return_value = np.zeros(7)
    engine.get_joint_velocities.return_value = np.zeros(7)
    engine.get_base_velocity.return_value = np.array([1.0, 0.0, 0.0])
    engine.get_base_position.return_value = np.array([0.0, 0.0, 0.9])
    engine.get_base_orientation.return_value = np.array([1.0, 0.0, 0.0, 0.0])
    engine.get_joint_torques.return_value = np.zeros(7)
    engine.get_imu_data.return_value = np.zeros(6)
    return engine


class TestHumanoidWalkEnv:
    """Smoke tests for the walking environment."""

    def test_rl_humanoid_envs_construction(self, mock_engine: MagicMock) -> None:
        env = HumanoidWalkEnv(engine=mock_engine, target_velocity=1.5)
        assert env.task_config.target_velocity[0] == pytest.approx(1.5)

    def test_rl_humanoid_envs_reset(self, mock_engine: MagicMock) -> None:
        env = HumanoidWalkEnv(engine=mock_engine)
        obs, info = env.reset(seed=42)
        assert obs is not None
        assert "step_count" in info

    def test_rl_humanoid_envs_step(self, mock_engine: MagicMock) -> None:
        env = HumanoidWalkEnv(engine=mock_engine)
        env.reset(seed=42)
        action = np.zeros(7, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(reward, float)

    def test_termination_on_fall(self, mock_engine: MagicMock) -> None:
        mock_engine.get_base_position.return_value = np.array([0.0, 0.0, 0.1])
        env = HumanoidWalkEnv(engine=mock_engine)
        env.reset(seed=42)
        _, _, terminated, _, _ = env.step(np.zeros(7, dtype=np.float32))
        assert terminated is True


class TestHumanoidStandEnv:
    """Smoke tests for the standing/balance environment."""

    def test_rl_humanoid_envs_construction(self, mock_engine: MagicMock) -> None:
        env = HumanoidStandEnv(engine=mock_engine, perturbation_force=10.0)
        assert env._perturbation_force == 10.0

    def test_rl_humanoid_envs_reset(self, mock_engine: MagicMock) -> None:
        env = HumanoidStandEnv(engine=mock_engine)
        obs, info = env.reset(seed=42)
        assert obs is not None

    def test_rl_humanoid_envs_step(self, mock_engine: MagicMock) -> None:
        env = HumanoidStandEnv(engine=mock_engine)
        env.reset(seed=42)
        action = np.zeros(7, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(reward, float)
        assert terminated is False
