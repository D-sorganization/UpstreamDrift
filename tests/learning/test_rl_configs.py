"""Smoke tests for learning.rl.configs module."""

from __future__ import annotations

import numpy as np
import pytest

from src.learning.rl.configs import (
    ActionConfig,
    ActionMode,
    ObservationConfig,
    RewardConfig,
    TaskConfig,
    TaskType,
)


class TestObservationConfig:
    """Tests for ObservationConfig."""

    def test_default_obs_dim(self) -> None:
        config = ObservationConfig()
        dim = config.get_obs_dim(n_joints=7)
        # joint_pos (7) + joint_vel (7) = 14
        assert dim == 14

    def test_obs_dim_with_all_features(self) -> None:
        config = ObservationConfig(
            include_joint_pos=True,
            include_joint_vel=True,
            include_joint_torque=True,
            include_ee_pos=True,
            include_ee_vel=True,
            include_imu=True,
        )
        dim = config.get_obs_dim(n_joints=7, n_ee=2)
        # 7 + 7 + 7 + 2*3 + 2*6 + 6 = 45
        assert dim == 45

    def test_obs_dim_with_history(self) -> None:
        config = ObservationConfig(history_length=3)
        dim = config.get_obs_dim(n_joints=7)
        assert dim == 14 * 3


class TestActionConfig:
    """Tests for ActionConfig."""

    def test_default_action_mode(self) -> None:
        config = ActionConfig()
        assert config.mode == ActionMode.TORQUE

    def test_process_action_clip(self) -> None:
        config = ActionConfig(action_clip=0.5, action_scale=1.0)
        action = np.array([1.0, -1.0, 0.3])
        processed = config.process_action(action, None)
        np.testing.assert_array_less(processed, 0.5 + 1e-10)
        np.testing.assert_array_less(-0.5 - 1e-10, processed)

    def test_process_action_scale(self) -> None:
        config = ActionConfig(action_clip=10.0, action_scale=2.0)
        action = np.array([1.0, 0.5])
        processed = config.process_action(action, None)
        np.testing.assert_allclose(processed, [2.0, 1.0])

    def test_process_action_smoothing(self) -> None:
        config = ActionConfig(smoothing_alpha=0.5)
        prev = np.array([1.0, 0.0])
        current = np.array([0.0, 1.0])
        processed = config.process_action(current, prev)
        np.testing.assert_allclose(processed, [0.5, 0.5])


class TestRewardConfig:
    """Tests for RewardConfig."""

    def test_energy_penalty(self) -> None:
        config = RewardConfig(energy_penalty_weight=0.01)
        torques = np.array([1.0, 2.0, 3.0])
        penalty = config.compute_energy_penalty(torques)
        assert penalty == pytest.approx(0.14)

    def test_smoothness_penalty_no_prev(self) -> None:
        config = RewardConfig()
        assert config.compute_smoothness_penalty(np.zeros(3), None) == 0.0

    def test_smoothness_penalty_with_prev(self) -> None:
        config = RewardConfig(smoothness_penalty_weight=1.0)
        action = np.array([1.0, 0.0])
        prev = np.array([0.0, 0.0])
        penalty = config.compute_smoothness_penalty(action, prev)
        assert penalty == pytest.approx(1.0)


class TestTaskConfig:
    """Tests for TaskConfig."""

    def test_default_task_type(self) -> None:
        config = TaskConfig()
        assert config.task_type == TaskType.LOCOMOTION

    def test_is_success(self) -> None:
        config = TaskConfig(success_threshold=0.1)
        assert config.is_success(0.05) is True
        assert config.is_success(0.15) is False

    def test_max_episode_steps(self) -> None:
        config = TaskConfig(max_episode_steps=500)
        assert config.max_episode_steps == 500
