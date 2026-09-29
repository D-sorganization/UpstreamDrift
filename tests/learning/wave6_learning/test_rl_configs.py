"""Wave 6 coverage: src.learning.rl.configs."""

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
    def test_default_dim(self) -> None:
        c = ObservationConfig()
        # joint_pos + joint_vel = 2 * n_joints
        assert c.get_obs_dim(7) == 14

    def test_all_flags(self) -> None:
        c = ObservationConfig(
            include_joint_pos=True,
            include_joint_vel=True,
            include_joint_torque=True,
            include_ee_pos=True,
            include_ee_vel=True,
            include_imu=True,
            history_length=2,
        )
        # 7 + 7 + 7 + 2*3 + 2*6 + 6 = 45, *2 = 90
        assert c.get_obs_dim(7, n_ee=2) == 90

    def test_none_joints_raises(self) -> None:
        c = ObservationConfig()
        with pytest.raises(ValueError):
            c.get_obs_dim(None)  # type: ignore[arg-type]


class TestActionConfig:
    def test_process_action_clip_scale(self) -> None:
        c = ActionConfig(mode=ActionMode.TORQUE, action_scale=2.0, action_clip=0.5)
        result = c.process_action(np.array([2.0, -2.0, 0.1]), None)
        # clipped to [-0.5, 0.5] then scaled by 2
        np.testing.assert_allclose(result, [1.0, -1.0, 0.2])

    def test_process_action_smoothing(self) -> None:
        c = ActionConfig(smoothing_alpha=0.5, action_scale=1.0, action_clip=10.0)
        prev = np.array([1.0])
        cur = np.array([3.0])
        out = c.process_action(cur, prev)
        # 0.5*1 + 0.5*3 = 2
        assert out[0] == pytest.approx(2.0)

    def test_process_action_none_raises(self) -> None:
        with pytest.raises(ValueError):
            ActionConfig().process_action(None, None)  # type: ignore[arg-type]

    def test_action_mode_values(self) -> None:
        assert ActionMode.TORQUE.value == "torque"
        assert ActionMode.IMPEDANCE.value == "impedance"


class TestRewardConfig:
    def test_energy_penalty(self) -> None:
        c = RewardConfig(energy_penalty_weight=0.5)
        out = c.compute_energy_penalty(np.array([1.0, 2.0]))
        assert out == pytest.approx(0.5 * 5.0)

    def test_smoothness_penalty(self) -> None:
        c = RewardConfig(smoothness_penalty_weight=1.0)
        out = c.compute_smoothness_penalty(np.array([1.0, 1.0]), np.array([0.0, 0.0]))
        assert out == pytest.approx(2.0)

    def test_smoothness_no_prev(self) -> None:
        c = RewardConfig()
        assert c.compute_smoothness_penalty(np.array([1.0]), None) == 0.0

    def test_smoothness_none_action_raises(self) -> None:
        with pytest.raises(ValueError):
            RewardConfig().compute_smoothness_penalty(None, None)  # type: ignore[arg-type]


class TestTaskConfig:
    def test_default_target_velocity(self) -> None:
        c = TaskConfig()
        np.testing.assert_array_equal(c.target_velocity, [1.0, 0.0, 0.0])

    def test_is_success(self) -> None:
        c = TaskConfig(success_threshold=0.1)
        assert c.is_success(0.05)
        assert not c.is_success(0.5)

    def test_task_type_values(self) -> None:
        assert TaskType.LOCOMOTION.value == "locomotion"
        assert TaskType.MANIPULATION.value == "manipulation"
