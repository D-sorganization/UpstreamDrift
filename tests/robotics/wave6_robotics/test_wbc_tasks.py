"""Tests for whole_body/task module: Task and task factories."""

from __future__ import annotations

import numpy as np
import pytest

from src.robotics.control.whole_body.task import (
    Task,
    TaskGains,
    TaskType,
    create_com_task,
    create_contact_constraint,
    create_ee_task,
    create_joint_limit_task,
    create_posture_task,
)


def test_task_basic() -> None:
    J = np.eye(3, 6)
    t = Task(
        name="x",
        task_type=TaskType.SOFT,
        priority=2,
        jacobian=J,
        target=np.zeros(3),
    )
    assert t.task_dim == 3
    assert t.config_dim == 6
    W = t.get_weight_matrix()
    assert np.allclose(W, np.eye(3))


def test_task_with_weight_matrix() -> None:
    t = Task(
        name="x",
        task_type=TaskType.SOFT,
        priority=2,
        jacobian=np.eye(3, 6),
        target=np.zeros(3),
        weight=np.array([1.0, 2.0, 3.0]),
    )
    W = t.get_weight_matrix()
    assert np.allclose(np.diag(W), [1.0, 2.0, 3.0])


def test_task_invalid_jacobian_ndim() -> None:
    with pytest.raises(ValueError, match="2D"):
        Task(
            name="x",
            task_type=TaskType.SOFT,
            priority=2,
            jacobian=np.zeros(3),
            target=np.zeros(3),
        )


def test_task_invalid_target_shape() -> None:
    with pytest.raises(ValueError, match="doesn't match"):
        Task(
            name="x",
            task_type=TaskType.SOFT,
            priority=2,
            jacobian=np.eye(3, 6),
            target=np.zeros(2),
        )


def test_task_invalid_weight_shape() -> None:
    with pytest.raises(ValueError, match="doesn't match"):
        Task(
            name="x",
            task_type=TaskType.SOFT,
            priority=2,
            jacobian=np.eye(3, 6),
            target=np.zeros(3),
            weight=np.zeros(5),
        )


def test_task_inequality_requires_bound() -> None:
    with pytest.raises(ValueError, match="at least one bound"):
        Task(
            name="x",
            task_type=TaskType.INEQUALITY,
            priority=2,
            jacobian=np.eye(3, 6),
            target=np.zeros(3),
        )


def test_task_nonfinite_jacobian() -> None:
    J = np.eye(3, 6)
    J[0, 0] = np.nan
    with pytest.raises(ValueError, match="Jacobian"):
        Task(
            name="x",
            task_type=TaskType.SOFT,
            priority=2,
            jacobian=J,
            target=np.zeros(3),
        )


def test_task_nonfinite_target() -> None:
    with pytest.raises(ValueError, match="Target"):
        Task(
            name="x",
            task_type=TaskType.SOFT,
            priority=2,
            jacobian=np.eye(3, 6),
            target=np.array([np.nan, 0.0, 0.0]),
        )


def test_compute_error_feedback() -> None:
    t = Task(
        name="x",
        task_type=TaskType.SOFT,
        priority=2,
        jacobian=np.eye(3, 6),
        target=np.zeros(3),
        gain_p=10.0,
        gain_d=2.0,
    )
    out = t.compute_error_feedback(np.ones(3), np.ones(3))
    assert np.allclose(out, 10.0 + 2.0)


def test_create_com_task_defaults() -> None:
    t = create_com_task(
        jacobian_com=np.zeros((3, 6)),
        com_current=np.zeros(3),
        com_target=np.array([0.1, 0.0, 0.0]),
        com_velocity=np.zeros(3),
    )
    assert t.name == "com_tracking"
    assert t.task_dim == 3


def test_create_com_task_with_gains_and_vel_target() -> None:
    g = TaskGains(weight=0.5, priority=2, gain_p=50.0, gain_d=10.0)
    t = create_com_task(
        jacobian_com=np.zeros((3, 6)),
        com_current=np.zeros(3),
        com_target=np.zeros(3),
        com_velocity=np.zeros(3),
        com_velocity_target=np.array([0.1, 0.0, 0.0]),
        gains=g,
    )
    assert t.gain_p == 50.0


def test_create_posture_task_defaults() -> None:
    t = create_posture_task(
        n_v=6,
        q_current=np.zeros(6),
        q_target=np.ones(6),
        v_current=np.zeros(6),
    )
    assert t.name == "posture"
    assert t.task_dim == 6


def test_create_posture_task_with_mask() -> None:
    mask = np.array([True, False, True, False, True, False])
    t = create_posture_task(
        n_v=6,
        q_current=np.zeros(6),
        q_target=np.ones(6),
        v_current=np.zeros(6),
        mask=mask,
    )
    assert t.task_dim == 3


def test_create_posture_task_q_larger_than_v() -> None:
    # n_q (7) > n_v (6) – common with quaternion roots
    t = create_posture_task(
        n_v=6,
        q_current=np.zeros(7),
        q_target=np.ones(7),
        v_current=np.zeros(6),
    )
    assert t.task_dim == 6


def test_create_ee_task_full_pose() -> None:
    t = create_ee_task(
        jacobian_ee=np.zeros((6, 6)),
        ee_current=np.zeros(6),
        ee_target=np.zeros(6),
        ee_velocity=np.zeros(6),
    )
    assert t.task_dim == 6


def test_create_ee_task_position_only() -> None:
    t = create_ee_task(
        jacobian_ee=np.zeros((6, 6)),
        ee_current=np.zeros(6),
        ee_target=np.zeros(6),
        ee_velocity=np.zeros(6),
        position_only=True,
    )
    assert t.task_dim == 3


def test_create_ee_task_position_only_with_vel_target() -> None:
    t = create_ee_task(
        jacobian_ee=np.zeros((6, 6)),
        ee_current=np.zeros(6),
        ee_target=np.zeros(6),
        ee_velocity=np.zeros(6),
        ee_velocity_target=np.zeros(6),
        position_only=True,
    )
    assert t.task_dim == 3


def test_create_contact_constraint() -> None:
    t = create_contact_constraint(
        jacobian_contact=np.zeros((3, 6)),
        contact_velocity=np.zeros(3),
    )
    assert t.task_type == TaskType.EQUALITY
    assert t.priority == 0


def test_create_joint_limit_task_not_near_limits_returns_none() -> None:
    t = create_joint_limit_task(
        n_v=3,
        q_current=np.zeros(3),
        v_current=np.zeros(3),
        q_min=-np.ones(3),
        q_max=np.ones(3),
        margin=0.1,
    )
    assert t is None


def test_create_joint_limit_task_near_lower() -> None:
    t = create_joint_limit_task(
        n_v=3,
        q_current=np.array([-0.95, 0.0, 0.0]),
        v_current=np.zeros(3),
        q_min=-np.ones(3),
        q_max=np.ones(3),
        margin=0.1,
    )
    assert t is not None
    assert t.task_dim == 1


def test_create_joint_limit_task_near_upper() -> None:
    t = create_joint_limit_task(
        n_v=3,
        q_current=np.array([0.0, 0.95, 0.0]),
        v_current=np.zeros(3),
        q_min=-np.ones(3),
        q_max=np.ones(3),
        margin=0.1,
    )
    assert t is not None
    assert t.task_dim == 1
