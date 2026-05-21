"""Tests for src.robotics.core.types dataclasses and validation."""

from __future__ import annotations

import numpy as np
import pytest

from src.robotics.core.types import (
    ContactState,
    ContactType,
    FootstepTarget,
    ForceTorqueReading,
    IMUReading,
    RobotState,
    SolverResult,
    TaskDescriptor,
    TaskPriority,
)


class TestContactState:
    def _base(self, **overrides: object) -> ContactState:
        kw = {
            "contact_id": 1,
            "body_a": "a",
            "body_b": "b",
            "position": [0.0, 0.0, 0.0],
            "normal": [0.0, 0.0, 2.0],  # not unit -> should normalize
        }
        kw.update(overrides)
        return ContactState(**kw)  # type: ignore[arg-type]

    def test_normal_is_normalized(self) -> None:
        c = self._base()
        assert np.allclose(np.linalg.norm(c.normal), 1.0)
        assert np.allclose(c.normal, [0, 0, 1])

    def test_invalid_position_shape(self) -> None:
        with pytest.raises(ValueError, match="position"):
            self._base(position=[0.0, 0.0])

    def test_invalid_normal_shape(self) -> None:
        with pytest.raises(ValueError, match="normal"):
            self._base(normal=[1.0, 0.0])

    def test_invalid_friction_force_shape(self) -> None:
        with pytest.raises(ValueError, match="friction_force"):
            self._base(friction_force=np.zeros(2))

    def test_negative_penetration(self) -> None:
        with pytest.raises(ValueError, match="penetration"):
            self._base(penetration=-0.1)

    def test_negative_normal_force(self) -> None:
        with pytest.raises(ValueError, match="normal_force"):
            self._base(normal_force=-1.0)

    def test_negative_friction_coef(self) -> None:
        with pytest.raises(ValueError, match="friction_coefficient"):
            self._base(friction_coefficient=-0.1)

    def test_get_wrench_force_normal_plus_friction(self) -> None:
        c = self._base(
            normal_force=10.0,
            friction_force=np.array([1.0, 2.0, 0.0]),
        )
        w = c.get_wrench()
        assert w.shape == (6,)
        assert np.allclose(w[:3], np.array([1.0, 2.0, 10.0]))
        assert np.allclose(w[3:], np.zeros(3))

    def test_is_sliding_true(self) -> None:
        c = self._base(
            normal_force=10.0,
            friction_force=np.array([5.0, 0.0, 0.0]),
            friction_coefficient=0.5,
        )
        assert c.is_sliding()

    def test_is_sliding_false(self) -> None:
        c = self._base(
            normal_force=10.0,
            friction_force=np.array([0.1, 0.0, 0.0]),
            friction_coefficient=0.5,
        )
        assert not c.is_sliding()

    def test_with_force_immutable_update(self) -> None:
        c = self._base(normal_force=1.0)
        c2 = c.with_force(5.0, friction_force=np.array([1.0, 0.0, 0.0]))
        assert c.normal_force == 1.0
        assert c2.normal_force == 5.0
        assert np.allclose(c2.friction_force, [1.0, 0.0, 0.0])
        # uses default friction force when none passed
        c3 = c.with_force(2.0)
        assert c3.normal_force == 2.0
        assert np.allclose(c3.friction_force, c.friction_force)

    def test_zero_normal_not_normalized(self) -> None:
        c = self._base(normal=[0.0, 0.0, 1e-12])
        # tiny normal stays as-is
        assert c.normal.shape == (3,)

    def test_contact_type_enum_default(self) -> None:
        c = self._base()
        assert c.contact_type == ContactType.POINT


class TestTaskDescriptor:
    def test_construct(self) -> None:
        J = np.eye(3, 6)
        t = TaskDescriptor(
            name="x",
            priority=TaskPriority.PRIMARY,
            jacobian=J,
            target=np.zeros(3),
        )
        assert t.task_dim == 3
        assert t.config_dim == 6

    def test_jacobian_not_2d(self) -> None:
        with pytest.raises(ValueError, match="2D"):
            TaskDescriptor(
                name="x",
                priority=TaskPriority.PRIMARY,
                jacobian=np.zeros(3),
                target=np.zeros(3),
            )

    def test_target_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="doesn't match"):
            TaskDescriptor(
                name="x",
                priority=TaskPriority.PRIMARY,
                jacobian=np.eye(3, 6),
                target=np.zeros(2),
            )

    def test_with_weight(self) -> None:
        J = np.eye(3, 6)
        t = TaskDescriptor(
            name="x",
            priority=TaskPriority.PRIMARY,
            jacobian=J,
            target=np.zeros(3),
            weight=np.ones(3),
        )
        assert isinstance(t.weight, np.ndarray)


class TestFootstepTarget:
    def _base(self, **overrides: object) -> FootstepTarget:
        kw = {
            "position": [0.0, 0.0, 0.0],
            "orientation": [1.0, 0.0, 0.0, 0.0],
            "foot": "left",
            "timing": 0.5,
            "duration": 1.0,
        }
        kw.update(overrides)
        return FootstepTarget(**kw)  # type: ignore[arg-type]

    def test_valid(self) -> None:
        f = self._base()
        assert f.foot == "left"

    def test_bad_position(self) -> None:
        with pytest.raises(ValueError, match="position"):
            self._base(position=[0.0, 0.0])

    def test_bad_orientation(self) -> None:
        with pytest.raises(ValueError, match="orientation"):
            self._base(orientation=[1.0, 0.0, 0.0])

    def test_bad_foot(self) -> None:
        with pytest.raises(ValueError, match="foot"):
            self._base(foot="middle")

    def test_negative_timing(self) -> None:
        with pytest.raises(ValueError, match="timing"):
            self._base(timing=-0.1)

    def test_zero_duration(self) -> None:
        with pytest.raises(ValueError, match="duration"):
            self._base(duration=0.0)


class TestRobotState:
    def test_n_q_n_v(self) -> None:
        rs = RobotState(timestamp=0.0, q=np.zeros(7), v=np.zeros(6))
        assert rs.n_q == 7
        assert rs.n_v == 6


class TestSolverResult:
    def test_defaults(self) -> None:
        r = SolverResult(success=True, solution=np.zeros(3))
        assert r.cost == float("inf")
        assert r.iterations == 0


class TestForceTorqueReading:
    def test_force_torque_decomp(self) -> None:
        r = ForceTorqueReading(
            timestamp=0.0,
            sensor_id="ft",
            wrench=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        )
        assert np.allclose(r.force, [1, 2, 3])
        assert np.allclose(r.torque, [4, 5, 6])

    def test_bad_wrench(self) -> None:
        with pytest.raises(ValueError, match="wrench"):
            ForceTorqueReading(timestamp=0.0, sensor_id="x", wrench=np.zeros(5))


class TestIMUReading:
    def test_valid_no_orient(self) -> None:
        r = IMUReading(
            timestamp=0.0,
            sensor_id="imu",
            linear_acceleration=np.zeros(3),
            angular_velocity=np.zeros(3),
        )
        assert r.orientation is None

    def test_valid_with_orient(self) -> None:
        r = IMUReading(
            timestamp=0.0,
            sensor_id="imu",
            linear_acceleration=np.zeros(3),
            angular_velocity=np.zeros(3),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        assert r.orientation is not None

    def test_bad_la(self) -> None:
        with pytest.raises(ValueError, match="linear_acceleration"):
            IMUReading(
                timestamp=0.0,
                sensor_id="imu",
                linear_acceleration=np.zeros(2),
                angular_velocity=np.zeros(3),
            )

    def test_bad_av(self) -> None:
        with pytest.raises(ValueError, match="angular_velocity"):
            IMUReading(
                timestamp=0.0,
                sensor_id="imu",
                linear_acceleration=np.zeros(3),
                angular_velocity=np.zeros(4),
            )

    def test_bad_orient(self) -> None:
        with pytest.raises(ValueError, match="orientation"):
            IMUReading(
                timestamp=0.0,
                sensor_id="imu",
                linear_acceleration=np.zeros(3),
                angular_velocity=np.zeros(3),
                orientation=np.zeros(3),
            )
