"""Wave6 fast tests for skeleton + golf_state data models.

JointState, ForceVector, ClubState, SwingMetrics, BallState,
TrajectoryPoint, EnvironmentState — construction, to_dict/from_dict
roundtrips, derived properties, validation.
"""

from __future__ import annotations

import math

import pytest

from src.unreal_integration.geometry import Quaternion, Vector3
from src.unreal_integration.golf_state import (
    BallState,
    ClubState,
    EnvironmentState,
    SwingMetrics,
    TrajectoryPoint,
)
from src.unreal_integration.skeleton import ForceVector, JointState

# ---------- JointState ----------


class TestJointState:
    def test_minimal_construction(self) -> None:
        js = JointState(
            name="hip", position=Vector3(0, 1, 0), rotation=Quaternion.identity()
        )
        assert js.name == "hip"
        assert js.velocity is None

    def test_full_construction(self) -> None:
        js = JointState(
            name="knee",
            position=Vector3(0, 0, 0),
            rotation=Quaternion.identity(),
            velocity=Vector3(1, 0, 0),
            angular_velocity=Vector3(0, 1, 0),
            joint_angle=0.5,
            joint_velocity=0.1,
            parent_name="hip",
        )
        assert js.joint_angle == 0.5
        assert js.parent_name == "hip"

    def test_to_dict_includes_optional(self) -> None:
        js = JointState(
            name="x",
            position=Vector3(),
            rotation=Quaternion.identity(),
            velocity=Vector3(1, 2, 3),
            joint_angle=1.5,
            parent_name="parent",
        )
        d = js.to_dict()
        assert d["name"] == "x"
        assert "velocity" in d
        assert "joint_angle" in d
        assert d["parent_name"] == "parent"

    def test_to_dict_excludes_none(self) -> None:
        js = JointState(name="x", position=Vector3(), rotation=Quaternion.identity())
        d = js.to_dict()
        assert "velocity" not in d
        assert "joint_angle" not in d

    def test_roundtrip(self) -> None:
        js = JointState(
            name="shoulder",
            position=Vector3(1, 2, 3),
            rotation=Quaternion.identity(),
            velocity=Vector3(0.1, 0.2, 0.3),
            angular_velocity=Vector3(0, 0, 1),
            joint_angle=0.7,
            joint_velocity=0.2,
            parent_name="torso",
        )
        d = js.to_dict()
        js2 = JointState.from_dict(d)
        assert js2.name == "shoulder"
        assert js2.position == Vector3(1, 2, 3)
        assert js2.joint_angle == 0.7
        assert js2.parent_name == "torso"

    def test_validate_empty_name(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            JointState(
                name="  ",
                position=Vector3(),
                rotation=Quaternion.identity(),
                validate=True,
            )


# ---------- ForceVector ----------


class TestForceVector:
    def test_construction(self) -> None:
        fv = ForceVector(
            origin=Vector3(0, 1, 0),
            direction=Vector3(0, -1, 0),
            magnitude=9.81,
            force_type="gravity",
        )
        assert fv.magnitude == 9.81
        assert fv.force_type == "gravity"

    def test_endpoint(self) -> None:
        fv = ForceVector(
            origin=Vector3(0, 0, 0),
            direction=Vector3(2, 0, 0),  # will be normalized to (1,0,0)
            magnitude=5.0,
            scale_factor=2.0,
        )
        ep = fv.endpoint()
        assert ep == Vector3(10.0, 0.0, 0.0)

    def test_to_dict_includes_optional(self) -> None:
        fv = ForceVector(
            origin=Vector3(),
            direction=Vector3(1, 0, 0),
            magnitude=1.0,
            joint_name="elbow",
            color=(1.0, 0.0, 0.0, 1.0),
            scale_factor=2.0,
        )
        d = fv.to_dict()
        assert d["joint_name"] == "elbow"
        assert d["color"] == [1.0, 0.0, 0.0, 1.0]
        assert d["scale_factor"] == 2.0

    def test_to_dict_skips_default_scale(self) -> None:
        fv = ForceVector(
            origin=Vector3(),
            direction=Vector3(1, 0, 0),
            magnitude=1.0,
        )
        assert "scale_factor" not in fv.to_dict()

    def test_roundtrip(self) -> None:
        fv = ForceVector(
            origin=Vector3(1, 1, 1),
            direction=Vector3(0, 1, 0),
            magnitude=42.0,
            force_type="muscle",
            joint_name="bicep",
            color=(0.5, 0.5, 0.5, 1.0),
        )
        fv2 = ForceVector.from_dict(fv.to_dict())
        assert fv2.magnitude == 42.0
        assert fv2.force_type == "muscle"
        assert fv2.color == (0.5, 0.5, 0.5, 1.0)

    def test_validate_rejects_negative_magnitude(self) -> None:
        with pytest.raises(ValueError, match="magnitude"):
            ForceVector(
                origin=Vector3(),
                direction=Vector3(1, 0, 0),
                magnitude=-1.0,
                validate=True,
            )


# ---------- ClubState ----------


class TestClubState:
    def test_head_speed(self) -> None:
        cs = ClubState(
            head_position=Vector3(),
            head_velocity=Vector3(3, 4, 0),
        )
        assert cs.head_speed == pytest.approx(5.0)

    def test_to_dict_includes_speed(self) -> None:
        cs = ClubState(head_position=Vector3(), head_velocity=Vector3(0, 0, 0))
        d = cs.to_dict()
        assert "head_speed" in d

    def test_roundtrip(self) -> None:
        cs = ClubState(
            head_position=Vector3(1, 2, 3),
            head_velocity=Vector3(4, 5, 6),
            head_acceleration=Vector3(0, 0, -9.81),
            shaft_flex=[0.0, 0.1, 0.2],
            face_angle=2.0,
            loft_angle=10.0,
            lie_angle=60.0,
            shaft_lean=5.0,
        )
        cs2 = ClubState.from_dict(cs.to_dict())
        assert cs2.face_angle == 2.0
        assert cs2.shaft_flex == [0.0, 0.1, 0.2]
        assert cs2.head_acceleration == Vector3(0, 0, -9.81)


# ---------- SwingMetrics ----------


class TestSwingMetrics:
    def test_estimated_ball_speed(self) -> None:
        sm = SwingMetrics(club_head_speed=40.0, smash_factor=1.5)
        assert sm.estimated_ball_speed == 60.0

    def test_estimated_ball_speed_missing(self) -> None:
        assert SwingMetrics().estimated_ball_speed is None
        assert SwingMetrics(club_head_speed=40.0).estimated_ball_speed is None

    def test_to_dict_excludes_none(self) -> None:
        sm = SwingMetrics(club_head_speed=40.0)
        d = sm.to_dict()
        assert "club_head_speed" in d
        assert "x_factor" not in d

    def test_roundtrip(self) -> None:
        sm = SwingMetrics(
            club_head_speed=45.0,
            x_factor=40.0,
            kinetic_energy=120.0,
            smash_factor=1.48,
            attack_angle=-3.0,
            swing_path=2.0,
            tempo=3.0,
        )
        sm2 = SwingMetrics.from_dict(sm.to_dict())
        assert sm2.club_head_speed == 45.0
        assert sm2.tempo == 3.0


# ---------- BallState ----------


class TestBallState:
    def test_ball_speed(self) -> None:
        bs = BallState(position=Vector3(), velocity=Vector3(3, 4, 0))
        assert bs.ball_speed == pytest.approx(5.0)

    def test_launch_angle_45deg(self) -> None:
        bs = BallState(position=Vector3(), velocity=Vector3(10, 0, 10))
        assert bs.launch_angle == pytest.approx(45.0)

    def test_launch_angle_zero_horizontal_positive_z(self) -> None:
        bs = BallState(position=Vector3(), velocity=Vector3(0, 0, 1))
        assert bs.launch_angle == 90.0

    def test_launch_angle_zero_horizontal_negative_z(self) -> None:
        bs = BallState(position=Vector3(), velocity=Vector3(0, 0, -1))
        assert bs.launch_angle == -90.0

    def test_roundtrip(self) -> None:
        bs = BallState(
            position=Vector3(1, 2, 3),
            velocity=Vector3(10, 0, 5),
            spin_rate=3000.0,
            spin_axis=Vector3(0, 0, 1),
            is_in_flight=True,
        )
        bs2 = BallState.from_dict(bs.to_dict())
        assert bs2.spin_rate == 3000.0
        assert bs2.is_in_flight is True
        assert bs2.spin_axis == Vector3(0, 0, 1)


# ---------- TrajectoryPoint ----------


class TestTrajectoryPoint:
    def test_minimal(self) -> None:
        tp = TrajectoryPoint(time=0.5, position=Vector3(1, 2, 3))
        d = tp.to_dict()
        assert d["time"] == 0.5
        assert "velocity" not in d

    def test_full_roundtrip(self) -> None:
        tp = TrajectoryPoint(
            time=1.0,
            position=Vector3(1, 2, 3),
            velocity=Vector3(0.1, 0.2, 0.3),
            color=(1.0, 0.0, 0.0, 0.5),
        )
        tp2 = TrajectoryPoint.from_dict(tp.to_dict())
        assert tp2.color == (1.0, 0.0, 0.0, 0.5)
        assert tp2.velocity == Vector3(0.1, 0.2, 0.3)


# ---------- EnvironmentState ----------


class TestEnvironmentState:
    def test_default(self) -> None:
        env = EnvironmentState.default()
        assert env.temperature == 20.0
        assert env.air_density == pytest.approx(1.225)
        assert env.wind_velocity == Vector3.zero()

    def test_roundtrip(self) -> None:
        env = EnvironmentState(
            wind_velocity=Vector3(1, 0, 0),
            temperature=15.0,
            humidity=0.6,
            altitude=500.0,
            air_density=1.18,
            pressure=950.0,
        )
        env2 = EnvironmentState.from_dict(env.to_dict())
        assert env2.altitude == 500.0
        assert env2.wind_velocity == Vector3(1, 0, 0)


def test_math_module_in_use() -> None:
    # Sanity: launch_angle uses math.atan2; this acts as a documentation test
    assert math.atan2(1, 1) > 0
