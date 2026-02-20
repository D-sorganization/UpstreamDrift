"""Unit tests for Unreal Engine data models.

Following TDD principles - tests written first to define expected behavior.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from src.unreal_integration.data_models import (
    BallState,
    ClubState,
    EnvironmentState,
    ForceVector,
    JointState,
    Quaternion,
    SwingMetrics,
    TrajectoryPoint,
    UnrealDataFrame,
    Vector3,
)


class TestVector3:
    """Tests for Vector3 data model."""

    def test_create_from_values(self):
        """Test Vector3 creation from individual values."""
        v = Vector3(x=1.0, y=2.0, z=3.0)
        assert v.x == 1.0
        assert v.y == 2.0
        assert v.z == 3.0

    def test_create_from_numpy(self):
        """Test Vector3 creation from numpy array."""
        arr = np.array([1.0, 2.0, 3.0])
        v = Vector3.from_numpy(arr)
        assert v.x == 1.0
        assert v.y == 2.0
        assert v.z == 3.0

    def test_to_numpy(self):
        """Test Vector3 conversion to numpy array."""
        v = Vector3(x=1.0, y=2.0, z=3.0)
        arr = v.to_numpy()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3,)
        np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])

    def test_magnitude(self):
        """Test Vector3 magnitude calculation."""
        v = Vector3(x=3.0, y=4.0, z=0.0)
        assert v.magnitude == pytest.approx(5.0)

    def test_normalized(self):
        """Test Vector3 normalization."""
        v = Vector3(x=3.0, y=4.0, z=0.0)
        n = v.normalized()
        assert n.magnitude == pytest.approx(1.0)
        assert n.x == pytest.approx(0.6)
        assert n.y == pytest.approx(0.8)

    def test_to_dict(self):
        """Test Vector3 serialization to dict."""
        v = Vector3(x=1.0, y=2.0, z=3.0)
        d = v.to_dict()
        assert d == {"x": 1.0, "y": 2.0, "z": 3.0}

    def test_from_dict(self):
        """Test Vector3 deserialization from dict."""
        d = {"x": 1.0, "y": 2.0, "z": 3.0}
        v = Vector3.from_dict(d)
        assert v.x == 1.0
        assert v.y == 2.0
        assert v.z == 3.0

    @pytest.mark.parametrize(
        "op, v1_args, v2_args, expected",
        [
            ("add", (1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (5.0, 7.0, 9.0)),
            ("sub", (4.0, 5.0, 6.0), (1.0, 2.0, 3.0), (3.0, 3.0, 3.0)),
        ],
        ids=["addition", "subtraction"],
    )
    def test_vector_arithmetic(self, op, v1_args, v2_args, expected):
        """Test Vector3 addition and subtraction."""
        v1 = Vector3(x=v1_args[0], y=v1_args[1], z=v1_args[2])
        v2 = Vector3(x=v2_args[0], y=v2_args[1], z=v2_args[2])
        result = (v1 + v2) if op == "add" else (v1 - v2)
        assert result.x == expected[0]
        assert result.y == expected[1]
        assert result.z == expected[2]

    def test_scalar_multiplication(self):
        """Test Vector3 scalar multiplication."""
        v = Vector3(x=1.0, y=2.0, z=3.0)
        result = v * 2.0
        assert result.x == 2.0
        assert result.y == 4.0
        assert result.z == 6.0

    def test_dot_product(self):
        """Test Vector3 dot product."""
        v1 = Vector3(x=1.0, y=2.0, z=3.0)
        v2 = Vector3(x=4.0, y=5.0, z=6.0)
        assert v1.dot(v2) == pytest.approx(32.0)

    def test_cross_product(self):
        """Test Vector3 cross product."""
        v1 = Vector3(x=1.0, y=0.0, z=0.0)
        v2 = Vector3(x=0.0, y=1.0, z=0.0)
        result = v1.cross(v2)
        assert result.x == pytest.approx(0.0)
        assert result.y == pytest.approx(0.0)
        assert result.z == pytest.approx(1.0)

    def test_zero_vector(self):
        """Test Vector3.zero() factory method."""
        v = Vector3.zero()
        assert v.x == 0.0
        assert v.y == 0.0
        assert v.z == 0.0


class TestQuaternion:
    """Tests for Quaternion data model."""

    @pytest.mark.parametrize(
        "factory, expected_w",
        [
            (lambda: Quaternion(w=1.0, x=0.0, y=0.0, z=0.0), 1.0),
            (lambda: Quaternion.identity(), 1.0),
        ],
        ids=["from-values", "identity"],
    )
    def test_creation(self, factory, expected_w):
        """Test Quaternion creation methods."""
        q = factory()
        assert q.w == expected_w
        assert q.x == 0.0
        assert q.y == 0.0
        assert q.z == 0.0

    def test_from_euler(self):
        """Test Quaternion creation from Euler angles."""
        # 90 degrees around Z axis
        q = Quaternion.from_euler(roll=0, pitch=0, yaw=math.pi / 2)
        assert q.w == pytest.approx(math.cos(math.pi / 4))
        assert q.z == pytest.approx(math.sin(math.pi / 4))

    @pytest.mark.parametrize(
        "roll, pitch, yaw",
        [(0.1, 0.2, 0.3), (0.0, 0.0, 0.0), (0.5, -0.3, 0.8)],
        ids=["small-angles", "zero", "mixed"],
    )
    def test_euler_roundtrip(self, roll, pitch, yaw):
        """Test Quaternion Euler conversion roundtrip."""
        q = Quaternion.from_euler(roll=roll, pitch=pitch, yaw=yaw)
        r, p, y = q.to_euler()
        assert r == pytest.approx(roll, abs=1e-6)
        assert p == pytest.approx(pitch, abs=1e-6)
        assert y == pytest.approx(yaw, abs=1e-6)

    def test_magnitude(self):
        """Test Quaternion magnitude calculation."""
        q = Quaternion.identity()
        assert q.magnitude == pytest.approx(1.0)

    def test_normalized(self):
        """Test Quaternion normalization."""
        q = Quaternion(w=2.0, x=0.0, y=0.0, z=0.0)
        n = q.normalized()
        assert n.magnitude == pytest.approx(1.0)
        assert n.w == pytest.approx(1.0)

    def test_conjugate(self):
        """Test Quaternion conjugate."""
        q = Quaternion(w=1.0, x=2.0, y=3.0, z=4.0)
        c = q.conjugate()
        assert c.w == 1.0
        assert c.x == -2.0
        assert c.y == -3.0
        assert c.z == -4.0

    def test_to_dict(self):
        """Test Quaternion serialization to dict."""
        q = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
        d = q.to_dict()
        assert d == {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}


class TestJointState:
    """Tests for JointState data model."""

    def test_create_joint_state(self):
        """Test JointState creation."""
        js = JointState(
            name="shoulder_L",
            position=Vector3(x=0.1, y=1.4, z=0.2),
            rotation=Quaternion.identity(),
            velocity=Vector3.zero(),
            angular_velocity=Vector3.zero(),
        )
        assert js.name == "shoulder_L"
        assert js.position.x == 0.1

    def test_joint_state_with_angle(self):
        """Test JointState with joint angle."""
        js = JointState(
            name="elbow_L",
            position=Vector3.zero(),
            rotation=Quaternion.identity(),
            joint_angle=1.57,  # 90 degrees in radians
        )
        assert js.joint_angle == pytest.approx(1.57)

    def test_joint_state_to_dict(self):
        """Test JointState serialization."""
        js = JointState(
            name="shoulder_L",
            position=Vector3(x=0.1, y=1.4, z=0.2),
            rotation=Quaternion.identity(),
        )
        d = js.to_dict()
        assert d["name"] == "shoulder_L"
        assert "position" in d
        assert "rotation" in d

    def test_joint_state_from_dict(self):
        """Test JointState deserialization."""
        d = {
            "name": "shoulder_L",
            "position": {"x": 0.1, "y": 1.4, "z": 0.2},
            "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
        }
        js = JointState.from_dict(d)
        assert js.name == "shoulder_L"
        assert js.position.x == 0.1


class TestForceVector:
    """Tests for ForceVector data model."""

    def test_create_force_vector(self):
        """Test ForceVector creation."""
        fv = ForceVector(
            origin=Vector3(x=0.0, y=1.0, z=0.0),
            direction=Vector3(x=0.0, y=-1.0, z=0.0),
            magnitude=9.81,
            force_type="gravity",
        )
        assert fv.magnitude == 9.81
        assert fv.force_type == "gravity"

    def test_force_vector_endpoint(self):
        """Test ForceVector endpoint calculation."""
        fv = ForceVector(
            origin=Vector3(x=0.0, y=0.0, z=0.0),
            direction=Vector3(x=1.0, y=0.0, z=0.0),
            magnitude=5.0,
        )
        endpoint = fv.endpoint()
        assert endpoint.x == pytest.approx(5.0)
        assert endpoint.y == pytest.approx(0.0)
        assert endpoint.z == pytest.approx(0.0)

    def test_force_vector_to_dict(self):
        """Test ForceVector serialization."""
        fv = ForceVector(
            origin=Vector3(x=0.0, y=1.0, z=0.0),
            direction=Vector3(x=0.0, y=-1.0, z=0.0),
            magnitude=9.81,
            force_type="gravity",
        )
        d = fv.to_dict()
        assert d["magnitude"] == 9.81
        assert d["force_type"] == "gravity"

    def test_torque_vector(self):
        """Test ForceVector for torque representation."""
        tv = ForceVector(
            origin=Vector3(x=0.0, y=0.0, z=0.0),
            direction=Vector3(x=0.0, y=0.0, z=1.0),
            magnitude=10.5,
            force_type="torque",
            joint_name="shoulder_L",
        )
        assert tv.force_type == "torque"
        assert tv.joint_name == "shoulder_L"


class TestClubState:
    """Tests for ClubState data model."""

    def test_create_club_state(self):
        """Test ClubState creation."""
        cs = ClubState(
            head_position=Vector3(x=0.5, y=0.8, z=0.1),
            head_velocity=Vector3(x=25.0, y=10.0, z=5.0),
            head_acceleration=Vector3(x=100.0, y=50.0, z=20.0),
            shaft_flex=[0.01, 0.02, 0.015, 0.01, 0.005],
            face_angle=2.5,
            loft_angle=10.0,
        )
        assert cs.head_position.x == 0.5
        assert len(cs.shaft_flex) == 5
        assert cs.face_angle == 2.5

    def test_club_head_speed(self):
        """Test ClubState head speed calculation."""
        cs = ClubState(
            head_position=Vector3.zero(),
            head_velocity=Vector3(x=30.0, y=40.0, z=0.0),
        )
        assert cs.head_speed == pytest.approx(50.0)  # 3-4-5 triangle

    def test_club_state_to_dict(self):
        """Test ClubState serialization."""
        cs = ClubState(
            head_position=Vector3(x=0.5, y=0.8, z=0.1),
            head_velocity=Vector3(x=25.0, y=10.0, z=5.0),
        )
        d = cs.to_dict()
        assert "head_position" in d
        assert "head_velocity" in d
        assert "head_speed" in d


class TestSwingMetrics:
    """Tests for SwingMetrics data model."""

    def test_create_swing_metrics(self):
        """Test SwingMetrics creation."""
        sm = SwingMetrics(
            club_head_speed=45.2,
            x_factor=52.3,
            kinetic_energy=1250.5,
            smash_factor=1.48,
            attack_angle=-2.5,
            swing_path=3.0,
            face_to_path=-1.5,
        )
        assert sm.club_head_speed == 45.2
        assert sm.x_factor == 52.3

    def test_swing_metrics_calculated_fields(self):
        """Test SwingMetrics with calculated ball speed."""
        sm = SwingMetrics(
            club_head_speed=100.0,
            smash_factor=1.5,
        )
        assert sm.estimated_ball_speed == pytest.approx(150.0)

    def test_swing_metrics_to_dict(self):
        """Test SwingMetrics serialization."""
        sm = SwingMetrics(
            club_head_speed=45.2,
            x_factor=52.3,
        )
        d = sm.to_dict()
        assert d["club_head_speed"] == 45.2
        assert d["x_factor"] == 52.3


class TestBallState:
    """Tests for BallState data model."""

    def test_create_ball_state(self):
        """Test BallState creation."""
        bs = BallState(
            position=Vector3(x=0.0, y=0.0, z=0.05),
            velocity=Vector3(x=60.0, y=20.0, z=40.0),
            spin_rate=2500.0,
            spin_axis=Vector3(x=0.0, y=0.1, z=1.0),
        )
        assert bs.spin_rate == 2500.0
        assert bs.velocity.x == 60.0

    def test_ball_launch_angle(self):
        """Test BallState launch angle calculation."""
        bs = BallState(
            position=Vector3.zero(),
            velocity=Vector3(x=100.0, y=0.0, z=100.0),  # 45 degree launch
        )
        assert bs.launch_angle == pytest.approx(45.0, abs=0.1)


class TestTrajectoryPoint:
    """Tests for TrajectoryPoint data model."""

    def test_create_trajectory_point(self):
        """Test TrajectoryPoint creation."""
        tp = TrajectoryPoint(
            time=0.5,
            position=Vector3(x=10.0, y=0.0, z=5.0),
            velocity=Vector3(x=50.0, y=0.0, z=25.0),
        )
        assert tp.time == 0.5
        assert tp.position.x == 10.0

    def test_trajectory_point_color(self):
        """Test TrajectoryPoint with color."""
        tp = TrajectoryPoint(
            time=0.5,
            position=Vector3(x=10.0, y=0.0, z=5.0),
            color=(1.0, 0.0, 0.0, 1.0),  # Red with full alpha
        )
        assert tp.color == (1.0, 0.0, 0.0, 1.0)


class TestEnvironmentState:
    """Tests for EnvironmentState data model."""

    def test_create_environment_state(self):
        """Test EnvironmentState creation."""
        env = EnvironmentState(
            wind_velocity=Vector3(x=5.0, y=0.0, z=0.0),
            temperature=20.0,
            humidity=0.6,
            altitude=100.0,
            air_density=1.225,
        )
        assert env.wind_velocity.x == 5.0
        assert env.temperature == 20.0

    def test_default_environment(self):
        """Test EnvironmentState.default() factory method."""
        env = EnvironmentState.default()
        assert env.temperature == 20.0
        assert env.air_density == pytest.approx(1.225)


class TestUnrealDataFrame:
    """Tests for UnrealDataFrame data model."""

    def test_create_data_frame(self):
        """Test UnrealDataFrame creation."""
        frame = UnrealDataFrame(
            timestamp=0.0167,
            frame_number=1,
            joints={
                "shoulder_L": JointState(
                    name="shoulder_L",
                    position=Vector3(x=0.1, y=1.4, z=0.2),
                    rotation=Quaternion.identity(),
                ),
            },
            forces=[
                ForceVector(
                    origin=Vector3.zero(),
                    direction=Vector3(x=0.0, y=-1.0, z=0.0),
                    magnitude=9.81,
                    force_type="gravity",
                ),
            ],
        )
        assert frame.timestamp == 0.0167
        assert frame.frame_number == 1
        assert "shoulder_L" in frame.joints

    def test_data_frame_with_club(self):
        """Test UnrealDataFrame with club state."""
        frame = UnrealDataFrame(
            timestamp=0.5,
            frame_number=30,
            joints={},
            club=ClubState(
                head_position=Vector3(x=0.5, y=0.8, z=0.1),
                head_velocity=Vector3(x=25.0, y=10.0, z=5.0),
            ),
        )
        assert frame.club is not None
        assert frame.club.head_position.x == 0.5

    def test_data_frame_with_metrics(self):
        """Test UnrealDataFrame with swing metrics."""
        frame = UnrealDataFrame(
            timestamp=1.0,
            frame_number=60,
            joints={},
            metrics=SwingMetrics(
                club_head_speed=45.2,
                x_factor=52.3,
            ),
        )
        assert frame.metrics is not None
        assert frame.metrics.club_head_speed == 45.2

    def test_data_frame_to_json(self):
        """Test UnrealDataFrame JSON serialization."""
        frame = UnrealDataFrame(
            timestamp=0.0167,
            frame_number=1,
            joints={
                "shoulder_L": JointState(
                    name="shoulder_L",
                    position=Vector3(x=0.1, y=1.4, z=0.2),
                    rotation=Quaternion.identity(),
                ),
            },
        )
        json_str = frame.to_json()
        data = json.loads(json_str)
        assert data["timestamp"] == 0.0167
        assert data["frame"] == 1
        assert "joints" in data

    def test_data_frame_from_json(self):
        """Test UnrealDataFrame JSON deserialization."""
        json_str = """{
            "timestamp": 0.0167,
            "frame": 1,
            "joints": {
                "shoulder_L": {
                    "name": "shoulder_L",
                    "position": {"x": 0.1, "y": 1.4, "z": 0.2},
                    "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}
                }
            }
        }"""
        frame = UnrealDataFrame.from_json(json_str)
        assert frame.timestamp == 0.0167
        assert frame.frame_number == 1
        assert "shoulder_L" in frame.joints

    def test_data_frame_from_physics_state(self):
        """Test UnrealDataFrame creation from physics state."""
        # Simulated physics state
        q = np.array([0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 1.0])  # 7 DOF
        v = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])  # 6 DOF velocities
        t = 0.5

        frame = UnrealDataFrame.from_physics_state(
            q=q,
            v=v,
            timestamp=t,
            frame_number=30,
            joint_names=["pelvis", "shoulder_L"],
        )
        assert frame.timestamp == 0.5
        assert frame.frame_number == 30
        assert len(frame.joints) >= 1

    def test_data_frame_protocol_message(self):
        """Test UnrealDataFrame protocol message format."""
        frame = UnrealDataFrame(
            timestamp=0.0167,
            frame_number=1,
            joints={},
        )
        msg = frame.to_protocol_message()
        assert msg["type"] == "frame"
        assert "data" in msg
        assert msg["data"]["timestamp"] == 0.0167


class TestDataModelContracts:
    """Tests for Design by Contract compliance."""

    @pytest.mark.parametrize(
        "x, match",
        [
            (float("nan"), "NaN"),
            (float("inf"), "infinite"),
        ],
        ids=["nan", "infinite"],
    )
    def test_vector3_invalid_values(self, x, match):
        """Test Vector3 rejects NaN and infinite values."""
        with pytest.raises(ValueError, match=match):
            Vector3(x=x, y=0.0, z=0.0, validate=True)

    def test_quaternion_normalization_check(self):
        """Test Quaternion validates normalization."""
        q = Quaternion(w=2.0, x=0.0, y=0.0, z=0.0, validate=True)
        # Should auto-normalize when validate=True
        assert q.magnitude == pytest.approx(1.0)

    def test_force_vector_positive_magnitude(self):
        """Test ForceVector requires positive magnitude."""
        with pytest.raises(ValueError, match="positive"):
            ForceVector(
                origin=Vector3.zero(),
                direction=Vector3(x=1.0, y=0.0, z=0.0),
                magnitude=-5.0,
                validate=True,
            )

    def test_joint_state_requires_name(self):
        """Test JointState requires non-empty name."""
        with pytest.raises(ValueError, match="name"):
            JointState(
                name="",
                position=Vector3.zero(),
                rotation=Quaternion.identity(),
                validate=True,
            )

    @pytest.mark.parametrize(
        "timestamp, frame_number, match",
        [
            (-1.0, 1, "timestamp"),
            (0.0, -1, "frame"),
        ],
        ids=["negative-timestamp", "negative-frame"],
    )
    def test_data_frame_invalid_values(self, timestamp, frame_number, match):
        """Test UnrealDataFrame rejects invalid timestamp and frame number."""
        with pytest.raises(ValueError, match=match):
            UnrealDataFrame(
                timestamp=timestamp,
                frame_number=frame_number,
                joints={},
                validate=True,
            )


class TestDataModelPerformance:
    """Performance tests for data models."""

    def test_large_joint_set_serialization(self):
        """Test serialization performance with many joints."""
        joints = {}
        for i in range(50):  # Standard humanoid skeleton has ~50 bones
            joints[f"joint_{i}"] = JointState(
                name=f"joint_{i}",
                position=Vector3(x=float(i), y=0.0, z=0.0),
                rotation=Quaternion.identity(),
            )

        frame = UnrealDataFrame(
            timestamp=0.0167,
            frame_number=1,
            joints=joints,
        )

        # Serialization should complete quickly
        json_str = frame.to_json()
        assert len(json_str) > 0

        # Deserialization should also be fast
        frame2 = UnrealDataFrame.from_json(json_str)
        assert len(frame2.joints) == 50

    def test_trajectory_with_many_points(self):
        """Test trajectory with many points."""
        points = []
        for i in range(1000):
            points.append(
                TrajectoryPoint(
                    time=i * 0.001,
                    position=Vector3(x=float(i), y=0.0, z=float(i) ** 2 * 0.0001),
                )
            )

        frame = UnrealDataFrame(
            timestamp=1.0,
            frame_number=60,
            joints={},
            trajectory=points,
        )

        json_str = frame.to_json()
        assert len(json_str) > 0
