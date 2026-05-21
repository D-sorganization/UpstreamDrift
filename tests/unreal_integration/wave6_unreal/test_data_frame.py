"""Wave6 fast tests for UnrealDataFrame.

Covers serialization, JSON encoding/decoding, protocol message wrapping,
physics-state construction, and validation.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.unreal_integration.data_frame import UnrealDataFrame
from src.unreal_integration.geometry import Quaternion, Vector3
from src.unreal_integration.golf_state import (
    BallState,
    ClubState,
    EnvironmentState,
    SwingMetrics,
    TrajectoryPoint,
)
from src.unreal_integration.skeleton import ForceVector, JointState


def _make_minimal_frame() -> UnrealDataFrame:
    return UnrealDataFrame(
        timestamp=0.0167,
        frame_number=1,
        joints={
            "hip": JointState(
                name="hip",
                position=Vector3(0, 0, 0),
                rotation=Quaternion.identity(),
            )
        },
    )


class TestUnrealDataFrameBasics:
    def test_construct_minimal(self) -> None:
        f = _make_minimal_frame()
        assert f.timestamp == pytest.approx(0.0167)
        assert f.frame_number == 1
        assert len(f.joints) == 1

    def test_to_dict_minimal(self) -> None:
        d = _make_minimal_frame().to_dict()
        assert d["timestamp"] == pytest.approx(0.0167)
        assert d["frame"] == 1
        assert "forces" not in d
        assert "club" not in d

    def test_to_json_is_valid_json(self) -> None:
        s = _make_minimal_frame().to_json()
        parsed = json.loads(s)
        assert parsed["frame"] == 1

    def test_to_protocol_message(self) -> None:
        msg = _make_minimal_frame().to_protocol_message()
        assert msg["type"] == "frame"
        assert "data" in msg

    def test_repr(self) -> None:
        r = repr(_make_minimal_frame())
        assert "UnrealDataFrame" in r and "frame=1" in r

    def test_validate_rejects_negative_timestamp(self) -> None:
        with pytest.raises(ValueError, match="timestamp"):
            UnrealDataFrame(timestamp=-1.0, frame_number=0, joints={}, validate=True)

    def test_validate_rejects_negative_frame_number(self) -> None:
        with pytest.raises(ValueError, match="frame_number"):
            UnrealDataFrame(timestamp=0.0, frame_number=-1, joints={}, validate=True)


class TestUnrealDataFrameSerialization:
    def test_full_roundtrip(self) -> None:
        frame = UnrealDataFrame(
            timestamp=1.0,
            frame_number=42,
            joints={
                "hip": JointState(
                    name="hip",
                    position=Vector3(0, 1, 0),
                    rotation=Quaternion.identity(),
                )
            },
            forces=[
                ForceVector(
                    origin=Vector3(),
                    direction=Vector3(0, 1, 0),
                    magnitude=10.0,
                )
            ],
            club=ClubState(head_position=Vector3(), head_velocity=Vector3(1, 0, 0)),
            ball=BallState(position=Vector3(), velocity=Vector3(10, 0, 5)),
            metrics=SwingMetrics(club_head_speed=45.0, smash_factor=1.5),
            trajectory=[TrajectoryPoint(time=0.1, position=Vector3(1, 0, 1))],
            environment=EnvironmentState.default(),
        )
        json_str = frame.to_json()
        frame2 = UnrealDataFrame.from_json(json_str)
        assert frame2.frame_number == 42
        assert frame2.club is not None
        assert frame2.ball is not None
        assert frame2.metrics is not None
        assert frame2.trajectory is not None and len(frame2.trajectory) == 1
        assert frame2.environment is not None
        assert frame2.forces is not None and len(frame2.forces) == 1

    def test_from_dict_minimal(self) -> None:
        d = {"timestamp": 0.0, "frame": 0, "joints": {}}
        f = UnrealDataFrame.from_dict(d)
        assert f.joints == {}
        assert f.forces is None


class TestFromPhysicsState:
    def test_creates_joints_from_q_v(self) -> None:
        q = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        v = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        frame = UnrealDataFrame.from_physics_state(
            q=q,
            v=v,
            timestamp=0.5,
            frame_number=5,
            joint_names=["a", "b"],
        )
        assert frame.timestamp == 0.5
        assert "a" in frame.joints
        assert "b" in frame.joints
        assert frame.joints["a"].position == Vector3(1.0, 2.0, 3.0)
        assert frame.joints["b"].velocity == Vector3(0.4, 0.5, 0.6)

    def test_no_joint_names_returns_empty_joints(self) -> None:
        f = UnrealDataFrame.from_physics_state(
            q=np.array([1.0]),
            v=np.array([0.0]),
            timestamp=0.0,
            frame_number=0,
        )
        assert f.joints == {}

    def test_short_q_skips_missing_joints(self) -> None:
        # joint_names has 3 entries but q only fits 1
        q = np.array([1.0, 2.0, 3.0])
        v = np.array([0.0, 0.0, 0.0])
        f = UnrealDataFrame.from_physics_state(
            q=q, v=v, timestamp=0.0, frame_number=0, joint_names=["a", "b", "c"]
        )
        # Only "a" should fit
        assert "a" in f.joints
        assert "b" not in f.joints
