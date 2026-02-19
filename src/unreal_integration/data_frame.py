"""UnrealDataFrame — the primary data structure for Unreal Engine streaming.

Contains the complete visualization frame that is serialized and sent
over WebSocket to Unreal Engine for rendering.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from .geometry import Quaternion, Vector3
from .golf_state import (
    BallState,
    ClubState,
    EnvironmentState,
    SwingMetrics,
    TrajectoryPoint,
)
from .skeleton import ForceVector, JointState


@dataclass
class UnrealDataFrame:
    """Complete data frame for Unreal Engine visualization.

    This is the primary data structure streamed to Unreal Engine.
    Contains all data needed to render a single frame.

    Attributes:
        timestamp: Simulation time in seconds.
        frame_number: Frame counter (0-indexed).
        joints: Dictionary of joint states by name.
        forces: List of force vectors to visualize.
        club: Golf club state (optional).
        ball: Golf ball state (optional).
        metrics: Swing analysis metrics (optional).
        trajectory: List of trajectory points (optional).
        environment: Environmental conditions (optional).

    Example:
        >>> frame = UnrealDataFrame(
        ...     timestamp=0.0167,
        ...     frame_number=1,
        ...     joints={"shoulder_L": JointState(...)},
        ... )
        >>> json_str = frame.to_json()
    """

    timestamp: float
    frame_number: int
    joints: dict[str, JointState]
    forces: list[ForceVector] | None = None
    club: ClubState | None = None
    ball: BallState | None = None
    metrics: SwingMetrics | None = None
    trajectory: list[TrajectoryPoint] | None = None
    environment: EnvironmentState | None = None

    def __new__(  # noqa: PLR0913
        cls,
        timestamp: float,
        frame_number: int,
        joints: dict[str, JointState],
        forces: list[ForceVector] | None = None,
        club: ClubState | None = None,
        ball: BallState | None = None,
        metrics: SwingMetrics | None = None,
        trajectory: list[TrajectoryPoint] | None = None,
        environment: EnvironmentState | None = None,
        validate: bool = False,
    ) -> UnrealDataFrame:
        """Create new UnrealDataFrame with optional validation."""
        instance = object.__new__(cls)
        return instance

    def __init__(  # noqa: PLR0913
        self,
        timestamp: float,
        frame_number: int,
        joints: dict[str, JointState],
        forces: list[ForceVector] | None = None,
        club: ClubState | None = None,
        ball: BallState | None = None,
        metrics: SwingMetrics | None = None,
        trajectory: list[TrajectoryPoint] | None = None,
        environment: EnvironmentState | None = None,
        validate: bool = False,
    ) -> None:
        """Initialize UnrealDataFrame.

        Args:
            timestamp: Simulation time.
            frame_number: Frame counter.
            joints: Dictionary of joint states.
            forces: List of force vectors.
            club: Golf club state.
            ball: Golf ball state.
            metrics: Swing metrics.
            trajectory: Trajectory points.
            environment: Environmental conditions.
            validate: If True, validate inputs.
        """
        if validate:
            if timestamp < 0:
                raise ValueError("timestamp must be non-negative")
            if frame_number < 0:
                raise ValueError("frame_number must be non-negative")
        self.timestamp = timestamp
        self.frame_number = frame_number
        self.joints = joints
        self.forces = forces
        self.club = club
        self.ball = ball
        self.metrics = metrics
        self.trajectory = trajectory
        self.environment = environment

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        result: dict[str, Any] = {
            "timestamp": self.timestamp,
            "frame": self.frame_number,
            "joints": {name: js.to_dict() for name, js in self.joints.items()},
        }
        if self.forces:
            result["forces"] = [f.to_dict() for f in self.forces]
        if self.club is not None:
            result["club"] = self.club.to_dict()
        if self.ball is not None:
            result["ball"] = self.ball.to_dict()
        if self.metrics is not None:
            result["metrics"] = self.metrics.to_dict()
        if self.trajectory:
            result["trajectory"] = [tp.to_dict() for tp in self.trajectory]
        if self.environment is not None:
            result["environment"] = self.environment.to_dict()
        return result

    def to_json(self) -> str:
        """Convert to JSON string.

        Returns:
            JSON string representation.
        """
        return json.dumps(self.to_dict())

    def to_protocol_message(self) -> dict[str, Any]:
        """Convert to WebSocket protocol message format.

        Returns:
            Protocol message with type and data fields.
        """
        return {
            "type": "frame",
            "data": self.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], validate: bool = False) -> UnrealDataFrame:
        """Create UnrealDataFrame from dictionary.

        Args:
            d: Dictionary representation.
            validate: If True, validate inputs.

        Returns:
            New UnrealDataFrame instance.
        """
        joints = {
            name: JointState.from_dict(js_dict)
            for name, js_dict in d.get("joints", {}).items()
        }
        forces = (
            [ForceVector.from_dict(f) for f in d.get("forces", [])]
            if "forces" in d
            else None
        )
        club = ClubState.from_dict(d["club"]) if "club" in d else None
        ball = BallState.from_dict(d["ball"]) if "ball" in d else None
        metrics = SwingMetrics.from_dict(d["metrics"]) if "metrics" in d else None
        trajectory = (
            [TrajectoryPoint.from_dict(tp) for tp in d.get("trajectory", [])]
            if "trajectory" in d
            else None
        )
        environment = (
            EnvironmentState.from_dict(d["environment"]) if "environment" in d else None
        )

        return cls(
            timestamp=float(d["timestamp"]),
            frame_number=int(d["frame"]),
            joints=joints,
            forces=forces,
            club=club,
            ball=ball,
            metrics=metrics,
            trajectory=trajectory,
            environment=environment,
            validate=validate,
        )

    @classmethod
    def from_json(cls, json_str: str, validate: bool = False) -> UnrealDataFrame:
        """Create UnrealDataFrame from JSON string.

        Args:
            json_str: JSON string representation.
            validate: If True, validate inputs.

        Returns:
            New UnrealDataFrame instance.
        """
        d = json.loads(json_str)
        return cls.from_dict(d, validate=validate)

    @classmethod
    def from_physics_state(  # noqa: PLR0913
        cls,
        q: np.ndarray,
        v: np.ndarray,
        timestamp: float,
        frame_number: int,
        joint_names: list[str] | None = None,
        validate: bool = False,
    ) -> UnrealDataFrame:
        """Create UnrealDataFrame from physics engine state.

        This is a convenience method for converting raw physics state
        into the Unreal Engine format.

        Args:
            q: Generalized coordinates (positions).
            v: Generalized velocities.
            timestamp: Simulation time.
            frame_number: Frame counter.
            joint_names: List of joint names (optional).
            validate: If True, validate inputs.

        Returns:
            New UnrealDataFrame instance.
        """
        joints: dict[str, JointState] = {}

        # Create joint states from physics state
        # This is a simplified mapping - real implementation would use
        # proper skeleton configuration
        if joint_names:
            for i, name in enumerate(joint_names):
                if i * 3 + 2 < len(q):
                    joints[name] = JointState(
                        name=name,
                        position=Vector3(
                            x=float(q[i * 3]) if i * 3 < len(q) else 0.0,
                            y=float(q[i * 3 + 1]) if i * 3 + 1 < len(q) else 0.0,
                            z=float(q[i * 3 + 2]) if i * 3 + 2 < len(q) else 0.0,
                        ),
                        rotation=Quaternion.identity(),
                        velocity=(
                            Vector3(
                                x=float(v[i * 3]) if i * 3 < len(v) else 0.0,
                                y=float(v[i * 3 + 1]) if i * 3 + 1 < len(v) else 0.0,
                                z=float(v[i * 3 + 2]) if i * 3 + 2 < len(v) else 0.0,
                            )
                            if len(v) > i * 3 + 2
                            else None
                        ),
                    )

        return cls(
            timestamp=timestamp,
            frame_number=frame_number,
            joints=joints,
            validate=validate,
        )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"UnrealDataFrame(t={self.timestamp:.4f}, frame={self.frame_number}, "
            f"joints={len(self.joints)}, forces={len(self.forces or [])})"
        )
