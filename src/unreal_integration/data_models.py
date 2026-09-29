"""Core data models for Unreal Engine integration.

This module re-exports all public symbols from the decomposed sub-modules
for backward compatibility.  New code should import directly from:

- ``src.unreal_integration.geometry``   – _validate_finite, Vector3, Quaternion
- ``src.unreal_integration.skeleton``   – JointState, ForceVector
- ``src.unreal_integration.golf_state`` – ClubState, SwingMetrics, BallState,
                                          TrajectoryPoint, EnvironmentState
- ``src.unreal_integration.data_frame`` – UnrealDataFrame

Data Flow:
    Physics Engine → Data Models → JSON/MessagePack → WebSocket → Unreal Engine

Usage:
    from src.unreal_integration.data_models import (
        UnrealDataFrame,
        JointState,
        ForceVector,
        Vector3,
    )

    frame = UnrealDataFrame(
        timestamp=0.0167,
        frame_number=1,
        joints={"shoulder_L": JointState(...)},
    )
    json_str = frame.to_json()
"""

from __future__ import annotations

from src.unreal_integration.data_frame import UnrealDataFrame
from src.unreal_integration.geometry import Quaternion, Vector3, _validate_finite
from src.unreal_integration.golf_state import (
    BallState,
    ClubState,
    EnvironmentState,
    SwingMetrics,
    TrajectoryPoint,
)
from src.unreal_integration.skeleton import ForceVector, JointState

__all__ = [
    "_validate_finite",
    "Vector3",
    "Quaternion",
    "JointState",
    "ForceVector",
    "ClubState",
    "SwingMetrics",
    "BallState",
    "TrajectoryPoint",
    "EnvironmentState",
    "UnrealDataFrame",
]
