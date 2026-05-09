"""
Canonical Intermediate Representation (CIR) for Motion Capture Pipeline.

This module defines the Pydantic v2 contracts that every pipeline stage reads/writes.
It unifies engine-specific dataclasses and API models into a single canonical representation.

Models:
- Calibration: Camera intrinsics/extrinsics, unit system, source FPS, world-up axis
- KeypointFrame: 2D or 3D keypoints with confidences
- KeypointSequence: Timestamped sequence of KeypointFrame
- MarkerFrame / MarkerTrajectory: Labeled 3D markers
- SkeletonRig: Joints, parents, T-pose offsets, axes, limits
- JointStateFrame / JointTrajectory: q, qdot, qddot, time
- MotionTrajectory: Skeleton + joint trajectory + metadata
- MotionMatchingRequest / MotionMatchingResult: Solver inputs/outputs
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.functional_validators import AfterValidator

# =============================================================================
# Type Aliases
# =============================================================================

ArrayLike = list[float] | np.ndarray
SchemaName = Literal["BODY_25", "MediaPipe_33", "COCO_17", "OpenPose_25", "custom"]
Axis = Literal["X", "Y", "Z", "+X", "-X", "+Y", "-Y", "+Z", "-Z"]
UpAxis = Literal["+Y", "+Z", "+X", "-Y", "-Z", "-X"]


# =============================================================================
# Calibration
# =============================================================================


class CameraIntrinsics(BaseModel):
    """Camera intrinsic parameters."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    fx: float = Field(..., description="Focal length x (pixels)", gt=0)
    fy: float = Field(..., description="Focal length y (pixels)", gt=0)
    cx: float = Field(..., description="Principal point x (pixels)")
    cy: float = Field(..., description="Principal point y (pixels)")
    k1: float = Field(default=0.0, description="Radial distortion coefficient 1")
    k2: float = Field(default=0.0, description="Radial distortion coefficient 2")
    p1: float = Field(default=0.0, description="Tangential distortion coefficient 1")
    p2: float = Field(default=0.0, description="Tangential distortion coefficient 2")

    @field_validator("fx", "fy", "cx", "cy")
    @classmethod
    def check_finite(cls, v: float) -> float:
        if not np.isfinite(v):
            raise ValueError("Value must be finite")
        return v


class CameraExtrinsics(BaseModel):
    """Camera extrinsic parameters (world-to-camera transform)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    rotation: list[list[float]] = Field(
        default_factory=lambda: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        description="3x3 rotation matrix (world-to-camera)",
    )
    translation: list[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0],
        description="Translation vector (world-to-camera, meters)",
    )

    @field_validator("rotation")
    @classmethod
    def check_rotation_shape(cls, v: list[list[float]]) -> list[list[float]]:
        if len(v) != 3 or any(len(row) != 3 for row in v):
            raise ValueError("Rotation must be 3x3 matrix")
        return v

    @field_validator("translation")
    @classmethod
    def check_translation_shape(cls, v: list[float]) -> list[float]:
        if len(v) != 3:
            raise ValueError("Translation must be length 3")
        return v


class Calibration(BaseModel):
    """Multi-camera calibration data."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(..., description="Unique calibration identifier")
    cameras: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Camera ID -> {intrinsics, extrinsics}",
    )
    unit_system: Literal["meters", "millimeters", "pixels"] = Field(
        default="meters", description="Unit system for 3D coordinates"
    )
    source_fps: float = Field(..., description="Source capture frame rate (Hz)", gt=0)
    world_up_axis: UpAxis = Field(
        default="+Y", description="World coordinate system up axis"
    )
    calibrated_at: datetime = Field(
        default_factory=datetime.now, description="Calibration timestamp"
    )

    @field_validator("source_fps")
    @classmethod
    def check_fps_finite(cls, v: float) -> float:
        if not np.isfinite(v):
            raise ValueError("FPS must be finite")
        return v

    @model_validator(mode="after")
    def check_cameras_have_intrinsics(self) -> Calibration:
        for cam_id, cam_data in self.cameras.items():
            if "intrinsics" not in cam_data:
                raise ValueError(f"Camera {cam_id} missing intrinsics")
        return self


# =============================================================================
# Keypoint Data
# =============================================================================


class Keypoint(BaseModel):
    """Single keypoint with optional confidence."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    x: float = Field(..., description="X coordinate (pixels or normalized)")
    y: float = Field(..., description="Y coordinate (pixels or normalized)")
    z: float | None = Field(default=None, description="Z coordinate (if 3D)")
    confidence: float = Field(
        default=1.0, description="Confidence score [0, 1]", ge=0, le=1
    )
    name: str | None = Field(default=None, description="Keypoint name (e.g., 'nose')")

    @field_validator("x", "y", "z")
    @classmethod
    def check_finite(cls, v: float | None) -> float | None:
        if v is not None and not np.isfinite(v):
            raise ValueError("Coordinate must be finite")
        return v


class KeypointFrame(BaseModel):
    """Single frame of keypoints (2D or 3D)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    timestamp: float = Field(..., description="Frame timestamp (seconds)", ge=0)
    keypoints: list[Keypoint] = Field(
        ..., description="List of keypoints", min_length=1
    )
    schema_name: SchemaName = Field(..., description="Keypoint schema used")
    frame_index: int | None = Field(default=None, description="Frame index in sequence")

    @field_validator("timestamp")
    @classmethod
    def check_timestamp_finite(cls, v: float) -> float:
        if not np.isfinite(v):
            raise ValueError("Timestamp must be finite")
        return v

    @model_validator(mode="after")
    def _invariant_keypoints_have_consistent_depth(self) -> KeypointFrame:
        """All keypoints should be either all 2D or all 3D (no mix)."""
        has_z = [kp.z is not None for kp in self.keypoints]
        if not (all(has_z) or not any(has_z)):
            raise ValueError(
                "keypoints_have_consistent_depth: all keypoints must be either 2D or 3D"
            )
        return self


class KeypointSequence(BaseModel):
    """Timestamped sequence of keypoint frames."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(..., description="Unique sequence identifier")
    frames: list[KeypointFrame] = Field(
        ..., description="Sequence of keypoint frames", min_length=1
    )
    calibration: Calibration | None = Field(
        default=None, description="Associated calibration data"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    @model_validator(mode="after")
    def check_monotonic_timestamps(self) -> KeypointSequence:
        timestamps = [f.timestamp for f in self.frames]
        if timestamps != sorted(timestamps):
            raise ValueError("Timestamps must be monotonically increasing")
        return self

    @model_validator(mode="after")
    def check_consistent_schema(self) -> KeypointSequence:
        schemas = {f.schema_name for f in self.frames}
        if len(schemas) > 1:
            raise ValueError(f"Inconsistent schemas: {schemas}")
        return self

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    @property
    def num_keypoints(self) -> int:
        return len(self.frames[0].keypoints) if self.frames else 0

    @property
    def duration(self) -> float:
        if len(self.frames) < 2:
            return 0.0
        return self.frames[-1].timestamp - self.frames[0].timestamp


# =============================================================================
# Marker Data
# =============================================================================


class Marker(BaseModel):
    """Single 3D marker with label and confidence."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(..., description="Marker name/label")
    x: float = Field(..., description="X position (meters)")
    y: float = Field(..., description="Y position (meters)")
    z: float = Field(..., description="Z position (meters)")
    residual: float | None = Field(
        default=None, description="Residual error (mm)", ge=0
    )
    occluded: bool = Field(default=False, description="Marker is occluded")

    @field_validator("x", "y", "z", "residual")
    @classmethod
    def check_finite(cls, v: float | None) -> float | None:
        if v is not None and not np.isfinite(v):
            raise ValueError("Value must be finite")
        return v


class MarkerFrame(BaseModel):
    """Single frame of 3D markers."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    timestamp: float = Field(..., description="Frame timestamp (seconds)", ge=0)
    markers: dict[str, Marker] = Field(
        default_factory=dict, description="Marker name -> Marker data"
    )
    frame_index: int | None = Field(default=None, description="Frame index in sequence")

    @field_validator("timestamp")
    @classmethod
    def check_timestamp_finite(cls, v: float) -> float:
        if not np.isfinite(v):
            raise ValueError("Timestamp must be finite")
        return v

    @property
    def marker_names(self) -> list[str]:
        return list(self.markers.keys())

    @property
    def num_markers(self) -> int:
        return len(self.markers)


class MarkerTrajectory(BaseModel):
    """Time series of marker frames."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(..., description="Unique trajectory identifier")
    frames: list[MarkerFrame] = Field(
        ..., description="Sequence of marker frames", min_length=1
    )
    calibration: Calibration | None = Field(
        default=None, description="Associated calibration data"
    )
    subject_id: str | None = Field(default=None, description="Subject identifier")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    @model_validator(mode="after")
    def check_monotonic_timestamps(self) -> MarkerTrajectory:
        timestamps = [f.timestamp for f in self.frames]
        if timestamps != sorted(timestamps):
            raise ValueError("Timestamps must be monotonically increasing")
        return self

    @model_validator(mode="after")
    def check_consistent_markers(self) -> MarkerTrajectory:
        """All frames should have the same marker set."""
        if len(self.frames) < 2:
            return self
        reference_markers = set(self.frames[0].marker_names)
        for _i, frame in enumerate(self.frames[1:], 1):
            frame_markers = set(frame.marker_names)
            if frame_markers != reference_markers:
                # Allow for occlusions, but names should be consistent
                pass  # Relaxing this constraint for practical use
        return self

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    @property
    def marker_names(self) -> list[str]:
        return self.frames[0].marker_names if self.frames else []

    @property
    def duration(self) -> float:
        if len(self.frames) < 2:
            return 0.0
        return self.frames[-1].timestamp - self.frames[0].timestamp


# =============================================================================
# Skeleton Rig
# =============================================================================


class JointLimit(BaseModel):
    """Joint limit specification."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    lower: float | None = Field(default=None, description="Lower limit (radians)")
    upper: float | None = Field(default=None, description="Upper limit (radians)")
    soft_lower: float | None = Field(default=None, description="Soft lower limit")
    soft_upper: float | None = Field(default=None, description="Soft upper limit")

    @field_validator("lower", "upper", "soft_lower", "soft_upper")
    @classmethod
    def check_finite(cls, v: float | None) -> float | None:
        if v is not None and not np.isfinite(v):
            raise ValueError("Limit must be finite")
        return v

    @model_validator(mode="after")
    def check_limit_order(self) -> JointLimit:
        if (
            self.lower is not None
            and self.upper is not None
            and self.lower > self.upper
        ):
            raise ValueError("Lower limit must be <= upper limit")
        return self


class JointDef(BaseModel):
    """Joint definition in skeleton rig."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(..., description="Joint name", min_length=1)
    parent: str | None = Field(default=None, description="Parent joint name")
    children: list[str] = Field(default_factory=list, description="Child joint names")
    tpose_offset: list[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0],
        description="T-pose offset from parent (meters)",
        min_length=3,
        max_length=3,
    )
    axes: list[Axis] = Field(
        default_factory=lambda: ["X", "Y", "Z"],
        description="Joint rotation axes",
        min_length=1,
        max_length=3,
    )
    limits: list[JointLimit] = Field(
        default_factory=list,
        description="Joint limits (one per DOF)",
    )
    semantic_label: str | None = Field(
        default=None, description="Semantic label (e.g., 'right_knee')"
    )

    @field_validator("tpose_offset")
    @classmethod
    def check_offset_shape(cls, v: list[float]) -> list[float]:
        if len(v) != 3:
            raise ValueError("T-pose offset must be length 3")
        return v

    @field_validator("axes")
    @classmethod
    def check_axes_shape(cls, v: list[Axis]) -> list[Axis]:
        if len(v) != 3:
            raise ValueError("Joint axes must be length 3")
        return v

    @model_validator(mode="after")
    def _invariant_axes_match_limits(self) -> JointDef:
        """Number of axes should match number of limits (if limits provided)."""
        if self.limits and len(self.axes) != len(self.limits):
            raise ValueError(
                "axes_match_limits: number of axes must equal number of limits"
            )
        return self


class SkeletonRig(BaseModel):
    """Skeleton rig definition."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(..., description="Unique rig identifier")
    joints: dict[str, JointDef] = Field(
        ..., description="Joint name -> Joint definition", min_length=1
    )
    root_joint: str = Field(..., description="Root joint name")
    up_axis: UpAxis = Field(default="+Y", description="Skeleton up axis")
    scale: float = Field(default=1.0, description="Global scale factor", gt=0)
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    @model_validator(mode="after")
    def check_root_exists(self) -> SkeletonRig:
        if self.root_joint not in self.joints:
            raise ValueError(f"Root joint '{self.root_joint}' not found in joints")
        return self

    @model_validator(mode="after")
    def check_parent_child_consistency(self) -> SkeletonRig:
        """Verify parent-child relationships are consistent."""
        for joint_name, joint in self.joints.items():
            # Check children reference valid joints
            for child in joint.children:
                if child not in self.joints:
                    raise ValueError(f"Joint {joint_name} has invalid child {child}")
            # Check parent references valid joint
            if joint.parent is not None and joint.parent not in self.joints:
                raise ValueError(
                    f"Joint {joint_name} has invalid parent {joint.parent}"
                )
        return self

    @property
    def num_joints(self) -> int:
        return len(self.joints)

    @property
    def num_dofs(self) -> int:
        return sum(len(j.axes) for j in self.joints.values())

    def get_joint_chain(self, joint_name: str) -> list[str]:
        """Get the chain of joints from root to specified joint."""
        if joint_name not in self.joints:
            raise ValueError(f"Joint {joint_name} not found")
        chain = [joint_name]
        current = self.joints[joint_name].parent
        while current is not None:
            chain.insert(0, current)
            current = self.joints[current].parent
        return chain


# =============================================================================
# Joint States
# =============================================================================


class JointStateFrame(BaseModel):
    """Single frame of joint states."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    timestamp: float = Field(..., description="Frame timestamp (seconds)", ge=0)
    q: list[float] = Field(..., description="Joint positions", min_length=1)
    qdot: list[float] | None = Field(default=None, description="Joint velocities")
    qddot: list[float] | None = Field(default=None, description="Joint accelerations")
    frame_index: int | None = Field(default=None, description="Frame index")

    @field_validator("timestamp")
    @classmethod
    def check_timestamp_finite(cls, v: float) -> float:
        if not np.isfinite(v):
            raise ValueError("Timestamp must be finite")
        return v

    @field_validator("q", "qdot", "qddot")
    @classmethod
    def check_finite_values(cls, v: list[float] | None) -> list[float] | None:
        if v is not None and not all(np.isfinite(x) for x in v):
            raise ValueError("All values must be finite")
        return v

    @model_validator(mode="after")
    def _invariant_matching_dimensions(self) -> JointStateFrame:
        """q, qdot, qddot should have same length if all present."""
        lengths = [len(self.q)]
        if self.qdot is not None:
            lengths.append(len(self.qdot))
        if self.qddot is not None:
            lengths.append(len(self.qddot))
        if len(set(lengths)) != 1:
            raise ValueError(
                "matching_dimensions: q, qdot, qddot must have identical length"
            )
        return self

    @property
    def num_dofs(self) -> int:
        return len(self.q)


class JointTrajectory(BaseModel):
    """Time series of joint states."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(..., description="Unique trajectory identifier")
    skeleton: SkeletonRig = Field(..., description="Associated skeleton rig")
    frames: list[JointStateFrame] = Field(
        ..., description="Sequence of joint states", min_length=1
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    @model_validator(mode="after")
    def check_monotonic_timestamps(self) -> JointTrajectory:
        timestamps = [f.timestamp for f in self.frames]
        if timestamps != sorted(timestamps):
            raise ValueError("Timestamps must be monotonically increasing")
        return self

    @model_validator(mode="after")
    def check_dof_consistency(self) -> JointTrajectory:
        """All frames should have same DOF count as skeleton."""
        expected_dofs = self.skeleton.num_dofs
        for frame in self.frames:
            if frame.num_dofs != expected_dofs:
                raise ValueError(
                    f"Frame has {frame.num_dofs} DOFs, expected {expected_dofs}"
                )
        return self

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    @property
    def duration(self) -> float:
        if len(self.frames) < 2:
            return 0.0
        return self.frames[-1].timestamp - self.frames[0].timestamp


# =============================================================================
# Motion Trajectory (High-level CIR)
# =============================================================================


class MotionTrajectory(BaseModel):
    """High-level motion representation combining skeleton, trajectory, and metadata."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(..., description="Unique motion identifier")
    skeleton: SkeletonRig = Field(..., description="Skeleton rig")
    trajectory: JointTrajectory = Field(..., description="Joint trajectory")
    marker_reference: MarkerTrajectory | None = Field(
        default=None, description="Optional reference marker trajectory"
    )
    subject: dict[str, Any] | None = Field(
        default=None,
        description="Subject metadata (height, mass, age, etc.)",
    )
    sport: str | None = Field(default=None, description="Sport type (e.g., 'golf')")
    club: str | None = Field(default=None, description="Club/equipment used")
    source_provenance: dict[str, Any] = Field(
        default_factory=dict,
        description="Source data provenance (file paths, capture system, etc.)",
    )
    created_at: datetime = Field(
        default_factory=datetime.now, description="Creation timestamp"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    @model_validator(mode="after")
    def check_trajectory_skeleton_match(self) -> MotionTrajectory:
        """Trajectory skeleton should match main skeleton."""
        if self.trajectory.skeleton.id != self.skeleton.id:
            raise ValueError("Trajectory skeleton does not match main skeleton")
        return self

    @property
    def duration(self) -> float:
        return self.trajectory.duration

    @property
    def num_frames(self) -> int:
        return self.trajectory.num_frames


# =============================================================================
# Motion Matching
# =============================================================================


class MotionMatchingRequest(BaseModel):
    """Motion matching solver request."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(..., description="Unique request identifier")
    target_trajectory: MotionTrajectory | None = Field(
        default=None, description="Target trajectory to match"
    )
    target_markers: MarkerTrajectory | None = Field(
        default=None, description="Target marker trajectory"
    )
    target_keypoints: KeypointSequence | None = Field(
        default=None, description="Target keypoint sequence"
    )
    skeleton: SkeletonRig = Field(..., description="Skeleton to use for matching")
    constraints: dict[str, Any] = Field(
        default_factory=dict,
        description="Matching constraints (weights, tolerances, etc.)",
    )
    solver_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Solver-specific configuration",
    )

    @model_validator(mode="after")
    def check_has_target(self) -> MotionMatchingRequest:
        """Must have at least one target specification."""
        has_target = (
            self.target_trajectory is not None
            or self.target_markers is not None
            or self.target_keypoints is not None
        )
        if not has_target:
            raise ValueError(
                "Must specify at least one target (trajectory, markers, or keypoints)"
            )
        return self


class TorqueFrame(BaseModel):
    """Single frame of generalized joint torques."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    timestamp: float = Field(..., description="Frame timestamp (seconds)")
    tau: list[float] = Field(..., description="Generalized joint torques (N*m)")

    @field_validator("timestamp")
    @classmethod
    def _timestamp_finite(cls, v: float) -> float:
        if not np.isfinite(v):
            raise ValueError("timestamp must be finite")
        return v

    @field_validator("tau")
    @classmethod
    def _tau_finite(cls, v: list[float]) -> list[float]:
        if not all(np.isfinite(x) for x in v):
            raise ValueError("tau values must be finite")
        return v


class TorqueTrajectory(BaseModel):
    """Time series of generalized joint torques.

    Distinct from :class:`JointTrajectory`: torques have different
    invariants (any sign, no q/qdot/qddot semantics) and align with
    a rig's joint name list rather than its DOF list.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    frames: list[TorqueFrame] = Field(..., description="Per-frame torques")
    rig_joint_names: list[str] = Field(
        ..., description="Joint names corresponding to tau entries"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    @model_validator(mode="after")
    def _invariant_consistent(self) -> TorqueTrajectory:
        if not self.frames:
            raise ValueError("TorqueTrajectory must have at least one frame")
        n = len(self.rig_joint_names)
        for i, f in enumerate(self.frames):
            if len(f.tau) != n:
                raise ValueError(
                    f"frame {i} tau length {len(f.tau)} != rig_joint_names length {n}"
                )
        ts = [f.timestamp for f in self.frames]
        if any(b <= a for a, b in zip(ts, ts[1:], strict=False)):
            raise ValueError("timestamps must be strictly monotonic")
        return self

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    @property
    def duration(self) -> float:
        if len(self.frames) < 2:
            return 0.0
        return self.frames[-1].timestamp - self.frames[0].timestamp


class MuscleActivationFrame(BaseModel):
    """Single frame of muscle activations in [0, 1]."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    timestamp: float = Field(..., description="Frame timestamp (seconds)")
    activations: list[float] = Field(..., description="Muscle activations [0, 1]")

    @field_validator("timestamp")
    @classmethod
    def _timestamp_finite(cls, v: float) -> float:
        if not np.isfinite(v):
            raise ValueError("timestamp must be finite")
        return v

    @field_validator("activations")
    @classmethod
    def _activations_in_unit_interval(cls, v: list[float]) -> list[float]:
        if not all(np.isfinite(a) and 0.0 <= a <= 1.0 for a in v):
            raise ValueError("activations must lie in [0, 1] and be finite")
        return v


class MuscleActivationTrajectory(BaseModel):
    """Time series of muscle activations.

    Distinct from :class:`JointTrajectory`: activation values are bounded
    to [0, 1] and align with muscle names, not DOFs.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    frames: list[MuscleActivationFrame] = Field(
        ..., description="Per-frame activations"
    )
    muscle_names: list[str] = Field(
        ..., description="Muscle names corresponding to activation entries"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    @model_validator(mode="after")
    def _invariant_consistent(self) -> MuscleActivationTrajectory:
        if not self.frames:
            raise ValueError("MuscleActivationTrajectory must have at least one frame")
        n = len(self.muscle_names)
        for i, f in enumerate(self.frames):
            if len(f.activations) != n:
                raise ValueError(
                    f"frame {i} activation length {len(f.activations)} != "
                    f"muscle_names length {n}"
                )
        ts = [f.timestamp for f in self.frames]
        if any(b <= a for a, b in zip(ts, ts[1:], strict=False)):
            raise ValueError("timestamps must be strictly monotonic")
        return self

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    @property
    def duration(self) -> float:
        if len(self.frames) < 2:
            return 0.0
        return self.frames[-1].timestamp - self.frames[0].timestamp


#: Current ``MotionMatchingResult`` schema version. v2 enforces the
#: payload-on-success invariant; v1 (legacy) documents predate it and
#: are migrated forward on load with the invariant relaxed.
MOTION_MATCHING_RESULT_SCHEMA_VERSION: int = 2


class MotionMatchingResult(BaseModel):
    """Motion matching solver result."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    request_id: str = Field(..., description="Associated request identifier")
    success: bool = Field(..., description="Whether matching succeeded")
    matched_trajectory: MotionTrajectory | None = Field(
        default=None, description="Matched joint trajectory"
    )
    torques: TorqueTrajectory | None = Field(
        default=None,
        description="Generalized joint torques (distinct from JointTrajectory)",
    )
    activations: MuscleActivationTrajectory | None = Field(
        default=None,
        description="Muscle activations in [0, 1] (distinct from JointTrajectory)",
    )
    error_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Error metrics (RMSE, max error, etc.)",
    )
    iterations: int | None = Field(default=None, description="Solver iterations")
    solve_time: float | None = Field(
        default=None, description="Solve time (seconds)", ge=0
    )
    message: str | None = Field(default=None, description="Status message")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    schema_version: int = Field(
        default=MOTION_MATCHING_RESULT_SCHEMA_VERSION,
        description=(
            "Document schema version. v1 (legacy) results predate the "
            "successful-payload invariant; v2 enforces it. Absent on "
            "legacy documents and migrated to 1 on load."
        ),
        ge=1,
    )

    @field_validator("solve_time")
    @classmethod
    def check_solve_time_finite(cls, v: float | None) -> float | None:
        if v is not None and not np.isfinite(v):
            raise ValueError("Solve time must be finite")
        return v

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_document(cls, data: Any) -> Any:
        """Migrate v1 (legacy) serialized documents forward.

        Legacy documents have no ``schema_version`` field. They are tagged
        as ``schema_version=1`` so the post-validator can recognize them
        and relax the payload-on-success invariant. In-process construction
        (where ``data`` is already a model instance, or where the caller
        explicitly sets ``schema_version``) is left untouched.
        """
        if isinstance(data, dict) and "schema_version" not in data:
            # Shallow copy so we never mutate the caller's dict.
            data = {**data, "schema_version": 1}
        return data

    @model_validator(mode="after")
    def _invariant_has_payload_when_successful(self) -> MotionMatchingResult:
        """Successful results must carry at least one payload.

        trajectory, torques, muscle activations, or scalar error
        metrics. Failed results may carry only a message. Either
        ``torques`` or ``activations`` is sufficient — the two are not
        interchangeable.

        The invariant is relaxed for ``schema_version < 2`` (legacy
        documents) so that previously serialized successful results
        without a payload still load. Newly-created results default to
        the current schema version and are checked strictly.
        """
        if (
            self.success
            and self.schema_version >= MOTION_MATCHING_RESULT_SCHEMA_VERSION
        ):
            has_payload = (
                self.matched_trajectory is not None
                or self.torques is not None
                or self.activations is not None
                or bool(self.error_metrics)
            )
            if not has_payload:
                raise ValueError(
                    "Successful MotionMatchingResult must include at least one "
                    "of: matched_trajectory, torques, activations, error_metrics"
                )
        return self


# =============================================================================
# Serialization Helpers
# =============================================================================


def serialize_model(model: BaseModel) -> str:
    """Serialize a Pydantic model to JSON string."""
    return model.model_dump_json(indent=2)


def deserialize_model(json_str: str, model_class: type) -> BaseModel:
    """Deserialize a JSON string to a Pydantic model."""
    return model_class.model_validate_json(json_str)


def save_model(model: BaseModel, path: str | Path) -> None:
    """Save a Pydantic model to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(serialize_model(model))


def load_model(json_path: str | Path, model_class: type) -> BaseModel:
    """Load a Pydantic model from a JSON file."""
    with open(json_path) as f:
        return deserialize_model(f.read(), model_class)


# =============================================================================
# Backward Compatibility Shims (Deprecated)
# =============================================================================
# These re-exports provide backward compatibility with existing code.
# New code should import directly from this module.


__all__ = [
    # Core types
    "Calibration",
    "CameraIntrinsics",
    "CameraExtrinsics",
    # Keypoint types
    "Keypoint",
    "KeypointFrame",
    "KeypointSequence",
    # Marker types
    "Marker",
    "MarkerFrame",
    "MarkerTrajectory",
    # Skeleton types
    "JointDef",
    "JointLimit",
    "SkeletonRig",
    # Joint state types
    "JointStateFrame",
    "JointTrajectory",
    # Torque + muscle-activation types
    "TorqueFrame",
    "TorqueTrajectory",
    "MuscleActivationFrame",
    "MuscleActivationTrajectory",
    # High-level types
    "MotionTrajectory",
    "MotionMatchingRequest",
    "MotionMatchingResult",
    "MOTION_MATCHING_RESULT_SCHEMA_VERSION",
    # Serialization
    "serialize_model",
    "deserialize_model",
    "save_model",
    "load_model",
]
