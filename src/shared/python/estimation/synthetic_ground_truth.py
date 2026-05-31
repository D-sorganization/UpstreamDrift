"""Synthetic camera observations from known trajectories.

The rig consumes the CIR motion-pipeline contracts on ``origin/main`` and keeps
the forward model pluggable. Pinocchio/CanonicalState adapters can implement the
same :class:`ForwardModel` protocol once those issue dependencies are merged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeAlias

import numpy as np
import numpy.typing as npt

from src.shared.python.core.contracts import check_finite, require
from src.shared.python.motion_pipeline import (
    Calibration,
    CameraExtrinsics,
    CameraIntrinsics,
    JointStateFrame,
    JointTrajectory,
    Keypoint,
    KeypointFrame,
    KeypointSequence,
    Marker,
    MarkerFrame,
    MarkerTrajectory,
    SkeletonRig,
)

FloatArray: TypeAlias = npt.NDArray[np.float64]


class ForwardModel(Protocol):
    """Map one known state frame to world-space landmark positions in metres."""

    def landmark_positions(self, frame: JointStateFrame) -> dict[str, FloatArray]:
        """Return ``landmark -> xyz`` world positions for a trajectory frame."""


@dataclass(frozen=True)
class SyntheticCamera:
    """Pinhole camera with world-to-camera extrinsics."""

    camera_id: str
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics

    def rotation_matrix(self) -> FloatArray:
        """Return a finite 3x3 world-to-camera rotation matrix."""
        rotation = np.asarray(self.extrinsics.rotation, dtype=np.float64)
        require(rotation.shape == (3, 3), "camera rotation must be 3x3")
        require(check_finite(rotation), "camera rotation must be finite")
        return rotation

    def translation_vector(self) -> FloatArray:
        """Return a finite 3-vector world-to-camera translation."""
        translation = np.asarray(self.extrinsics.translation, dtype=np.float64)
        require(translation.shape == (3,), "camera translation must be length 3")
        require(check_finite(translation), "camera translation must be finite")
        return translation


@dataclass(frozen=True)
class NoiseModel:
    """Pixel-space Gaussian observation noise."""

    sigma_px: float = 0.0

    def __post_init__(self) -> None:
        require(np.isfinite(self.sigma_px), "sigma_px must be finite")
        require(self.sigma_px >= 0.0, "sigma_px must be non-negative")


@dataclass(frozen=True)
class ObservationPolicy:
    """Synthetic visibility controls."""

    dropout_probability: float = 0.0
    occlusion_radius_m: float = 0.0
    occluder_centers_m: tuple[tuple[float, float, float], ...] = ()
    min_depth_m: float = 1.0e-6

    def __post_init__(self) -> None:
        require(
            0.0 <= self.dropout_probability <= 1.0,
            "dropout_probability must be in [0, 1]",
        )
        require(self.occlusion_radius_m >= 0.0, "occlusion_radius_m >= 0 required")
        require(self.min_depth_m > 0.0, "min_depth_m must be positive")
        centers = np.asarray(self.occluder_centers_m, dtype=np.float64)
        if centers.size:
            require(centers.ndim == 2 and centers.shape[1] == 3, "occluders are xyz")
            require(check_finite(centers), "occluder centers must be finite")


@dataclass(frozen=True)
class ProjectionRecord:
    """One projected landmark before conversion to CIR keypoints."""

    camera_id: str
    frame_index: int
    timestamp: float
    name: str
    x_px: float
    y_px: float
    depth_m: float
    confidence: float
    occluded: bool
    dropped: bool


@dataclass(frozen=True)
class GroundTruthRigResult:
    """Synthetic observations plus the underlying world-space truth."""

    observations_by_camera: dict[str, KeypointSequence]
    ground_truth_markers: MarkerTrajectory
    projection_records: tuple[ProjectionRecord, ...]


class SkeletonRigForwardModel:
    """Deterministic FK for CIR ``SkeletonRig`` fixtures.

    Each joint state entry rotates one local offset about the first declared
    joint axis. This lightweight model is intentionally simple; production
    engine adapters should implement :class:`ForwardModel` directly.
    """

    def __init__(self, skeleton: SkeletonRig) -> None:
        require(skeleton.num_joints > 0, "skeleton must contain joints")
        self._skeleton = skeleton
        self._dof_offsets = self._build_dof_offsets(skeleton)

    def landmark_positions(self, frame: JointStateFrame) -> dict[str, FloatArray]:
        """Return one point per skeleton joint in world coordinates."""
        q = np.asarray(frame.q, dtype=np.float64)
        require(q.shape == (self._skeleton.num_dofs,), "frame q length mismatch")
        require(check_finite(q), "frame q must be finite")
        return self._forward_positions(q)

    @staticmethod
    def _build_dof_offsets(skeleton: SkeletonRig) -> dict[str, int]:
        offsets: dict[str, int] = {}
        cursor = 0
        for name, joint in skeleton.joints.items():
            offsets[name] = cursor
            cursor += len(joint.axes)
        return offsets

    def _forward_positions(self, q: FloatArray) -> dict[str, FloatArray]:
        positions: dict[str, FloatArray] = {}
        rotations: dict[str, FloatArray] = {}
        root_name = self._skeleton.root_joint
        for name in self._topological_names(root_name):
            joint = self._skeleton.joints[name]
            local_offset = np.asarray(joint.tpose_offset, dtype=np.float64)
            local_rotation = self._joint_rotation(name, joint.axes[0], q)
            if joint.parent is None:
                positions[name] = local_offset * self._skeleton.scale
                rotations[name] = local_rotation
                continue
            parent_position = positions[joint.parent]
            parent_rotation = rotations[joint.parent]
            positions[name] = parent_position + parent_rotation @ local_offset
            rotations[name] = parent_rotation @ local_rotation
        return positions

    def _topological_names(self, root_name: str) -> list[str]:
        ordered: list[str] = []
        pending = [root_name]
        while pending:
            name = pending.pop(0)
            ordered.append(name)
            pending.extend(self._skeleton.joints[name].children)
        return ordered

    def _joint_rotation(
        self, joint_name: str, axis_name: str, q: FloatArray
    ) -> FloatArray:
        angle = q[self._dof_offsets[joint_name]]
        axis = _axis_vector(axis_name)
        return _axis_angle_rotation(axis, float(angle))


class SyntheticObservationRig:
    """Generate reproducible multi-camera observations from known motion."""

    def __init__(
        self,
        cameras: tuple[SyntheticCamera, ...],
        forward_model: ForwardModel,
        *,
        noise: NoiseModel | None = None,
        policy: ObservationPolicy | None = None,
        seed: int | None = None,
    ) -> None:
        require(len(cameras) > 0, "at least one camera is required")
        require(len({c.camera_id for c in cameras}) == len(cameras), "unique cameras")
        self._cameras = cameras
        self._forward_model = forward_model
        self._noise = noise or NoiseModel()
        self._policy = policy or ObservationPolicy()
        self._rng = np.random.default_rng(seed)

    @classmethod
    def from_calibration(
        cls,
        calibration: Calibration,
        forward_model: ForwardModel,
        *,
        noise: NoiseModel | None = None,
        policy: ObservationPolicy | None = None,
        seed: int | None = None,
    ) -> SyntheticObservationRig:
        """Build cameras from a CIR calibration payload."""
        cameras = tuple(
            _camera_from_payload(k, v) for k, v in calibration.cameras.items()
        )
        return cls(cameras, forward_model, noise=noise, policy=policy, seed=seed)

    def generate(self, trajectory: JointTrajectory) -> GroundTruthRigResult:
        """Project every known trajectory frame into each synthetic camera."""
        require(trajectory.num_frames > 0, "trajectory must contain frames")
        marker_frames: list[MarkerFrame] = []
        keypoint_frames: dict[str, list[KeypointFrame]] = {
            camera.camera_id: [] for camera in self._cameras
        }
        records: list[ProjectionRecord] = []
        for frame_number, frame in enumerate(trajectory.frames):
            landmarks = self._forward_model.landmark_positions(frame)
            marker_frames.append(_marker_frame(frame, landmarks))
            for camera in self._cameras:
                projected = self._project_frame(camera, frame_number, frame, landmarks)
                records.extend(projected)
                keypoint_frames[camera.camera_id].append(
                    _keypoint_frame(frame, projected)
                )
        sequences = {
            camera_id: KeypointSequence(
                id=f"{trajectory.id}-{camera_id}-synthetic-2d",
                frames=frames,
                metadata={"source": "synthetic-ground-truth", "issue": 6790},
            )
            for camera_id, frames in keypoint_frames.items()
        }
        truth = MarkerTrajectory(
            id=f"{trajectory.id}-ground-truth-markers",
            frames=marker_frames,
            metadata={"source": "synthetic-ground-truth", "issue": 6790},
        )
        return GroundTruthRigResult(sequences, truth, tuple(records))

    def _project_frame(
        self,
        camera: SyntheticCamera,
        frame_index: int,
        frame: JointStateFrame,
        landmarks: dict[str, FloatArray],
    ) -> list[ProjectionRecord]:
        records: list[ProjectionRecord] = []
        for name, xyz_world in landmarks.items():
            x_px, y_px, depth_m = project_world_point(camera, xyz_world)
            occluded = self._is_occluded(np.asarray(xyz_world, dtype=np.float64))
            behind_camera = depth_m <= self._policy.min_depth_m
            dropped = bool(self._rng.random() < self._policy.dropout_probability)
            if self._noise.sigma_px:
                x_px += float(self._rng.normal(0.0, self._noise.sigma_px))
                y_px += float(self._rng.normal(0.0, self._noise.sigma_px))
            invisible = occluded or dropped or behind_camera
            records.append(
                ProjectionRecord(
                    camera_id=camera.camera_id,
                    frame_index=frame_index,
                    timestamp=frame.timestamp,
                    name=name,
                    x_px=x_px,
                    y_px=y_px,
                    depth_m=depth_m,
                    confidence=0.0 if invisible else 1.0,
                    occluded=occluded or behind_camera,
                    dropped=dropped,
                )
            )
        return records

    def _is_occluded(self, xyz_world: FloatArray) -> bool:
        if self._policy.occlusion_radius_m <= 0.0:
            return False
        for center in self._policy.occluder_centers_m:
            delta = xyz_world - np.asarray(center, dtype=np.float64)
            if float(np.linalg.norm(delta)) <= self._policy.occlusion_radius_m:
                return True
        return False


def project_world_point(
    camera: SyntheticCamera, xyz_world: npt.ArrayLike
) -> tuple[float, float, float]:
    """Project one finite world-space point to pixel coordinates."""
    xyz = np.asarray(xyz_world, dtype=np.float64)
    require(xyz.shape == (3,), "xyz_world must be a 3-vector")
    require(check_finite(xyz), "xyz_world must be finite")
    xyz_camera = camera.rotation_matrix() @ xyz + camera.translation_vector()
    depth = float(xyz_camera[2])
    require(np.isfinite(depth), "projected depth must be finite")
    z = depth if abs(depth) > 1.0e-12 else np.copysign(1.0e-12, depth or 1.0)
    x_norm = float(xyz_camera[0] / z)
    y_norm = float(xyz_camera[1] / z)
    intr = camera.intrinsics
    radius_2 = x_norm * x_norm + y_norm * y_norm
    radial = 1.0 + intr.k1 * radius_2 + intr.k2 * radius_2 * radius_2
    x_dist = x_norm * radial + 2.0 * intr.p1 * x_norm * y_norm
    x_dist += intr.p2 * (radius_2 + 2.0 * x_norm * x_norm)
    y_dist = y_norm * radial + intr.p1 * (radius_2 + 2.0 * y_norm * y_norm)
    y_dist += 2.0 * intr.p2 * x_norm * y_norm
    return intr.fx * x_dist + intr.cx, intr.fy * y_dist + intr.cy, depth


def _camera_from_payload(camera_id: str, payload: dict[str, object]) -> SyntheticCamera:
    intrinsics = CameraIntrinsics.model_validate(payload["intrinsics"])
    extrinsics_payload = payload.get("extrinsics", {})
    extrinsics = CameraExtrinsics.model_validate(extrinsics_payload)
    return SyntheticCamera(camera_id, intrinsics, extrinsics)


def _marker_frame(
    frame: JointStateFrame, landmarks: dict[str, FloatArray]
) -> MarkerFrame:
    markers = {
        name: Marker(name=name, x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))
        for name, xyz in landmarks.items()
    }
    return MarkerFrame(
        timestamp=frame.timestamp,
        markers=markers,
        frame_index=frame.frame_index,
    )


def _keypoint_frame(
    frame: JointStateFrame, projected: list[ProjectionRecord]
) -> KeypointFrame:
    keypoints = [
        Keypoint(
            x=record.x_px,
            y=record.y_px,
            confidence=record.confidence,
            name=record.name,
        )
        for record in projected
        if not record.dropped
    ]
    return KeypointFrame(
        timestamp=frame.timestamp,
        keypoints=keypoints or [Keypoint(x=0.0, y=0.0, confidence=0.0, name="empty")],
        schema_name="custom",
        frame_index=frame.frame_index,
    )


def _axis_vector(axis_name: str) -> FloatArray:
    sign = -1.0 if axis_name.startswith("-") else 1.0
    axis = axis_name[-1]
    if axis == "X":
        return np.asarray([sign, 0.0, 0.0], dtype=np.float64)
    if axis == "Y":
        return np.asarray([0.0, sign, 0.0], dtype=np.float64)
    if axis == "Z":
        return np.asarray([0.0, 0.0, sign], dtype=np.float64)
    raise ValueError(f"unsupported axis {axis_name!r}")


def _axis_angle_rotation(axis: FloatArray, angle: float) -> FloatArray:
    x, y, z = axis / np.linalg.norm(axis)
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    one_c = 1.0 - c
    return np.asarray(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )
