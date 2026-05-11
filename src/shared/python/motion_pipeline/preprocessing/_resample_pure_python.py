"""
Resampling for motion capture data.

Part of issue #4564. Frame-rate conversion with anti-aliasing.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..contracts import KeypointFrame, KeypointSequence, MarkerFrame, MarkerTrajectory


def resample(
    data: KeypointSequence | MarkerTrajectory,
    target_fps: float,
    source_fps: float | None = None,
) -> KeypointSequence | MarkerTrajectory:
    """
    Resample motion capture data to target frame rate.

    Args:
        data: Input keypoint sequence or marker trajectory
        target_fps: Target frame rate in Hz
        source_fps: Source frame rate (auto-detected if not provided)

    Returns:
        Resampled data at target_fps

    Raises:
        ValueError: If data type is unsupported
    """
    if isinstance(data, KeypointSequence):
        return _resample_keypoints(data, target_fps, source_fps)
    if isinstance(data, MarkerTrajectory):
        return _resample_markers(data, target_fps, source_fps)
    raise ValueError(f"Unsupported data type: {type(data)}")


def _resample_keypoints(
    seq: KeypointSequence,
    target_fps: float,
    source_fps: float | None,
) -> KeypointSequence:
    """Resample keypoint sequence to target FPS."""
    if len(seq.frames) < 2:
        return seq

    # Auto-detect source FPS
    if source_fps is None:
        source_fps = _estimate_fps(seq.frames)

    # Compute new timestamps
    duration = seq.frames[-1].timestamp - seq.frames[0].timestamp
    num_new_frames = int(duration * target_fps) + 1
    new_timestamps = np.linspace(
        seq.frames[0].timestamp, seq.frames[-1].timestamp, num_new_frames
    )

    # Extract data arrays
    data = _keypoints_to_array(seq.frames)
    timestamps = np.array([f.timestamp for f in seq.frames])

    # Resample each keypoint dimension
    resampled_data = np.zeros((num_new_frames, data.shape[1], 3))
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            resampled_data[:, i, j] = np.interp(
                new_timestamps, timestamps, data[:, i, j]
            )

    # Reconstruct frames
    new_frames = _array_to_keypoint_frames_at_timestamps(
        seq, resampled_data, new_timestamps
    )

    return KeypointSequence(
        id=seq.id,
        frames=new_frames,
        calibration=seq.calibration,
        metadata={
            **seq.metadata,
            "resampled": True,
            "source_fps": source_fps,
            "target_fps": target_fps,
        },
    )


def _resample_markers(
    traj: MarkerTrajectory,
    target_fps: float,
    source_fps: float | None,
) -> MarkerTrajectory:
    """Resample marker trajectory to target FPS."""
    if len(traj.frames) < 2:
        return traj

    # Auto-detect source FPS
    if source_fps is None:
        source_fps = _estimate_fps(traj.frames)

    # Compute new timestamps
    duration = traj.frames[-1].timestamp - traj.frames[0].timestamp
    num_new_frames = int(duration * target_fps) + 1
    new_timestamps = np.linspace(
        traj.frames[0].timestamp, traj.frames[-1].timestamp, num_new_frames
    )

    # Extract data arrays
    data = _markers_to_array(traj.frames)
    timestamps = np.array([f.timestamp for f in traj.frames])

    # Resample each marker dimension
    resampled_data = np.zeros((num_new_frames, data.shape[1], 3))
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            resampled_data[:, i, j] = np.interp(
                new_timestamps, timestamps, data[:, i, j]
            )

    # Reconstruct frames
    new_frames = _array_to_marker_frames_at_timestamps(
        traj, resampled_data, new_timestamps
    )

    return MarkerTrajectory(
        id=traj.id,
        frames=new_frames,
        calibration=traj.calibration,
        subject_id=traj.subject_id,
        metadata={
            **traj.metadata,
            "resampled": True,
            "source_fps": source_fps,
            "target_fps": target_fps,
        },
    )


def _estimate_fps(frames: list) -> float:
    """Estimate FPS from frame timestamps."""
    if len(frames) < 2:
        return 30.0

    timestamps = [f.timestamp for f in frames]
    dt = np.mean(np.diff(timestamps))

    if dt <= 0:
        return 30.0

    return 1.0 / dt


def _keypoints_to_array(frames: list[KeypointFrame]) -> np.ndarray:
    """Convert keypoint frames to array."""
    if not frames:
        return np.array([])

    num_frames = len(frames)
    num_keypoints = len(frames[0].keypoints)

    data = np.zeros((num_frames, num_keypoints, 3))

    for i, frame in enumerate(frames):
        for j, kp in enumerate(frame.keypoints):
            data[i, j, 0] = kp.x
            data[i, j, 1] = kp.y
            if kp.z is not None:
                data[i, j, 2] = kp.z

    return data


def _markers_to_array(frames: list[MarkerFrame]) -> np.ndarray:
    """Convert marker frames to array."""
    if not frames:
        return np.array([])

    num_frames = len(frames)
    marker_names = list(frames[0].markers.keys())
    num_markers = len(marker_names)

    data = np.zeros((num_frames, num_markers, 3))

    for i, frame in enumerate(frames):
        for j, name in enumerate(marker_names):
            if name in frame.markers:
                m = frame.markers[name]
                data[i, j, 0] = m.x
                data[i, j, 1] = m.y
                data[i, j, 2] = m.z

    return data


def _array_to_keypoint_frames_at_timestamps(
    seq: KeypointSequence,
    data: np.ndarray,
    timestamps: np.ndarray,
) -> list[KeypointFrame]:
    """Convert array back to keypoint frames at specified timestamps."""
    new_frames = []

    for i, ts in enumerate(timestamps):
        new_keypoints = []
        for j in range(data.shape[1]):
            kp = (
                seq.frames[0].keypoints[j] if j < len(seq.frames[0].keypoints) else None
            )
            if kp:
                new_kp = Keypoint(
                    x=data[i, j, 0],
                    y=data[i, j, 1],
                    z=data[i, j, 2] if kp.z is not None else None,
                    confidence=kp.confidence,
                    name=kp.name,
                )
                new_keypoints.append(new_kp)

        new_frames.append(
            KeypointFrame(
                timestamp=float(ts),
                keypoints=new_keypoints,
                schema_name=seq.frames[0].schema_name,
                frame_index=i,
            )
        )

    return new_frames


def _array_to_marker_frames_at_timestamps(
    traj: MarkerTrajectory,
    data: np.ndarray,
    timestamps: np.ndarray,
) -> list[MarkerFrame]:
    """Convert array back to marker frames at specified timestamps."""
    new_frames = []
    marker_names = list(traj.frames[0].markers.keys())

    for i, ts in enumerate(timestamps):
        new_markers = {}
        for j, name in enumerate(marker_names):
            if name in traj.frames[0].markers:
                m = traj.frames[0].markers[name]
                new_markers[name] = Marker(
                    name=name,
                    x=data[i, j, 0],
                    y=data[i, j, 1],
                    z=data[i, j, 2],
                    residual=m.residual,
                    occluded=m.occluded,
                )

        new_frames.append(
            MarkerFrame(
                timestamp=float(ts),
                markers=new_markers,
                frame_index=i,
            )
        )

    return new_frames


# Import for type hints
from ..contracts import Keypoint, Marker
