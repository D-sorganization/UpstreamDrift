"""
Resampling for motion capture data.

Part of issue #4564. Frame-rate conversion with anti-aliasing.
"""

from __future__ import annotations


import numpy as np

from ..contracts import KeypointSequence, MarkerTrajectory
from ._frame_arrays import (
    array_to_keypoint_frames_at_timestamps as _array_to_keypoint_frames_at_timestamps,
    array_to_marker_frames_at_timestamps as _array_to_marker_frames_at_timestamps,
    estimate_fps as _estimate_fps,
    keypoints_to_array as _keypoints_to_array,
    markers_to_array as _markers_to_array,
    vectorized_interp_axes as _vectorized_interp_axes,
)


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
    resampled_data = _vectorized_interp_axes(new_timestamps, timestamps, data)

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
    resampled_data = _vectorized_interp_axes(new_timestamps, timestamps, data)

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
