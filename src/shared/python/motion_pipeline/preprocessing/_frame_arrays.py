"""Shared frame-to-array helpers for motion preprocessing."""

from __future__ import annotations

import numpy as np

from ..contracts import Keypoint, KeypointFrame, Marker, MarkerFrame


def keypoints_to_array(frames: list[KeypointFrame]) -> np.ndarray:
    """Convert keypoint frames to ``(frames, keypoints, xyz)`` coordinates."""
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


def markers_to_array(frames: list[MarkerFrame]) -> np.ndarray:
    """Convert marker frames to ``(frames, markers, xyz)`` coordinates."""
    if not frames:
        return np.array([])

    num_frames = len(frames)
    marker_names = list(frames[0].markers.keys())
    num_markers = len(marker_names)
    data = np.zeros((num_frames, num_markers, 3))

    for i, frame in enumerate(frames):
        for j, name in enumerate(marker_names):
            if name in frame.markers:
                marker = frame.markers[name]
                data[i, j, 0] = marker.x
                data[i, j, 1] = marker.y
                data[i, j, 2] = marker.z

    return data


def estimate_fps(frames: list) -> float:
    """Estimate FPS from frame timestamps."""
    if len(frames) < 2:
        return 30.0

    timestamps = [frame.timestamp for frame in frames]
    dt = float(np.mean(np.diff(timestamps)))

    if dt <= 0:
        return 30.0

    return 1.0 / dt


def array_to_keypoint_frames(
    frames: list[KeypointFrame],
    data: np.ndarray,
) -> list[KeypointFrame]:
    """Convert array data back to keypoint frames."""
    new_frames: list[KeypointFrame] = []

    for i, frame in enumerate(frames):
        new_keypoints = []
        for j, kp in enumerate(frame.keypoints):
            new_keypoints.append(
                Keypoint(
                    x=float(data[i, j, 0]),
                    y=float(data[i, j, 1]),
                    z=float(data[i, j, 2]) if kp.z is not None else None,
                    confidence=kp.confidence,
                    name=kp.name,
                )
            )

        new_frames.append(
            KeypointFrame(
                timestamp=frame.timestamp,
                keypoints=new_keypoints,
                schema_name=frame.schema_name,
                frame_index=frame.frame_index,
            )
        )

    return new_frames


def array_to_marker_frames(
    frames: list[MarkerFrame],
    data: np.ndarray,
) -> list[MarkerFrame]:
    """Convert array data back to marker frames."""
    new_frames: list[MarkerFrame] = []
    marker_names = list(frames[0].markers.keys())

    for i, frame in enumerate(frames):
        new_markers = {}
        for j, name in enumerate(marker_names):
            if name in frame.markers:
                marker = frame.markers[name]
                new_markers[name] = Marker(
                    name=name,
                    x=float(data[i, j, 0]),
                    y=float(data[i, j, 1]),
                    z=float(data[i, j, 2]),
                    residual=marker.residual,
                    occluded=marker.occluded,
                )

        new_frames.append(
            MarkerFrame(
                timestamp=frame.timestamp,
                markers=new_markers,
                frame_index=frame.frame_index,
            )
        )

    return new_frames


def array_to_keypoint_frames_at_timestamps(
    seq,
    data: np.ndarray,
    timestamps: np.ndarray,
) -> list[KeypointFrame]:
    """Convert array data to keypoint frames at specified timestamps."""
    new_frames: list[KeypointFrame] = []

    for i, timestamp in enumerate(timestamps):
        new_keypoints = []
        for j in range(data.shape[1]):
            kp = (
                seq.frames[0].keypoints[j] if j < len(seq.frames[0].keypoints) else None
            )
            if kp is not None:
                new_keypoints.append(
                    Keypoint(
                        x=float(data[i, j, 0]),
                        y=float(data[i, j, 1]),
                        z=float(data[i, j, 2]) if kp.z is not None else None,
                        confidence=kp.confidence,
                        name=kp.name,
                    )
                )

        new_frames.append(
            KeypointFrame(
                timestamp=float(timestamp),
                keypoints=new_keypoints,
                schema_name=seq.frames[0].schema_name,
                frame_index=i,
            )
        )

    return new_frames


def array_to_marker_frames_at_timestamps(
    traj,
    data: np.ndarray,
    timestamps: np.ndarray,
) -> list[MarkerFrame]:
    """Convert array data to marker frames at specified timestamps."""
    new_frames: list[MarkerFrame] = []
    marker_names = list(traj.frames[0].markers.keys())

    for i, timestamp in enumerate(timestamps):
        new_markers = {}
        for j, name in enumerate(marker_names):
            if name in traj.frames[0].markers:
                marker = traj.frames[0].markers[name]
                new_markers[name] = Marker(
                    name=name,
                    x=float(data[i, j, 0]),
                    y=float(data[i, j, 1]),
                    z=float(data[i, j, 2]),
                    residual=marker.residual,
                    occluded=marker.occluded,
                )

        new_frames.append(
            MarkerFrame(
                timestamp=float(timestamp),
                markers=new_markers,
                frame_index=i,
            )
        )

    return new_frames
