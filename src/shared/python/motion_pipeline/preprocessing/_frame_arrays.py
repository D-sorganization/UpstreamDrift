"""Shared frame-to-array helpers for motion preprocessing."""

from __future__ import annotations

import numpy as np

from ..contracts import KeypointFrame, MarkerFrame


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
