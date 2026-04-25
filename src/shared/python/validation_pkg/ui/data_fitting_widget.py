"""GUI-facing pose-to-marker conversion for the A3 fitting pipeline.

Extracted from data_fitting.py (issue #3060).

This module contains helpers consumed by gui_pkg (video_pose_pipeline).
No Qt/widget dependencies are required; the name reflects its GUI-facing role.
"""

from __future__ import annotations

import numpy as np


def convert_poses_to_markers(
    pose_keypoints: np.ndarray,
    keypoint_names: list[str],
    target_markers: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Convert pose estimation keypoints to biomechanical marker format.

    Maps OpenPose/MediaPipe keypoints to standard marker positions.

    Args:
        pose_keypoints: Keypoint positions [N x 3] or [N x 2]
        keypoint_names: Names of each keypoint
        target_markers: Optional list of target marker names to output

    Returns:
        Tuple of (marker_positions [M x 3], marker_names [M]).
    """
    if pose_keypoints is None:
        raise ValueError("pose_keypoints must be provided")

    # Standard mapping from pose estimation to biomechanical markers
    pose_to_marker_map = {
        # MediaPipe / OpenPose keypoint names -> Biomechanics marker names
        "left_shoulder": "LSHO",
        "right_shoulder": "RSHO",
        "left_elbow": "LELB",
        "right_elbow": "RELB",
        "left_wrist": "LWRI",
        "right_wrist": "RWRI",
        "left_hip": "LASI",
        "right_hip": "RASI",
        "left_knee": "LKNE",
        "right_knee": "RKNE",
        "left_ankle": "LANK",
        "right_ankle": "RANK",
        # Additional mappings
        "nose": "NOSE",
        "left_ear": "LEAR",
        "right_ear": "REAR",
    }

    # Ensure 3D coordinates
    if pose_keypoints.shape[1] == 2:
        # Add zero z-coordinate for 2D keypoints
        pose_keypoints = np.hstack([pose_keypoints, np.zeros((len(pose_keypoints), 1))])

    # Filter and reorder keypoints
    marker_positions = []
    marker_names = []

    for i, keypoint_name in enumerate(keypoint_names):
        marker_name = pose_to_marker_map.get(keypoint_name.lower())

        if marker_name is None:
            continue

        if target_markers is not None and marker_name not in target_markers:
            continue

        marker_positions.append(pose_keypoints[i])
        marker_names.append(marker_name)

    return np.array(marker_positions), marker_names
