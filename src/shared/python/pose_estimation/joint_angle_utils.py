"""Shared joint angle computation utilities for pose estimators.

Provides common 3D angle calculations used by both MediaPipe and OpenPose
estimators. Centralizes the biomechanical angle logic so both backends
produce consistent results.

Issue #759: Complete motion matching pipeline.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


def _angle_between(
    v1: np.ndarray,
    v2: np.ndarray,
) -> float:
    """Compute angle between two 3D vectors in radians.

    Args:
        v1: First vector (3,)
        v2: Second vector (3,)

    Returns:
        Angle in radians, or NaN if either vector is zero-length.
    """
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return float("nan")
    cos_angle = np.dot(v1, v2) / (n1 * n2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.arccos(cos_angle))


def _compute_flexion(
    proximal: np.ndarray,
    joint: np.ndarray,
    distal: np.ndarray,
) -> float:
    """Compute flexion angle at a joint from three keypoints.

    The angle is measured between the vector from joint->proximal and
    joint->distal.  A straight limb gives ~pi radians.

    Args:
        proximal: Position of the proximal landmark (3,)
        joint: Position of the joint landmark (3,)
        distal: Position of the distal landmark (3,)

    Returns:
        Flexion angle in radians.
    """
    return _angle_between(proximal - joint, distal - joint)


def compute_joint_angles(
    keypoints: dict[str, np.ndarray],
    keypoint_mapping: dict[str, str] | None = None,
) -> dict[str, float]:
    """Compute biomechanical joint angles from 3D keypoint positions.

    Supports both MediaPipe and OpenPose keypoint naming conventions via
    the ``keypoint_mapping`` parameter.

    The following angles are computed when sufficient keypoints exist:
    - right/left elbow flexion
    - right/left shoulder flexion
    - right/left hip flexion
    - right/left knee flexion
    - right/left ankle dorsiflexion
    - trunk rotation (X-factor between shoulder and hip lines)

    Args:
        keypoints: Mapping of keypoint name to 3D position array.
        keypoint_mapping: Optional mapping from canonical names
            (e.g. ``right_shoulder``) to the actual keypoint names
            used in ``keypoints``.  If *None*, the canonical names
            are used directly (MediaPipe convention).

    Returns:
        Dictionary of joint angle name -> angle in radians.
    """
    angles: dict[str, float] = {}

    def _get(name: str) -> np.ndarray | None:
        key = keypoint_mapping.get(name, name) if keypoint_mapping else name
        kp = keypoints.get(key)
        if kp is None:
            return None
        return np.asarray(kp, dtype=float)

    # Bilateral flexion angles (proximal-joint-distal triplets)
    _JOINT_DEFS: list[tuple[str, str, str, str]] = [
        ("elbow_flexion", "shoulder", "elbow", "wrist"),
        ("shoulder_flexion", "hip", "shoulder", "elbow"),
        ("hip_flexion", "shoulder", "hip", "knee"),
        ("knee_flexion", "hip", "knee", "ankle"),
    ]
    _compute_bilateral_flexion(angles, _get, _JOINT_DEFS)

    # Trunk rotation (X-factor)
    _compute_trunk_rotation(angles, _get)

    return angles


def _compute_bilateral_flexion(
    angles: dict[str, float],
    getter: Callable[[str], np.ndarray | None],
    joint_defs: list[tuple[str, str, str, str]],
) -> None:
    """Compute bilateral (right/left) flexion angles for each joint definition."""
    for angle_name, proximal_name, joint_name, distal_name in joint_defs:
        for side in ("right", "left"):
            proximal = getter(f"{side}_{proximal_name}")
            joint = getter(f"{side}_{joint_name}")
            distal = getter(f"{side}_{distal_name}")
            if proximal is not None and joint is not None and distal is not None:
                angle = _compute_flexion(proximal, joint, distal)
                if not np.isnan(angle):
                    angles[f"{side}_{angle_name}"] = angle


def _compute_trunk_rotation(
    angles: dict[str, float],
    getter: Callable[[str], np.ndarray | None],
) -> None:
    """Compute trunk rotation (X-factor) from shoulder and hip lines."""
    l_shoulder = getter("left_shoulder")
    r_shoulder = getter("right_shoulder")
    l_hip = getter("left_hip")
    r_hip = getter("right_hip")

    if l_shoulder is None or r_shoulder is None or l_hip is None or r_hip is None:
        return

    shoulder_vec = r_shoulder[:2] - l_shoulder[:2]
    hip_vec = r_hip[:2] - l_hip[:2]
    n1 = np.linalg.norm(shoulder_vec)
    n2 = np.linalg.norm(hip_vec)
    if n1 > 1e-12 and n2 > 1e-12:
        cos_a = np.dot(shoulder_vec, hip_vec) / (n1 * n2)
        cos_a = np.clip(cos_a, -1.0, 1.0)
        angles["trunk_rotation"] = float(np.arccos(cos_a))


# ------------------------------------------------------------------
# OpenPose BODY_25 -> canonical name mapping
# ------------------------------------------------------------------
OPENPOSE_TO_CANONICAL: dict[str, str] = {
    "right_shoulder": "RShoulder",
    "left_shoulder": "LShoulder",
    "right_elbow": "RElbow",
    "left_elbow": "LElbow",
    "right_wrist": "RWrist",
    "left_wrist": "LWrist",
    "right_hip": "RHip",
    "left_hip": "LHip",
    "right_knee": "RKnee",
    "left_knee": "LKnee",
    "right_ankle": "RAnkle",
    "left_ankle": "LAnkle",
}
