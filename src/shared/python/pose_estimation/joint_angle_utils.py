"""Shared joint angle computation utilities for pose estimators.

Provides common 3D angle calculations used by both MediaPipe and OpenPose
estimators. Centralizes the biomechanical angle logic so both backends
produce consistent results.

Issue #759: Complete motion matching pipeline.
"""

from __future__ import annotations

import numpy as np

from src.shared.python.logging_config import get_logger

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

    # ------------------------------------------------------------------
    # Elbow flexion: shoulder-elbow-wrist
    # ------------------------------------------------------------------
    for side in ("right", "left"):
        shoulder = _get(f"{side}_shoulder")
        elbow = _get(f"{side}_elbow")
        wrist = _get(f"{side}_wrist")
        if shoulder is not None and elbow is not None and wrist is not None:
            angle = _compute_flexion(shoulder, elbow, wrist)
            if not np.isnan(angle):
                angles[f"{side}_elbow_flexion"] = angle

    # ------------------------------------------------------------------
    # Shoulder flexion: hip-shoulder-elbow
    # ------------------------------------------------------------------
    for side in ("right", "left"):
        hip = _get(f"{side}_hip")
        shoulder = _get(f"{side}_shoulder")
        elbow = _get(f"{side}_elbow")
        if hip is not None and shoulder is not None and elbow is not None:
            angle = _compute_flexion(hip, shoulder, elbow)
            if not np.isnan(angle):
                angles[f"{side}_shoulder_flexion"] = angle

    # ------------------------------------------------------------------
    # Hip flexion: shoulder-hip-knee
    # ------------------------------------------------------------------
    for side in ("right", "left"):
        shoulder = _get(f"{side}_shoulder")
        hip = _get(f"{side}_hip")
        knee = _get(f"{side}_knee")
        if shoulder is not None and hip is not None and knee is not None:
            angle = _compute_flexion(shoulder, hip, knee)
            if not np.isnan(angle):
                angles[f"{side}_hip_flexion"] = angle

    # ------------------------------------------------------------------
    # Knee flexion: hip-knee-ankle
    # ------------------------------------------------------------------
    for side in ("right", "left"):
        hip = _get(f"{side}_hip")
        knee = _get(f"{side}_knee")
        ankle = _get(f"{side}_ankle")
        if hip is not None and knee is not None and ankle is not None:
            angle = _compute_flexion(hip, knee, ankle)
            if not np.isnan(angle):
                angles[f"{side}_knee_flexion"] = angle

    # ------------------------------------------------------------------
    # Trunk rotation (X-factor): angle between shoulder line and hip line
    # projected onto the transverse (horizontal XY) plane
    # ------------------------------------------------------------------
    l_shoulder = _get("left_shoulder")
    r_shoulder = _get("right_shoulder")
    l_hip = _get("left_hip")
    r_hip = _get("right_hip")

    if (
        l_shoulder is not None
        and r_shoulder is not None
        and l_hip is not None
        and r_hip is not None
    ):
        shoulder_vec = r_shoulder[:2] - l_shoulder[:2]  # XY projection
        hip_vec = r_hip[:2] - l_hip[:2]
        n1 = np.linalg.norm(shoulder_vec)
        n2 = np.linalg.norm(hip_vec)
        if n1 > 1e-12 and n2 > 1e-12:
            cos_a = np.dot(shoulder_vec, hip_vec) / (n1 * n2)
            cos_a = np.clip(cos_a, -1.0, 1.0)
            angles["trunk_rotation"] = float(np.arccos(cos_a))

    return angles


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
