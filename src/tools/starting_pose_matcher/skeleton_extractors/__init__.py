"""Engine-specific skeleton extractors for starting-pose initialisation.

Each module provides a ``extract_skeleton(source) -> Skeleton`` function
conforming to the abstract ``ISkeletonExtractor`` contract.

Modules:
    drake: Skeleton extractor for Drake rigid-body systems.
    mediapipe: Skeleton extractor using Google MediaPipe landmarks.
    mujoco: Skeleton extractor for MuJoCo models.
    openpose: Skeleton extractor using OpenPose keypoints.
    opensim: Skeleton extractor for OpenSim musculoskeletal models.
    pinocchio: Skeleton extractor for Pinocchio URDF models.
"""

__all__ = [
    "drake",
    "mediapipe",
    "mujoco",
    "openpose",
    "opensim",
    "pinocchio",
]
