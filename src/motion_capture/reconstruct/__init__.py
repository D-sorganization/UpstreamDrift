"""Multi-view reconstruction evaluation for the markerless rig (epic #9619).

This package is the *consumer-side* half of the self-calibrating pipeline
described in ``docs/motion_capture/self_calibrating_pipeline.md``: a pinhole
camera model that bridges to the ADR-0041 calibration records, a rigid
synthetic skeleton with fixed segment lengths, a scene renderer that produces
per-view observations (in the same schema ``motion_capture.rig ingest``
writes) together with ground truth, and the metrics every reconstruction
algorithm is judged by. Reference geometry (calibration solvers,
triangulation, bundle adjustment) is owned by Tools under ADR-0041; the
ownership of the golf-specific fitter is decided in #9630. Until then nothing
here reconstructs — it makes reconstruction *testable* (#9629).

Modules:

- :mod:`.cameras` — pinhole projection, ``look_at`` extrinsics, record bridge
- :mod:`.skeleton` — rigid 15-joint skeleton, forward kinematics, swing motion
- :mod:`.synthetic` — scene rendering with noise, occlusion and outliers
- :mod:`.metrics` — camera pose, bone length, joint position, outlier scores
"""

from .cameras import PinholeCamera, look_at
from .metrics import (
    bone_length_errors,
    camera_pose_error,
    joint_position_errors,
    outlier_flag_scores,
)
from .skeleton import RigidSkeleton, swing_trajectory
from .synthetic import RenderOptions, SyntheticScene, SyntheticTruth

__all__ = [
    "PinholeCamera",
    "RenderOptions",
    "RigidSkeleton",
    "SyntheticScene",
    "SyntheticTruth",
    "bone_length_errors",
    "camera_pose_error",
    "joint_position_errors",
    "look_at",
    "outlier_flag_scores",
    "swing_trajectory",
]
