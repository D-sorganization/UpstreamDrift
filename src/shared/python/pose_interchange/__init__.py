"""Canonical pose interchange — engine-agnostic skeleton pose representation.

Foundation for the Pose Studio EPIC (#4895). Establishes a single
canonical pose convention so per-engine adapters do not have to
round-trip through every other engine's convention pairwise.

Public surface:

- :class:`CanonicalPose` — frozen dataclass holding pelvis SE(3) +
  per-joint angles in the canonical convention (intrinsic XYZ Euler in
  degrees, joint names matching :func:`reference_golfer_setup`).
- :class:`PoseConventionAdapter` — runtime-checkable :class:`Protocol`
  every engine adapter implements.
- :class:`JointSlot` — describes one joint's slot in an engine's
  ``q`` vector (for adapters that need layout metadata).
- :func:`canonical_zero_pose` — the all-zero canonical pose.
- :func:`canonical_from_reference_setup` — the canonical address pose
  derived from :func:`reference_golfer_setup`.

The canonical convention is documented in
`docs/adr/0012-canonical-pose-interchange.md`.
"""

from __future__ import annotations

from src.shared.python.pose_interchange.canonical import (
    CONVENTION_TAG,
    CanonicalPose,
    canonical_from_reference_setup,
    canonical_zero_pose,
)
from src.shared.python.pose_interchange.live_kinematics import (
    CapabilityError,
    LiveKinematicsService,
    ServiceCapabilities,
)
from src.shared.python.pose_interchange.protocol import (
    JointSlot,
    PoseConventionAdapter,
)
from src.shared.python.pose_interchange.se3 import (
    compose_se3,
    inverse_se3,
    se3_from_xyz_xyz_deg,
    se3_to_xyz_xyz_deg,
)

__all__ = [
    "CONVENTION_TAG",
    "CanonicalPose",
    "CapabilityError",
    "JointSlot",
    "LiveKinematicsService",
    "PoseConventionAdapter",
    "ServiceCapabilities",
    "canonical_from_reference_setup",
    "canonical_zero_pose",
    "compose_se3",
    "inverse_se3",
    "se3_from_xyz_xyz_deg",
    "se3_to_xyz_xyz_deg",
]
