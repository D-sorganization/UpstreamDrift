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

The canonical pose convention is documented in
`docs/adr/0012-canonical-pose-interchange.md`. The additive dynamic-state
surface is documented in `docs/conventions/canonical-v2.md` and
`docs/adr/0026-canonical-dynamic-state-v2.md`.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

# Numeric consumers must not import application reference-pose services.
if TYPE_CHECKING:
    from src.shared.python.pose_interchange.canonical import (
        CONVENTION_TAG,
        CanonicalPose,
        canonical_from_reference_setup,
        canonical_zero_pose,
    )
    from src.shared.python.pose_interchange.canonical_state import (
        CONVENTION_TAG_V2,
        CanonicalState,
        canonical_state_zero,
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
        euler_xyz_deg_to_quat_wxyz,
        inverse_se3,
        quat_exp,
        quat_log,
        quat_to_matrix,
        se3_from_xyz_xyz_deg,
        se3_to_xyz_xyz_deg,
    )

    from .frame_transport import FixedFrameTransport
    from .joint_chart import SerialRotationChart, SingularChartError
    from .native_joint_state import (
        NativeJointStateAdapter,
        NativeManifoldState,
        NativeRotationGroup,
        RotationState,
    )

    from .native_motion_sequence import (
        NativeMotionSequence,
        export_native_motion,
        restore_native_motion,
    )

    from .native_motion_io import (
        NativeMotionDocument,
        load_native_motion,
        save_native_motion,
    )

_EXPORT_MODULES = {
    "CONVENTION_TAG": "src.shared.python.pose_interchange.canonical",
    "CanonicalPose": "src.shared.python.pose_interchange.canonical",
    "canonical_from_reference_setup": "src.shared.python.pose_interchange.canonical",
    "canonical_zero_pose": "src.shared.python.pose_interchange.canonical",
    "CONVENTION_TAG_V2": "src.shared.python.pose_interchange.canonical_state",
    "CanonicalState": "src.shared.python.pose_interchange.canonical_state",
    "canonical_state_zero": "src.shared.python.pose_interchange.canonical_state",
    "CapabilityError": "src.shared.python.pose_interchange.live_kinematics",
    "LiveKinematicsService": "src.shared.python.pose_interchange.live_kinematics",
    "ServiceCapabilities": "src.shared.python.pose_interchange.live_kinematics",
    "JointSlot": "src.shared.python.pose_interchange.protocol",
    "PoseConventionAdapter": "src.shared.python.pose_interchange.protocol",
    "compose_se3": "src.shared.python.pose_interchange.se3",
    "euler_xyz_deg_to_quat_wxyz": "src.shared.python.pose_interchange.se3",
    "inverse_se3": "src.shared.python.pose_interchange.se3",
    "quat_exp": "src.shared.python.pose_interchange.se3",
    "quat_log": "src.shared.python.pose_interchange.se3",
    "quat_to_matrix": "src.shared.python.pose_interchange.se3",
    "se3_from_xyz_xyz_deg": "src.shared.python.pose_interchange.se3",
    "se3_to_xyz_xyz_deg": "src.shared.python.pose_interchange.se3",
    "FixedFrameTransport": ".frame_transport",
    "SerialRotationChart": ".joint_chart",
    "SingularChartError": ".joint_chart",
    "NativeJointStateAdapter": ".native_joint_state",
    "NativeManifoldState": ".native_joint_state",
    "NativeRotationGroup": ".native_joint_state",
    "RotationState": ".native_joint_state",
    "NativeMotionSequence": ".native_motion_sequence",
    "export_native_motion": ".native_motion_sequence",
    "restore_native_motion": ".native_motion_sequence",
    "NativeMotionDocument": ".native_motion_io",
    "load_native_motion": ".native_motion_io",
    "save_native_motion": ".native_motion_io",
}


def __getattr__(name: str) -> Any:
    """Load the owning provider only when its public export is requested."""
    module = _EXPORT_MODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


# Public API version (SemVer MAJOR.MINOR.PATCH).
#
# Bump rules (per issue #5917, ADR-0012):
# - MAJOR: breaking change to ``CanonicalPose`` fields, the
#   ``PoseConventionAdapter`` protocol, or the canonical convention
#   itself (introduce ``canonical-v2`` instead when possible).
# - MINOR: backwards-compatible additions (new helpers, new adapter
#   methods with defaults, new optional fields).
# - PATCH: bug fixes that do not change the public surface.
#
# 2.0.0 (CC-2, ADR-0026): adds the ``canonical-v2`` dynamic state surface
# (``CanonicalState`` + manifold ops). The ``canonical-v1`` pose API below is
# unchanged and remains valid for pose-only callers.
__version__ = "2.1.0"

# Canonical *pose* (v1) schema version. Mirrors ``CONVENTION_TAG`` for
# downstream consumers that prefer numeric comparison; the string tag
# (``canonical-v1``) remains the on-the-wire identifier. The canonical-v2
# dynamic state carries its own ``CONVENTION_TAG_V2`` ("canonical-v2").
SCHEMA_VERSION = "1.0.0"

__all__ = [
    "CONVENTION_TAG",
    "CONVENTION_TAG_V2",
    "CanonicalPose",
    "CanonicalState",
    "FixedFrameTransport",
    "SerialRotationChart",
    "NativeJointStateAdapter",
    "NativeManifoldState",
    "NativeRotationGroup",
    "RotationState",
    "NativeMotionSequence",
    "export_native_motion",
    "restore_native_motion",
    "NativeMotionDocument",
    "load_native_motion",
    "save_native_motion",
    "SingularChartError",
    "CapabilityError",
    "JointSlot",
    "LiveKinematicsService",
    "PoseConventionAdapter",
    "SCHEMA_VERSION",
    "ServiceCapabilities",
    "__version__",
    "canonical_from_reference_setup",
    "canonical_state_zero",
    "canonical_zero_pose",
    "compose_se3",
    "euler_xyz_deg_to_quat_wxyz",
    "inverse_se3",
    "quat_exp",
    "quat_log",
    "quat_to_matrix",
    "se3_from_xyz_xyz_deg",
    "se3_to_xyz_xyz_deg",
]
