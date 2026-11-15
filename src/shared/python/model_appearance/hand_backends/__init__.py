"""Shared hand posing. Visual backends keep their own registries.

This package poses a kinematic chain onto a grip and reports contact quality.
It does not register a visual backend. Sibling hand PRs own ``registry.py``;
merging them will conflict on this ``__init__`` and should keep both exports.
"""

from src.shared.python.model_appearance.hand_backends.chain import (
    JointDescription,
    KinematicChain,
    build_default_chain,
    curl_sign_for,
    joint_name,
)
from src.shared.python.model_appearance.hand_backends.gb2_adapter import (
    GB2_BRANCH,
    GB2_ISSUE,
    gb2_curl_status,
)
from src.shared.python.model_appearance.hand_backends.grip_solid import (
    GripSolid,
    default_grip,
)
from src.shared.python.model_appearance.hand_backends.metrics import (
    CONTACT_QUALITY_METRIC_KEYS,
    SCHEMA,
    contact_quality_metrics,
    metrics_from_tables,
    palm_contact_area_m2,
)
from src.shared.python.model_appearance.hand_backends.posing import (
    pose_chain,
    pose_grip,
)
from src.shared.python.model_appearance.hand_backends.report import (
    FingerOverride,
    FlatPlacement,
    GripPoseSolution,
    HandPose,
    SegmentPose,
)
from src.shared.python.model_appearance.hand_backends.settings import (
    DEFAULT_SETTINGS_PATH,
    PoseSettings,
    load_pose_settings,
    settings_from_dict,
)
from src.shared.python.model_appearance.hand_backends.solve import surface_gap_m

__all__ = [
    "CONTACT_QUALITY_METRIC_KEYS",
    "DEFAULT_SETTINGS_PATH",
    "GB2_BRANCH",
    "GB2_ISSUE",
    "SCHEMA",
    "FingerOverride",
    "FlatPlacement",
    "GripPoseSolution",
    "GripSolid",
    "HandPose",
    "JointDescription",
    "KinematicChain",
    "PoseSettings",
    "SegmentPose",
    "build_default_chain",
    "contact_quality_metrics",
    "curl_sign_for",
    "default_grip",
    "gb2_curl_status",
    "joint_name",
    "load_pose_settings",
    "metrics_from_tables",
    "palm_contact_area_m2",
    "pose_chain",
    "pose_grip",
    "settings_from_dict",
    "surface_gap_m",
]
