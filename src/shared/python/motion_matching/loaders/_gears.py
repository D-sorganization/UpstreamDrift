"""Deprecated alias for :mod:`._marker_clusters` (issue #4480).

This shim preserves the old import path for one release.  New code must
import from :mod:`src.shared.python.motion_matching.loaders._marker_clusters`.
"""

from __future__ import annotations

import warnings

from ._marker_clusters import *  # noqa: F401,F403
from ._marker_clusters import (  # noqa: F401
    CLUBHEAD_CLUSTER,
    EXCLUDED_MARKERS,
    GRIP_CLUSTER,
    MAX_GAP_FRAMES,
    OCCLUDED_MARKERS,
    SENTINEL_MARKERS,
    ClusterClubPose,
    extract_cluster_club_pose,
    fill_short_gaps,
    has_marker_clusters,
    pose_from_cluster,
    y_up_to_z_up,
    y_up_to_z_up_rotation,
)

warnings.warn(
    "src.shared.python.motion_matching.loaders._gears is deprecated; "
    "import from src.shared.python.motion_matching.loaders._marker_clusters "
    "instead. The old module name will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

# Backwards-compat aliases for the renamed identifiers.
GearsClubPose = ClusterClubPose
is_gears_schema = has_marker_clusters
extract_gears_pose = extract_cluster_club_pose
