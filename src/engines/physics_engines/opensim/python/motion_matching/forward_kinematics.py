"""Forward-kinematics extraction for the OpenSim golf humanoid.

.. note:: Collapsed in issue #6938.

    This module used to carry a **second, independent** copy of the OpenSim
    FK extractor (grip / clubhead world poses + quaternion conversion). That
    copy had drifted: it resolved the grip via ``/bodyset/Club/club_grip_offset``
    and the clubhead via ``/bodyset/Club/club_head_offset`` — frame paths that
    do **not** exist in the shipped ``golf_humanoid.osim`` (the grip frame is
    ``hand_r_grip_offset`` under the ``hand_r_to_club`` weld joint; this was the
    exact #4191 ``hand_left`` / ``hand_right`` style bug). Carrying two
    extractors meant the motion-matching and ``opensim_golf`` clubhead-pose math
    could silently diverge.

    The canonical extractor lives in
    :mod:`src.engines.physics_engines.opensim.python.opensim_golf.fk`, which
    reconciles every prior FK path and reads the correct
    ``/jointset/hand_r_to_club/...`` ``PhysicalOffsetFrame`` components. This
    module is now a thin re-export of that single source of truth, so the two
    paths can no longer drift.

Canonical conventions (per ``CROSS_ENGINE_PARITY_SPEC.md`` §2.2):

* Position: 3-vector ``[x, y, z]`` in metres, world frame.
* Quaternion: 4-vector ``[w, x, y, z]``, unit norm, ``w >= 0``.
"""

from __future__ import annotations

from src.engines.physics_engines.opensim.python.opensim_golf.fk import (
    CANONICAL_LANDMARKS,
    _ensure_position_realised,
    _frame_pose_in_ground,
    _require_opensim,
    _rotmat_to_quat,
    extract_clubhead_pose,
    extract_full_pose,
    extract_grip_pose,
)

# Back-compat alias: this module historically exported the quaternion helper
# under a longer name. Keep it pointing at the canonical implementation.
_rotation_matrix_to_quat_wxyz = _rotmat_to_quat

__all__ = [
    "CANONICAL_LANDMARKS",
    "extract_clubhead_pose",
    "extract_full_pose",
    "extract_grip_pose",
]

# Re-exported private helpers are intentionally referenced so linters do not
# flag the imports as unused; downstream code/tests reach them by name.
_REEXPORTED = (
    _ensure_position_realised,
    _frame_pose_in_ground,
    _require_opensim,
    _rotation_matrix_to_quat_wxyz,
)
