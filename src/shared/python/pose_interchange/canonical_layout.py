"""Single source of truth for the canonical-v2 floating-base layout.

The canonical-v2 spec (``docs/conventions/canonical-v2.md`` §2) fixes the
generalized-velocity layout for the floating base as ``[linear; angular]``:

    v = [ base_vx, base_vy, base_vz,    # base linear velocity, world frame
          base_wx, base_wy, base_wz,    # base angular velocity, BODY frame
          dj_0, ... ]                   # joint velocities

and the configuration as ``[xyz, quat_wxyz]``. Every pose_interchange adapter
imports the slices and sizes declared here rather than embedding index literals,
so the layout cannot silently drift between adapters (DRY + single source of
truth for the CC-7 conformance contract).
"""

from __future__ import annotations

# Floating-base configuration: [x, y, z, qw, qx, qy, qz]
FREE_FLYER_Q = 7
# Floating-base velocity / acceleration: [vx, vy, vz, wx, wy, wz]
FREE_FLYER_V = 6

# Configuration slices.
BASE_POSITION = slice(0, 3)
BASE_QUATERNION = slice(3, 7)

# Velocity / acceleration slices (canonical-v2 order is linear-first).
LINEAR = slice(0, 3)
ANGULAR = slice(3, 6)

__all__ = [
    "ANGULAR",
    "BASE_POSITION",
    "BASE_QUATERNION",
    "FREE_FLYER_Q",
    "FREE_FLYER_V",
    "LINEAR",
]
