"""Model contracts and canonical body definitions across physics engines.

Defines canonical body sets and verification helpers so cross-engine code
can rely on consistent segment naming across MuJoCo, Drake, and Pinocchio.
"""

from __future__ import annotations

#: Canonical set of body names required for cross-engine golf models.
#: All default golfer models in the core tier (MuJoCo, Drake, Pinocchio)
#: must expose these body names.
REQUIRED_GOLFER_BODIES: tuple[str, ...] = (
    "pelvis",
    "forearm_left",
    "forearm_right",
    "hand_left",
    "hand_right",
    "club_shaft",
    "club_head",
)
