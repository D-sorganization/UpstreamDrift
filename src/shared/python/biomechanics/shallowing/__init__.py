"""Shallowing analysis sub-package.

Provides tools for analysing the shallowing phase of the golf downswing,
including hand-path plane computation (Phase 1 of epic #5422) and passive
squaring torque + plane classification (Phase 2 of epic #5422).
"""

from __future__ import annotations

from .hand_path_plane import (
    Plane3D,
    compute_hand_path_plane,
    extract_lead_hand_trajectory,
)
from .passive_squaring import (
    ShallowingMetrics,
    classify_swing_plane,
    compute_club_com_offset,
    compute_passive_squaring_torque,
)

__all__: list[str] = [
    "Plane3D",
    "ShallowingMetrics",
    "classify_swing_plane",
    "compute_club_com_offset",
    "compute_hand_path_plane",
    "compute_passive_squaring_torque",
    "extract_lead_hand_trajectory",
]
