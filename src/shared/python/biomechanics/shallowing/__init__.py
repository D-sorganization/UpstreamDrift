"""Shallowing analysis sub-package.

Provides tools for analysing the shallowing phase of the golf downswing,
including hand-path plane computation (Phase 1 of epic #5422).
"""

from __future__ import annotations

from .hand_path_plane import (
    Plane3D,
    compute_hand_path_plane,
    extract_lead_hand_trajectory,
)

__all__: list[str] = [
    "Plane3D",
    "compute_hand_path_plane",
    "extract_lead_hand_trajectory",
]
