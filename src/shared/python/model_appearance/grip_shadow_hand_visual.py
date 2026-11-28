"""Backward-compatible re-exports for Shadow hand grip visuals (#15122).

Implementations live in :mod:`hand_backends.shadow`; use
:mod:`hand_backends.registry` for backend dispatch.
"""

from __future__ import annotations

from src.shared.python.model_appearance.hand_backends.shadow import (
    ShadowHandVisualSettings,
    apply_wrist_offsets,
    build_shadow_grip_hand_mesh,
    extract_shadow_hand_mesh,
    map_hand_grip_spec_to_wrist,
    penetration_metrics,
    solve_shadow_hand_qpos,
    visual_settings,
)

__all__ = [
    "ShadowHandVisualSettings",
    "apply_wrist_offsets",
    "build_shadow_grip_hand_mesh",
    "extract_shadow_hand_mesh",
    "map_hand_grip_spec_to_wrist",
    "penetration_metrics",
    "solve_shadow_hand_qpos",
    "visual_settings",
]
