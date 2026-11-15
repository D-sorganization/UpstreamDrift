"""Configurable tolerances and materials for hand visual backends (#15122)."""

from __future__ import annotations

from dataclasses import dataclass

from src.shared.python.model_appearance.schema import DEFAULT_SKIN_TONE

DEFAULT_VISUAL_GRIP_RADIUS_M = 0.014

# sRGB floats for skin_2 (default golfer), resolved at render time via library.
DEFAULT_HAND_SKIN_MATERIAL = DEFAULT_SKIN_TONE


@dataclass(frozen=True)
class HandVisualSettings:
    """Non-hard-coded limits for penetration checks and mesh quality."""

    shaft_penetration_tolerance_m: float = 0.0005
    finger_penetration_tolerance_m: float = 0.0015
    max_shaft_penetration_fraction: float = 0.0
    max_finger_penetration_fraction: float = 0.02
    penetration_sample_count: int = 400
    procedural_skin_rings: int = 18
    procedural_skin_sides: int = 24
    palm_grid_rings: int = 12
    palm_grid_sides: int = 28
    skin_material: str = DEFAULT_HAND_SKIN_MATERIAL


DEFAULT_HAND_VISUAL_SETTINGS = HandVisualSettings()
