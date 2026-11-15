"""Configurable tolerances and materials for hand visual backends (#15122)."""

from __future__ import annotations

from dataclasses import dataclass

from src.shared.python.model_appearance.schema import DEFAULT_SKIN_TONE

DEFAULT_VISUAL_GRIP_RADIUS_M = 0.014

DEFAULT_HAND_SKIN_MATERIAL = DEFAULT_SKIN_TONE


@dataclass(frozen=True)
class HandAnthropometryReference:
    """Target adult proportions (metres) for visual acceptance metrics."""

    reference_hand_length_m: float = 0.19
    reference_palm_width_m: float = 0.085
    finger_diameter_base_m: float = 0.018
    finger_diameter_tip_m: float = 0.014
    middle_finger_length_m: float = 0.08
    ratio_tolerance_rel: float = 0.15

    @property
    def finger_radius_base_m(self) -> float:
        return 0.5 * self.finger_diameter_base_m

    @property
    def finger_radius_tip_m(self) -> float:
        return 0.5 * self.finger_diameter_tip_m


@dataclass(frozen=True)
class HandVisualSettings:
    """Non-hard-coded limits for penetration checks and mesh quality."""

    shaft_penetration_tolerance_m: float = 0.0005
    finger_penetration_tolerance_m: float = 0.0015
    max_shaft_penetration_fraction: float = 0.0
    max_finger_penetration_fraction: float = 0.02
    penetration_sample_count: int = 400
    procedural_skin_sides: int = 22
    sdf_grid_resolution: int = 96
    sdf_smooth_union_m: float = 0.0075
    thenar_radius_m: float = 0.012
    skin_material: str = DEFAULT_HAND_SKIN_MATERIAL
    lead_skin_material: str = "skin_2"
    trail_skin_material: str = "skin_3"
    anthropometry: HandAnthropometryReference = HandAnthropometryReference()

    def skin_material_for_role(self, role: str) -> str:
        if role == "lead":
            return self.lead_skin_material
        if role == "trail":
            return self.trail_skin_material
        return self.skin_material


DEFAULT_HAND_VISUAL_SETTINGS = HandVisualSettings()
