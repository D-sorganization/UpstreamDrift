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
    reference_palm_length_m: float = 0.102
    finger_diameter_base_m: float = 0.018
    finger_diameter_tip_m: float = 0.014
    middle_finger_length_m: float = 0.08
    ratio_tolerance_rel: float = 0.15
    finger_base_diameter_range_m: tuple[float, float] = (0.0165, 0.0205)
    finger_tip_diameter_range_m: tuple[float, float] = (0.0128, 0.0162)
    hand_length_range_m: tuple[float, float] = (0.180, 0.205)
    palm_width_range_m: tuple[float, float] = (0.078, 0.094)
    palm_length_range_m: tuple[float, float] = (0.092, 0.112)
    knuckle_attachment_tolerance_m: float = 0.0045
    finger_base_diameters_m: tuple[tuple[str, float], ...] = (
        ("index", 0.0188),
        ("middle", 0.0196),
        ("ring", 0.0182),
        ("little", 0.0170),
    )
    finger_tip_diameters_m: tuple[tuple[str, float], ...] = (
        ("index", 0.0144),
        ("middle", 0.0150),
        ("ring", 0.0140),
        ("little", 0.0132),
    )
    thumb_base_diameter_m: float = 0.0210
    thumb_tip_diameter_m: float = 0.0162

    @property
    def finger_radius_base_m(self) -> float:
        return 0.5 * self.finger_diameter_base_m

    @property
    def finger_radius_tip_m(self) -> float:
        return 0.5 * self.finger_diameter_tip_m

    def base_radius_for(self, finger: str) -> float:
        """Return base skin radius (m) for ``finger`` or default base radius."""
        mapping = dict(self.finger_base_diameters_m)
        return 0.5 * mapping.get(finger, self.finger_diameter_base_m)

    def tip_radius_for(self, finger: str) -> float:
        """Return distal tip skin radius (m) for ``finger`` or default tip radius."""
        mapping = dict(self.finger_tip_diameters_m)
        return 0.5 * mapping.get(finger, self.finger_diameter_tip_m)


@dataclass(frozen=True)
class HandVisualSettings:
    """Non-hard-coded limits for penetration checks and mesh quality."""

    shaft_penetration_tolerance_m: float = 0.0005
    finger_penetration_tolerance_m: float = 0.0015
    pad_contact_tolerance_m: float = 0.0030
    max_shaft_penetration_fraction: float = 0.0
    max_finger_penetration_fraction: float = 0.02
    penetration_sample_count: int = 400
    procedural_skin_sides: int = 40
    sdf_grid_resolution: int = 96
    sdf_smooth_union_m: float = 0.0075
    thenar_radius_m: float = 0.012
    palm_thickness_m: float = 0.022
    wrist_thickness_m: float = 0.031
    wrist_width_fraction: float = 0.68
    knuckle_dome_height_m: float = 0.0045
    interdigital_valley_m: float = 0.0022
    palmar_cup_base_m: float = 0.0025
    palmar_cup_curl_gain_m: float = 0.0040
    palm_cup_depth_min_m: float = 0.0015
    palmar_pad_flatness: float = 0.86
    knuckle_socket_inset_m: float = 0.010
    inter_finger_min_gap_m: float = 0.0004
    open_hand_lift_m: float = 0.012
    full_contact_palm_finger_depth: float = 0.5
    mcp_from_v_deg: float = 110.0
    mcp_span_fraction: float = 0.82
    overlap_finger_clearance_m: float = 0.0115
    thenar_half_extents_m: tuple[float, float, float] = (0.011, 0.0085, 0.018)
    thumb_base_opposition_rad: float = 0.62
    thenar_flare_scale: float = 1.06
    distal_pad_t_start: float = 0.65
    distal_cap_shift_m: float = 0.0060
    render_frame_px: int = 1024
    render_frame_fill: float = 0.70
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
