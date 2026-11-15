"""Settings and constants for the Shadow Dexterous Hand visual backend."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as SciRot

from src.shared.python.core.handedness import Handedness

Array = NDArray[np.float64]

_CONFIG_PATH = (
    Path(__file__).resolve().parents[5] / "scripts/config/grip_shadow_hand_visual.json"
)
_TANG_REF_X_MODEL = 0.366

_FINGER_CHAINS: dict[str, tuple[str, ...]] = {
    "index": ("FFJ4", "FFJ3", "FFJ2", "FFJ1"),
    "middle": ("MFJ4", "MFJ3", "MFJ2", "MFJ1"),
    "ring": ("RFJ4", "RFJ3", "RFJ2", "RFJ1"),
    "little": ("LFJ5", "LFJ4", "LFJ3", "LFJ2", "LFJ1"),
    "thumb": ("THJ5", "THJ4", "THJ3", "THJ2", "THJ1"),
}

_FINGER_BODY_TAG: dict[str, str] = {
    "index": "ff",
    "middle": "mf",
    "ring": "rf",
    "little": "lf",
    "thumb": "th",
}

_CURL_STAGE_FINGERS: dict[str, tuple[str, ...]] = {
    "lead_fingers_index_to_pinky": ("index", "middle", "ring", "little"),
    "trail_fingers": ("index", "middle", "ring", "little"),
}


@dataclass(frozen=True)
class ShadowHandPose:
    """MuJoCo qpos, affine grip-frame mount, and golfer handedness for one hand."""

    qpos: Array
    mount_matrix: Array
    handedness: Handedness = "right"


@dataclass(frozen=True)
class ShadowHandVisualSettings:
    """All tolerances, anthropometric bands, contouring, and colour settings."""

    club_penetration_tolerance_m: float
    club_assembly_axial_overhang_m: float
    finger_finger_penetration_tolerance_m: float
    finger_contact_tolerance_m: float
    pad_clearance_m: float
    grip_radius_range_m: tuple[float, float]
    render_frame_px: int
    render_frame_fill: float
    curl_bisect_iterations: int
    curl_joint_step_rad: float
    grip_radius_nominal_m: float
    mount_matrix: Array
    palm_center_offset_deg: float
    palm_diagonal_tilt_gain: float
    wrist_gain: dict[str, float]
    palm_cup_gain: dict[str, float]
    skin_tone_rgba: tuple[float, float, float, float]
    hidden_body_substrings: tuple[str, ...]
    hidden_mesh_substrings: tuple[str, ...]
    native_hand_length_m: float
    native_hand_breadth_m: float
    reference_hand_length_m: float
    reference_hand_breadth_m: float
    length_scale: float
    breadth_scale: float
    hand_length_range_m: tuple[float, float]
    palm_breadth_range_m: tuple[float, float]
    finger_diameter_range_m: tuple[float, float]
    knuckle_line_max_gap_m: float
    thumb_min_radial_clearance_m: float
    palm_pad_min_radial_clearance_m: float
    palm_palmar_cap_z_m: float
    palm_palmar_softness_m: float
    palm_dorsal_cap_z_m: float
    palm_dorsal_softness_m: float
    palm_tangency_x_m: float
    wrist_taper_min_scale: float
    mcp_bridge_radius_m: float
    palm_transverse_arch_m: float
    palm_arch_half_breadth_m: float
    palm_diagonal_max_deg: float


def _pair(raw: Any, default: tuple[float, float]) -> tuple[float, float]:
    vals = raw if raw is not None else default
    return (float(vals[0]), float(vals[1]))


def _load_settings() -> ShadowHandVisualSettings:
    raw = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    rot_cfg = raw["mount_rotation_deg"]
    rot = SciRot.from_euler(
        rot_cfg["axis1"], rot_cfg["angle1"], degrees=True
    ) * SciRot.from_euler(rot_cfg["axis2"], rot_cfg["angle2"], degrees=True)
    mount = np.eye(4)
    mount[:3, :3] = rot.as_matrix()
    mount[:3, 3] = np.asarray(raw.get("mount_translation_m", (0.0, 0.0, 0.0)), float)
    anthro = raw.get("anthropometry", {})
    topo = raw.get("topology", {})
    return ShadowHandVisualSettings(
        club_penetration_tolerance_m=float(raw["club_penetration_tolerance_m"]),
        club_assembly_axial_overhang_m=float(
            raw.get("club_assembly_axial_overhang_m", 0.22)
        ),
        finger_finger_penetration_tolerance_m=float(
            raw["finger_finger_penetration_tolerance_m"]
        ),
        finger_contact_tolerance_m=float(raw.get("finger_contact_tolerance_m", 0.004)),
        pad_clearance_m=float(raw["pad_clearance_m"]),
        grip_radius_range_m=_pair(raw.get("grip_radius_range_m"), (0.0105, 0.0135)),
        render_frame_px=int(raw.get("render_frame_px", 1024)),
        render_frame_fill=float(raw.get("render_frame_fill", 0.70)),
        curl_bisect_iterations=int(raw["curl_bisect_iterations"]),
        curl_joint_step_rad=float(raw["curl_joint_step_rad"]),
        grip_radius_nominal_m=float(raw["grip_radius_nominal_m"]),
        mount_matrix=mount,
        palm_center_offset_deg=float(raw.get("palm_center_offset_deg", 55.0)),
        palm_diagonal_tilt_gain=float(raw.get("palm_diagonal_tilt_gain", 0.45)),
        wrist_gain=dict(raw["wrist_from_spec"]),
        palm_cup_gain=dict(raw.get("palm_cup_from_spec", {})),
        skin_tone_rgba=tuple(
            float(x) for x in raw.get("skin_tone_rgba", (0.83, 0.66, 0.55, 1.0))
        ),  # type: ignore[arg-type]
        hidden_body_substrings=tuple(
            raw.get("hidden_body_substrings", ("forearm", "_wrist"))
        ),
        hidden_mesh_substrings=tuple(
            raw.get(
                "hidden_mesh_substrings",
                ("forearm", "wrist", "mounting_plate", "f_knuckle"),
            )
        ),
        native_hand_length_m=float(anthro.get("native_hand_length_m", 0.2205)),
        native_hand_breadth_m=float(anthro.get("native_hand_breadth_m", 0.0843)),
        reference_hand_length_m=float(anthro.get("reference_hand_length_m", 0.1932)),
        reference_hand_breadth_m=float(anthro.get("reference_hand_breadth_m", 0.08675)),
        length_scale=float(anthro.get("length_scale", 1.0)),
        breadth_scale=float(anthro.get("breadth_scale", 1.0)),
        hand_length_range_m=_pair(anthro.get("hand_length_range_m"), (0.175, 0.215)),
        palm_breadth_range_m=_pair(anthro.get("palm_breadth_range_m"), (0.078, 0.098)),
        finger_diameter_range_m=_pair(
            anthro.get("finger_diameter_range_m"), (0.013, 0.022)
        ),
        knuckle_line_max_gap_m=float(topo.get("knuckle_line_max_gap_m", 0.014)),
        thumb_min_radial_clearance_m=float(
            topo.get("thumb_min_radial_clearance_m", 0.0005)
        ),
        palm_pad_min_radial_clearance_m=float(
            topo.get("palm_pad_min_radial_clearance_m", 0.0005)
        ),
        palm_palmar_cap_z_m=float(topo.get("palm_palmar_cap_z_m", 0.0185)),
        palm_palmar_softness_m=float(topo.get("palm_palmar_softness_m", 0.0022)),
        palm_dorsal_cap_z_m=float(topo.get("palm_dorsal_cap_z_m", -0.0112)),
        palm_dorsal_softness_m=float(topo.get("palm_dorsal_softness_m", 0.0018)),
        palm_tangency_x_m=float(topo.get("palm_tangency_x_m", 0.338)),
        wrist_taper_min_scale=float(topo.get("wrist_taper_min_scale", 0.80)),
        mcp_bridge_radius_m=float(topo.get("mcp_bridge_radius_m", 0.0082)),
        palm_transverse_arch_m=float(topo.get("palm_transverse_arch_m", 0.0028)),
        palm_arch_half_breadth_m=float(topo.get("palm_arch_half_breadth_m", 0.038)),
        palm_diagonal_max_deg=float(topo.get("palm_diagonal_max_deg", 7.0)),
    )


@lru_cache(maxsize=1)
def visual_settings() -> ShadowHandVisualSettings:
    """Cached settings loaded from ``grip_shadow_hand_visual.json``."""
    return _load_settings()
