"""JSON-driven settings for hand visual backends."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.shared.python.contracts import require

_REPO_ROOT = Path(__file__).resolve().parents[5]
_DEFAULT_MYOHAND_CONFIG = _REPO_ROOT / "scripts/config/hand_backend_myohand.json"


def _rgb_triple(raw: Any, name: str) -> tuple[float, float, float]:
    require(isinstance(raw, list) and len(raw) == 3, f"{name} must be length-3 list")
    return tuple(float(v) for v in raw)  # type: ignore[return-value]


@dataclass(frozen=True)
class MyoHandVisualSettings:
    """Tolerances, calibration and colours for :mod:`hand_backends.myohand`."""

    schema_version: int
    grip_radius_m: float
    penetration_tolerance_m: float
    finger_pair_tolerance_m: float
    club_cylinder_radius_m: float
    research_only: bool
    license_gate_issue: int
    bone_rgb: tuple[float, float, float]
    grip_rgb: tuple[float, float, float]
    shaft_rgb: tuple[float, float, float]
    butt_cap_rgb: tuple[float, float, float]
    background_rgb: tuple[float, float, float]
    anthropometric_reference_hand_length_m: float
    anthropometric_scale_exponent: float
    width_scale_from_breadth: bool
    model_xml_relpath: str
    hand_body_names: tuple[str, ...]
    palm_anchor_bodies: tuple[str, ...]
    registration_mode: str
    mcp_azimuth_offset_deg: float
    mcp_tangential_bias_m: float
    palm_axial_offset_m: float
    palm_seat_clearance_m: float
    bone_target_clearance_m: float
    bone_safe_clearance_m: float
    distal_contact_tolerance_m: float
    middle_contact_tolerance_m: float
    thumb_contact_tolerance_m: float
    thumb_max_z_m: float
    palm_finger_depth_flexion_gain: float
    palm_finger_depth_deviation_gain: float
    thumb_straightness_range_rad: tuple[float, float]
    render_frame_fill: float

    @property
    def model_xml_path(self) -> Path:
        return _REPO_ROOT / self.model_xml_relpath


def load_myohand_settings(path: Path | None = None) -> MyoHandVisualSettings:
    cfg_path = _DEFAULT_MYOHAND_CONFIG if path is None else path
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    palm = payload["palm_finger_depth_coupling"]
    thumb_rng = payload["thumb_straightness_range_rad"]
    return MyoHandVisualSettings(
        schema_version=int(payload["schema_version"]),
        grip_radius_m=float(payload["grip_radius_m"]),
        penetration_tolerance_m=float(payload["penetration_tolerance_m"]),
        finger_pair_tolerance_m=float(payload["finger_pair_tolerance_m"]),
        club_cylinder_radius_m=float(payload["club_cylinder_radius_m"]),
        research_only=bool(payload["research_only"]),
        license_gate_issue=int(payload["license_gate_issue"]),
        bone_rgb=_rgb_triple(payload["bone_rgb"], "bone_rgb"),
        grip_rgb=_rgb_triple(payload["grip_rgb"], "grip_rgb"),
        shaft_rgb=_rgb_triple(payload["shaft_rgb"], "shaft_rgb"),
        butt_cap_rgb=_rgb_triple(payload["butt_cap_rgb"], "butt_cap_rgb"),
        background_rgb=_rgb_triple(payload["background_rgb"], "background_rgb"),
        anthropometric_reference_hand_length_m=float(
            payload["anthropometric_reference_hand_length_m"]
        ),
        anthropometric_scale_exponent=float(payload["anthropometric_scale_exponent"]),
        width_scale_from_breadth=bool(payload["width_scale_from_breadth"]),
        model_xml_relpath=str(payload["model_xml_relpath"]),
        hand_body_names=tuple(str(n) for n in payload["hand_body_names"]),
        palm_anchor_bodies=tuple(str(n) for n in payload["palm_anchor_bodies"]),
        registration_mode=str(payload["registration_mode"]),
        mcp_azimuth_offset_deg=float(payload["mcp_azimuth_offset_deg"]),
        mcp_tangential_bias_m=float(payload["mcp_tangential_bias_m"]),
        palm_axial_offset_m=float(payload["palm_axial_offset_m"]),
        palm_seat_clearance_m=float(payload["palm_seat_clearance_m"]),
        bone_target_clearance_m=float(payload["bone_target_clearance_m"]),
        bone_safe_clearance_m=float(payload["bone_safe_clearance_m"]),
        distal_contact_tolerance_m=float(payload["distal_contact_tolerance_m"]),
        middle_contact_tolerance_m=float(payload["middle_contact_tolerance_m"]),
        thumb_contact_tolerance_m=float(payload["thumb_contact_tolerance_m"]),
        thumb_max_z_m=float(payload["thumb_max_z_m"]),
        palm_finger_depth_flexion_gain=float(palm["flexion_gain"]),
        palm_finger_depth_deviation_gain=float(palm["deviation_gain"]),
        thumb_straightness_range_rad=(
            float(thumb_rng[0]),
            float(thumb_rng[1]),
        ),
        render_frame_fill=float(payload["render_frame_fill"]),
    )


@lru_cache(maxsize=1)
def default_myohand_settings() -> MyoHandVisualSettings:
    return load_myohand_settings()


__all__ = [
    "MyoHandVisualSettings",
    "default_myohand_settings",
    "load_myohand_settings",
]
