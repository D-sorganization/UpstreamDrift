"""Settings and asset provenance verification for the Wonik Allegro backend."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.shared.python.contracts import ensure, require
from src.shared.python.core.handedness import Handedness
from src.shared.python.model_appearance.hand_backends.types import Array

REPO_ROOT = Path(__file__).resolve().parents[5]
_CONFIG_PATH = REPO_ROOT / "scripts/config/hand_backend_allegro.json"
ALLEGRO_ASSETS_ROOT = (
    REPO_ROOT
    / "src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/hand_assets/wonik_allegro"
)

_DIGIT_TAG: dict[str, str] = {
    "index": "ff",
    "middle": "mf",
    "ring": "rf",
    "thumb": "th",
}

_SEGMENTS: tuple[str, ...] = ("base", "proximal", "medial", "distal", "tip")


@dataclass(frozen=True)
class AllegroHandPose:
    """MuJoCo qpos, affine grip-frame mount (after shaft roll), and handedness."""

    qpos: Array
    mount_matrix: Array
    handedness: Handedness = "right"


@dataclass(frozen=True)
class AllegroHandVisualSettings:
    """All tolerances, anthropometric bands, solver counts, and colours from JSON."""

    backend_name: str
    grip_radius_nominal_m: float
    grip_radius_range_m: tuple[float, float]
    pad_clearance_m: float
    proximal_stage_offset_m: float
    club_penetration_tolerance_m: float
    finger_contact_tolerance_m: float
    finger_finger_penetration_tolerance_m: float
    contact_percentile: float
    solver_vertex_stride: int
    visible_geom_group: int
    hidden_body_substrings: tuple[str, ...]
    hidden_mesh_substrings: tuple[str, ...]
    palm_center_offset_deg: float
    palm_diagonal_tilt_gain: float
    palm_diagonal_max_deg: float
    palm_tangency_x_m: float
    palm_tangency_z_m: float
    mount_radial_search_range_m: tuple[float, float]
    mount_radial_bisect_iterations: int
    curl_coarse_samples: int
    curl_bisect_iterations: int
    curl_joint_step_rad: float
    relax_max_iterations: int
    thumb_min_radial_clearance_m: float
    thumb_index_min_clearance_m: float
    thumb_z_base_m: float
    thumb_z_straight_gain_m: float
    thumb_max_z_m: float
    thumb_coarse_grid_n: int
    thumb_curl_presets_rad: tuple[tuple[float, float], ...]
    thumb_refine_steps_rad: tuple[float, ...]
    finger_abduction_inward_rad: float
    skin_tone_rgba: tuple[float, float, float, float]
    tip_pad_rgba: tuple[float, float, float, float]
    native_hand_length_m: float
    native_hand_breadth_m: float
    reference_hand_length_m: float
    reference_hand_breadth_m: float
    length_scale: float
    breadth_scale: float
    hand_length_range_m: tuple[float, float]
    palm_breadth_range_m: tuple[float, float]
    finger_diameter_range_m: tuple[float, float]
    topology_knuckle_line_max_gap_m: float
    render_frame_px: int
    render_frame_fill: float
    both_frame_fill: float
    club_axial_overhang_m: float


def _pair(raw: Any) -> tuple[float, float]:
    return (float(raw[0]), float(raw[1]))


def _rgba(raw: Any) -> tuple[float, float, float, float]:
    return (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))


@lru_cache(maxsize=1)
def visual_settings() -> AllegroHandVisualSettings:
    """Load and cache all Allegro backend settings from JSON config."""
    require(_CONFIG_PATH.is_file(), f"Allegro config missing: {_CONFIG_PATH}")
    raw = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    anthro = raw["anthropometry"]
    topo = raw["topology"]
    rend = raw["rendering"]
    return AllegroHandVisualSettings(
        backend_name=str(raw["backend_name"]),
        grip_radius_nominal_m=float(raw["grip_radius_nominal_m"]),
        grip_radius_range_m=_pair(raw["grip_radius_range_m"]),
        pad_clearance_m=float(raw["pad_clearance_m"]),
        proximal_stage_offset_m=float(raw["proximal_stage_offset_m"]),
        club_penetration_tolerance_m=float(raw["club_penetration_tolerance_m"]),
        finger_contact_tolerance_m=float(raw["finger_contact_tolerance_m"]),
        finger_finger_penetration_tolerance_m=float(
            raw["finger_finger_penetration_tolerance_m"]
        ),
        contact_percentile=float(raw["contact_percentile"]),
        solver_vertex_stride=int(raw["solver_vertex_stride"]),
        visible_geom_group=int(raw["visible_geom_group"]),
        hidden_body_substrings=tuple(str(s) for s in raw["hidden_body_substrings"]),
        hidden_mesh_substrings=tuple(str(s) for s in raw["hidden_mesh_substrings"]),
        palm_center_offset_deg=float(raw["palm_center_offset_deg"]),
        palm_diagonal_tilt_gain=float(raw["palm_diagonal_tilt_gain"]),
        palm_diagonal_max_deg=float(raw["palm_diagonal_max_deg"]),
        palm_tangency_x_m=float(raw["palm_tangency_x_m"]),
        palm_tangency_z_m=float(raw["palm_tangency_z_m"]),
        mount_radial_search_range_m=_pair(raw["mount_radial_search_range_m"]),
        mount_radial_bisect_iterations=int(raw["mount_radial_bisect_iterations"]),
        curl_coarse_samples=int(raw["curl_coarse_samples"]),
        curl_bisect_iterations=int(raw["curl_bisect_iterations"]),
        curl_joint_step_rad=float(raw["curl_joint_step_rad"]),
        relax_max_iterations=int(raw["relax_max_iterations"]),
        thumb_min_radial_clearance_m=float(raw["thumb_min_radial_clearance_m"]),
        thumb_index_min_clearance_m=float(raw["thumb_index_min_clearance_m"]),
        thumb_z_base_m=float(raw["thumb_z_base_m"]),
        thumb_z_straight_gain_m=float(raw["thumb_z_straight_gain_m"]),
        thumb_max_z_m=float(raw["thumb_max_z_m"]),
        thumb_coarse_grid_n=int(raw["thumb_coarse_grid_n"]),
        thumb_curl_presets_rad=tuple(
            (float(pair[0]), float(pair[1])) for pair in raw["thumb_curl_presets_rad"]
        ),
        thumb_refine_steps_rad=tuple(
            float(step) for step in raw["thumb_refine_steps_rad"]
        ),
        finger_abduction_inward_rad=float(raw["finger_abduction_inward_rad"]),
        skin_tone_rgba=_rgba(raw["skin_tone_rgba"]),
        tip_pad_rgba=_rgba(raw["tip_pad_rgba"]),
        native_hand_length_m=float(anthro["native_hand_length_m"]),
        native_hand_breadth_m=float(anthro["native_hand_breadth_m"]),
        reference_hand_length_m=float(anthro["reference_hand_length_m"]),
        reference_hand_breadth_m=float(anthro["reference_hand_breadth_m"]),
        length_scale=float(anthro["length_scale"]),
        breadth_scale=float(anthro["breadth_scale"]),
        hand_length_range_m=_pair(anthro["hand_length_range_m"]),
        palm_breadth_range_m=_pair(anthro["palm_breadth_range_m"]),
        finger_diameter_range_m=_pair(anthro["finger_diameter_range_m"]),
        topology_knuckle_line_max_gap_m=float(topo["knuckle_line_max_gap_m"]),
        render_frame_px=int(rend["frame_px"]),
        render_frame_fill=float(rend["frame_fill"]),
        both_frame_fill=float(rend["both_frame_fill"]),
        club_axial_overhang_m=float(rend["club_axial_overhang_m"]),
    )


@lru_cache(maxsize=1)
def load_allegro_provenance() -> dict[str, Any]:
    """Verify Wonik Allegro BSD-2-Clause LICENSE, NOTICE, and STL SHA256 hashes."""
    prov_path = ALLEGRO_ASSETS_ROOT / "PROVENANCE.json"
    lic_path = ALLEGRO_ASSETS_ROOT / "LICENSE"
    notice_path = ALLEGRO_ASSETS_ROOT / "NOTICE"
    require(prov_path.is_file(), f"Allegro PROVENANCE.json missing: {prov_path}")
    require(lic_path.is_file(), f"Allegro LICENSE missing: {lic_path}")
    require(notice_path.is_file(), f"Allegro NOTICE missing: {notice_path}")

    record = json.loads(prov_path.read_text(encoding="utf-8"))
    require(
        record.get("spdx_license") == "BSD-2-Clause",
        "Allegro SPDX license must be BSD-2-Clause",
    )
    lic_hash = hashlib.sha256(lic_path.read_bytes()).hexdigest()
    require(
        lic_hash == record.get("license_sha256"),
        f"Allegro LICENSE sha256 mismatch: {lic_hash}",
    )
    for rel_path, expected in record.get("stl_files_sha256", {}).items():
        stl_path = ALLEGRO_ASSETS_ROOT / rel_path
        require(stl_path.is_file(), f"Allegro STL asset missing: {stl_path}")
        actual = hashlib.sha256(stl_path.read_bytes()).hexdigest()
        require(
            actual == expected,
            f"Allegro STL {rel_path} sha256 {actual} != {expected}",
        )
    ensure(len(record.get("stl_files_sha256", {})) >= 14, "expected 14 Allegro STLs")
    return record


def allegro_mjcf_path(*, trail: bool, handedness: str = "right") -> Path:
    """Return left_hand.xml or right_hand.xml matching golf role and handedness."""
    load_allegro_provenance()
    require(
        handedness in ("right", "left"),
        f"handedness must be 'right' or 'left', got {handedness!r}",
    )
    use_right = bool(trail) ^ (handedness == "left")
    filename = "right_hand.xml" if use_right else "left_hand.xml"
    path = ALLEGRO_ASSETS_ROOT / filename
    require(path.is_file(), f"Allegro MJCF missing: {path}")
    return path


__all__ = [
    "ALLEGRO_ASSETS_ROOT",
    "AllegroHandPose",
    "AllegroHandVisualSettings",
    "_DIGIT_TAG",
    "_SEGMENTS",
    "allegro_mjcf_path",
    "load_allegro_provenance",
    "visual_settings",
]
