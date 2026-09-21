"""Anatomical visual asset specifications and semantic body mapping (MV-02, #10478).

Provides decoupled visual asset bindings for multi-engine models. Supports selectable
visual skins (inertia ellipsoids vs. anatomical meshes/capsules) and multi-layer rendering
without modifying any underlying physical parameters (mass, inertia, contact geometry).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_ASSET_DIR = "assets/body_part_shapes/default"
CC0_LICENSE = "CC0-1.0"
PROVENANCE_MANIFEST = "assets/body_part_shapes/default/manifest.json"


class VisualSkinMode(str, Enum):
    """Supported presentation-level visual skin modes."""

    INERTIA_ELLIPSOIDS = "inertia_ellipsoids"
    ANATOMICAL_MESH = "anatomical_mesh"
    ANATOMICAL_CAPSULE = "anatomical_capsule"


@dataclass(frozen=True)
class VisualAssetBinding:
    """Rigid visual asset binding definition for a model link or solid."""

    semantic_body: str
    mesh_relative_path: str
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    local_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    local_orientation_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    units: str = "meters"
    color: tuple[float, float, float, float] = (0.18, 0.55, 0.8, 1.0)
    license_name: str = CC0_LICENSE
    provenance: str = PROVENANCE_MANIFEST
    is_fallback: bool = False
    fallback_geometry_type: str = "box"


def get_diagnostic_fallback_binding(
    semantic_body: str, missing_file: str = "fallback.stl"
) -> VisualAssetBinding:
    """Generate a highly-visible diagnostic fallback binding (bright magenta)."""
    return VisualAssetBinding(
        semantic_body=semantic_body,
        mesh_relative_path=f"diagnostics/{missing_file}",
        scale=(0.05, 0.05, 0.05),
        color=(1.0, 0.0, 1.0, 1.0),
        license_name=CC0_LICENSE,
        provenance="internal:diagnostic_fallback",
        is_fallback=True,
        fallback_geometry_type="box",
    )


def _match_club_solid(solid_lower: str) -> VisualAssetBinding | None:
    """Map club components to dedicated visual geometries and colors."""
    if "clubhead" in solid_lower:
        return VisualAssetBinding(
            semantic_body="clubhead",
            mesh_relative_path=f"{DEFAULT_ASSET_DIR}/clubhead.stl",
            scale=(0.11, 0.09, 0.06),
            color=(0.95, 0.6, 0.12, 1.0),
            license_name=CC0_LICENSE,
            provenance="procedural:clubhead",
            fallback_geometry_type="box",
        )
    if "shaft" in solid_lower:
        return VisualAssetBinding(
            semantic_body="shaft",
            mesh_relative_path=f"{DEFAULT_ASSET_DIR}/shaft.stl",
            scale=(0.015, 0.015, 1.15),
            color=(0.7, 0.7, 0.75, 1.0),
            license_name=CC0_LICENSE,
            provenance="procedural:shaft",
            fallback_geometry_type="cylinder",
        )
    if "grip" in solid_lower:
        return VisualAssetBinding(
            semantic_body="grip",
            mesh_relative_path=f"{DEFAULT_ASSET_DIR}/grip.stl",
            scale=(0.03, 0.03, 0.28),
            color=(0.15, 0.15, 0.15, 1.0),
            license_name=CC0_LICENSE,
            provenance="procedural:grip",
            fallback_geometry_type="cylinder",
        )
    return None


def _match_humanoid_body(
    combined: str,
) -> tuple[str, str, tuple[float, float, float]] | None:
    """Determine semantic body name, stl filename, and default scale from body/solid text."""
    if "head" in combined:
        return "head", "head.stl", (1.0, 1.0, 1.0)
    if any(k in combined for k in ("torso", "comrod", "trunk", "pelvis")):
        return "torso", "torso.stl", (1.0, 1.0, 1.0)
    if any(
        k in combined
        for k in ("lupperarm", "rupperarm", "upper_arm", "hubtols", "hubtors")
    ):
        return "upper_arm", "upper_arm.stl", (1.0, 1.0, 1.0)
    if any(k in combined for k in ("forearm", "elbow")):
        return "forearm", "forearm.stl", (1.0, 1.0, 1.0)
    if any(k in combined for k in ("hand", "standoff")):
        return "hand", "hand.stl", (1.0, 1.0, 1.0)
    if "femur" in combined or "thigh" in combined:
        return "thigh", "thigh.stl", (1.0, 1.0, 1.0)
    if "tibia" in combined or "shin" in combined or "shank" in combined:
        return "shin", "shin.stl", (1.0, 1.0, 1.0)
    if any(k in combined for k in ("calcn", "talus", "toes", "foot")):
        return "foot", "foot.stl", (1.0, 1.0, 1.0)
    return None


def resolve_anatomical_visual(
    body_name: str, solid_name: str | None = None
) -> VisualAssetBinding | None:
    """Resolve a body and optional solid name to a qualified anatomical visual binding."""
    b_lower = body_name.lower()
    s_lower = (solid_name or "").lower()
    combined = f"{b_lower} {s_lower}"

    if "world" in b_lower:
        return None

    if "club" in combined:
        club_match = _match_club_solid(s_lower)
        if club_match is not None:
            return club_match

    matched = _match_humanoid_body(combined)
    if matched is not None:
        semantic_name, stl_file, scale = matched
        return VisualAssetBinding(
            semantic_body=semantic_name,
            mesh_relative_path=f"{DEFAULT_ASSET_DIR}/{stl_file}",
            scale=scale,
            color=(0.18, 0.55, 0.8, 1.0),
            license_name=CC0_LICENSE,
            provenance=PROVENANCE_MANIFEST,
        )

    return get_diagnostic_fallback_binding(body_name)
