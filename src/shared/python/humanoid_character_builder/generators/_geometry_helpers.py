from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

from humanoid_character_builder.core.segment_definitions import (
    GeometryType,
    SegmentDefinition,
)


def create_geometry_dict(
    segment_def: SegmentDefinition,
    dimensions: dict[str, float],
    is_collision: bool,
) -> dict[str, Any]:
    geom_spec = (
        segment_def.get_collision_geometry()
        if is_collision
        else segment_def.visual_geometry
    )

    length = dimensions.get("length", 0.1)
    width = dimensions.get("width", 0.05)
    depth = dimensions.get("depth", 0.05)

    if geom_spec.geometry_type == GeometryType.BOX:
        return {
            "type": "box",
            "size": (width, depth, length),
        }
    if geom_spec.geometry_type == GeometryType.CYLINDER:
        radius = (width + depth) / 4
        return {
            "type": "cylinder",
            "radius": radius,
            "length": length,
        }
    if geom_spec.geometry_type == GeometryType.SPHERE:
        radius = length / 2
        return {
            "type": "sphere",
            "radius": radius,
        }
    if geom_spec.geometry_type == GeometryType.CAPSULE:
        radius = (width + depth) / 4
        return {
            "type": "cylinder",
            "radius": radius,
            "length": max(0.01, length - 2 * radius),
        }
    if geom_spec.geometry_type == GeometryType.MESH:
        return {
            "type": "mesh",
            "filename": geom_spec.mesh_path,
            "scale": geom_spec.mesh_scale,
        }
    return {
        "type": "box",
        "size": (width, depth, length),
    }


def add_geometry_element(parent: ET.Element, geom: dict[str, Any]) -> None:
    geometry = ET.SubElement(parent, "geometry")

    geom_type = geom["type"]
    if geom_type == "box":
        size = geom["size"]
        ET.SubElement(
            geometry, "box", size=f"{size[0]:.6f} {size[1]:.6f} {size[2]:.6f}"
        )
    elif geom_type == "cylinder":
        ET.SubElement(
            geometry,
            "cylinder",
            radius=f"{geom['radius']:.6f}",
            length=f"{geom['length']:.6f}",
        )
    elif geom_type == "sphere":
        ET.SubElement(geometry, "sphere", radius=f"{geom['radius']:.6f}")
    elif geom_type == "mesh":
        scale = geom.get("scale", (1.0, 1.0, 1.0))
        ET.SubElement(
            geometry,
            "mesh",
            filename=geom["filename"],
            scale=f"{scale[0]:.6f} {scale[1]:.6f} {scale[2]:.6f}",
        )
