"""Visible head and neck for the MeshCat and OpenSim backends (visual only).

Builds the shared parametric head (``model_appearance.head``) in the spec's own
head body frame, writes one OBJ per part and returns what each backend needs to
register a mesh visual with a colour. A spec without a head body gets none
(the MuJoCo layer has a torso fallback; these native viewers draw the spec as
is). Nothing here touches dynamics.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.model_appearance import (
    build_head_parts,
    head_frame_in_body,
    library_materials,
    place_parts,
    resolve_head_anchor,
)
from src.shared.python.model_appearance.head import (
    HAIR_PART,
    drawn_head_length,
    part_material_name,
)
from src.shared.python.model_appearance.body_skin_mesh import vertex_normals
from src.shared.python.model_appearance.schema import AppearanceDocument


@dataclass(frozen=True)
class HeadMeshFile:
    name: str
    body: str  # spec body name
    path: Path
    rgba: tuple[float, float, float, float]


def write_obj(path: Path, vertices: Any, faces: Any, *, smooth: bool = False) -> None:
    """Write triangles; smooth output references one unit normal per vertex.

    Smooth meshes must have finite, nonzero vertex normals. Coordinates and
    topology are preserved; normals are evaluated in the supplied mesh frame.
    """
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices]
    if smooth:
        normals = vertex_normals(np.asarray(vertices, dtype=float), np.asarray(faces))
        if not np.isfinite(normals).all() or not np.allclose(
            np.linalg.norm(normals, axis=1), 1.0
        ):
            raise ValueError("Smooth OBJ requires finite, nonzero vertex normals")
        lines += [f"vn {x:.6f} {y:.6f} {z:.6f}" for x, y, z in normals]
        lines.append("s 1")
        lines += ["f " + " ".join(f"{i + 1}//{i + 1}" for i in face) for face in faces]
    else:
        lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def head_mesh_files(
    spec: Mapping[str, Any], directory: Path, doc: AppearanceDocument | None = None
) -> list[HeadMeshFile]:
    """OBJ files of the head parts in the spec head body frame (may be empty)."""
    doc = doc or AppearanceDocument()
    anchor = resolve_head_anchor(spec)
    if anchor is None or not doc.head.enabled:
        return []
    materials = library_materials(doc)
    directory.mkdir(parents=True, exist_ok=True)
    parts = place_parts(
        build_head_parts(drawn_head_length(spec, anchor, doc.head), doc.head),
        head_frame_in_body(anchor, doc.head),
    )
    out = []
    for part in parts:
        path = directory / f"head_{part.name}.obj"
        write_obj(
            path, part.mesh.vertices, part.mesh.faces, smooth=part.name == HAIR_PART
        )
        rgba = materials[part_material_name(part, doc)].rgba
        out.append(HeadMeshFile(part.name, anchor.body, path, rgba))
    return out
