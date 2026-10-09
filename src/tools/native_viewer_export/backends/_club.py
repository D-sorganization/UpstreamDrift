"""Club meshes for the native MeshCat backends (shared by Drake and Pinocchio).

Both backends draw the spec robot with the shared visual skeleton, whose club
head is an ellipsoid hint. When the spec describes a club, these helpers give
the shaft, grip and parametric head meshes in the club-body frame instead, so
the exported view shows the same head as the MuJoCo and OpenSim layers.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from src.shared.python.model_appearance import library
from src.shared.python.model_appearance.club_assembly import (
    PART_MATERIALS,
    assembly_from_spec,
    assembly_meshes,
    club_body_name,
)
from src.shared.python.model_appearance.geometry import Mesh

DEFAULT_FINISH = "satin_steel"


@dataclass(frozen=True)
class ClubPart:
    """One club mesh in the club-body frame with its display colour."""

    name: str
    mesh: Mesh
    rgba: tuple[float, float, float, float]


def club_parts(
    spec: Mapping[str, Any], finish: str = DEFAULT_FINISH
) -> tuple[str, list[ClubPart]] | None:
    """``(club body name, [shaft, grip, head])`` for ``spec``, or ``None``.

    Postcondition: ``None`` exactly when the spec has no club assembly.
    """
    if finish not in library.MATERIALS:
        raise ValueError(f"unknown club finish {finish!r}")
    club = assembly_from_spec(spec)
    body = club_body_name(spec)
    if club is None or body is None:
        return None
    parts = []
    for name, mesh in assembly_meshes(club).items():
        rgba = library.MATERIALS[PART_MATERIALS[name] or finish].rgba
        parts.append(ClubPart(f"club_{name}", mesh, tuple(float(c) for c in rgba)))  # type: ignore[arg-type]
    return body, parts


def write_obj(mesh: Mesh, directory: str, stem: str) -> str:
    """Write ``mesh`` as a Wavefront OBJ under ``directory``; return its path."""
    from pathlib import Path

    from src.tools.native_viewer_export.backends._head import write_obj as write_file

    path = Path(directory) / f"{stem}.obj"
    write_file(path, mesh.vertices, mesh.faces)
    return str(path)
