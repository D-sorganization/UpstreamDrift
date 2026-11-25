"""Club meshes for the native MeshCat backends (shared by Drake and Pinocchio).

Both backends draw the spec robot with the shared visual skeleton, whose club
head is an ellipsoid hint. When the spec describes a club, these helpers give
the shaft, grip, parametric head and both grip-hand meshes in the club-body
frame instead, so the exported view shows the same head and hands as the
MuJoCo layer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from src.shared.python.model_appearance import library
from src.shared.python.model_appearance.club_assembly import (
    assembly_from_spec,
    club_body_name,
)
from src.shared.python.model_appearance.club_chain_meshes import club_meshes_by_body
from src.shared.python.model_appearance.defaults import default_appearance
from src.shared.python.model_appearance.schema import AppearanceDocument
from src.shared.python.model_appearance.geometry import Mesh

DEFAULT_FINISH = "satin_steel"


@dataclass(frozen=True)
class ClubPart:
    """One club mesh in the club-body frame with its display colour."""

    name: str
    mesh: Mesh
    rgba: tuple[float, float, float, float]


def club_parts_by_body(
    spec: Mapping[str, Any],
    finish: str = DEFAULT_FINISH,
    appearance: AppearanceDocument | None = None,
) -> dict[str, list[ClubPart]]:
    """Club meshes per spec body, each in that body's frame (empty: no club).

    A rigid club has every part on the club body. A chain shaft (#12686) has
    the grip, hands and root shaft piece on the club body, one shaft piece on
    each link and the head on the tip link, so the bent shaft and the head
    that follows it are drawn where the physics puts them (#15150).
    ``appearance`` (default: the contract default) makes the hands bare skin
    or gloved. Raises ``ValueError`` for an unknown ``finish``.
    """
    doc = appearance if appearance is not None else default_appearance()
    if finish not in library.MATERIALS:
        raise ValueError(f"unknown club finish {finish!r}")
    club = assembly_from_spec(spec)
    if club is None or club_body_name(spec) is None:
        return {}
    materials = library.library_materials(doc)
    return {
        body: [
            ClubPart(
                f"club_{label}",
                mesh,
                tuple(float(c) for c in materials[material].rgba),  # type: ignore[arg-type]
            )
            for label, mesh, material in parts
        ]
        for body, parts in club_meshes_by_body(spec, club, finish, doc).items()
    }


def club_parts(
    spec: Mapping[str, Any],
    finish: str = DEFAULT_FINISH,
    appearance: AppearanceDocument | None = None,
) -> tuple[str, list[ClubPart]] | None:
    """``(club body, [shaft, grip, head, lead_hand, trail_hand])`` or ``None``.

    The two curled-finger hands are wrapped on the grip (OSV-2,
    ``model_appearance.grip_hands``) and ride the club body like the rest.
    For a chain shaft only the parts that stay on the club body are returned;
    :func:`club_parts_by_body` has the links and the head.

    Postcondition: ``None`` exactly when the spec has no club assembly.
    """
    by_body = club_parts_by_body(spec, finish, appearance)
    body = club_body_name(spec)
    return None if body is None or body not in by_body else (body, by_body[body])


def write_obj(mesh: Mesh, directory: str, stem: str) -> str:
    """Write ``mesh`` as a Wavefront OBJ under ``directory``; return its path."""
    from pathlib import Path

    from src.tools.native_viewer_export.backends._head import write_obj as write_file

    path = Path(directory) / f"{stem}.obj"
    write_file(path, mesh.vertices, mesh.faces)
    return str(path)
