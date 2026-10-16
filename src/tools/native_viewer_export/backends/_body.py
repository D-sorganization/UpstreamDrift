"""Shared body meshes as OBJ files for the MeshCat and OpenSim backends.

Mirrors ``_head.py``: writes one OBJ per resolved
``model_appearance.body_meshes.BodyMesh`` (skin, garment, blend, shoe and
club parts) in its SPEC body frame, with the library RGBA of its material.
The backends register these as mesh visuals in R9a/R9b (#12688, #12689);
each must place an OBJ in the spec frame of ``BodyMeshFile.body`` and apply
its own body-frame transform explicitly. Visual only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from src.shared.python.model_appearance import document_from_dict
from src.shared.python.model_appearance.body_meshes import (
    SPEC_BODY_FRAME,
    ResolvedAppearance,
)
from src.shared.python.model_appearance.schema import AppearanceDocument
from src.tools.native_viewer_export.backends._head import write_obj

_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]+")
#: The default appearance document every native-viewer backend renders.
DEFAULT_APPEARANCE = {"schema_version": "appearance-v1"}


def default_appearance() -> AppearanceDocument:
    """The shared default appearance document (one source for every backend)."""
    return document_from_dict(DEFAULT_APPEARANCE)


@dataclass(frozen=True)
class BodyMeshFile:
    body: str  # spec body name
    label: str
    path: Path
    material_name: str
    rgba: tuple[float, float, float, float]


def _file_stem(index: int, body: str, label: str) -> str:
    leaf = _UNSAFE.sub("_", body.rsplit("/", 1)[-1]).strip("_") or "body"
    return f"body_{index:03d}_{leaf}_{label}"


def body_mesh_files(
    resolved: ResolvedAppearance, directory: Path
) -> list[BodyMeshFile]:
    """One OBJ per resolved body mesh (spec body frame); empty when none.

    Precondition: ``resolved`` is in the spec body frame. Postcondition: one
    distinct file per mesh, in ``resolved.meshes`` order, vertices written to
    1e-6 m.
    """
    if resolved.frame != SPEC_BODY_FRAME:
        raise ValueError(f"expected spec-body-frame meshes, got {resolved.frame!r}")
    if not resolved.meshes:
        return []
    directory.mkdir(parents=True, exist_ok=True)
    out = []
    for index, item in enumerate(resolved.meshes):
        path = directory / f"{_file_stem(index, item.body, item.label)}.obj"
        mesh = item.mesh_in_body_frame
        write_obj(path, mesh.vertices, mesh.faces)
        rgba = resolved.rgba(item.material_name)
        out.append(BodyMeshFile(item.body, item.label, path, item.material_name, rgba))
    return out
