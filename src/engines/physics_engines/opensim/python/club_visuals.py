"""Visible club for OpenSim models: generated shaft, grip and head meshes (OSV-1).

The three meshes come from the shared club assembly
(``model_appearance.club_assembly``: Tools parametric head with a committed
STL fallback, tapered shaft and grip), are written once as binary STL under
``models/geometry/club/`` with a provenance manifest, and referenced from
``<Mesh>`` elements. Geometry is visual only: body masses, mass centres and
inertias are never touched.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any
from pathlib import Path
import xml.etree.ElementTree as ET  # noqa: S405  # nosemgrep: python.lang.security.use-defused-xml.use-defused-xml  # construction only

from src.shared.python.model_appearance import library
from src.shared.python.model_appearance.schema import Material
from src.shared.python.model_appearance.club_assembly import (
    PART_MATERIALS,
    ClubAssembly,
    assembly_meshes,
)
from src.shared.python.model_appearance.club_head_mesh import load_club_head
from src.shared.python.model_appearance.mesh_io import stl_bytes

OPENSIM_DIR = Path(__file__).resolve().parents[1]
GEOMETRY_DIR = OPENSIM_DIR / "models" / "geometry" / "club"
PROVENANCE_NAME = "provenance.json"
PARTS = ("shaft", "grip", "head")
DEFAULT_FINISH = "satin_steel"
SCHEMA = "opensim-club-visuals-v1"


def asset_filename(alias: str, part: str) -> str:
    """STL file name for ``part`` of the club with head alias ``alias``."""
    if part not in PARTS:
        raise ValueError(f"unknown club part {part!r}; expected one of {PARTS}")
    if not alias or not alias.replace("_", "").isalnum():
        raise ValueError(f"club alias {alias!r} must be alphanumeric")
    return f"club_{alias}_{part}.stl"


def asset_paths(alias: str, directory: Path = GEOMETRY_DIR) -> dict[str, Path]:
    """Absolute path of each part's STL in ``directory``."""
    return {part: Path(directory) / asset_filename(alias, part) for part in PARTS}


def write_club_assets(
    club: ClubAssembly, directory: Path = GEOMETRY_DIR
) -> dict[str, Path]:
    """Write the club's STLs and merge their provenance into the manifest.

    Postconditions: three files exist; the manifest entry for the club records
    the generator, head source, head library name, assembly parameters and the
    sha256 of every file.
    """
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    head = load_club_head(club.head_alias)
    paths = asset_paths(club.head_alias, out)
    digests: dict[str, str] = {}
    for part, mesh in assembly_meshes(club).items():
        payload = stl_bytes(mesh, f"{club.head_alias} {part}; club frame; units=m")
        paths[part].write_bytes(payload)
        digests[part] = hashlib.sha256(payload).hexdigest()
    params = {
        "head_alias": club.head_alias,
        "length_m": club.length_m,
        "grip_length_m": club.grip_length_m,
        "shaft_radius_m": club.shaft_radius_m,
        "axis_offset_m": club.axis_offset_m,
        "face_roll_deg": club.face_roll_deg,
    }
    spec_hash = hashlib.sha256(
        json.dumps(params, sort_keys=True).encode("utf-8")
    ).hexdigest()
    manifest_path = out / PROVENANCE_NAME
    manifest: dict[str, Any] = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {"schema": SCHEMA, "clubs": {}}
    )
    manifest["generator"] = (
        "src.engines.physics_engines.opensim.python.club_visuals.write_club_assets"
    )
    manifest["frame"] = "club body: origin at sole point on shaft axis, shaft -y"
    manifest["units"] = "m"
    manifest["clubs"][club.head_alias] = {
        "spec_sha256": spec_hash,
        "assembly": params,
        "head_library_name": head.library_name,
        "head_source": head.source,
        "files": {p: asset_filename(club.head_alias, p) for p in PARTS},
        "sha256": digests,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return paths


def mesh_element(
    name: str,
    mesh_file: str,
    rgb: tuple[float, float, float],
    socket_frame: str = "..",
) -> ET.Element:
    """One OpenSim ``<Mesh>`` (unit scale, opaque) on ``socket_frame``."""
    mesh = ET.Element("Mesh", attrib={"name": name})
    ET.SubElement(mesh, "socket_frame").text = socket_frame
    ET.SubElement(mesh, "scale_factors").text = "1 1 1"
    app = ET.SubElement(mesh, "Appearance")
    ET.SubElement(app, "opacity").text = "1"
    ET.SubElement(app, "color").text = " ".join(format(c, ".4g") for c in rgb)
    ET.SubElement(mesh, "mesh_file").text = mesh_file
    return mesh


def club_mesh_elements(
    alias: str,
    mesh_dir_ref: str,
    finish: str = DEFAULT_FINISH,
    materials: Mapping[str, Material] | None = None,
    socket_frame: str = "..",
) -> list[ET.Element]:
    """``<Mesh>`` elements for shaft, grip and head, files under ``mesh_dir_ref``.

    ``mesh_dir_ref`` is the directory as the saved model should reference it
    (absolute, or ``""`` for bare file names). OpenSim 4.6 does not resolve
    relative paths containing ``..``, so committed models use bare names and
    loaders call :func:`register_geometry_path`. Colours follow the appearance
    library: club finish for shaft and head, ``grip_rubber`` for the grip.
    """
    table = materials if materials is not None else library.MATERIALS
    if finish not in table:
        raise ValueError(f"unknown club finish {finish!r}")
    elements = []
    for part in PARTS:
        material = table[PART_MATERIALS[part] or finish]
        rgb = (material.rgba[0], material.rgba[1], material.rgba[2])
        name = asset_filename(alias, part)
        file_ref = f"{mesh_dir_ref.rstrip('/')}/{name}" if mesh_dir_ref else name
        elements.append(mesh_element(f"club_{part}_geom", file_ref, rgb, socket_frame))
    return elements


def attach_club_meshes(
    body: ET.Element,
    alias: str,
    mesh_dir_ref: str,
    finish: str = DEFAULT_FINISH,
) -> int:
    """Replace ``body``'s ``attached_geometry`` club meshes; returns the count."""
    geom = body.find("attached_geometry")
    if geom is None:
        geom = ET.SubElement(body, "attached_geometry")
    for old in [m for m in geom if str(m.get("name", "")).startswith("club_")]:
        geom.remove(old)
    new = club_mesh_elements(alias, mesh_dir_ref, finish)
    geom.extend(new)
    return len(new)


def register_geometry_path(directory: Path = GEOMETRY_DIR) -> bool:
    """Add ``directory`` to OpenSim's geometry search path before loading a model.

    Returns ``False`` when the ``opensim`` package is not installed. Idempotent.
    """
    try:
        import opensim
    except ImportError:
        return False
    opensim.ModelVisualizer.addDirToGeometrySearchPaths(str(Path(directory).resolve()))
    return True
