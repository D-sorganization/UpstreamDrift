"""Engine-neutral visual scene for the Drake and Pinocchio MeshCat backends.

R9a (#12688, epic #12678). Both backends used to carry their own palette
copies and near-identical capsule/shape loops. This module builds the scene
ONCE from the shared appearance spec (``model_appearance``); each backend is a
thin adapter that translates :class:`VisualItem` records into its own geometry
API (Law of Demeter: adapters never read the spec, the clothing tables or the
appearance document).

Modes
    * Appearance (``appearance`` is a document): the resolved skin, garment,
      blend, shoe and club meshes (``body_meshes``, spec body frame) plus the
      parametric head (or the R7 head slot) replace every capsule and hint
      ellipsoid. Colours are the library materials, the floor is the
      document's ground material.
    * Plain (``appearance`` is ``None``): the grey capsule skeleton and hint
      shapes in ``library.PLAIN_PALETTE``; the head and club meshes stay.

Frame contract
    Items are expressed in the SPEC body frame of ``VisualItem.body``
    (``""`` is the world). Drake's URDF links and Pinocchio's body poses
    differ in how they reach that frame, so each adapter composes its own body
    transform explicitly. Visual only: nothing here touches dynamics.

Design by contract
    The scene is validated at the boundary and fails closed with
    ``ValueError``: an unknown item kind, a colour outside [0, 1], an item on a
    body the engine does not have, or an appearance document that dresses
    nothing while the skeleton has capsules.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.model_appearance import library
from src.shared.python.model_appearance.body_meshes import resolve_appearance
from src.shared.python.model_appearance.club_assembly import assembly_from_spec
from src.shared.python.model_appearance.schema import (
    AppearanceDocument,
    document_from_dict,
)
from src.shared.python.motion_matching.visual_skeleton import derive_visual_skeleton
from src.tools.native_viewer_export.backends._body import body_mesh_files
from src.tools.native_viewer_export.backends._club import club_parts, write_obj
from src.tools.native_viewer_export.backends._head import head_mesh_files
from src.tools.native_viewer_export.backends._scene import z_axis_frame

__all__ = [
    "KINDS",
    "SceneVisuals",
    "VisualItem",
    "build_scene_visuals",
    "check_bodies",
    "default_appearance",
]

KINDS = ("mesh", "capsule", "ellipsoid", "box")
WORLD = ""
RGBA = tuple[float, float, float, float]
Array = NDArray[np.float64]


@dataclass(frozen=True)
class VisualItem:
    """One visual in a spec body frame.

    ``size`` is ``(radius, length)`` for a capsule, the half extents for an
    ellipsoid and the FULL extents for a box (the engines' own conventions);
    a mesh carries ``mesh_path`` instead. ``material`` is the library material
    (or ``plain:<key>``) the colour came from, or ``None`` for head parts.
    """

    name: str
    body: str
    kind: str
    rgba: RGBA
    rotation: Array = field(default_factory=lambda: np.eye(3))
    translation: Array = field(default_factory=lambda: np.zeros(3))
    size: tuple[float, ...] = ()
    mesh_path: Path | None = None
    material: str | None = None
    label: str = ""

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown visual kind {self.kind!r}; known: {KINDS}")
        if len(self.rgba) != 4 or not all(
            math.isfinite(c) and 0.0 <= c <= 1.0 for c in self.rgba
        ):
            raise ValueError(f"{self.name}: rgba must be 4 values in [0, 1]")
        if (self.kind == "mesh") != (self.mesh_path is not None):
            raise ValueError(f"{self.name}: exactly the mesh kind carries a path")
        if self.kind != "mesh" and (
            not self.size or not all(math.isfinite(s) and s > 0 for s in self.size)
        ):
            raise ValueError(f"{self.name}: {self.kind} needs positive finite sizes")


@dataclass(frozen=True)
class SceneVisuals:
    """Every visual of the spec robot plus the floor."""

    items: tuple[VisualItem, ...]
    ground_height_m: float
    floor_rgba: RGBA
    appearance: bool

    def bodies(self) -> set[str]:
        return {item.body for item in self.items}


def default_appearance() -> AppearanceDocument:
    """The default appearance document every engine renders (as MuJoCo does)."""
    return document_from_dict({"schema_version": "appearance-v1"})


def check_bodies(scene: SceneVisuals, known: Collection[str]) -> None:
    """Fail closed when an item sits on a body the engine does not have."""
    missing = sorted(b for b in scene.bodies() if b != WORLD and b not in known)
    if missing:
        raise ValueError(f"visuals on bodies the engine lacks: {missing}")


def _rgba(values: Any) -> RGBA:
    r, g, b, a = (float(v) for v in values)
    return (r, g, b, a)


def _mesh_item(
    index: int,
    body: str,
    label: str,
    path: Path,
    rgba: RGBA,
    material: str | None,
) -> VisualItem:
    return VisualItem(
        f"{index:03d}_{label}",
        body,
        "mesh",
        _rgba(rgba),
        mesh_path=path,
        material=material,
        label=label,
    )


def _ground_rgba(doc: AppearanceDocument) -> RGBA:
    environment = doc.environment
    materials = library.library_materials(doc)
    name = environment.ground_material
    if name not in materials:
        raise ValueError(f"ground material {name!r} is not in the library")
    return _rgba(materials[name].rgba)


def _head_items(
    spec: Mapping[str, Any], directory: Path, doc: AppearanceDocument | None
) -> list[VisualItem]:
    files = head_mesh_files(spec, directory / "head", doc)
    return [
        _mesh_item(0, f.body, f"head_{f.name}", f.path, f.rgba, None) for f in files
    ]


def _appearance_items(
    spec: Mapping[str, Any], directory: Path, doc: AppearanceDocument
) -> list[VisualItem]:
    """Head parts plus every resolved body, garment, shoe and club mesh."""
    resolved = resolve_appearance(spec, doc, assembly_from_spec(spec))
    files = body_mesh_files(resolved, directory / "body")
    items = _head_items(spec, directory, doc)
    items += [
        _mesh_item(i, f.body, f.label, f.path, f.rgba, f.material_name)
        for i, f in enumerate(files, start=1)
    ]
    return items


def _club_items(spec: Mapping[str, Any], directory: Path) -> list[VisualItem]:
    club = club_parts(spec)
    if club is None:
        return []
    body, parts = club
    (directory / "club").mkdir(parents=True, exist_ok=True)
    out = []
    for i, part in enumerate(parts):
        path = Path(write_obj(part.mesh, str(directory / "club"), part.name))
        out.append(_mesh_item(i, body, part.name, path, part.rgba, None))
    return out


def _plain_items(
    spec: Mapping[str, Any], directory: Path, heads: list[VisualItem]
) -> list[VisualItem]:
    """The grey skeleton; head and club meshes replace their capsules/hints."""
    skeleton = derive_visual_skeleton(spec)
    club = _club_items(spec, directory)
    head_body = heads[0].body if heads else None
    club_body = club[0].body if club else None
    items = list(heads) + club
    capsule = library.plain_rgba("capsule")
    shape_rgba = library.plain_rgba("shape")
    for n, cap in enumerate(skeleton.capsules):
        if cap.body == head_body:
            continue
        rot, centre, length = z_axis_frame(cap.start_m, cap.end_m)
        items.append(
            VisualItem(
                f"cap{n}",
                cap.body,
                "capsule",
                capsule,
                rot,
                centre,
                (cap.radius_m, length),
                material="plain:capsule",
                label=f"cap{n}",
            )
        )
    for n, shp in enumerate(skeleton.shapes):
        if shp.body == club_body:
            continue
        half = tuple(float(h) for h in shp.half_size_m)
        ellipsoid = shp.kind == "ellipsoid"
        items.append(
            VisualItem(
                f"shp{n}",
                shp.body,
                "ellipsoid" if ellipsoid else "box",
                shape_rgba,
                translation=np.asarray(shp.center_m, dtype=float),
                size=half if ellipsoid else tuple(2.0 * h for h in half),
                material="plain:shape",
                label=f"shp{n}",
            )
        )
    return items


def build_scene_visuals(
    spec: Mapping[str, Any],
    directory: Path,
    appearance: AppearanceDocument | None,
) -> SceneVisuals:
    """The visuals of ``spec`` under ``appearance`` (``None`` = plain mode).

    Preconditions: ``spec`` is a full-body spec; ``appearance`` names only
    library or document materials. Postconditions: deterministic; OBJ files
    written under ``directory``; in appearance mode no capsule or hint shape
    remains; fails closed when the document dresses nothing.
    """
    directory.mkdir(parents=True, exist_ok=True)
    skeleton = derive_visual_skeleton(spec)
    if appearance is None:
        items = _plain_items(spec, directory, _head_items(spec, directory, None))
        floor = library.plain_rgba("floor")
    else:
        items = _appearance_items(spec, directory, appearance)
        if skeleton.capsules and not any(i.material for i in items):
            raise ValueError("the appearance document dressed no body of the spec")
        floor = _ground_rgba(appearance)
    return SceneVisuals(
        tuple(items), float(skeleton.ground.height_m), floor, appearance is not None
    )
