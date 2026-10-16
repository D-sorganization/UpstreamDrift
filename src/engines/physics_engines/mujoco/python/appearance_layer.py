"""MuJoCo translation of an engine-agnostic appearance document.

Emits ``<asset>`` textures/materials/meshes, smooth visual-only mesh geoms
(skin, garments, shoes, club shaft, grip, mesh head and grip hands), a skybox, a textured ground, shadowed
studio lights and a 960x720 offscreen buffer. Every geom is class ``visual``
(group 1, no contacts, zero mass), so inertias and dynamics are untouched.

The body meshes themselves come resolved from the shared, engine-neutral
``model_appearance.body_meshes`` (R8, #12685); this module only translates
them. They are in the SPEC body frame, and the MJCF body is the joint
follower frame, so every vertex is mapped explicitly through the exporter's
``offsets[body]`` (``inv(child_to_follower)``) before it is written.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET  # nosec B405 # nosemgrep: python.lang.security.use-defused-xml.use-defused-xml - construction only; parsing is defused
from collections.abc import Mapping
from typing import Any

import numpy as np

from src.shared.python.model_appearance import geometry, library
from src.shared.python.model_appearance.body_meshes import (
    SPEC_BODY_FRAME,
    ResolvedAppearance,
    club_meshes,
    resolve_appearance,
)
from src.shared.python.model_appearance.club_assembly import ClubAssembly
from src.shared.python.model_appearance.schema import (
    AppearanceDocument,
    Material,
    Texture,
    document_from_dict,
)

OFFSCREEN_SIZE = (960, 720)
_CLASS = "visual"


def _nums(values: Any) -> str:
    return " ".join(format(float(v), ".6g") for v in np.asarray(values).ravel())


def _rgb(color: Any) -> str:
    return _nums(np.asarray(color, dtype=float)[:3])


class _Assets:
    """Collects deduplicated texture/material/mesh assets for one document."""

    def __init__(self, root: ET.Element, doc: AppearanceDocument) -> None:
        asset = root.find("asset")
        if asset is None:
            asset = ET.SubElement(root, "asset")
        if asset is None:  # pragma: no cover - SubElement always returns an element
            raise ValueError("MJCF root has no asset element")
        self.asset: ET.Element = asset
        self.materials = library.library_materials(doc)
        self._made: set[str] = set()
        self.n_meshes = 0

    @property
    def n_materials(self) -> int:
        return len(self._made)

    def material(self, name: str, *, plane: bool = False) -> str:
        key = f"{name}:plane" if plane else name
        mat_name = f"mat_{name}{'_plane' if plane else ''}"
        if key in self._made:
            return mat_name
        material = self.materials[name]
        attrs = _material_attrs(material, plane)
        tex = material.texture
        if tex is not None and tex.kind != "flat":
            tex_name = f"tex_{name}{'_plane' if plane else ''}"
            ET.SubElement(
                self.asset,
                "texture",
                attrib=dict(_texture_attrs(tex_name, material, plane)),
            )
            attrs["texture"] = tex_name
            if plane:
                attrs["texrepeat"] = f"{tex.repeat:g} {tex.repeat:g}"
                attrs["texuniform"] = "true"
        ET.SubElement(self.asset, "material", attrib={"name": mat_name, **attrs})
        self._made.add(key)
        return mat_name

    def mesh(self, name: str, mesh: geometry.Mesh) -> None:
        ET.SubElement(
            self.asset,
            "mesh",
            name=name,
            vertex=_nums(mesh.vertices),
            face=" ".join(str(int(i)) for i in mesh.faces.ravel()),
        )
        self.n_meshes += 1


def _material_attrs(material: Material, plane: bool) -> dict[str, str]:
    smooth = 1.0 - material.roughness
    attrs = {
        "rgba": _nums(material.rgba),
        "specular": _nums(min(1.0, 0.1 + 0.55 * smooth + 0.35 * material.metallic)),
        "shininess": _nums(max(0.02, smooth**1.5)),
        "metallic": _nums(material.metallic),
        "roughness": _nums(material.roughness),
    }
    if plane:
        attrs["reflectance"] = _nums(0.12 * smooth)
    return attrs


def _texture_attrs(name: str, material: Material, plane: bool) -> dict[str, str]:
    tex: Texture = material.texture  # type: ignore[assignment]
    a = tex.color_a or material.base_color
    b = tex.color_b or tuple(min(1.0, c * 1.12) for c in material.base_color)
    attrs = {
        "name": name,
        "type": "2d" if plane else "cube",
        "width": "512" if plane else "128",
        "height": "512" if plane else "128",
        "rgb1": _rgb(a),
        "rgb2": _rgb(b),
    }
    if tex.kind == "checker":
        attrs["builtin"] = "checker"
    elif tex.kind == "gradient":
        attrs["builtin"] = "gradient"
    else:  # noise, or a file fallback that has no procedural form
        attrs["builtin"] = "flat"
        attrs["mark"] = "random"
        attrs["markrgb"] = _rgb(b)
        attrs["random"] = _nums(max(0.005, tex.noise))
    if tex.kind == "file":
        raise ValueError(
            "File textures are not supported by the MuJoCo layer; use a "
            "procedural texture kind"
        )
    return attrs


def _add_geom(body: ET.Element, name: str, mesh: str, material: str) -> None:
    ET.SubElement(
        body,
        "geom",
        name=name,
        type="mesh",
        mesh=mesh,
        material=material,
        attrib={"class": _CLASS},
    )


def _world_points(offset: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ offset[:3, :3].T + offset[:3, 3]


def _mesh_to_mjcf(offset: np.ndarray, mesh: geometry.Mesh) -> geometry.Mesh:
    return geometry.Mesh(_world_points(offset, mesh.vertices), mesh.faces)


def _add_environment(
    root: ET.Element,
    assets: _Assets,
    doc: AppearanceDocument,
) -> str:
    env = doc.environment
    ET.SubElement(
        assets.asset,
        "texture",
        name="skybox",
        type="skybox",
        builtin="gradient",
        rgb1=_rgb(env.sky_top),
        rgb2=_rgb(env.sky_bottom),
        width="512",
        height="3072",
    )
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    lit = env.lighting != "flat"
    ET.SubElement(
        visual,
        "global",
        offwidth=str(OFFSCREEN_SIZE[0]),
        offheight=str(OFFSCREEN_SIZE[1]),
    )
    ET.SubElement(
        visual,
        "headlight",
        ambient="0.55 0.55 0.58" if lit else "0.7 0.7 0.7",
        diffuse="0.35 0.35 0.35" if lit else "0.2 0.2 0.2",
        specular="0.05 0.05 0.05",
    )
    ET.SubElement(visual, "quality", shadowsize="4096", offsamples="8")
    haze = _rgb(env.sky_bottom) + " 1"
    ET.SubElement(visual, "rgba", haze=haze)
    ET.SubElement(visual, "map", haze="0.25", znear="0.02", zfar="40")
    return assets.material(env.ground_material, plane=True)


def _attach_body_meshes(
    assets: _Assets,
    elements: Mapping[str, ET.Element],
    offsets: Mapping[str, np.ndarray],
    resolved: ResolvedAppearance,
) -> int:
    """Translate every resolved body mesh into an MJCF mesh geom; garment count.

    Precondition: the resolution is in the spec body frame (DbC); each mesh is
    mapped into the MJCF body frame with ``offsets[body]``.
    """
    if resolved.frame != SPEC_BODY_FRAME:
        raise ValueError(f"expected spec-body-frame meshes, got {resolved.frame!r}")
    index_of: dict[str, int] = {}
    garments = 0
    for item in resolved.meshes:
        index = index_of.setdefault(item.body, len(index_of))
        _add_mesh_visual(
            assets,
            elements[item.body],
            f"vmesh_{index}_{item.label}",
            offsets[item.body],
            item.mesh_in_body_frame,
            item.material_name,
        )
        garments += item.label.startswith("garment")
    return garments


def attach_appearance(
    root: ET.Element,
    elements: Mapping[str, ET.Element],
    offsets: Mapping[str, np.ndarray],
    doc: AppearanceDocument,
    club: ClubAssembly | None = None,
    *,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Add materials, textures, smooth meshes and scene dressing.

    ``spec`` is the body/joint document; ``model_appearance.body_meshes``
    resolves its meshes (trunk sized from its ``visual_hints``, R8 #12685)
    and its landmarks orient the trunk, pelvis and shoe cross-sections (R1).

    Returns the summary (mesh and material counts, ground material name via
    ``ground_material``) merged into the exporter metadata. The caller adds
    the ground, lights and camera that use the returned material.
    """
    if doc.body_model != "ellipsoid":
        raise ValueError(
            f"body_model {doc.body_model!r} needs a rigid-bind mesh set that is "
            "not available yet (CMB-6 #11657); use 'ellipsoid'"
        )
    assets = _Assets(root, doc)
    ground_material = _add_environment(root, assets, doc)
    garments = _attach_body_meshes(
        assets, elements, offsets, resolve_appearance(spec, doc, club)
    )
    return {
        "appearance": doc.name,
        "meshes": assets.n_meshes,
        "garments": garments,
        "materials": assets.n_materials,
        "ground_material": ground_material,
        "club_head": None if club is None else club.head_alias,
        "offscreen": list(OFFSCREEN_SIZE),
    }


def _add_mesh_visual(
    assets: _Assets,
    parent: ET.Element,
    mesh_name: str,
    offset: np.ndarray,
    mesh: Any,
    material: Any,
) -> None:
    """Register ``mesh`` as an asset and attach a visual geom to ``parent``."""
    assets.mesh(mesh_name, _mesh_to_mjcf(offset, mesh))
    _add_geom(parent, f"visual_{mesh_name}", mesh_name, assets.material(material))


def attach_club_meshes(
    root: ET.Element,
    elements: Mapping[str, ET.Element],
    offsets: Mapping[str, np.ndarray],
    body: str,
    club: ClubAssembly,
    finish: str = "satin_steel",
) -> int:
    """Add only the shaft, grip, mesh head and grip hands of ``club`` (plain layer).

    For the plain ``visual=True`` export, which keeps its capsule skeleton but
    should still show the real head (GCV-11) and both hands wrapped on the
    grip (OSV-2). Visual only, massless,
    non-colliding. Returns the number of meshes added.
    """
    doc = document_from_dict({"schema_version": "appearance-v1"})
    assets = _Assets(root, doc)
    for label, mesh, material in club_meshes(club, finish):
        _add_mesh_visual(
            assets, elements[body], f"vmesh_club_{label}", offsets[body], mesh, material
        )
    return assets.n_meshes
