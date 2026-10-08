"""MuJoCo translation of an engine-agnostic appearance document.

Emits ``<asset>`` textures/materials/meshes, smooth visual-only mesh geoms
(skin, garments, shoes, club shaft, grip and mesh head), a skybox, a textured ground, shadowed
studio lights and a 960x720 offscreen buffer. Every geom is class ``visual``
(group 1, no contacts, zero mass), so inertias and dynamics are untouched.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET  # nosec B405 # nosemgrep: python.lang.security.use-defused-xml.use-defused-xml - construction only; parsing is defused
from collections.abc import Mapping
from typing import Any

import numpy as np

from src.shared.python.model_appearance import geometry, library
from src.shared.python.model_appearance.club_assembly import (
    PART_MATERIALS,
    ClubAssembly,
    assembly_meshes,
)
from src.shared.python.model_appearance.schema import (
    AppearanceDocument,
    Material,
    Texture,
    document_from_dict,
)
from src.shared.python.motion_matching.visual_skeleton import (
    Capsule,
    VisualSkeleton,
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


def _club_meshes(
    club: ClubAssembly, finish: str
) -> list[tuple[str, geometry.Mesh, str]]:
    """Shaft, grip and mesh head (GCV-11) in the club-body frame."""
    return [
        (label, mesh, PART_MATERIALS[label] or finish)
        for label, mesh in assembly_meshes(club).items()
    ]


def _body_meshes(
    doc: AppearanceDocument,
    body: str,
    capsules: list[Capsule],
    club: ClubAssembly | None = None,
) -> list[tuple[str, geometry.Mesh, str]]:
    """(label, mesh in spec body frame, material name) for one body."""
    rule = library.rule_for_body(doc, body)
    if rule is not None and rule.mesh == "none":
        return []
    part = library.classify_body(body)
    base = library.material_name_for(doc, body)
    if part == "club" and club is not None:
        return _club_meshes(club, base)
    garment = library.garment_name_for(doc, body)
    scale = 1.0 if rule is None else rule.radius_scale
    out: list[tuple[str, geometry.Mesh, str]] = []
    for i, cap in enumerate(capsules):
        start, end = np.asarray(cap.start_m), np.asarray(cap.end_m)
        radius = scale * (
            cap.radius_m
            if part in ("club", "other")
            else library.PART_RADIUS_M.get(part, cap.radius_m)
        )
        radius = min(radius, 0.6 * cap.length_m())  # keep short links from ballooning
        aspect = library.PART_ASPECT.get(part, 1.0)
        out.append(
            (
                f"skin{i}",
                geometry.lofted_segment(
                    start, end, radius, geometry.LoftOptions(aspect=aspect)
                ),
                base,
            )
        )
        if garment is not None:
            band = library.GARMENT_COVERAGE.get(part, (0.0, 1.0))
            out.append(
                (
                    f"garment{i}",
                    geometry.lofted_segment(
                        start,
                        end,
                        radius,
                        geometry.LoftOptions(
                            aspect=aspect,
                            coverage=band,
                            thickness=library.GARMENT_THICKNESS,
                        ),
                    ),
                    garment,
                )
            )
    return out


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


def attach_appearance(
    root: ET.Element,
    elements: Mapping[str, ET.Element],
    offsets: Mapping[str, np.ndarray],
    skeleton: VisualSkeleton,
    doc: AppearanceDocument,
    club: ClubAssembly | None = None,
) -> dict[str, Any]:
    """Add materials, textures, smooth meshes and scene dressing.

    Returns the summary (mesh and material counts, ground material name via
    ``ground_material``) merged into the exporter metadata. The caller adds
    the ground, lights and camera that use the returned material.
    """
    assets = _Assets(root, doc)
    ground_material = _add_environment(root, assets, doc)
    by_body: dict[str, list[Capsule]] = {}
    for cap in skeleton.capsules:
        by_body.setdefault(cap.body, []).append(cap)
    garments = 0
    for index, (body, caps) in enumerate(by_body.items()):
        for label, mesh, material in _body_meshes(doc, body, caps, club):
            mesh_name = f"vmesh_{index}_{label}"
            assets.mesh(mesh_name, _mesh_to_mjcf(offsets[body], mesh))
            _add_geom(
                elements[body],
                f"visual_{mesh_name}",
                mesh_name,
                assets.material(material),
            )
            garments += label.startswith("garment")
    return {
        "appearance": doc.name,
        "meshes": assets.n_meshes,
        "garments": garments,
        "materials": assets.n_materials,
        "ground_material": ground_material,
        "club_head": None if club is None else club.head_alias,
        "offscreen": list(OFFSCREEN_SIZE),
    }


def attach_club_meshes(
    root: ET.Element,
    elements: Mapping[str, ET.Element],
    offsets: Mapping[str, np.ndarray],
    body: str,
    club: ClubAssembly,
    finish: str = "satin_steel",
) -> int:
    """Add only the shaft, grip and mesh head of ``club`` (plain visual layer).

    For the plain ``visual=True`` export, which keeps its capsule skeleton but
    should still show the real head (GCV-11). Visual only, massless,
    non-colliding. Returns the number of meshes added.
    """
    doc = document_from_dict({"schema_version": "appearance-v1"})
    assets = _Assets(root, doc)
    for label, mesh, material in _club_meshes(club, finish):
        mesh_name = f"vmesh_club_{label}"
        assets.mesh(mesh_name, _mesh_to_mjcf(offsets[body], mesh))
        _add_geom(
            elements[body],
            f"visual_{mesh_name}",
            mesh_name,
            assets.material(material),
        )
    return assets.n_meshes
