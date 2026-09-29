"""URDF generation driven by a body_part_viz ShapeLibrary (#4765).

Exercises the end-to-end wiring: ``URDFGeneratorConfig.shape_library`` ->
``HumanoidURDFGenerator._lookup_library_shape`` ->
``urdf_geometry.create_geometry_dict(shape=...)`` ->
``urdf_bridge.shape_to_urdf_visual``.
"""

from __future__ import annotations

import defusedxml.ElementTree as ET  # nosec B405 — defusedxml is the safe parser
from humanoid_character_builder.core.body_parameters import BodyParameters
from humanoid_character_builder.generators.urdf_config import URDFGeneratorConfig
from humanoid_character_builder.generators.urdf_generator import HumanoidURDFGenerator

from src.shared.python.body_part_viz.asset_library import ShapeLibrary


def _strip_xml_declaration(xml_str: str) -> str:
    if xml_str.startswith("<?xml"):
        return xml_str[xml_str.index("?>") + 2 :]
    return xml_str


def _link_visual(root: ET.Element, link_name: str) -> ET.Element | None:
    for link in root.findall("link"):
        if link.get("name") == link_name:
            return link.find("visual/geometry")
    return None


def test_default_library_drives_link_visuals() -> None:
    library = ShapeLibrary.default()
    config = URDFGeneratorConfig(shape_library=library, generate_collision=False)
    generator = HumanoidURDFGenerator(config)
    params = BodyParameters()

    urdf_xml = generator.generate(params)
    root = ET.fromstring(_strip_xml_declaration(urdf_xml))

    # Head should be a mesh referencing the bundled head asset.
    head_geom = _link_visual(root, "head")
    assert head_geom is not None
    head_mesh = head_geom.find("mesh")
    assert head_mesh is not None
    assert "head.stl" in head_mesh.get("filename", "")
    assert "package://body_part_viz/" in head_mesh.get("filename", "")

    # Both upper_arm sides should resolve to the same library asset.
    for link_name in ("left_upper_arm", "right_upper_arm"):
        geom = _link_visual(root, link_name)
        assert geom is not None, f"missing visual for {link_name}"
        mesh = geom.find("mesh")
        assert mesh is not None, f"{link_name} visual is not a mesh"
        assert "upper_arm.stl" in mesh.get("filename", "")


def test_no_shape_library_keeps_legacy_payload() -> None:
    config = URDFGeneratorConfig(shape_library=None, generate_collision=False)
    generator = HumanoidURDFGenerator(config)
    urdf_xml = generator.generate(BodyParameters())
    root = ET.fromstring(_strip_xml_declaration(urdf_xml))

    # Without a library the existing behaviour should be untouched: the
    # head segment uses a primitive (sphere) per the legacy mapping.
    head_geom = _link_visual(root, "head")
    assert head_geom is not None
    # Either sphere (legacy) or another primitive — just assert no mesh.
    assert head_geom.find("mesh") is None


def test_segments_without_library_entry_fall_back() -> None:
    library = ShapeLibrary.default()
    config = URDFGeneratorConfig(shape_library=library, generate_collision=False)
    generator = HumanoidURDFGenerator(config)
    urdf_xml = generator.generate(BodyParameters())
    root = ET.fromstring(_strip_xml_declaration(urdf_xml))

    # The bundled library doesn't have a "pelvis" entry; the link must
    # still exist with a non-mesh primitive payload.
    pelvis_geom = _link_visual(root, "pelvis")
    assert pelvis_geom is not None
    assert pelvis_geom.find("mesh") is None
