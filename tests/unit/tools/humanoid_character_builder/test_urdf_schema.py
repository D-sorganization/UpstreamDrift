"""Test URDF schema validation for humanoid character builder.

lxml is optional; skip the whole module when missing. Core schema
contracts are also covered in test_urdf_quality.py.
"""

import pytest

etree = pytest.importorskip("lxml.etree")

from humanoid_character_builder import CharacterBuilder  # noqa: E402
from humanoid_character_builder.core.body_parameters import BodyParameters  # noqa: E402
from humanoid_character_builder.presets.loader import (
    list_available_presets,
)  # noqa: E402

# URDF XML namespace and root element validation
URDF_ROOT_TAG = "robot"
REQUIRED_ELEMENTS = ["link", "joint"]


class TestURDFSchemaValidation:
    """Test that generated URDFs are valid XML conforming to URDF spec."""

    @pytest.mark.parametrize("preset", list_available_presets())
    def test_urdf_is_valid_xml(self, preset: str) -> None:
        """Generated URDF should be valid XML."""
        builder = CharacterBuilder()
        params = builder.create_from_preset(preset)
        urdf_xml = builder.generate_urdf(params)

        # This will raise etree.XMLSyntaxError if invalid
        tree = etree.fromstring(urdf_xml.encode("utf-8"))
        assert tree is not None, f"URDF for preset '{preset}' is not valid XML"

    @pytest.mark.parametrize("preset", list_available_presets())
    def test_urdf_has_robot_root(self, preset: str) -> None:
        """Generated URDF should have <robot> as root element."""
        builder = CharacterBuilder()
        params = builder.create_from_preset(preset)
        urdf_xml = builder.generate_urdf(params)

        tree = etree.fromstring(urdf_xml.encode("utf-8"))
        assert (
            tree.tag == URDF_ROOT_TAG
        ), f"URDF for preset '{preset}' should have <robot> root, got <{tree.tag}>"

    @pytest.mark.parametrize("preset", list_available_presets())
    def test_urdf_has_required_elements(self, preset: str) -> None:
        """Generated URDF should have required link and joint elements."""
        builder = CharacterBuilder()
        params = builder.create_from_preset(preset)
        urdf_xml = builder.generate_urdf(params)

        tree = etree.fromstring(urdf_xml.encode("utf-8"))

        links = tree.findall(".//link")
        joints = tree.findall(".//joint")

        assert len(links) > 0, f"URDF for preset '{preset}' has no link elements"
        assert len(joints) > 0, f"URDF for preset '{preset}' has no joint elements"

    def test_urdf_has_valid_link_names(self) -> None:
        """All link elements should have valid name attributes."""
        builder = CharacterBuilder()
        params = BodyParameters(height_m=1.75, mass_kg=75.0)
        urdf_xml = builder.generate_urdf(params)

        tree = etree.fromstring(urdf_xml.encode("utf-8"))
        links = tree.findall(".//link")

        for link in links:
            name = link.get("name")
            assert name is not None, "Link element missing name attribute"
            assert len(name) > 0, "Link name attribute is empty"

    def test_urdf_has_valid_joint_structure(self) -> None:
        """All joint elements should have required attributes and children."""
        builder = CharacterBuilder()
        params = BodyParameters(height_m=1.75, mass_kg=75.0)
        urdf_xml = builder.generate_urdf(params)

        tree = etree.fromstring(urdf_xml.encode("utf-8"))
        joints = tree.findall(".//joint")

        for joint in joints:
            name = joint.get("name")
            assert name is not None, "Joint element missing name attribute"

            joint_type = joint.get("type")
            assert joint_type is not None, f"Joint '{name}' missing type attribute"
            assert joint_type in (
                "revolute",
                "continuous",
                "prismatic",
                "fixed",
                "floating",
                "planar",
            ), f"Joint '{name}' has invalid type: {joint_type}"

            # Check for parent and child elements
            parent = joint.find("parent")
            child = joint.find("child")
            assert parent is not None, f"Joint '{name}' missing parent element"
            assert child is not None, f"Joint '{name}' missing child element"
            assert (
                parent.get("link") is not None
            ), f"Joint '{name}' parent missing link attribute"
            assert (
                child.get("link") is not None
            ), f"Joint '{name}' child missing link attribute"

    def test_urdf_no_duplicate_link_names(self) -> None:
        """All link names should be unique."""
        builder = CharacterBuilder()
        params = BodyParameters(height_m=1.75, mass_kg=75.0)
        urdf_xml = builder.generate_urdf(params)

        tree = etree.fromstring(urdf_xml.encode("utf-8"))
        links = tree.findall(".//link")
        link_names = [link.get("name") for link in links]

        assert len(link_names) == len(set(link_names)), "Duplicate link names found"

    def test_urdf_no_duplicate_joint_names(self) -> None:
        """All joint names should be unique."""
        builder = CharacterBuilder()
        params = BodyParameters(height_m=1.75, mass_kg=75.0)
        urdf_xml = builder.generate_urdf(params)

        tree = etree.fromstring(urdf_xml.encode("utf-8"))
        joints = tree.findall(".//joint")
        joint_names = [joint.get("name") for joint in joints]

        assert len(joint_names) == len(set(joint_names)), "Duplicate joint names found"

    def test_urdf_xml_encoding_is_utf8(self) -> None:
        """URDF should be valid UTF-8 encoded XML."""
        builder = CharacterBuilder()
        params = BodyParameters(height_m=1.75, mass_kg=75.0)
        urdf_xml = builder.generate_urdf(params)

        # Should not raise UnicodeDecodeError
        urdf_xml.encode("utf-8").decode("utf-8")

        # Parse should succeed
        tree = etree.fromstring(urdf_xml.encode("utf-8"))
        assert tree is not None
