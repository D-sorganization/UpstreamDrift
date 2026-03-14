"""
Tests for URDF XML formatting behavior.

Verifies that the pretty-print path does NOT inject an XML declaration
(minidom's well-known issue), produces properly indented output, and
generates valid XML that can be round-tripped through ElementTree.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest
from humanoid_character_builder.core.body_parameters import BodyParameters
from humanoid_character_builder.generators.urdf_generator import (
    HumanoidURDFGenerator,
    URDFGeneratorConfig,
)


@pytest.fixture()
def default_params() -> BodyParameters:
    return BodyParameters(name="test_robot")


@pytest.fixture()
def pretty_generator() -> HumanoidURDFGenerator:
    config = URDFGeneratorConfig(pretty_print=True, indent="  ")
    return HumanoidURDFGenerator(config)


@pytest.fixture()
def compact_generator() -> HumanoidURDFGenerator:
    config = URDFGeneratorConfig(pretty_print=False)
    return HumanoidURDFGenerator(config)


class TestPrettyPrintNoXmlDeclaration:
    """When pretty_print=True, output must NOT contain an XML declaration."""

    def test_no_xml_declaration(
        self, pretty_generator: HumanoidURDFGenerator, default_params: BodyParameters
    ) -> None:
        urdf_xml = pretty_generator.generate(default_params)
        assert not urdf_xml.startswith("<?xml"), (
            "pretty_print output must not start with an XML declaration "
            "(<?xml ...?>) — use ET.indent() instead of minidom.toprettyxml()"
        )

    def test_no_xml_declaration_anywhere(
        self, pretty_generator: HumanoidURDFGenerator, default_params: BodyParameters
    ) -> None:
        urdf_xml = pretty_generator.generate(default_params)
        assert (
            "<?xml" not in urdf_xml
        ), "XML declaration must not appear anywhere in pretty_print output"


class TestPrettyPrintIndentation:
    """When pretty_print=True, output must be properly indented."""

    def test_output_is_indented(
        self, pretty_generator: HumanoidURDFGenerator, default_params: BodyParameters
    ) -> None:
        urdf_xml = pretty_generator.generate(default_params)
        lines = urdf_xml.splitlines()
        # At least some lines should start with spaces (indentation)
        indented_lines = [line for line in lines if line.startswith("  ")]
        assert (
            len(indented_lines) > 0
        ), "pretty_print output must contain indented lines"

    def test_child_elements_are_indented_under_robot(
        self, pretty_generator: HumanoidURDFGenerator, default_params: BodyParameters
    ) -> None:
        urdf_xml = pretty_generator.generate(default_params)
        lines = urdf_xml.splitlines()
        # <link and <joint elements should be indented (child of <robot>)
        link_lines = [line for line in lines if "<link " in line]
        assert len(link_lines) > 0, "Should have link elements"
        for line in link_lines:
            assert line.startswith(
                "  "
            ), f"<link> element should be indented with 2 spaces, got: {line!r}"


class TestPrettyPrintNoBlankLines:
    """Output must not contain blank lines (a known minidom issue)."""

    def test_no_blank_lines_in_pretty_output(
        self, pretty_generator: HumanoidURDFGenerator, default_params: BodyParameters
    ) -> None:
        urdf_xml = pretty_generator.generate(default_params)
        blank_lines = [line for line in urdf_xml.splitlines() if line.strip() == ""]
        assert len(blank_lines) == 0, (
            f"pretty_print output must not contain blank lines; "
            f"found {len(blank_lines)} blank line(s)"
        )


class TestCompactOutput:
    """When pretty_print=False, output must be compact (no indentation)."""

    def test_compact_has_no_leading_whitespace_on_elements(
        self,
        compact_generator: HumanoidURDFGenerator,
        default_params: BodyParameters,
    ) -> None:
        urdf_xml = compact_generator.generate(default_params)
        # Compact output should not contain newline-indented child elements
        assert "\n  " not in urdf_xml, "compact output must not contain indented lines"

    def test_compact_no_xml_declaration(
        self,
        compact_generator: HumanoidURDFGenerator,
        default_params: BodyParameters,
    ) -> None:
        urdf_xml = compact_generator.generate(default_params)
        assert (
            "<?xml" not in urdf_xml
        ), "compact output must not contain an XML declaration"


class TestValidXmlOutput:
    """Output must be valid XML that can be parsed by ElementTree."""

    def test_pretty_output_is_valid_xml(
        self, pretty_generator: HumanoidURDFGenerator, default_params: BodyParameters
    ) -> None:
        urdf_xml = pretty_generator.generate(default_params)
        # Should not raise
        root = ET.fromstring(urdf_xml)
        assert root.tag == "robot"

    def test_compact_output_is_valid_xml(
        self,
        compact_generator: HumanoidURDFGenerator,
        default_params: BodyParameters,
    ) -> None:
        urdf_xml = compact_generator.generate(default_params)
        root = ET.fromstring(urdf_xml)
        assert root.tag == "robot"

    def test_pretty_output_has_links_and_joints(
        self, pretty_generator: HumanoidURDFGenerator, default_params: BodyParameters
    ) -> None:
        urdf_xml = pretty_generator.generate(default_params)
        root = ET.fromstring(urdf_xml)
        links = root.findall("link")
        joints = root.findall("joint")
        assert len(links) > 0, "Must have at least one link"
        assert len(joints) > 0, "Must have at least one joint"
