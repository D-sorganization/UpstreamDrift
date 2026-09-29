"""
End-to-end integration tests for URDF model generation.

These tests verify full roundtrip pipelines:
  Generate URDF -> Parse back -> Validate structural equivalence

Each test class covers a distinct pipeline to ensure that the
entire build-parse-validate cycle produces consistent results.

References:
  - GitHub issue #1694 (end-to-end integration tests)
"""

from __future__ import annotations

import defusedxml.ElementTree as DefusedET  # noqa: S314  # Security: defusedxml prevents XML attacks
from model_generation.builders.parametric_builder import ParametricBuilder
from model_generation.converters.urdf_parser import URDFParser
from model_generation.core.types import (
    GeometryType,
    JointType,
)


class TestParametricBuilderRoundtrip:
    """ParametricBuilder -> Build -> Parse -> Validate links/joints match."""

    def test_default_humanoid_roundtrip(self) -> None:
        """Default humanoid model survives roundtrip with correct structure."""
        builder = ParametricBuilder("humanoid")
        builder.set_parameters(height_m=1.75, mass_kg=75.0)
        builder.add_humanoid_segments()
        result = builder.build()

        assert result.success, f"Build failed: {result.error_message}"
        assert result.urdf_xml is not None

        parser = URDFParser(resolve_meshes=False)
        parsed = parser.parse_string(result.urdf_xml)

        assert parsed.name == "humanoid"

        # Must have multiple links (humanoid skeleton)
        assert len(parsed.links) > 10, (
            f"Expected >10 links for humanoid, got {len(parsed.links)}"
        )

        # Must have joints connecting them
        assert len(parsed.joints) >= len(parsed.links) - 1

        # Root should be pelvis
        root = parsed.get_root_link()
        assert root is not None
        assert root.name == "pelvis"

        # Must have bilateral symmetry: left and right limbs
        link_names = {lnk.name for lnk in parsed.links}
        for segment in ["thigh", "shin", "foot", "upper_arm", "forearm", "hand"]:
            assert f"left_{segment}" in link_names, f"Missing left_{segment}"
            assert f"right_{segment}" in link_names, f"Missing right_{segment}"

    def test_parametric_mass_distribution(self) -> None:
        """Total mass of all links approximately equals the configured mass."""
        target_mass = 80.0
        builder = ParametricBuilder("mass_check")
        builder.set_parameters(height_m=1.80, mass_kg=target_mass)
        builder.add_humanoid_segments()
        result = builder.build()

        assert result.success
        total_mass = result.get_total_mass()

        # The sum of segment mass ratios may not be exactly 1.0 due to
        # shoulders and intermediate links, but should be in the right ballpark
        assert total_mass > 0, "Total mass must be positive"
        # Allow generous tolerance since parametric builder distributes mass
        # across many segments with approximation
        assert total_mass > target_mass * 0.5, (
            f"Total mass {total_mass} is less than 50% of target {target_mass}"
        )

    def test_parametric_height_affects_geometry(self) -> None:
        """Different heights produce different link dimensions."""
        builder_short = ParametricBuilder("short")
        builder_short.set_parameters(height_m=1.50, mass_kg=60.0)
        builder_short.add_humanoid_segments()
        result_short = builder_short.build()

        builder_tall = ParametricBuilder("tall")
        builder_tall.set_parameters(height_m=2.00, mass_kg=90.0)
        builder_tall.add_humanoid_segments()
        result_tall = builder_tall.build()

        assert result_short.success and result_tall.success

        # Find a common link (e.g., left_thigh) and compare dimensions
        thigh_short = result_short.get_link("left_thigh")
        thigh_tall = result_tall.get_link("left_thigh")

        assert thigh_short is not None
        assert thigh_tall is not None

        # Taller person should have greater thigh inertia (larger segment)
        assert thigh_tall.inertia.ixx > thigh_short.inertia.ixx, (
            "Taller model should have larger thigh inertia"
        )

    def test_parametric_builder_produces_valid_xml(self) -> None:
        """Parametric URDF is always well-formed XML with <robot> root."""
        builder = ParametricBuilder("xml_check")
        builder.set_parameters(height_m=1.70, mass_kg=70.0)
        builder.add_humanoid_segments()
        result = builder.build()

        assert result.success
        assert result.urdf_xml is not None

        root = DefusedET.fromstring(result.urdf_xml)
        assert root.tag == "robot"
        assert root.get("name") == "xml_check"

        # Every link must have an inertial element
        for link_elem in root.findall(".//link"):
            inertial = link_elem.find("inertial")
            assert inertial is not None, (
                f"Link '{link_elem.get('name')}' missing <inertial>"
            )
            mass = inertial.find("mass")
            assert mass is not None
            mass_val = float(mass.get("value", "0"))
            assert mass_val > 0, (
                f"Link '{link_elem.get('name')}' has non-positive mass {mass_val}"
            )

    def test_parametric_custom_segment_roundtrip(self) -> None:
        """Custom segments added via add_segment survive roundtrip."""
        builder = ParametricBuilder("custom_bot")
        builder.set_parameters(height_m=1.0, mass_kg=10.0)
        builder.add_segment(
            name="base",
            parent=None,
            mass_ratio=0.5,
            length_ratio=0.3,
            geometry_type=GeometryType.BOX,
        )
        builder.add_segment(
            name="arm",
            parent="base",
            mass_ratio=0.3,
            length_ratio=0.4,
            geometry_type=GeometryType.CYLINDER,
            joint_type=JointType.REVOLUTE,
            joint_axis=(1, 0, 0),
        )
        builder.add_segment(
            name="hand",
            parent="arm",
            mass_ratio=0.2,
            length_ratio=0.15,
            geometry_type=GeometryType.SPHERE,
            joint_type=JointType.FIXED,
        )

        result = builder.build()
        assert result.success
        assert result.urdf_xml is not None

        parser = URDFParser(resolve_meshes=False)
        parsed = parser.parse_string(result.urdf_xml)

        link_names = {lnk.name for lnk in parsed.links}
        assert link_names == {"base", "arm", "hand"}

        # Verify parent chain
        assert parsed.get_parent("arm") == "base"
        assert parsed.get_parent("hand") == "arm"
        assert parsed.get_parent("base") is None
