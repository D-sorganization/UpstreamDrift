"""
Unit tests for models module.

Tests model XML generation functions.
"""

import pytest
from mujoco_humanoid_golf.models import (
    DOUBLE_PENDULUM_XML,
    GRAVITY_M_S2,
    TRIPLE_PENDULUM_XML,
    generate_flexible_club_xml,
    generate_rigid_club_xml,
)


class TestModelXML:
    """Tests for model XML strings."""

    def test_double_pendulum_xml_exists(self) -> None:
        """Test double pendulum XML is defined."""
        assert DOUBLE_PENDULUM_XML is not None
        assert isinstance(DOUBLE_PENDULUM_XML, str)
        assert len(DOUBLE_PENDULUM_XML) > 0
        assert "mujoco" in DOUBLE_PENDULUM_XML.lower()

    def test_triple_pendulum_xml_exists(self) -> None:
        """Test triple pendulum XML is defined."""
        assert TRIPLE_PENDULUM_XML is not None
        assert isinstance(TRIPLE_PENDULUM_XML, str)
        assert len(TRIPLE_PENDULUM_XML) > 0
        assert "mujoco" in TRIPLE_PENDULUM_XML.lower()

    def test_gravity_constant(self) -> None:
        """Test gravity constant is defined."""
        # Check that gravity constant is a reasonable value (standard Earth gravity)
        assert isinstance(GRAVITY_M_S2, int | float)
        # Standard Earth gravity is 9.81 m/s² (source: NIST)
        assert 9.0 < GRAVITY_M_S2 < 10.0

    def test_xml_contains_gravity(self) -> None:
        """Test XML strings contain gravity value."""
        assert str(GRAVITY_M_S2) in DOUBLE_PENDULUM_XML
        assert str(GRAVITY_M_S2) in TRIPLE_PENDULUM_XML


class TestClubXMLGeneration:
    """Tests for club XML generation functions."""

    def test_generate_rigid_club_xml_default(self) -> None:
        """Test generating rigid club XML with default parameters."""
        xml = generate_rigid_club_xml()
        assert isinstance(xml, str)
        assert len(xml) > 0
        # Function returns body fragment, not complete MuJoCo XML
        assert "driver" in xml.lower() or "club" in xml.lower()
        assert "body" in xml.lower()

    def test_generate_rigid_club_xml_driver(self) -> None:
        """Test generating rigid club XML for driver."""
        xml = generate_rigid_club_xml("driver")
        assert isinstance(xml, str)
        assert len(xml) > 0

    def test_generate_rigid_club_xml_iron(self) -> None:
        """Test generating rigid club XML for iron."""
        xml = generate_rigid_club_xml("iron_7")
        assert isinstance(xml, str)
        assert len(xml) > 0

    def test_generate_flexible_club_xml_default(self) -> None:
        """Test generating flexible club XML with default parameters."""
        xml = generate_flexible_club_xml()
        assert isinstance(xml, str)
        assert len(xml) > 0
        # Function returns body fragment, not complete MuJoCo XML
        assert "body" in xml.lower()
        assert "driver" in xml.lower() or "club" in xml.lower()

    def test_generate_flexible_club_xml_segments(self) -> None:
        """Test generating flexible club XML with different segment counts."""
        xml_3 = generate_flexible_club_xml("driver", 3)
        xml_5 = generate_flexible_club_xml("driver", 5)
        assert isinstance(xml_3, str)
        assert isinstance(xml_5, str)
        # More segments should produce longer XML
        assert len(xml_5) >= len(xml_3)

    def test_generate_flexible_club_xml_different_clubs(self) -> None:
        """Test generating flexible club XML for different club types."""
        club_types = ["driver", "iron_7", "wedge"]
        for club_type in club_types:
            xml = generate_flexible_club_xml(club_type, 3)
            assert isinstance(xml, str)
            assert len(xml) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
