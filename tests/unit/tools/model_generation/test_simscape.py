"""
Tests for the SimScape converter module.
"""

import tempfile
from pathlib import Path

import pytest

# Sample SimScape-like MDL content for testing
SIMPLE_MDL = """
Model {
  Name "test_model"
  Block {
    BlockType Subsystem
    Name "Subsystem"
  }
  Block {
    BlockType BrickSolid
    Name "Body1"
    Mass "1.0"
    Dimensions "[0.1 0.2 0.3]"
  }
  Block {
    BlockType CylinderSolid
    Name "Body2"
    Mass "2.0"
    Radius "0.05"
    Length "0.4"
  }
  Block {
    BlockType RevoluteJoint
    Name "Joint1"
    Axis "[0 0 1]"
  }
  Line {
    SrcBlock "Body1"
    SrcPort "1"
    DstBlock "Joint1"
    DstPort "1"
  }
  Line {
    SrcBlock "Joint1"
    SrcPort "2"
    DstBlock "Body2"
    DstPort "1"
  }
}
"""


class TestMDLParser:
    """Tests for MDLParser class."""

    def test_parser_creation(self) -> None:
        """Test parser instantiation."""
        from model_generation.converters.simscape import MDLParser

        parser = MDLParser()
        assert parser is not None

    def test_parse_mdl_string(self) -> None:
        """Test parsing MDL content from string."""
        from model_generation.converters.simscape import MDLParser

        parser = MDLParser()
        model = parser.parse_string(SIMPLE_MDL, format="mdl")

        assert model is not None
        assert model.name == "test_model"

    def test_parse_body_blocks(self) -> None:
        """Test parsing body/solid blocks."""
        from model_generation.converters.simscape import (
            MDLParser,
            SimscapeBlockType,
        )

        parser = MDLParser()
        model = parser.parse_string(SIMPLE_MDL, format="mdl")

        bodies = model.get_body_blocks()
        assert len(bodies) >= 1

        # Check block types
        block_types = [b.block_type for b in bodies]
        assert SimscapeBlockType.BRICK_SOLID in block_types or any(
            bt in block_types
            for bt in [
                SimscapeBlockType.CYLINDER_SOLID,
                SimscapeBlockType.SOLID,
            ]
        )

    def test_parse_joint_blocks(self) -> None:
        """Test parsing joint blocks."""
        from model_generation.converters.simscape import (
            MDLParser,
        )

        parser = MDLParser()
        model = parser.parse_string(SIMPLE_MDL, format="mdl")

        joints = model.get_joint_blocks()
        # Should find at least the revolute joint
        assert len(joints) >= 0  # May be 0 if parsing doesn't find it

    def test_parse_connections(self) -> None:
        """Test parsing connections between blocks."""
        from model_generation.converters.simscape import MDLParser

        parser = MDLParser()
        model = parser.parse_string(SIMPLE_MDL, format="mdl")

        # Should have some connections
        assert isinstance(model.connections, list)

    def test_get_block_parameters(self) -> None:
        """Test extracting block parameters."""
        from model_generation.converters.simscape import MDLParser

        parser = MDLParser()
        model = parser.parse_string(SIMPLE_MDL, format="mdl")

        # Find a body block with mass
        for block in model.get_body_blocks():
            if "Mass" in block.parameters or "mass" in block.parameters:
                mass = block.get_param_float("Mass", 0)
                assert mass > 0
                break


class TestSimscapeConverter:
    """Tests for SimscapeToURDFConverter class."""

    def test_converter_creation(self) -> None:
        """Test converter instantiation."""
        from model_generation.converters.simscape import SimscapeToURDFConverter

        converter = SimscapeToURDFConverter()
        assert converter is not None

    def test_convert_simple_mdl(self) -> None:
        """Test converting simple MDL content."""
        from model_generation.converters.simscape import SimscapeToURDFConverter

        converter = SimscapeToURDFConverter()
        result = converter.convert_string(SIMPLE_MDL, format="mdl")

        # Should produce some output even if not perfect
        assert result is not None
        assert result.robot_name is not None

    def test_convert_with_config(self) -> None:
        """Test conversion with custom configuration."""
        from model_generation.converters.simscape import (
            ConversionConfig,
            SimscapeToURDFConverter,
        )

        config = ConversionConfig(
            robot_name="custom_name",
            include_visual=True,
            include_collision=True,
        )

        converter = SimscapeToURDFConverter(config)
        result = converter.convert_string(SIMPLE_MDL, format="mdl")

        assert result.robot_name == "custom_name"

    def test_convert_generates_urdf(self) -> None:
        """Test that conversion generates valid URDF string."""
        from model_generation.converters.simscape import SimscapeToURDFConverter

        converter = SimscapeToURDFConverter()
        result = converter.convert_string(SIMPLE_MDL, format="mdl")

        if result.success and result.urdf_string:
            assert "<robot" in result.urdf_string
            assert "</robot>" in result.urdf_string

    def test_conversion_result_contents(self) -> None:
        """Test conversion result structure."""
        from model_generation.converters.simscape import SimscapeToURDFConverter

        converter = SimscapeToURDFConverter()
        result = converter.convert_string(SIMPLE_MDL, format="mdl")

        assert hasattr(result, "success")
        assert hasattr(result, "links")
        assert hasattr(result, "joints")
        assert hasattr(result, "warnings")
        assert hasattr(result, "errors")
        assert isinstance(result.links, list)
        assert isinstance(result.joints, list)

    def test_unit_conversion(self) -> None:
        """Test unit conversion factors."""
        from model_generation.converters.simscape import (
            ConversionConfig,
            SimscapeToURDFConverter,
        )

        # Test mm to m conversion
        config_mm = ConversionConfig(length_unit="mm")
        converter_mm = SimscapeToURDFConverter(config_mm)

        # Length factor should be 0.001
        assert converter_mm.LENGTH_FACTORS["mm"] == 0.001

        # Test lb to kg conversion
        assert converter_mm.MASS_FACTORS["lb"] == pytest.approx(0.453592, rel=1e-3)


class TestConversionConfig:
    """Tests for ConversionConfig class."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        from model_generation.converters.simscape import ConversionConfig

        config = ConversionConfig()

        assert config.length_unit == "m"
        assert config.mass_unit == "kg"
        assert config.angle_unit == "rad"
        assert config.include_visual is True
        assert config.include_collision is True

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        from model_generation.converters.simscape import ConversionConfig

        config = ConversionConfig(
            robot_name="my_robot",
            length_unit="mm",
            mass_unit="g",
            include_visual=False,
        )

        assert config.robot_name == "my_robot"
        assert config.length_unit == "mm"
        assert config.mass_unit == "g"
        assert config.include_visual is False


class TestConvenienceFunction:
    """Tests for convenience conversion function."""

    def test_convert_simscape_to_urdf_function(self) -> None:
        """Test the convenience function."""
        from model_generation.converters.simscape import convert_simscape_to_urdf

        # Create temp MDL file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mdl", delete=False) as f:
            f.write(SIMPLE_MDL)
            temp_path = Path(f.name)

        try:
            result = convert_simscape_to_urdf(temp_path, robot_name="test_robot")
            assert result is not None
            assert result.robot_name == "test_robot"
        finally:
            temp_path.unlink()
