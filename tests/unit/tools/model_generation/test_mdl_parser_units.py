"""Unit tests for the SimScape MDL/SLX parser.

Covers issue #7001:
- parse_string on minimal .mdl fixture -> blocks / connections / params
- SimscapeParameter.as_float / as_vector (valid + malformed -> default)
- get_body_blocks / get_joint_blocks / get_connections_to
- _get_block_type mapping
- bad file extension -> error
"""

from __future__ import annotations

from pathlib import Path

import pytest
from model_generation.converters.simscape.mdl_parser import (
    MDLParser,
    SimscapeBlockType,
    SimscapeParameter,
)

MINIMAL_MDL = """
Model {
  Name "robot_model"
  Block {
    BlockType BrickSolid
    Name "Body1"
    Mass "1.5"
    Dimensions "[0.1 0.2 0.3]"
  }
  Block {
    BlockType CylinderSolid
    Name "Body2"
    Mass "2.0"
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


@pytest.fixture
def parser() -> MDLParser:
    return MDLParser()


class TestParseString:
    """parse_string extracts model name, blocks, params, and connections."""

    def test_model_name_extracted(self, parser: MDLParser) -> None:
        model = parser.parse_string(MINIMAL_MDL, format="mdl")
        assert model.name == "robot_model"

    def test_blocks_extracted(self, parser: MDLParser) -> None:
        model = parser.parse_string(MINIMAL_MDL, format="mdl")
        assert set(model.blocks) == {"Body1", "Body2", "Joint1"}

    def test_block_parameters_extracted(self, parser: MDLParser) -> None:
        model = parser.parse_string(MINIMAL_MDL, format="mdl")
        body1 = model.blocks["Body1"]
        assert body1.block_type == SimscapeBlockType.BRICK_SOLID
        assert body1.get_param("Mass") == "1.5"
        assert body1.get_param_float("Mass") == pytest.approx(1.5)

    def test_connections_extracted(self, parser: MDLParser) -> None:
        model = parser.parse_string(MINIMAL_MDL, format="mdl")
        assert len(model.connections) == 2
        first = model.connections[0]
        assert first.source_block == "Body1"
        assert first.dest_block == "Joint1"

    def test_none_content_raises(self, parser: MDLParser) -> None:
        with pytest.raises(ValueError, match="content must be provided"):
            parser.parse_string(None)  # type: ignore[arg-type]


class TestModelQueries:
    """SimscapeModel helper queries classify blocks correctly."""

    def test_get_body_blocks(self, parser: MDLParser) -> None:
        model = parser.parse_string(MINIMAL_MDL, format="mdl")
        names = {b.name for b in model.get_body_blocks()}
        assert names == {"Body1", "Body2"}

    def test_get_joint_blocks(self, parser: MDLParser) -> None:
        model = parser.parse_string(MINIMAL_MDL, format="mdl")
        names = {b.name for b in model.get_joint_blocks()}
        assert names == {"Joint1"}

    def test_get_connections_to(self, parser: MDLParser) -> None:
        model = parser.parse_string(MINIMAL_MDL, format="mdl")
        to_joint = model.get_connections_to("Joint1")
        assert len(to_joint) == 1
        assert to_joint[0].source_block == "Body1"

    def test_get_connections_from(self, parser: MDLParser) -> None:
        model = parser.parse_string(MINIMAL_MDL, format="mdl")
        from_joint = model.get_connections_from("Joint1")
        assert len(from_joint) == 1
        assert from_joint[0].dest_block == "Body2"


class TestSimscapeParameter:
    """as_float / as_vector handle scalars, vectors, and malformed input."""

    def test_as_float_scalar(self) -> None:
        assert SimscapeParameter("m", "3.14").as_float() == pytest.approx(3.14)

    def test_as_float_single_element_vector(self) -> None:
        assert SimscapeParameter("m", "[2.5]").as_float() == pytest.approx(2.5)

    def test_as_float_vector_returns_magnitude(self) -> None:
        # [3 4] -> sqrt(9 + 16) = 5
        assert SimscapeParameter("v", "[3 4]").as_float() == pytest.approx(5.0)

    def test_as_float_uses_evaluated_value(self) -> None:
        param = SimscapeParameter("m", "garbage", evaluated_value=9.0)
        assert param.as_float() == pytest.approx(9.0)

    def test_as_float_malformed_returns_default(self) -> None:
        assert SimscapeParameter("m", "not_a_number").as_float(default=1.23) == (
            pytest.approx(1.23)
        )

    def test_as_vector_valid(self) -> None:
        result = SimscapeParameter("v", "[1 2 3]").as_vector()
        assert result == pytest.approx((1.0, 2.0, 3.0))

    def test_as_vector_scalar(self) -> None:
        assert SimscapeParameter("v", "7").as_vector() == pytest.approx((7.0,))

    def test_as_vector_malformed_returns_default(self) -> None:
        result = SimscapeParameter("v", "[a b c]").as_vector(default=(9.0, 9.0, 9.0))
        assert result == pytest.approx((9.0, 9.0, 9.0))


class TestGetBlockType:
    """_get_block_type maps source-block paths and type strings."""

    @pytest.mark.parametrize(
        ("type_str", "expected"),
        [
            ("RevoluteJoint", SimscapeBlockType.REVOLUTE_JOINT),
            ("PrismaticJoint", SimscapeBlockType.PRISMATIC_JOINT),
            ("BrickSolid", SimscapeBlockType.BRICK_SOLID),
            ("SphereSolid", SimscapeBlockType.SPHERE_SOLID),
            ("WorldFrame", SimscapeBlockType.WORLD_FRAME),
            ("Subsystem", SimscapeBlockType.SUBSYSTEM),
        ],
    )
    def test_type_string_mapping(
        self,
        parser: MDLParser,
        type_str: str,
        expected: SimscapeBlockType,
    ) -> None:
        assert parser._get_block_type(type_str, "") == expected

    def test_source_block_mapping(self, parser: MDLParser) -> None:
        result = parser._get_block_type("", "sm_lib/Joints/Revolute Joint")
        assert result == SimscapeBlockType.REVOLUTE_JOINT

    def test_unknown_type_returns_unknown(self, parser: MDLParser) -> None:
        assert parser._get_block_type("Frobnicator", "") == SimscapeBlockType.UNKNOWN

    def test_none_type_str_raises(self, parser: MDLParser) -> None:
        with pytest.raises(ValueError, match="block_type_str must be provided"):
            parser._get_block_type(None, "")  # type: ignore[arg-type]


class TestParseFile:
    """parse() dispatches on suffix and rejects unknown extensions."""

    def test_unsupported_extension_raises(
        self, parser: MDLParser, tmp_path: Path
    ) -> None:
        bad = tmp_path / "model.txt"
        bad.write_text('Model { Name "x" }')
        with pytest.raises(ValueError, match="Unsupported file format"):
            parser.parse(bad)

    def test_missing_file_raises(self, parser: MDLParser, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            parser.parse(tmp_path / "does_not_exist.mdl")

    def test_parse_mdl_file_roundtrip(self, parser: MDLParser, tmp_path: Path) -> None:
        mdl = tmp_path / "robot.mdl"
        mdl.write_text(MINIMAL_MDL)
        model = parser.parse(mdl)
        assert model.name == "robot_model"
        assert set(model.blocks) == {"Body1", "Body2", "Joint1"}


class TestParseStringXml:
    """parse_string in xml mode parses blocks and tolerates malformed XML."""

    def test_xml_blocks_parsed(self, parser: MDLParser) -> None:
        xml = (
            "<root><Block Name='B1' BlockType='RevoluteJoint'>"
            "<P Name='Axis'>[0 0 1]</P></Block></root>"
        )
        model = parser.parse_string(xml, format="xml")
        assert any(b.name == "B1" for b in model.blocks.values())

    def test_malformed_xml_records_warning(self, parser: MDLParser) -> None:
        model = parser.parse_string("<root><Block></root>", format="xml")
        assert any("parse error" in w.lower() for w in model.warnings)
