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
import pytest
from model_generation.builders.manual_builder import ManualBuilder
from model_generation.builders.parametric_builder import ParametricBuilder
from model_generation.converters.urdf_parser import ParsedModel, URDFParser
from model_generation.core.types import (
    Geometry,
    GeometryType,
    Inertia,
    Joint,
    JointType,
    Link,
    Material,
    Origin,
)


class TestParsedModelOperations:
    """Test ParsedModel helper methods on roundtripped models."""

    @pytest.fixture()
    def parsed_chain(self) -> ParsedModel:
        """Build and parse a 4-link chain: A -> B -> C -> D."""
        builder = ManualBuilder("chain_model", validate_on_add=False)
        names = ["A", "B", "C", "D"]
        for name in names:
            builder.add_link(
                Link(
                    name=name,
                    inertia=Inertia.from_box(2.0, 0.2, 0.2, 0.2),
                )
            )
        for i in range(len(names) - 1):
            builder.add_joint(
                Joint(
                    name=f"{names[i]}_to_{names[i + 1]}",
                    joint_type=JointType.REVOLUTE,
                    parent=names[i],
                    child=names[i + 1],
                    axis=(0, 0, 1),
                )
            )

        result = builder.build()
        assert result.solver_status == "success"
        parser = URDFParser(resolve_meshes=False)
        return parser.parse_string(result.urdf_xml)

    def test_get_root_link(self, parsed_chain: ParsedModel) -> None:
        """Root of A->B->C->D is A."""
        root = parsed_chain.get_root_link()
        assert root is not None
        assert root.name == "A"

    def test_get_children(self, parsed_chain: ParsedModel) -> None:
        """A has child B, B has child C, D has no children."""
        assert parsed_chain.get_children("A") == ["B"]
        assert parsed_chain.get_children("B") == ["C"]
        assert parsed_chain.get_children("C") == ["D"]
        assert parsed_chain.get_children("D") == []

    def test_get_parent(self, parsed_chain: ParsedModel) -> None:
        """B's parent is A, D's parent is C, A has no parent."""
        assert parsed_chain.get_parent("A") is None
        assert parsed_chain.get_parent("B") == "A"
        assert parsed_chain.get_parent("C") == "B"
        assert parsed_chain.get_parent("D") == "C"

    def test_get_subtree(self, parsed_chain: ParsedModel) -> None:
        """Subtree rooted at B includes B, C, D."""
        subtree = parsed_chain.get_subtree("B")
        assert set(subtree) == {"B", "C", "D"}

    def test_copy_is_independent(self, parsed_chain: ParsedModel) -> None:
        """A copy of a ParsedModel is structurally identical but independent."""
        copy = parsed_chain.copy()

        assert copy.name == parsed_chain.name
        assert len(copy.links) == len(parsed_chain.links)
        assert len(copy.joints) == len(parsed_chain.joints)

        # Modify copy, original unaffected
        copy.links[0] = Link(
            name="modified",
            inertia=Inertia.from_box(1, 0.1, 0.1, 0.1),
        )
        assert parsed_chain.links[0].name == "A"

    def test_to_urdf_produces_valid_xml(self, parsed_chain: ParsedModel) -> None:
        """ParsedModel.to_urdf() produces parseable XML."""
        urdf = parsed_chain.to_urdf()
        root = DefusedET.fromstring(urdf)
        assert root.tag == "robot"
        assert len(root.findall(".//link")) == 4
        assert len(root.findall(".//joint")) == 3
