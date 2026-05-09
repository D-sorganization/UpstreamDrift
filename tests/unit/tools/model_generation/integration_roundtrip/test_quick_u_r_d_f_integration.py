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


class TestQuickURDFIntegration:
    """Test the convenience function quick_urdf end-to-end."""

    def test_quick_urdf_produces_parseable_output(self) -> None:
        """quick_urdf() output is valid URDF XML."""
        from model_generation import quick_urdf

        urdf = quick_urdf(height_m=1.80, mass_kg=80.0)
        assert isinstance(urdf, str)
        assert len(urdf) > 100  # non-trivial

        root = DefusedET.fromstring(urdf)
        assert root.tag == "robot"

    def test_quick_urdf_preset_produces_valid_model(self) -> None:
        """quick_urdf with preset 'athletic' produces a valid humanoid."""
        from model_generation import quick_urdf

        urdf = quick_urdf(height_m=1.85, preset="athletic")
        parser = URDFParser(resolve_meshes=False)
        parsed = parser.parse_string(urdf)

        assert len(parsed.links) > 10
        link_names = {lnk.name for lnk in parsed.links}
        assert "pelvis" in link_names
        assert "head" in link_names

    def test_quick_urdf_roundtrip_equivalence(self) -> None:
        """quick_urdf -> parse -> to_urdf -> parse produces structurally same model."""
        from model_generation import quick_urdf

        urdf1 = quick_urdf(height_m=1.75, mass_kg=75.0)
        parser = URDFParser(resolve_meshes=False)
        parsed1 = parser.parse_string(urdf1)

        # Reserialize
        urdf2 = parsed1.to_urdf()
        parsed2 = parser.parse_string(urdf2)

        # Structural equivalence: same links and joints
        names1 = {lnk.name for lnk in parsed1.links}
        names2 = {lnk.name for lnk in parsed2.links}
        assert names1 == names2

        joint_names1 = {j.name for j in parsed1.joints}
        joint_names2 = {j.name for j in parsed2.joints}
        assert joint_names1 == joint_names2
