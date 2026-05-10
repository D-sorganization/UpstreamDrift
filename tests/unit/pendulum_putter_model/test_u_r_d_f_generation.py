"""Unit tests for PendulumPutterModel - TDD approach.

Tests the Perfy-style pendulum putter model based on Dave Pelz's design.
The model consists of a rigid stand with pendulum arms holding an
interchangeable putter club.

Tests follow the Pragmatic Programmer principles:
- Small, focused test functions
- Test one thing at a time
- Clear assertions with descriptive messages
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    pass

# Skip entire module if model_generation.models.pendulum_putter is not available
# pendulum_putter submodule only exists at src/tools/model_generation in this repo.
pytest.importorskip(
    "model_generation.models.pendulum_putter",
    reason="model_generation.models.pendulum_putter package not available",
)


class TestURDFGeneration:
    """Test URDF generation and compatibility."""

    def test_generates_valid_urdf_xml(self) -> None:
        """Generated URDF should be valid XML."""
        import defusedxml.ElementTree as ET
        from model_generation.models.pendulum_putter import (
            PendulumPutterModelBuilder,
        )

        builder = PendulumPutterModelBuilder()
        result = builder.build()

        # Should parse as valid XML
        root = ET.fromstring(result.urdf_xml)
        assert root.tag == "robot", "Assertion failed: root.tag == robot"
        assert root.attrib["name"] == "pendulum_putter", (
            "Assertion failed: root.attrib[name] == pendulum_putter"
        )

    def test_urdf_has_all_required_elements(self) -> None:
        """URDF should have all required elements for physics engines."""
        import defusedxml.ElementTree as ET
        from model_generation.models.pendulum_putter import (
            PendulumPutterModelBuilder,
        )

        builder = PendulumPutterModelBuilder()
        result = builder.build()

        root = ET.fromstring(result.urdf_xml)

        # Check for links
        links = root.findall("link")
        assert len(links) >= 6, "Assertion failed: len(links) >= 6"

        # Check for joints
        joints = root.findall("joint")
        assert len(joints) >= 5, "Assertion failed: len(joints) >= 5"

        # Check each link has required elements
        for link in links:
            if link.attrib["name"] != "world":
                # Should have inertial (except world)
                inertial = link.find("inertial")
                assert inertial is not None, (
                    f"Link {link.attrib['name']} missing inertial"
                )

    def test_can_save_to_file(self, tmp_path: Path) -> None:
        """Should be able to save URDF to file."""
        from model_generation.models.pendulum_putter import (
            PendulumPutterModelBuilder,
        )

        builder = PendulumPutterModelBuilder()
        output_path = tmp_path / "pendulum_putter.urdf"

        builder.save(output_path)

        assert output_path.exists(), "Assertion failed: output_path.exists()"
        content = output_path.read_text()
        assert "<robot" in content, "Assertion failed: <robot in content"
        assert "pendulum_putter" in content, (
            "Assertion failed: pendulum_putter in content"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
