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
# Note: model_generation can resolve from Tools repo via sys.path, but the
# pendulum_putter submodule only exists at src/tools/model_generation in this repo.
pytest.importorskip(
    "model_generation.models.pendulum_putter",
    reason="model_generation.models.pendulum_putter package not available",
)


class TestPendulumPutterModelConstruction:
    """Test model construction and structure."""

    def test_model_builds_successfully(self):
        """Model should build without errors."""
        from model_generation.models.pendulum_putter import (
            PendulumPutterModelBuilder,
        )

        builder = PendulumPutterModelBuilder()
        result = builder.build()

        assert result.success, f"Build failed: {result.error_message}"
        assert result.urdf_xml is not None

    def test_model_has_correct_link_count(self):
        """Model should have expected number of links."""
        from model_generation.models.pendulum_putter import (
            PendulumPutterModelBuilder,
        )

        builder = PendulumPutterModelBuilder()
        result = builder.build()

        # Expected links: world, base, vertical_post, shoulder_mount,
        # pendulum_arm, club_mount, plus club links (grip, shaft, head)
        assert len(result.links) >= 6, "Should have at least 6 links"

    def test_model_has_single_dof_pendulum_joint(self):
        """Model should have exactly 1 DOF for pendulum motion."""
        from model_generation.models.pendulum_putter import (
            PendulumPutterModelBuilder,
        )

        builder = PendulumPutterModelBuilder()
        result = builder.build()

        # Count revolute joints (DOF contributors)
        revolute_joints = [j for j in result.joints if j.joint_type.value == "revolute"]

        assert len(revolute_joints) == 1, "Should have exactly 1 revolute joint"
        assert result.get_total_dof() == 1, "Total DOF should be 1"

    def test_pendulum_joint_rotates_about_y_axis(self):
        """Pendulum joint should rotate about Y-axis for X-Z plane swing."""
        from model_generation.models.pendulum_putter import (
            PendulumPutterModelBuilder,
        )

        builder = PendulumPutterModelBuilder()
        result = builder.build()

        pendulum_joint = result.get_joint("pendulum_joint")
        assert pendulum_joint is not None, "Should have pendulum_joint"

        # Y-axis rotation
        assert pendulum_joint.axis == (0, 1, 0), "Should rotate about Y-axis"


class TestPendulumPutterModelPhysics:
    """Test physical properties of the model."""

    def test_model_has_reasonable_total_mass(self):
        """Total mass should be reasonable for a putting robot."""
        from model_generation.models.pendulum_putter import (
            PendulumPutterModelBuilder,
        )

        builder = PendulumPutterModelBuilder()
        result = builder.build()

        total_mass = result.get_total_mass()

        # Perfy-style robot: ~5-15 kg total
        assert 1.0 < total_mass < 30.0, f"Mass {total_mass} kg seems unreasonable"

    def test_base_is_heaviest_component(self):
        """Base should be heavy for stability."""
        from model_generation.models.pendulum_putter import (
            PendulumPutterModelBuilder,
        )

        builder = PendulumPutterModelBuilder()
        result = builder.build()

        base_link = result.get_link("base_link")
        assert base_link is not None

        # Base should be at least 30% of total mass for stability
        base_mass = base_link.inertia.mass
        total_mass = result.get_total_mass()
        assert base_mass > 0.3 * total_mass, "Base should be heavy for stability"

    def test_all_inertias_are_physically_valid(self):
        """All inertia tensors should be positive definite."""
        from model_generation.models.pendulum_putter import (
            PendulumPutterModelBuilder,
        )

        builder = PendulumPutterModelBuilder()
        result = builder.build()

        for link in result.links:
            if link.inertia.mass > 1e-6:  # Skip negligible mass links
                assert (
                    link.inertia.is_positive_definite()
                ), f"Link {link.name} has non-positive-definite inertia"

    def test_pendulum_joint_has_appropriate_limits(self):
        """Pendulum joint should have reasonable angle limits."""
        from model_generation.models.pendulum_putter import (
            PendulumPutterModelBuilder,
        )

        builder = PendulumPutterModelBuilder()
        result = builder.build()

        joint = result.get_joint("pendulum_joint")
        assert joint is not None
        assert joint.limits is not None

        # Putting stroke: typically ±30-45 degrees max
        assert joint.limits.lower >= -math.pi / 2, "Lower limit too extreme"
        assert joint.limits.upper <= math.pi / 2, "Upper limit too extreme"

    def test_pendulum_joint_has_low_damping(self):
        """Pendulum should have low damping for free swing."""
        from model_generation.models.pendulum_putter import (
            PendulumPutterModelBuilder,
        )

        builder = PendulumPutterModelBuilder()
        result = builder.build()

        joint = result.get_joint("pendulum_joint")
        assert joint is not None

        # Low damping for pendulum behavior
        assert joint.dynamics.damping < 0.5, "Damping too high for pendulum"


class TestPendulumPutterModelConfiguration:
    """Test model configuration and customization."""

    def test_can_set_arm_length(self):
        """Should be able to configure arm length."""
        from model_generation.models.pendulum_putter import (
            PendulumPutterModelBuilder,
        )

        short_builder = PendulumPutterModelBuilder(arm_length_m=0.3)
        long_builder = PendulumPutterModelBuilder(arm_length_m=0.5)

        short_result = short_builder.build()
        long_result = long_builder.build()

        assert short_result.success
        assert long_result.success

        # Verify different configurations
        short_arm = short_result.get_link("pendulum_arm")
        long_arm = long_result.get_link("pendulum_arm")

        assert short_arm is not None
        assert long_arm is not None

    def test_can_set_shoulder_height(self):
        """Should be able to configure shoulder height."""
        from model_generation.models.pendulum_putter import (
            PendulumPutterModelBuilder,
        )

        builder = PendulumPutterModelBuilder(shoulder_height_m=1.0)
        result = builder.build()

        assert result.success

    def test_can_set_pendulum_damping(self):
        """Should be able to configure pendulum damping."""
        from model_generation.models.pendulum_putter import (
            PendulumPutterModelBuilder,
        )

        builder = PendulumPutterModelBuilder(damping=0.1)
        result = builder.build()

        joint = result.get_joint("pendulum_joint")
        assert joint.dynamics.damping == pytest.approx(0.1)


class TestInterchangeableClub:
    """Test club interchangeability feature."""

    def test_default_club_is_attached(self):
        """Model should have a default putter attached."""
        from model_generation.models.pendulum_putter import (
            PendulumPutterModelBuilder,
        )

        builder = PendulumPutterModelBuilder()
        result = builder.build()

        # Should have club-related links
        club_links = [
            link
            for link in result.links
            if "club" in link.name.lower()
            or "putter" in link.name.lower()
            or "grip" in link.name.lower()
            or "shaft" in link.name.lower()
            or "head" in link.name.lower()
        ]
        assert len(club_links) >= 1, "Should have club links attached"

    def test_can_build_without_club(self):
        """Should be able to build model without a club."""
        from model_generation.models.pendulum_putter import (
            PendulumPutterModelBuilder,
        )

        builder = PendulumPutterModelBuilder(include_club=False)
        result = builder.build()

        assert result.success

        # Should end with club_mount
        club_mount = result.get_link("club_mount")
        assert club_mount is not None

    def test_can_attach_custom_club(self):
        """Should be able to attach a custom club configuration."""
        from model_generation.models.pendulum_putter import (
            ClubConfig,
            PendulumPutterModelBuilder,
        )

        custom_club = ClubConfig(
            grip_length_m=0.25,
            shaft_length_m=0.85,
            head_mass_kg=0.35,
        )

        builder = PendulumPutterModelBuilder(club_config=custom_club)
        result = builder.build()

        assert result.success


class TestURDFGeneration:
    """Test URDF generation and compatibility."""

    def test_generates_valid_urdf_xml(self):
        """Generated URDF should be valid XML."""
        import defusedxml.ElementTree as ET
        from model_generation.models.pendulum_putter import (
            PendulumPutterModelBuilder,
        )

        builder = PendulumPutterModelBuilder()
        result = builder.build()

        # Should parse as valid XML
        root = ET.fromstring(result.urdf_xml)
        assert root.tag == "robot"
        assert root.attrib["name"] == "pendulum_putter"

    def test_urdf_has_all_required_elements(self):
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
        assert len(links) >= 6

        # Check for joints
        joints = root.findall("joint")
        assert len(joints) >= 5

        # Check each link has required elements
        for link in links:
            if link.attrib["name"] != "world":
                # Should have inertial (except world)
                inertial = link.find("inertial")
                assert (
                    inertial is not None
                ), f"Link {link.attrib['name']} missing inertial"

    def test_can_save_to_file(self, tmp_path: Path):
        """Should be able to save URDF to file."""
        from model_generation.models.pendulum_putter import (
            PendulumPutterModelBuilder,
        )

        builder = PendulumPutterModelBuilder()
        output_path = tmp_path / "pendulum_putter.urdf"

        builder.save(output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "<robot" in content
        assert "pendulum_putter" in content


class TestModelPortability:
    """Test that model can be moved around environments."""

    def test_world_link_is_root(self):
        """World link should be the root for attachment."""
        from model_generation.models.pendulum_putter import (
            PendulumPutterModelBuilder,
        )

        builder = PendulumPutterModelBuilder()
        result = builder.build()

        root = result.get_root_link()
        assert root is not None
        assert root.name == "world"

    def test_base_attached_to_world_via_fixed_joint(self):
        """Base should be attached to world via fixed joint for positioning."""
        from model_generation.models.pendulum_putter import (
            PendulumPutterModelBuilder,
        )

        builder = PendulumPutterModelBuilder()
        result = builder.build()

        # Find joint connecting world to base
        world_to_base = result.get_joint("world_to_base")
        assert world_to_base is not None
        assert world_to_base.joint_type.value == "fixed"
        assert world_to_base.parent == "world"
        assert world_to_base.child == "base_link"


class TestPendulumPhysicsAnalytical:
    """Test analytical physics properties of the pendulum model."""

    def test_natural_frequency_calculable(self):
        """Should be able to calculate natural frequency from model params."""
        from model_generation.models.pendulum_putter import (
            PendulumPutterModelBuilder,
        )

        builder = PendulumPutterModelBuilder(arm_length_m=0.4)
        result = builder.build()

        # Get pendulum arm length and mass distribution
        # ω = sqrt(g/L) for simple pendulum
        # For compound pendulum: ω = sqrt(m*g*d / I)
        # where d = distance from pivot to COM, I = moment of inertia about pivot

        pendulum_arm = result.get_link("pendulum_arm")
        assert pendulum_arm is not None

        # Basic sanity check - mass should allow pendulum motion
        assert pendulum_arm.inertia.mass > 0


class TestValidation:
    """Test model validation."""

    def test_validation_passes_for_default_model(self):
        """Default model should pass all validations."""
        from model_generation.models.pendulum_putter import (
            PendulumPutterModelBuilder,
        )

        builder = PendulumPutterModelBuilder()
        result = builder.build()

        assert result.validation is not None
        assert (
            result.validation.is_valid
        ), f"Validation failed: {result.validation.get_error_messages()}"

    def test_validation_catches_invalid_parameters(self):
        """Should catch invalid configuration parameters."""
        from model_generation.models.pendulum_putter import (
            PendulumPutterModelBuilder,
        )

        # Negative arm length should raise
        with pytest.raises(ValueError, match="arm_length"):
            PendulumPutterModelBuilder(arm_length_m=-0.5)

        # Negative shoulder height should raise
        with pytest.raises(ValueError, match="shoulder_height"):
            PendulumPutterModelBuilder(shoulder_height_m=-1.0)


class TestMetadata:
    """Test model metadata for documentation and traceability."""

    def test_metadata_includes_model_name(self):
        """Metadata should include model identification."""
        from model_generation.models.pendulum_putter import (
            PendulumPutterModelBuilder,
        )

        builder = PendulumPutterModelBuilder()
        result = builder.build()

        assert "robot_name" in result.metadata
        assert result.metadata["robot_name"] == "pendulum_putter"

    def test_metadata_includes_configuration(self):
        """Metadata should include configuration parameters."""
        from model_generation.models.pendulum_putter import (
            PendulumPutterModelBuilder,
        )

        builder = PendulumPutterModelBuilder(
            arm_length_m=0.35,
            shoulder_height_m=0.9,
        )
        result = builder.build()

        assert "arm_length_m" in result.metadata
        assert "shoulder_height_m" in result.metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
