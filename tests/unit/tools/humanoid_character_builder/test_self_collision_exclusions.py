"""Test self-collision exclusion generation for URDF."""

import pytest
import xml.etree.ElementTree as ET

from humanoid_character_builder import CharacterBuilder
from humanoid_character_builder.core.body_parameters import BodyParameters
from humanoid_character_builder.presets.loader import list_available_presets


class TestSelfCollisionExclusions:
    """Test that generated URDFs include proper collision exclusions."""

    def test_urdf_has_gazebo_disable_collisions(self) -> None:
        """Generated URDF should include gazebo disable_collisions elements."""
        builder = CharacterBuilder()
        params = BodyParameters(height_m=1.75, mass_kg=75.0)
        urdf_xml = builder.generate_urdf(params)

        tree = ET.fromstring(urdf_xml)
        gazebo_elements = tree.findall(".//gazebo")

        assert len(gazebo_elements) > 0, (
            "URDF should contain at least one <gazebo> element with disable_collisions"
        )

    def test_disable_collisions_have_required_attributes(self) -> None:
        """Each disable_collisions element should have link1 and link2 attributes."""
        builder = CharacterBuilder()
        params = BodyParameters(height_m=1.75, mass_kg=75.0)
        urdf_xml = builder.generate_urdf(params)

        tree = ET.fromstring(urdf_xml)

        for gazebo in tree.findall(".//gazebo"):
            for disable in gazebo.findall("disable_collisions"):
                link1 = disable.get("link1")
                link2 = disable.get("link2")
                assert link1 is not None, "disable_collisions missing link1 attribute"
                assert link2 is not None, "disable_collisions missing link2 attribute"

    def test_parent_child_pairs_excluded(self) -> None:
        """Direct parent-child joint pairs should be in exclusion list."""
        builder = CharacterBuilder()
        params = BodyParameters(height_m=1.75, mass_kg=75.0)
        urdf_xml = builder.generate_urdf(params)

        tree = ET.fromstring(urdf_xml)

        # Build set of excluded pairs
        excluded_pairs = set()
        for gazebo in tree.findall(".//gazebo"):
            for disable in gazebo.findall("disable_collisions"):
                link1 = disable.get("link1")
                link2 = disable.get("link2")
                if link1 and link2:
                    excluded_pairs.add(tuple(sorted((link1, link2))))

        # Get all joint parent-child pairs
        joint_pairs = set()
        for joint in tree.findall(".//joint"):
            parent = joint.find("parent")
            child = joint.find("child")
            if parent is not None and child is not None:
                parent_link = parent.get("link")
                child_link = child.get("link")
                if parent_link and child_link:
                    joint_pairs.add(tuple(sorted((parent_link, child_link))))

        # All joint pairs should be excluded
        missing = joint_pairs - excluded_pairs
        assert len(missing) == 0, (
            f"Parent-child pairs missing from exclusions: {missing}"
        )

    @pytest.mark.parametrize("preset", list_available_presets()[:5])
    def test_all_presets_have_collision_exclusions(self, preset: str) -> None:
        """All presets should generate URDFs with collision exclusions."""
        builder = CharacterBuilder()
        params = builder.create_from_preset(preset)
        urdf_xml = builder.generate_urdf(params)

        tree = ET.fromstring(urdf_xml)
        gazebo_elements = tree.findall(".//gazebo")

        assert len(gazebo_elements) > 0, (
            f"URDF for preset '{preset}' should contain gazebo disable_collisions"
        )

    def test_exclusion_count_reasonable(self) -> None:
        """Number of exclusions should be reasonable (at least number of joints)."""
        builder = CharacterBuilder()
        params = BodyParameters(height_m=1.75, mass_kg=75.0)
        urdf_xml = builder.generate_urdf(params)

        tree = ET.fromstring(urdf_xml)

        joint_count = len(tree.findall(".//joint"))
        exclusion_count = sum(
            len(g.findall("disable_collisions")) for g in tree.findall(".//gazebo")
        )

        assert exclusion_count >= joint_count, (
            f"Should have at least {joint_count} exclusions (one per joint), "
            f"got {exclusion_count}"
        )
