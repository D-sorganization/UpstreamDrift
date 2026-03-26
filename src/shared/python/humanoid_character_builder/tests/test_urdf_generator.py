"""
Tests for URDF generator module.
"""

import tempfile
from pathlib import Path

import defusedxml.ElementTree as ET
from humanoid_character_builder.core.body_parameters import BodyParameters
from humanoid_character_builder.generators.urdf_generator import (
    HumanoidURDFGenerator,
    URDFGeneratorConfig,
    generate_humanoid_urdf,
)


class TestHumanoidURDFGenerator:
    """Tests for HumanoidURDFGenerator class."""

    def test_init_default(self):
        generator = HumanoidURDFGenerator()
        assert generator.config.default_density > 0

    def test_generate_default_params(self):
        generator = HumanoidURDFGenerator()
        params = BodyParameters()

        urdf = generator.generate(params)

        assert urdf is not None
        assert '<robot name="humanoid"' in urdf
        assert "<link" in urdf
        assert "<joint" in urdf

    def test_generate_custom_params(self):
        generator = HumanoidURDFGenerator()
        params = BodyParameters(name="athlete", height_m=1.9, mass_kg=90.0)

        urdf = generator.generate(params)

        assert '<robot name="athlete"' in urdf

    def test_generate_valid_xml(self):
        generator = HumanoidURDFGenerator()
        params = BodyParameters()

        urdf = generator.generate(params)

        # Should parse without error
        root = ET.fromstring(urdf)
        assert root.tag == "robot"

    def test_generate_has_links(self):
        generator = HumanoidURDFGenerator()
        params = BodyParameters()
        urdf = generator.generate(params)
        root = ET.fromstring(urdf)

        links = root.findall("link")
        link_names = [link.get("name") for link in links]

        assert "pelvis" in link_names
        assert "head" in link_names
        assert "left_foot" in link_names

    def test_generate_has_joints(self):
        generator = HumanoidURDFGenerator()
        params = BodyParameters()
        urdf = generator.generate(params)
        root = ET.fromstring(urdf)

        joints = root.findall("joint")
        # Should have plenty of joints
        assert len(joints) > 10

    def test_generate_inertial_properties(self):
        generator = HumanoidURDFGenerator()
        params = BodyParameters()
        urdf = generator.generate(params)
        root = ET.fromstring(urdf)

        for link in root.findall("link"):
            inertial = link.find("inertial")
            if inertial is not None:
                mass = inertial.find("mass")
                inertia = inertial.find("inertia")
                assert mass is not None
                assert inertia is not None
                assert float(mass.get("value")) > 0

    def test_generate_visual_geometry(self):
        generator = HumanoidURDFGenerator()
        params = BodyParameters()
        urdf = generator.generate(params)
        root = ET.fromstring(urdf)

        for link in root.findall("link"):
            visual = link.find("visual")
            assert visual is not None
            geometry = visual.find("geometry")
            assert geometry is not None
            # Should be box, cylinder, sphere, or mesh
            assert (
                geometry.find("box") is not None
                or geometry.find("cylinder") is not None
                or geometry.find("sphere") is not None
                or geometry.find("mesh") is not None
            )

    def test_generate_collision_geometry(self):
        config = URDFGeneratorConfig(generate_collision=True)
        generator = HumanoidURDFGenerator(config)
        params = BodyParameters()
        urdf = generator.generate(params)
        root = ET.fromstring(urdf)

        # Most links should have collision
        collision_count = 0
        for link in root.findall("link"):
            if link.find("collision") is not None:
                collision_count += 1

        assert collision_count > 0

    def test_generate_no_collision(self):
        config = URDFGeneratorConfig(generate_collision=False)
        generator = HumanoidURDFGenerator(config)
        params = BodyParameters()
        urdf = generator.generate(params)
        root = ET.fromstring(urdf)

        # No links should have collision
        for link in root.findall("link"):
            assert link.find("collision") is None

    def test_generate_write_to_file(self):
        generator = HumanoidURDFGenerator()
        params = BodyParameters()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_robot.urdf"
            generator.generate(params, output_path=output_path)

            assert output_path.exists()
            assert output_path.stat().st_size > 0

    def test_generate_joint_limits(self):
        generator = HumanoidURDFGenerator()
        params = BodyParameters()
        urdf = generator.generate(params)
        root = ET.fromstring(urdf)

        for joint in root.findall("joint"):
            if joint.get("type") in ("revolute", "prismatic"):
                limit = joint.find("limit")
                assert limit is not None
                assert "lower" in limit.attrib
                assert "upper" in limit.attrib
                assert "effort" in limit.attrib
                assert "velocity" in limit.attrib

    def test_generate_joint_dynamics(self):
        generator = HumanoidURDFGenerator()
        params = BodyParameters()
        urdf = generator.generate(params)
        root = ET.fromstring(urdf)

        for joint in root.findall("joint"):
            dynamics = joint.find("dynamics")
            if dynamics is not None:
                assert "damping" in dynamics.attrib
                assert "friction" in dynamics.attrib


class TestGenerateHumanoidURDF:
    """Tests for convenience function."""

    def test_basic_call(self):
        params = BodyParameters()
        urdf = generate_humanoid_urdf(params)
        assert "<robot" in urdf

    def test_with_config(self):
        params = BodyParameters()
        config = URDFGeneratorConfig(pretty_print=False)
        urdf = generate_humanoid_urdf(params, config=config)
        assert "\n" not in urdf  # Should be one line if not pretty printed (mostly)

    def test_with_output_path(self):
        params = BodyParameters()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "humanoid.urdf"
            generate_humanoid_urdf(params, output_path=output_path)

            assert output_path.exists()


class TestCompositeJointExpansion:
    """Tests for composite joint expansion."""

    def test_gimbal_joint_expansion(self):
        # The neck_to_head joint is typically a gimbal joint
        generator = HumanoidURDFGenerator()
        params = BodyParameters()
        urdf = generator.generate(params)
        root = ET.fromstring(urdf)

        # Check for expanded joints
        joint_names = [j.get("name") for j in root.findall("joint")]

        # Look for _x, _y, _z suffixes if gimbal was expanded
        # Note: names might differ depending on joint definition
        expanded = any("_x" in name or "_y" in name or "_z" in name for name in joint_names)
        assert expanded

    def test_no_expansion(self):
        config = URDFGeneratorConfig(expand_composite_joints=False)
        generator = HumanoidURDFGenerator(config)
        params = BodyParameters()
        urdf = generator.generate(params)
        root = ET.fromstring(urdf)

        # Should find gimbal type if not expanded (though standard URDF parsers might fail)
        # Our generator maps GIMBAL to 'fixed' if not expanded?
        # Let's check the map_joint_type logic.
        # "revolute" if not expanded? Wait, _map_joint_type handles standard types.
        # Composite types are only handled via expansion in _generate_joint.
        # If expand=False, it calls _generate_single_joint.
        # _map_joint_type maps GIMBAL to 'revolute'.
        # So we should see a single revolute joint for the gimbal joint.

        # Specifically neck_to_head
        neck_joints = [j for j in root.findall("joint") if j.get("name").startswith("neck_to_head")]
        assert len(neck_joints) == 1
        assert neck_joints[0].get("type") == "revolute"


class TestProportionFactors:
    """Tests for proportion scaling."""

    def test_tall_character(self):
        generator = HumanoidURDFGenerator()

        # Generate standard and tall
        params_std = BodyParameters(height_m=1.70)
        params_tall = BodyParameters(height_m=2.00)

        urdf_std = generator.generate(params_std)
        urdf_tall = generator.generate(params_tall)

        # Extract total length of a leg chain to compare
        # This is complex to parse from URDF without a kinematic solver.
        # Instead, we can inspect the generated link lengths in the generator logic?
        # Or check the <cylinder length="..."> in the XML.

        def get_total_cylinder_length(xml_str):
            root = ET.fromstring(xml_str)
            total = 0.0
            for geom in root.findall(".//geometry/cylinder"):
                total += float(geom.get("length"))
            return total

        len_std = get_total_cylinder_length(urdf_std)
        len_tall = get_total_cylinder_length(urdf_tall)

        assert len_tall > len_std

    def test_wide_shoulders(self):
        generator = HumanoidURDFGenerator()

        params = BodyParameters(shoulder_width_factor=1.5)
        urdf = generator.generate(params)

        # Hard to verify without parsing positions, but execution should succeed
        assert urdf is not None

    def test_muscular_build(self):
        generator = HumanoidURDFGenerator()

        params = BodyParameters(muscularity=0.8, body_fat_factor=0.1)
        urdf = generator.generate(params)

        assert urdf is not None
