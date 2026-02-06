"""
Unit tests for URDF generator module.
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
from humanoid_character_builder.mesh.inertia_calculator import InertiaMode


class TestURDFGeneratorConfig:
    """Tests for URDFGeneratorConfig."""

    def test_default_values(self):
        config = URDFGeneratorConfig()
        assert config.inertia_mode == InertiaMode.PRIMITIVE_APPROXIMATION
        assert config.generate_collision is True
        assert config.expand_composite_joints is True

    def test_custom_values(self):
        config = URDFGeneratorConfig(
            inertia_mode=InertiaMode.MESH_UNIFORM_DENSITY,
            default_density=1100.0,
            generate_collision=False,
        )
        assert config.inertia_mode == InertiaMode.MESH_UNIFORM_DENSITY
        assert config.default_density == 1100.0
        assert config.generate_collision is False


class TestHumanoidURDFGenerator:
    """Tests for HumanoidURDFGenerator."""

    def test_generate_default_params(self):
        generator = HumanoidURDFGenerator()
        params = BodyParameters()

        urdf_xml = generator.generate(params)

        assert urdf_xml is not None
        assert len(urdf_xml) > 0
        assert "<?xml" in urdf_xml
        assert "<robot" in urdf_xml

    def test_generate_custom_params(self):
        generator = HumanoidURDFGenerator()
        params = BodyParameters(
            height_m=1.90,
            mass_kg=90.0,
            name="tall_humanoid",
        )

        urdf_xml = generator.generate(params)

        assert 'name="tall_humanoid"' in urdf_xml

    def test_generate_valid_xml(self):
        generator = HumanoidURDFGenerator()
        params = BodyParameters()

        urdf_xml = generator.generate(params)

        # Remove XML declaration for parsing
        xml_content = urdf_xml
        if xml_content.startswith("<?xml"):
            xml_content = xml_content[xml_content.index("?>") + 2 :]

        # Should parse without errors
        root = ET.fromstring(xml_content)
        assert root.tag == "robot"

    def test_generate_has_links(self):
        generator = HumanoidURDFGenerator()
        params = BodyParameters()

        urdf_xml = generator.generate(params)

        # Parse and check for links
        xml_content = urdf_xml
        if xml_content.startswith("<?xml"):
            xml_content = xml_content[xml_content.index("?>") + 2 :]

        root = ET.fromstring(xml_content)
        links = root.findall("link")

        # Should have multiple links
        assert len(links) > 10

        # Check for expected links
        link_names = {link.get("name") for link in links}
        assert "pelvis" in link_names
        assert "head" in link_names
        assert "left_thigh" in link_names

    def test_generate_has_joints(self):
        generator = HumanoidURDFGenerator()
        params = BodyParameters()

        urdf_xml = generator.generate(params)

        xml_content = urdf_xml
        if xml_content.startswith("<?xml"):
            xml_content = xml_content[xml_content.index("?>") + 2 :]

        root = ET.fromstring(xml_content)
        joints = root.findall("joint")

        # Should have multiple joints
        assert len(joints) > 10

    def test_generate_inertial_properties(self):
        generator = HumanoidURDFGenerator()
        params = BodyParameters()

        urdf_xml = generator.generate(params)

        xml_content = urdf_xml
        if xml_content.startswith("<?xml"):
            xml_content = xml_content[xml_content.index("?>") + 2 :]

        root = ET.fromstring(xml_content)

        # Check that links have inertial properties
        for link in root.findall("link"):
            inertial = link.find("inertial")
            assert inertial is not None, f"Link {link.get('name')} missing inertial"

            mass = inertial.find("mass")
            assert mass is not None

            inertia = inertial.find("inertia")
            assert inertia is not None

            # Check inertia has required attributes
            assert "ixx" in inertia.attrib
            assert "iyy" in inertia.attrib
            assert "izz" in inertia.attrib

    def test_generate_visual_geometry(self):
        generator = HumanoidURDFGenerator()
        params = BodyParameters()

        urdf_xml = generator.generate(params)

        xml_content = urdf_xml
        if xml_content.startswith("<?xml"):
            xml_content = xml_content[xml_content.index("?>") + 2 :]

        root = ET.fromstring(xml_content)

        # Most links should have visual geometry
        visual_count = 0
        for link in root.findall("link"):
            visual = link.find("visual")
            if visual is not None:
                geometry = visual.find("geometry")
                assert geometry is not None
                visual_count += 1

        assert visual_count > 10

    def test_generate_collision_geometry(self):
        config = URDFGeneratorConfig(generate_collision=True)
        generator = HumanoidURDFGenerator(config)
        params = BodyParameters()

        urdf_xml = generator.generate(params)

        xml_content = urdf_xml
        if xml_content.startswith("<?xml"):
            xml_content = xml_content[xml_content.index("?>") + 2 :]

        root = ET.fromstring(xml_content)

        # Most links should have collision geometry
        collision_count = 0
        for link in root.findall("link"):
            collision = link.find("collision")
            if collision is not None:
                collision_count += 1

        assert collision_count > 10

    def test_generate_no_collision(self):
        config = URDFGeneratorConfig(generate_collision=False)
        generator = HumanoidURDFGenerator(config)
        params = BodyParameters()

        urdf_xml = generator.generate(params)

        # Should still be valid but without collision
        assert "<collision>" not in urdf_xml

    def test_generate_write_to_file(self):
        generator = HumanoidURDFGenerator()
        params = BodyParameters()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.urdf"
            urdf_xml = generator.generate(params, output_path=output_path)

            assert output_path.exists()
            content = output_path.read_text()
            assert content == urdf_xml

    def test_generate_joint_limits(self):
        generator = HumanoidURDFGenerator()
        params = BodyParameters()

        urdf_xml = generator.generate(params)

        xml_content = urdf_xml
        if xml_content.startswith("<?xml"):
            xml_content = xml_content[xml_content.index("?>") + 2 :]

        root = ET.fromstring(xml_content)

        # Revolute joints should have limits
        for joint in root.findall("joint"):
            if joint.get("type") == "revolute":
                limit = joint.find("limit")
                assert limit is not None, f"Joint {joint.get('name')} missing limit"
                assert "lower" in limit.attrib
                assert "upper" in limit.attrib

    def test_generate_joint_dynamics(self):
        generator = HumanoidURDFGenerator()
        params = BodyParameters()

        urdf_xml = generator.generate(params)

        xml_content = urdf_xml
        if xml_content.startswith("<?xml"):
            xml_content = xml_content[xml_content.index("?>") + 2 :]

        root = ET.fromstring(xml_content)

        # Non-fixed joints should have dynamics
        dynamics_count = 0
        for joint in root.findall("joint"):
            dynamics = joint.find("dynamics")
            if dynamics is not None:
                dynamics_count += 1
                assert "damping" in dynamics.attrib

        assert dynamics_count > 0


class TestGenerateHumanoidURDF:
    """Tests for convenience function."""

    def test_basic_call(self):
        params = BodyParameters()
        urdf = generate_humanoid_urdf(params)

        assert urdf is not None
        assert "<robot" in urdf

    def test_with_config(self):
        params = BodyParameters()
        config = URDFGeneratorConfig(generate_collision=False)

        urdf = generate_humanoid_urdf(params, config=config)

        assert "<collision>" not in urdf

    def test_with_output_path(self):
        params = BodyParameters()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "humanoid.urdf"
            _ = generate_humanoid_urdf(params, output_path=output_path)

            assert output_path.exists()


class TestCompositeJointExpansion:
    """Tests for composite joint expansion."""

    def test_gimbal_joint_expansion(self):
        config = URDFGeneratorConfig(expand_composite_joints=True)
        generator = HumanoidURDFGenerator(config)
        params = BodyParameters()

        urdf_xml = generator.generate(params)

        # Gimbal joints should be expanded to 3 revolute joints
        # Check for intermediate links and joints
        assert "_z" in urdf_xml or "_y" in urdf_xml or "_x" in urdf_xml

    def test_no_expansion(self):
        config = URDFGeneratorConfig(expand_composite_joints=False)
        generator = HumanoidURDFGenerator(config)
        params = BodyParameters()

        urdf_xml = generator.generate(params)

        # Should still produce valid URDF
        assert "<robot" in urdf_xml


class TestProportionFactors:
    """Tests for body proportion factors."""

    def test_tall_character(self):
        generator = HumanoidURDFGenerator()

        # Normal height
        normal_params = BodyParameters(height_m=1.75)
        normal_urdf = generator.generate(normal_params)

        # Tall character
        tall_params = BodyParameters(height_m=1.95)
        tall_urdf = generator.generate(tall_params)

        # Both should be valid
        assert "<robot" in normal_urdf
        assert "<robot" in tall_urdf

    def test_wide_shoulders(self):
        generator = HumanoidURDFGenerator()

        params = BodyParameters(shoulder_width_factor=1.2)
        urdf = generator.generate(params)

        assert "<robot" in urdf

    def test_muscular_build(self):
        generator = HumanoidURDFGenerator()

        params = BodyParameters(muscularity=0.8, body_fat_factor=0.1)
        urdf = generator.generate(params)

        assert "<robot" in urdf
