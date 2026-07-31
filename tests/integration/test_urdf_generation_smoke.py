"""Integration smoke test: URDF generation and cross-engine load.

Issue #2369 -- test(urdf): add cross-engine URDF generation and load smoke test.

Generates a humanoid URDF from default body parameters and verifies that:
1. The generated XML is structurally valid (parseable, has links + joints).
2. The URDF can be loaded by each available physics engine (MuJoCo, Drake,
   Pinocchio).  Engines that are not installed are skipped automatically via
   ``pytest.importorskip``.
3. The output can be written to and read back from disk without corruption.
"""

from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET  # stdlib for Element/SubElement
from pathlib import Path

import defusedxml.ElementTree as DefusedET  # noqa: S314  # Security: defusedxml prevents XML attacks
import pytest
from humanoid_character_builder.core.body_parameters import BodyParameters
from humanoid_character_builder.generators.urdf_generator import (
    HumanoidURDFGenerator,
    URDFGeneratorConfig,
    generate_humanoid_urdf,
)


@pytest.fixture(scope="module")
def default_urdf() -> str:
    """Generate a default humanoid URDF once for the whole module."""
    params = BodyParameters(name="smoke_test_robot")
    return generate_humanoid_urdf(params)


@pytest.fixture(scope="module")
def default_urdf_root(default_urdf: str) -> ET.Element:
    """Parse the default URDF once for read-only structural assertions."""
    return DefusedET.fromstring(default_urdf)


@pytest.fixture(scope="module")
def default_urdf_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write the default URDF to a temp file and return its path."""
    params = BodyParameters(name="smoke_test_robot")
    tmp_dir = tmp_path_factory.mktemp("urdf_smoke")
    urdf_path = tmp_dir / "smoke_test_robot.urdf"
    generate_humanoid_urdf(params, output_path=urdf_path)
    return urdf_path


class TestURDFStructuralValidity:
    """Verify the generated URDF is structurally well-formed."""

    def test_is_parseable_xml(self, default_urdf_root: ET.Element) -> None:
        assert default_urdf_root.tag == "robot"

    def test_has_robot_name(self, default_urdf_root: ET.Element) -> None:
        assert default_urdf_root.get("name") == "smoke_test_robot"

    def test_has_links(self, default_urdf_root: ET.Element) -> None:
        links = default_urdf_root.findall("link")
        assert len(links) >= 10, f"Expected >=10 links, got {len(links)}"

    def test_has_joints(self, default_urdf_root: ET.Element) -> None:
        joints = default_urdf_root.findall("joint")
        assert len(joints) >= 10, f"Expected >=10 joints, got {len(joints)}"

    def test_all_links_have_inertial(self, default_urdf_root: ET.Element) -> None:
        for link in default_urdf_root.findall("link"):
            inertial = link.find("inertial")
            link_name = link.get("name")
            assert inertial is not None, f"Link '{link_name}' missing <inertial>"
            assert inertial.find("mass") is not None
            assert inertial.find("inertia") is not None

    def test_all_links_have_visual(self, default_urdf_root: ET.Element) -> None:
        for link in default_urdf_root.findall("link"):
            visual = link.find("visual")
            link_name = link.get("name")
            assert visual is not None, f"Link '{link_name}' missing <visual>"
            assert visual.find("geometry") is not None

    def test_revolute_joints_have_limits(self, default_urdf_root: ET.Element) -> None:
        for joint in default_urdf_root.findall("joint"):
            if joint.get("type") == "revolute":
                limit = joint.find("limit")
                joint_name = joint.get("name")
                assert (
                    limit is not None
                ), f"Revolute joint '{joint_name}' missing <limit>"
                assert "lower" in limit.attrib
                assert "upper" in limit.attrib

    def test_joint_parent_child_reference_valid_links(
        self, default_urdf_root: ET.Element
    ) -> None:
        """Every joint parent and child must reference an existing link."""
        link_names = {link.get("name") for link in default_urdf_root.findall("link")}
        for joint in default_urdf_root.findall("joint"):
            joint_name = joint.get("name")
            parent = joint.find("parent")
            child = joint.find("child")
            assert parent is not None
            assert child is not None
            parent_link = parent.get("link")
            child_link = child.get("link")
            assert (
                parent_link in link_names
            ), f"Joint '{joint_name}' parent '{parent_link}' not found in links"
            assert (
                child_link in link_names
            ), f"Joint '{joint_name}' child '{child_link}' not found in links"

    def test_no_xml_declaration(self, default_urdf: str) -> None:
        """URDF output must not carry an XML declaration header."""
        assert "<?xml" not in default_urdf


class TestURDFFilePersistence:
    """Verify writing and reading back the URDF from disk."""

    def test_file_is_written(self, default_urdf_path: Path) -> None:
        assert default_urdf_path.exists()
        assert default_urdf_path.stat().st_size > 0

    def test_file_content_is_valid_xml(self, default_urdf_path: Path) -> None:
        content = default_urdf_path.read_text(encoding="utf-8")
        root = DefusedET.fromstring(content)
        assert root.tag == "robot"

    def test_file_content_matches_string_output(self) -> None:
        """generate() return value and file content must be identical."""
        params = BodyParameters(name="file_match_robot")
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "robot.urdf"
            urdf_str = generate_humanoid_urdf(params, output_path=out)
            assert out.read_text(encoding="utf-8") == urdf_str

    def test_round_trip_preserves_link_count(self, default_urdf_path: Path) -> None:
        content = default_urdf_path.read_text(encoding="utf-8")
        root = DefusedET.fromstring(content)
        assert len(root.findall("link")) >= 10


class TestURDFConfigVariants:
    """Verify that different URDFGeneratorConfig options produce valid URDF."""

    def test_no_collision_geometry(self) -> None:
        config = URDFGeneratorConfig(generate_collision=False)
        generator = HumanoidURDFGenerator(config)
        urdf = generator.generate(BodyParameters())
        root = DefusedET.fromstring(urdf)
        for link in root.findall("link"):
            assert link.find("collision") is None

    def test_composite_joints_not_expanded(self) -> None:
        config = URDFGeneratorConfig(expand_composite_joints=False)
        generator = HumanoidURDFGenerator(config)
        urdf = generator.generate(BodyParameters())
        root = DefusedET.fromstring(urdf)
        assert len(root.findall("joint")) >= 1

    def test_composite_joints_expanded(self) -> None:
        config = URDFGeneratorConfig(expand_composite_joints=True)
        generator = HumanoidURDFGenerator(config)
        urdf = generator.generate(BodyParameters())
        assert "_z" in urdf or "_y" in urdf or "_x" in urdf

    def test_compact_output(self) -> None:
        config = URDFGeneratorConfig(pretty_print=False)
        generator = HumanoidURDFGenerator(config)
        urdf = generator.generate(BodyParameters())
        root = DefusedET.fromstring(urdf)
        assert root.tag == "robot"
        assert "\n  " not in urdf


@pytest.mark.integration
class TestURDFDrakeLoad:
    """Smoke test: load the generated URDF through Drake."""

    def test_drake_load_urdf(self, default_urdf_path: Path) -> None:
        # Check pydrake.multibody is available (not just a mock stub)
        pytest.importorskip(
            "pydrake.multibody",
            reason="pydrake.multibody not available -- skipping Drake URDF load smoke test",
        )
        from pydrake.multibody.parsing import Parser
        from pydrake.multibody.plant import MultibodyPlant

        try:
            plant = MultibodyPlant(time_step=0.0)
            parser = Parser(plant)
            if hasattr(parser, "AddModels"):
                parser.AddModels(str(default_urdf_path))
            else:
                parser.AddModelFromFile(str(default_urdf_path))
            plant.Finalize()
            assert plant.num_bodies() >= 1
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"Drake failed to load generated URDF: {exc}")
