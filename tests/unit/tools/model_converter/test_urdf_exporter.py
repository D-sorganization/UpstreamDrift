"""Unit tests for URDF model exporter (Issue #9965)."""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.model_converter.schema_validator import validate_canonical_model
from tools.model_converter.urdf_exporter import export_urdf

CANONICAL_YAML = (
    REPO_ROOT
    / "src"
    / "engines"
    / "physics_engines"
    / "pinocchio"
    / "models"
    / "spec"
    / "golfer_canonical.yaml"
)


@pytest.fixture(scope="module")
def canonical_model():
    return validate_canonical_model(CANONICAL_YAML)


@pytest.fixture(scope="module")
def urdf_root(canonical_model):
    xml_str = export_urdf(canonical_model)
    return ET.fromstring(xml_str)


class TestUrdfExporter:
    """Test exported URDF structural contracts."""

    def test_xml_is_well_formed(self, canonical_model) -> None:
        xml_str = export_urdf(canonical_model)
        assert xml_str.startswith("<?xml")
        root = ET.fromstring(xml_str)
        assert root.tag == "robot"
        assert root.attrib.get("name") == "golfer"

    def test_pelvis_is_root_without_parent(self, urdf_root) -> None:
        links = {link.attrib["name"] for link in urdf_root.findall("link")}
        assert "pelvis" in links
        child_links = {
            joint.find("child").attrib["link"]  # type: ignore[union-attr]
            for joint in urdf_root.findall("joint")
        }
        assert "pelvis" not in child_links, (
            "pelvis must be the root link with no parent joint"
        )

    def test_pelvis_inertial_properties(self, urdf_root) -> None:
        for link in urdf_root.findall("link"):
            if link.attrib.get("name") == "pelvis":
                inertial = link.find("inertial")
                assert inertial is not None
                mass = float(inertial.find("mass").attrib["value"])  # type: ignore[union-attr]
                assert mass == 11.7
                inertia = inertial.find("inertia")
                assert float(inertia.attrib["ixx"]) == 0.1337  # type: ignore[union-attr]
                assert float(inertia.attrib["iyy"]) == 0.1337  # type: ignore[union-attr]
                assert float(inertia.attrib["izz"]) == 0.1337  # type: ignore[union-attr]
                break
        else:
            pytest.fail("pelvis link not found")

    def test_mid_hands_link_and_grip_welding(self, urdf_root) -> None:
        links = {link.attrib["name"] for link in urdf_root.findall("link")}
        assert "mid_hands" in links
        assert "club_shaft" in links
        assert "club_head" in links

        joints = {j.attrib["name"]: j for j in urdf_root.findall("joint")}
        assert "thorax3_to_mid_hands" in joints
        mid_joint = joints["thorax3_to_mid_hands"]
        assert mid_joint.attrib["type"] == "fixed"
        assert mid_joint.find("parent").attrib["link"] == "thorax3"  # type: ignore[union-attr]
        assert mid_joint.find("child").attrib["link"] == "mid_hands"  # type: ignore[union-attr]

        assert "mid_hands_to_club_shaft" in joints
        shaft_joint = joints["mid_hands_to_club_shaft"]
        assert shaft_joint.attrib["type"] == "fixed"
        assert shaft_joint.find("parent").attrib["link"] == "mid_hands"  # type: ignore[union-attr]
        assert shaft_joint.find("child").attrib["link"] == "club_shaft"  # type: ignore[union-attr]

    def test_universal_joint_decomposition(self, urdf_root) -> None:
        links = {link.attrib["name"] for link in urdf_root.findall("link")}
        assert "lumbar1_intermediate" in links
        joints = {j.attrib["name"]: j for j in urdf_root.findall("joint")}
        assert "pelvis_to_lumbar1_intermediate" in joints
        assert "lumbar1_intermediate_to_lumbar1" in joints

    def test_gimbal_joint_decomposition(self, urdf_root) -> None:
        links = {link.attrib["name"] for link in urdf_root.findall("link")}
        assert "right_thigh_gimbal_z" in links
        assert "right_thigh_gimbal_y" in links
        joints = {j.attrib["name"]: j for j in urdf_root.findall("joint")}
        assert "pelvis_to_right_thigh_gimbal_z" in joints
        assert "right_thigh_gimbal_z_to_right_thigh_gimbal_y" in joints
        assert "right_thigh_gimbal_y_to_right_thigh" in joints

    def test_export_to_file(self, canonical_model, tmp_path: Path) -> None:
        out_file = tmp_path / "test_golfer.urdf"
        xml_str = export_urdf(canonical_model, out_path=out_file)
        assert out_file.exists()
        assert out_file.read_text(encoding="utf-8") == xml_str
