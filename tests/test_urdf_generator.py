import xml.etree.ElementTree as ET
from typing import Any

import pytest

from src.tools.urdf_generator.urdf_builder import URDFBuilder


class TestURDFBuilder:
    """Test suite for the URDFBuilder class."""

    def setup_method(self) -> None:
        """Initialize the builder before each test."""
        self.builder = URDFBuilder()

    def test_initialization(self) -> None:
        """Test initial state of the builder."""
        assert len(self.builder.segments) == 0
        assert len(self.builder.materials) == 0
        assert self.builder.robot_name == "golf_robot"

    def test_set_robot_name(self) -> None:
        """Test setting the robot name."""
        self.builder.set_robot_name("test_robot")
        assert self.builder.robot_name == "test_robot"
        xml = self.builder.get_urdf()
        assert 'name="test_robot"' in xml

    def test_add_segment_valid(self) -> None:
        """Test adding a valid segment."""
        segment_data = {
            "name": "base_link",
            "geometry": {
                "shape": "box",
                "dimensions": {"length": 1.0, "width": 0.5, "height": 0.5},
                "position": {"x": 0, "y": 0, "z": 0},
                "orientation": {"roll": 0, "pitch": 0, "yaw": 0},
            },
            "physics": {"mass": 10.0, "inertia": {"ixx": 0.1, "iyy": 0.1, "izz": 0.1}},
        }
        self.builder.add_segment(segment_data)
        assert len(self.builder.segments) == 1
        assert "base_link" in self.builder.get_segment_names()

    def test_add_segment_duplicate_name(self) -> None:
        """Test that adding a segment with a duplicate name raises an error."""
        segment_data = {"name": "link1", "geometry": {}}
        self.builder.add_segment(segment_data)

        with pytest.raises(ValueError, match="already exists"):
            self.builder.add_segment(segment_data)

    def test_add_segment_missing_name(self) -> None:
        """Test that adding a segment without a name raises an error."""
        segment_data: dict[str, Any] = {"geometry": {}}
        with pytest.raises(ValueError, match="must have a name"):
            # segment_data is intentionally missing 'name' to test validation hooks
            self.builder.add_segment(segment_data)  # type: ignore[arg-type]

    def test_remove_segment(self) -> None:
        """Test removing a segment."""
        self.builder.add_segment({"name": "link1", "geometry": {}})
        self.builder.remove_segment("link1")
        assert len(self.builder.segments) == 0

    def test_remove_unknown_segment(self) -> None:
        """Test removing a non-existent segment."""
        with pytest.raises(ValueError, match="not found"):
            self.builder.remove_segment("ghost_link")

    def test_remove_recursive(self) -> None:
        """Test that removing a parent removes its children."""
        self.builder.add_segment({"name": "parent", "geometry": {}})
        self.builder.add_segment({"name": "child", "parent": "parent", "geometry": {}})
        self.builder.add_segment(
            {"name": "grandchild", "parent": "child", "geometry": {}}
        )

        assert len(self.builder.segments) == 3
        self.builder.remove_segment("parent")
        assert len(self.builder.segments) == 0

    def test_modify_segment(self) -> None:
        """Test modifying an existing segment."""
        self.builder.add_segment({"name": "link1", "geometry": {"shape": "box"}})

        new_data = {
            "name": "link1",
            "geometry": {"shape": "sphere", "dimensions": {"length": 2.0}},
        }
        self.builder.modify_segment(new_data)

        assert self.builder.segments[0]["geometry"]["shape"] == "sphere"

    def test_modify_unknown_segment(self) -> None:
        """Test modifying a non-existent segment."""
        with pytest.raises(ValueError, match="not found"):
            self.builder.modify_segment({"name": "ghost", "geometry": {}})

    def test_urdf_generation_structure(self) -> None:
        """Test the structure of the generated URDF XML."""
        # Add a base link
        self.builder.add_segment(
            {
                "name": "base",
                "geometry": {
                    "shape": "cylinder",
                    "dimensions": {"width": 0.2, "length": 1.0},
                },
                "physics": {"mass": 5.0},
            }
        )

        # Add a child link with a joint
        self.builder.add_segment(
            {
                "name": "arm",
                "parent": "base",
                "geometry": {"shape": "box"},
                "joint": {
                    "type": "revolute",
                    "limits": {"lower": -90, "upper": 90},
                    "axis": {"x": 0, "y": 1, "z": 0},
                },
            }
        )

        xml_str = self.builder.get_urdf()

        # Parse validation
        root = ET.fromstring(xml_str)
        assert root.tag == "robot"
        assert root.attrib["name"] == "golf_robot"

        # Check links
        links = root.findall("link")
        assert len(links) == 2
        link_names = {link.attrib["name"] for link in links}
        assert "base" in link_names
        assert "arm" in link_names

        # Check joint
        joint = root.find("joint")
        assert joint is not None
        assert joint.attrib["name"] == "base_to_arm"
        assert joint.attrib["type"] == "revolute"

        parent = joint.find("parent")
        child = joint.find("child")
        assert parent is not None
        assert child is not None
        assert parent.attrib["link"] == "base"
        assert child.attrib["link"] == "arm"

        limit = joint.find("limit")
        assert limit is not None
        # -90 deg is approx -1.57 rad
        assert float(limit.attrib["lower"]) < -1.5

    def test_validation_orphans(self) -> None:
        """Test validation detects orphaned segments."""
        self.builder.add_segment(
            {"name": "orphan", "parent": "missing_parent", "geometry": {}}
        )

        errors = self.builder.validate_urdf()
        assert len(errors) == 1
        assert "non-existent parent" in errors[0]

    def test_validation_circular_dependency(self) -> None:
        """Test validation detects circular dependencies."""
        self.builder.add_segment({"name": "linkA", "parent": "linkB", "geometry": {}})
        self.builder.add_segment({"name": "linkB", "parent": "linkA", "geometry": {}})

        errors = self.builder.validate_urdf()
        assert len(errors) > 0
        assert "Circular dependency" in errors[0]

    def test_materials_handling(self) -> None:
        """Test that materials are correctly added and managed."""
        self.builder.add_segment(
            {
                "name": "link1",
                "geometry": {},
                "physics": {
                    "material": {
                        "name": "UserBlue",
                        "color": {"r": 0, "g": 0, "b": 1, "a": 1},
                    }
                },
            }
        )

        assert "UserBlue" in self.builder.materials
        xml_str = self.builder.get_urdf()
        root = ET.fromstring(xml_str)
        mat = root.find("material")
        assert mat is not None
        assert mat.attrib["name"] == "UserBlue"

    def test_empty_urdf_fallback(self) -> None:
        """Test that get_urdf returns a valid empty structure if no segments exist."""
        xml_str = self.builder.get_urdf()
        assert "base_link" in xml_str  # Checks fallback link
