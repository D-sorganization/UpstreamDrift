"""Additional URDFBuilder coverage: validation, handedness, joints, geometry."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from src.tools.model_explorer.urdf_builder import Handedness, URDFBuilder
from tests.tools.model_explorer._fixtures import make_segment


class TestURDFBuilderValidation:
    def setup_method(self) -> None:
        self.b = URDFBuilder()

    def test_reject_zero_mass(self) -> None:
        seg = make_segment("a")
        seg["physics"]["mass"] = 0.0
        with pytest.raises(ValueError, match="Mass must be positive"):
            self.b.add_segment(seg)

    def test_reject_negative_mass(self) -> None:
        seg = make_segment("a")
        seg["physics"]["mass"] = -1.0
        with pytest.raises(ValueError, match="Mass must be positive"):
            self.b.add_segment(seg)

    def test_reject_negative_inertia(self) -> None:
        seg = make_segment("a")
        seg["physics"]["inertia"]["ixx"] = -0.01
        with pytest.raises(ValueError, match="diagonal"):
            self.b.add_segment(seg)

    def test_reject_non_positive_definite_inertia(self) -> None:
        seg = make_segment("a")
        # huge off-diagonal makes it non PD
        seg["physics"]["inertia"]["ixy"] = 10.0
        with pytest.raises(ValueError, match="positive-definite"):
            self.b.add_segment(seg)

    def test_reject_missing_name(self) -> None:
        with pytest.raises(ValueError, match="must have a name"):
            self.b.add_segment({})

    def test_reject_duplicate_name(self) -> None:
        self.b.add_segment(make_segment("a"))
        with pytest.raises(ValueError, match="already exists"):
            self.b.add_segment(make_segment("a"))


class TestURDFBuilderRemovalAndModify:
    def setup_method(self) -> None:
        self.b = URDFBuilder()
        self.b.add_segment(make_segment("root"))
        self.b.add_segment(make_segment("arm", parent="root"))
        self.b.add_segment(make_segment("hand", parent="arm"))

    def test_remove_segment_removes_descendants(self) -> None:
        self.b.remove_segment("arm")
        names = self.b.get_segment_names()
        assert "arm" not in names
        assert "hand" not in names
        assert "root" in names

    def test_remove_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="not found"):
            self.b.remove_segment("ghost")

    def test_modify_segment(self) -> None:
        updated = make_segment("arm", parent="root")
        updated["physics"]["mass"] = 5.0
        # add_segment already validated; modify_segment does not validate
        self.b.modify_segment(updated)

    def test_modify_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="not found"):
            self.b.modify_segment(make_segment("ghost"))


class TestURDFBuilderGeneration:
    def setup_method(self) -> None:
        self.b = URDFBuilder()

    def test_empty_urdf_has_base_link(self) -> None:
        xml = self.b.get_urdf()
        assert 'name="base_link"' in xml

    def test_validate_finds_orphan_parent(self) -> None:
        # bypass add_segment for orphan detection
        self.b.segments.append({"name": "x", "parent": "ghost"})
        errors = self.b.validate_urdf()
        assert any("non-existent parent" in e for e in errors)

    def test_validate_clean(self) -> None:
        self.b.add_segment(make_segment("a"))
        self.b.add_segment(make_segment("b", parent="a"))
        assert self.b.validate_urdf() == []

    @pytest.mark.parametrize(
        "shape", ["Box", "Cylinder", "Sphere", "Capsule", "Unknown"]
    )
    def test_geometry_shapes(self, shape: str) -> None:
        seg = make_segment("a")
        seg["geometry"]["shape"] = shape
        self.b.add_segment(seg)
        xml = self.b.get_urdf()
        root = ET.fromstring(xml)
        link = root.find("link")
        assert link is not None
        geom = link.find("visual/geometry")
        assert geom is not None
        # Unknown shape falls back to box
        if shape == "Unknown":
            assert geom.find("box") is not None
        else:
            assert geom.find(shape.lower()) is not None

    def test_joint_revolute_and_prismatic_emits_limits(self) -> None:
        self.b.add_segment(make_segment("root"))
        prism = make_segment("p", parent="root")
        prism["joint"]["type"] = "prismatic"
        self.b.add_segment(prism)
        xml = self.b.get_urdf()
        root = ET.fromstring(xml)
        joints = root.findall("joint")
        assert any(j.get("type") == "prismatic" for j in joints)
        for j in joints:
            if j.get("type") in {"revolute", "prismatic"}:
                assert j.find("limit") is not None
                assert j.find("axis") is not None

    def test_continuous_joint_no_limits(self) -> None:
        self.b.add_segment(make_segment("root"))
        cont = make_segment("c", parent="root")
        cont["joint"]["type"] = "continuous"
        self.b.add_segment(cont)
        xml = self.b.get_urdf()
        root = ET.fromstring(xml)
        cj = next(j for j in root.findall("joint") if j.get("type") == "continuous")
        assert cj.find("axis") is not None
        assert cj.find("limit") is None  # no limit for continuous


class TestHandedness:
    def test_default_right(self) -> None:
        b = URDFBuilder()
        assert b.get_handedness() == Handedness.RIGHT

    def test_set_handedness(self) -> None:
        b = URDFBuilder()
        b.set_handedness(Handedness.LEFT)
        assert b.get_handedness() == Handedness.LEFT

    def test_mirror_toggles_handedness(self) -> None:
        b = URDFBuilder()
        b.add_segment(make_segment("left_arm"))
        b.mirror_for_handedness()
        assert b.get_handedness() == Handedness.LEFT
        names = b.get_segment_names()
        assert "right_arm" in names

    def test_mirror_y_position(self) -> None:
        b = URDFBuilder()
        seg = make_segment("a")
        seg["geometry"]["position"]["y"] = 0.5
        b.add_segment(seg)
        b.mirror_for_handedness()
        assert b.segments[0]["geometry"]["position"]["y"] == -0.5

    def test_get_mirrored_urdf_does_not_change_state(self) -> None:
        b = URDFBuilder()
        b.add_segment(make_segment("left_arm"))
        original_h = b.get_handedness()
        original_names = list(b.get_segment_names())
        b.get_mirrored_urdf(Handedness.LEFT)
        assert b.get_handedness() == original_h
        assert b.get_segment_names() == original_names

    def test_get_mirrored_urdf_same_handedness_unchanged(self) -> None:
        b = URDFBuilder()
        b.add_segment(make_segment("a"))
        xml = b.get_mirrored_urdf(Handedness.RIGHT)
        assert "<robot" in xml


class TestClearAndMisc:
    def test_clear(self) -> None:
        b = URDFBuilder()
        b.add_segment(make_segment("a"))
        b.clear()
        assert b.get_segment_count() == 0
        assert b.materials == {}

    def test_set_robot_name(self) -> None:
        b = URDFBuilder()
        b.set_robot_name("nameOfBot")
        xml = b.get_urdf()
        assert 'name="nameOfBot"' in xml
