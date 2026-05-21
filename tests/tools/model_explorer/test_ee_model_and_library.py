"""Tests for _ee_model.EndEffector and _ee_library.EndEffectorLibrary."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from src.tools.model_explorer._ee_library import EndEffectorLibrary
from src.tools.model_explorer._ee_model import EndEffector
from tests.tools.model_explorer._fixtures import EE_URDF


def _link(name: str) -> ET.Element:
    return ET.Element("link", {"name": name})


def _joint(name: str, jtype: str = "fixed") -> ET.Element:
    return ET.Element("joint", {"name": name, "type": jtype})


class TestEndEffector:
    def test_get_all_link_names(self) -> None:
        root = _link("root_link")
        ee = EndEffector(
            name="root_link",
            link_element=root,
            joint_element=None,
            child_links=[_link("c1"), _link("c2")],
            child_joints=[],
        )
        assert ee.get_all_link_names() == ["root_link", "c1", "c2"]

    def test_joint_type_default_fixed(self) -> None:
        ee = EndEffector(
            name="x",
            link_element=_link("x"),
            joint_element=None,
            child_links=[],
            child_joints=[],
        )
        assert ee.get_attachment_joint_type() == "fixed"

    def test_joint_type_from_element(self) -> None:
        ee = EndEffector(
            name="x",
            link_element=_link("x"),
            joint_element=_joint("j", "revolute"),
            child_links=[],
            child_joints=[],
        )
        assert ee.get_attachment_joint_type() == "revolute"

    def test_to_xml_elements_deepcopies(self) -> None:
        root = _link("r")
        c1 = _link("c1")
        j = _joint("attach", "fixed")
        cj = _joint("inner", "fixed")
        ee = EndEffector(
            name="r",
            link_element=root,
            joint_element=j,
            child_links=[c1],
            child_joints=[cj],
            source_file=Path("/tmp/x.urdf"),
        )
        links, joints = ee.to_xml_elements()
        assert len(links) == 2 and len(joints) == 2
        # ensure deep copy: not the same object
        assert links[0] is not root
        assert joints[0] is not j


class TestEndEffectorLibrary:
    def test_builtin_names_nonempty(self) -> None:
        lib = EndEffectorLibrary()
        names = lib.get_builtin_names()
        assert "simple_gripper" in names
        assert "tool_flange" in names
        assert "golf_club_attachment" in names

    def test_get_builtin_returns_end_effector(self) -> None:
        lib = EndEffectorLibrary()
        ee = lib.get_builtin("simple_gripper")
        assert ee is not None
        assert ee.link_element.get("name") == "gripper_base"
        assert len(ee.child_links) == 2
        assert len(ee.child_joints) == 2

    def test_get_builtin_unknown_returns_none(self) -> None:
        lib = EndEffectorLibrary()
        assert lib.get_builtin("nope") is None

    def test_get_builtin_info(self) -> None:
        lib = EndEffectorLibrary()
        info = lib.get_builtin_info("tool_flange")
        assert info is not None
        assert info["name"] == "Tool Flange"
        assert "description" in info
        assert lib.get_builtin_info("nope") is None

    def test_extract_from_urdf(self) -> None:
        lib = EndEffectorLibrary()
        ee = lib.extract_from_urdf(EE_URDF, "wrist", source_file=Path("x.urdf"))
        assert ee is not None
        assert ee.name == "wrist"
        # joint connecting base->wrist
        assert ee.joint_element is not None
        assert ee.joint_element.get("name") == "base_wrist"
        # two fingers
        child_names = [link.get("name") for link in ee.child_links]
        assert set(child_names) == {"finger_a", "finger_b"}
        assert ee.source_file == Path("x.urdf")

    def test_extract_from_urdf_missing_link(self) -> None:
        lib = EndEffectorLibrary()
        assert lib.extract_from_urdf(EE_URDF, "nope") is None

    def test_extract_from_urdf_bad_xml(self) -> None:
        lib = EndEffectorLibrary()
        assert lib.extract_from_urdf("<not xml", "wrist") is None

    def test_add_and_remove_from_library(self) -> None:
        lib = EndEffectorLibrary()
        ee = lib.get_builtin("tool_flange")
        assert ee is not None
        lib.add_to_library("flange", ee)
        assert "flange" in lib.end_effectors
        assert lib.remove_from_library("flange") is True
        assert lib.remove_from_library("flange") is False

    def test_precondition_get_builtin_none(self) -> None:
        lib = EndEffectorLibrary()
        with pytest.raises((ValueError, AssertionError, AttributeError)):
            lib.get_builtin(None)  # type: ignore[arg-type]
