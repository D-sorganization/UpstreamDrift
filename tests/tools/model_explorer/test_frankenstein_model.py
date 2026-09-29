"""Tests for _frankenstein_model.URDFModel."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path


from src.tools.model_explorer._frankenstein_model import URDFModel
from tests.tools.model_explorer._fixtures import SIMPLE_URDF


def _link(name: str) -> ET.Element:
    e = ET.Element("link", {"name": name})
    return e


def _joint(name: str, parent: str, child: str, jtype: str = "fixed") -> ET.Element:
    j = ET.Element("joint", {"name": name, "type": jtype})
    ET.SubElement(j, "parent", {"link": parent})
    ET.SubElement(j, "child", {"link": child})
    return j


class TestURDFModel:
    def test_create_empty(self) -> None:
        m = URDFModel.create_empty("bot")
        assert m.robot_name == "bot"
        assert m.links == {} and m.joints == {} and m.materials == {}
        assert m.file_path is None
        assert m.is_modified is False

    def test_from_element(self) -> None:
        root = ET.fromstring(SIMPLE_URDF)
        m = URDFModel.from_element(root)
        assert m.robot_name == "simple"
        assert set(m.links.keys()) == {"base", "arm", "hand"}
        assert set(m.joints.keys()) == {"base_to_arm", "arm_to_hand"}
        assert "red" in m.materials

    def test_from_file_and_to_xml_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "robot.urdf"
        path.write_text(SIMPLE_URDF, encoding="utf-8")
        m = URDFModel.from_file(path)
        assert m.file_path == path

        xml = m.to_xml()
        assert "<robot" in xml and 'name="simple"' in xml
        # roundtrip via fresh parse
        m2 = URDFModel.from_element(ET.fromstring(xml))
        assert set(m2.links.keys()) == set(m.links.keys())
        assert set(m2.joints.keys()) == set(m.joints.keys())

    def test_add_link_unique_naming(self) -> None:
        m = URDFModel.create_empty()
        n1 = m.add_link(_link("arm"))
        n2 = m.add_link(_link("arm"))
        n3 = m.add_link(_link("arm"))
        assert n1 == "arm"
        assert n2 == "arm_1"
        assert n3 == "arm_2"
        assert m.is_modified

    def test_add_link_with_rename(self) -> None:
        m = URDFModel.create_empty()
        n = m.add_link(_link("orig"), new_name="renamed")
        assert n == "renamed"
        assert m.links["renamed"].get("name") == "renamed"

    def test_add_link_default_name_when_missing(self) -> None:
        m = URDFModel.create_empty()
        elem = ET.Element("link")  # no name attr
        n = m.add_link(elem)
        assert n == "unnamed_link"

    def test_add_joint_with_parent_mapping(self) -> None:
        m = URDFModel.create_empty()
        j = _joint("j", "old_p", "old_c")
        mapping = {"old_p": "new_p", "old_c": "new_c"}
        name = m.add_joint(j, parent_mapping=mapping)
        added = m.joints[name]
        assert added.find("parent").get("link") == "new_p"
        assert added.find("child").get("link") == "new_c"

    def test_add_joint_dedup(self) -> None:
        m = URDFModel.create_empty()
        m.add_joint(_joint("j", "a", "b"))
        n2 = m.add_joint(_joint("j", "a", "b"))
        assert n2 == "j_1"

    def test_add_material_dedup(self) -> None:
        m = URDFModel.create_empty()
        mat = ET.Element("material", {"name": "blue"})
        n1 = m.add_material(mat)
        n2 = m.add_material(mat)
        assert n1 == "blue" and n2 == "blue"
        # second add does not duplicate
        assert len(m.materials) == 1

    def test_remove_link_also_removes_connected_joints(self) -> None:
        m = URDFModel.create_empty()
        m.add_link(_link("a"))
        m.add_link(_link("b"))
        m.add_link(_link("c"))
        m.add_joint(_joint("ab", "a", "b"))
        m.add_joint(_joint("bc", "b", "c"))

        assert m.remove_link("b") is True
        assert "b" not in m.links
        # both joints touching "b" are gone
        assert "ab" not in m.joints
        assert "bc" not in m.joints

    def test_remove_link_unknown_returns_false(self) -> None:
        m = URDFModel.create_empty()
        assert m.remove_link("nope") is False

    def test_remove_joint(self) -> None:
        m = URDFModel.create_empty()
        m.add_joint(_joint("j", "a", "b"))
        assert m.remove_joint("j") is True
        assert m.remove_joint("j") is False

    def test_get_names(self) -> None:
        m = URDFModel.create_empty()
        m.add_link(_link("x"))
        m.add_joint(_joint("y", "x", "z"))
        assert m.get_link_names() == ["x"]
        assert m.get_joint_names() == ["y"]
