"""Tests for non-GUI parts of component_library.py — URDFComponent, ComponentLibrary."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path


from src.tools.model_explorer.component_library import (
    ComponentLibrary,
    ComponentType,
    URDFComponent,
)
from tests.tools.model_explorer._fixtures import SIMPLE_URDF


class TestURDFComponent:
    def test_is_read_only_flag(self) -> None:
        c = URDFComponent(
            component_type=ComponentType.LINK,
            name="a",
            xml_content="<link name='a'/>",
            is_library=True,
        )
        assert c.is_read_only is True

    def test_get_hash_stable(self) -> None:
        c = URDFComponent(ComponentType.LINK, "x", "<link name='x'/>")
        h1 = c.get_hash()
        h2 = c.get_hash()
        assert h1 == h2
        assert len(h1) == 8

    def test_to_dict(self) -> None:
        c = URDFComponent(
            ComponentType.JOINT,
            "j",
            "<joint name='j'/>",
            source_file=Path("/tmp/a.urdf"),
            is_library=True,
            metadata={"k": "v"},
        )
        d = c.to_dict()
        assert d["type"] == "joint"
        assert d["name"] == "j"
        assert d["is_library"] is True
        assert d["metadata"] == {"k": "v"}
        assert d["source_file"] == str(Path("/tmp/a.urdf"))

    def test_to_dict_no_source(self) -> None:
        c = URDFComponent(ComponentType.LINK, "a", "<link/>")
        assert c.to_dict()["source_file"] is None

    def test_from_xml_element_link(self) -> None:
        elem = ET.fromstring(
            "<link name='base'>"
            "<visual><geometry><box size='1 1 1'/></geometry></visual>"
            "<inertial><mass value='2.5'/></inertial>"
            "</link>"
        )
        c = URDFComponent.from_xml_element(elem)
        assert c.component_type == ComponentType.LINK
        assert c.name == "base"
        assert c.metadata.get("geometry_type") == "box"
        assert c.metadata.get("mass") == "2.5"

    def test_from_xml_element_joint(self) -> None:
        elem = ET.fromstring(
            "<joint name='j' type='revolute'>"
            "<parent link='a'/><child link='b'/></joint>"
        )
        c = URDFComponent.from_xml_element(elem)
        assert c.component_type == ComponentType.JOINT
        assert c.metadata["joint_type"] == "revolute"
        assert c.metadata["parent"] == "a"
        assert c.metadata["child"] == "b"

    def test_from_xml_element_unknown_tag_defaults_to_link(self) -> None:
        elem = ET.fromstring("<weird name='x'/>")
        c = URDFComponent.from_xml_element(elem)
        assert c.component_type == ComponentType.LINK
        assert c.name == "x"

    def test_from_xml_element_unnamed(self) -> None:
        elem = ET.fromstring("<link/>")
        c = URDFComponent.from_xml_element(elem)
        assert c.name == "unnamed_link"


class TestComponentLibrary:
    def _write_urdf(self, tmp_path: Path) -> Path:
        p = tmp_path / "r.urdf"
        p.write_text(SIMPLE_URDF, encoding="utf-8")
        return p

    def test_load_urdf_as_library(self, tmp_path: Path) -> None:
        lib = ComponentLibrary()
        path = self._write_urdf(tmp_path)
        comps = lib.load_urdf_as_library(path)
        assert len(comps) >= 5  # 3 links + 2 joints + 1 material
        assert all(c.is_library for c in comps)
        assert path in lib.get_source_files()

    def test_load_urdf_as_library_missing_returns_empty(self, tmp_path: Path) -> None:
        lib = ComponentLibrary()
        result = lib.load_urdf_as_library(tmp_path / "nope.urdf")
        assert result == []

    def test_load_urdf_bad_xml_returns_empty(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.urdf"
        p.write_text("<not really xml", encoding="utf-8")
        lib = ComponentLibrary()
        assert lib.load_urdf_as_library(p) == []

    def test_load_urdf_as_working_marks_editable(self, tmp_path: Path) -> None:
        lib = ComponentLibrary()
        comps = lib.load_urdf_as_working(self._write_urdf(tmp_path))
        assert all(not c.is_library for c in comps)
        assert lib.get_working_components()

    def test_copy_to_working_with_rename(self, tmp_path: Path) -> None:
        lib = ComponentLibrary()
        lib.load_urdf_as_library(self._write_urdf(tmp_path))
        # find a known key
        keys = list(lib.get_library_items().keys())
        key = next(k for k in keys if k.endswith(":base"))
        new = lib.copy_to_working(key, new_name="base_copy")
        assert new is not None
        assert new.name == "base_copy"
        assert new.is_library is False
        assert 'name="base_copy"' in new.xml_content

    def test_copy_to_working_missing_key(self) -> None:
        lib = ComponentLibrary()
        assert lib.copy_to_working("ghost:x") is None

    def test_filter_by_type(self, tmp_path: Path) -> None:
        lib = ComponentLibrary()
        lib.load_urdf_as_library(self._write_urdf(tmp_path))
        joints = lib.get_library_components(filter_type=ComponentType.JOINT)
        assert all(c.component_type == ComponentType.JOINT for c in joints)
        assert len(joints) == 2

    def test_get_component_by_name(self, tmp_path: Path) -> None:
        lib = ComponentLibrary()
        lib.load_urdf_as_library(self._write_urdf(tmp_path))
        c = lib.get_component("base", from_library=True)
        assert c is not None and c.name == "base"
        assert lib.get_component("ghost", from_library=True) is None
        assert lib.get_component("ghost", from_library=False) is None

    def test_update_working_component(self, tmp_path: Path) -> None:
        lib = ComponentLibrary()
        lib.load_urdf_as_working(self._write_urdf(tmp_path))
        ok = lib.update_working_component("base", "<link name='base'/>")
        assert ok
        assert lib.get_component("base").xml_content == "<link name='base'/>"

    def test_update_working_unknown(self) -> None:
        lib = ComponentLibrary()
        assert lib.update_working_component("ghost", "<x/>") is False

    def test_remove_working_component(self, tmp_path: Path) -> None:
        lib = ComponentLibrary()
        lib.load_urdf_as_working(self._write_urdf(tmp_path))
        assert lib.remove_working_component("base") is True
        assert lib.remove_working_component("base") is False

    def test_export_working_to_urdf(self, tmp_path: Path) -> None:
        lib = ComponentLibrary()
        lib.load_urdf_as_working(self._write_urdf(tmp_path))
        xml = lib.export_working_to_urdf("exported")
        assert 'name="exported"' in xml
        root = ET.fromstring(xml)
        assert len(root.findall("link")) >= 3

    def test_export_with_bad_component_xml(self, tmp_path: Path) -> None:
        lib = ComponentLibrary()
        bad = URDFComponent(ComponentType.LINK, "x", "<not xml")
        lib._working_components["x"] = bad  # noqa: SLF001
        xml = lib.export_working_to_urdf()
        # bad component is skipped, valid output still produced
        assert "<robot" in xml

    def test_clear_working_and_library(self, tmp_path: Path) -> None:
        lib = ComponentLibrary()
        lib.load_urdf_as_library(self._write_urdf(tmp_path))
        lib.load_urdf_as_working(self._write_urdf(tmp_path))
        lib.clear_working()
        assert lib.get_working_components() == []
        lib.clear_library()
        assert lib.get_library_components() == []
        assert lib.get_source_files() == []

    def test_get_library_component_by_key(self, tmp_path: Path) -> None:
        lib = ComponentLibrary()
        lib.load_urdf_as_library(self._write_urdf(tmp_path))
        key = next(iter(lib.get_library_items().keys()))
        assert lib.get_library_component_by_key(key) is not None
        assert lib.get_library_component_by_key("ghost:x") is None
