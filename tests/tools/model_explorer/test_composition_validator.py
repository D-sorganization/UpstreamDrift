"""Composition validation tests for the first-party Frankenstein editor."""

from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET  # nosemgrep: python.lang.security.use-defused-xml.use-defused-xml

import pytest

from src.tools.model_explorer.composition_validator import (
    CompositionValidationError,
    CompositionValidator,
)
from src.tools.model_explorer.frankenstein_editor.model import URDFModel


def _link(name: str, *, mass: float | None = None) -> ET.Element:
    link = ET.Element("link", {"name": name})
    if mass is not None:
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "mass", {"value": str(mass)})
        ET.SubElement(
            inertial,
            "inertia",
            {
                "ixx": "0.1",
                "iyy": "0.1",
                "izz": "0.1",
                "ixy": "0",
                "ixz": "0",
                "iyz": "0",
            },
        )
    return link


def _box_link(
    name: str,
    *,
    size: str,
    origin_xyz: str = "0 0 0",
    mass: float | None = None,
) -> ET.Element:
    link = _link(name, mass=mass)
    visual = ET.SubElement(link, "visual")
    ET.SubElement(visual, "origin", {"xyz": origin_xyz})
    geometry = ET.SubElement(visual, "geometry")
    ET.SubElement(geometry, "box", {"size": size})
    return link


def _joint(
    name: str,
    parent: str,
    child: str,
    *,
    joint_type: str = "fixed",
    origin_xyz: str = "0 0 0",
) -> ET.Element:
    joint = ET.Element("joint", {"name": name, "type": joint_type})
    ET.SubElement(joint, "origin", {"xyz": origin_xyz})
    ET.SubElement(joint, "parent", {"link": parent})
    ET.SubElement(joint, "child", {"link": child})
    return joint


def _robot(*children: ET.Element) -> ET.Element:
    root = ET.Element("robot", {"name": "test_robot"})
    root.extend(children)
    return root


def _codes(result: object) -> set[str]:
    return {finding.code for finding in result.errors}


def _warning_codes(result: object) -> set[str]:
    return {finding.code for finding in result.warnings}


def test_valid_tree_has_no_errors() -> None:
    root = _robot(
        _link("base"),
        _link("arm", mass=1.0),
        _link("hand", mass=0.5),
        _joint("base_to_arm", "base", "arm", joint_type="revolute"),
        _joint("arm_to_hand", "arm", "hand", joint_type="continuous"),
    )

    result = CompositionValidator().validate_xml_root(root)

    assert result.ok
    assert result.errors == ()


def test_duplicate_raw_link_names_are_actionable() -> None:
    root = _robot(_link("base"), _link("base"))

    result = CompositionValidator().validate_xml_root(root)

    assert "duplicate_link_name" in _codes(result)
    assert "base" in result.errors[0].message


def test_cycle_detected_with_joint_names() -> None:
    root = _robot(
        _link("base"),
        _link("arm"),
        _joint("base_to_arm", "base", "arm"),
        _joint("arm_to_base", "arm", "base"),
    )

    result = CompositionValidator().validate_xml_root(root)

    assert "topology_cycle" in _codes(result)
    assert any("base_to_arm" in finding.message for finding in result.errors)
    assert any("arm_to_base" in finding.message for finding in result.errors)


def test_orphan_joint_names_missing_link() -> None:
    root = _robot(
        _link("base"),
        _joint("base_to_ghost", "base", "ghost", joint_type="fixed"),
    )

    result = CompositionValidator().validate_xml_root(root)

    assert "orphan_joint" in _codes(result)
    assert any("base_to_ghost" in finding.message for finding in result.errors)
    assert any("ghost" in finding.message for finding in result.errors)


def test_non_fixed_child_requires_positive_mass() -> None:
    root = _robot(
        _link("base"),
        _link("arm", mass=0.0),
        _joint("base_to_arm", "base", "arm", joint_type="revolute"),
    )

    result = CompositionValidator().validate_xml_root(root)

    assert "invalid_mass" in _codes(result)
    assert any("arm" in finding.message for finding in result.errors)


def test_non_fixed_child_requires_positive_semidefinite_inertia() -> None:
    arm = _link("arm", mass=1.0)
    inertia = arm.find("inertial/inertia")
    assert inertia is not None
    inertia.set("ixx", "-1")
    root = _robot(
        _link("base"),
        arm,
        _joint("base_to_arm", "base", "arm", joint_type="revolute"),
    )

    result = CompositionValidator().validate_xml_root(root)

    assert "invalid_inertia" in _codes(result)
    assert any("arm" in finding.message for finding in result.errors)


def test_heavy_attached_subtree_reports_mass_ratio_warning() -> None:
    root = _robot(
        _link("base", mass=1.0),
        _link("heavy_tool", mass=3.0),
        _link("payload", mass=1.0),
        _joint("base_to_tool", "base", "heavy_tool"),
        _joint("tool_to_payload", "heavy_tool", "payload"),
    )

    result = CompositionValidator().validate_xml_root(root)

    assert result.ok
    assert "subtree_mass_ratio" in _warning_codes(result)
    assert any("heavy_tool" in finding.message for finding in result.warnings)


def test_overlapping_attachment_geometry_reports_warning() -> None:
    root = _robot(
        _box_link("base", size="2 2 2"),
        _box_link("tool", size="1 1 1"),
        _joint("base_to_tool", "base", "tool", origin_xyz="0.25 0 0"),
    )

    result = CompositionValidator().validate_xml_root(root)

    assert result.ok
    assert "geometry_overlap" in _warning_codes(result)
    assert any("base_to_tool" in finding.message for finding in result.warnings)


def test_separated_attachment_geometry_has_no_overlap_warning() -> None:
    root = _robot(
        _box_link("base", size="2 2 2"),
        _box_link("tool", size="1 1 1"),
        _joint("base_to_tool", "base", "tool", origin_xyz="3 0 0"),
    )

    result = CompositionValidator().validate_xml_root(root)

    assert "geometry_overlap" not in _warning_codes(result)


def test_model_export_blocks_invalid_cycle_and_force_allows_export() -> None:
    model = URDFModel.from_element(
        _robot(
            _link("base"),
            _link("arm"),
            _joint("base_to_arm", "base", "arm"),
            _joint("arm_to_base", "arm", "base"),
        )
    )

    with pytest.raises(CompositionValidationError) as exc_info:
        model.to_xml()

    assert "base_to_arm" in str(exc_info.value)
    assert "arm_to_base" in str(exc_info.value)
    forced_xml = model.to_xml(force=True)
    assert 'joint name="base_to_arm"' in forced_xml


def test_model_panel_surfaces_validation_findings() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    if "PyQt6.QtWidgets" in sys.modules:
        qt_widgets = sys.modules["PyQt6.QtWidgets"]
    else:
        qt_widgets = pytest.importorskip("PyQt6.QtWidgets")
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    _ = app
    from src.tools.model_explorer.frankenstein_editor.panel import ModelPanel

    panel = ModelPanel("Working Model")
    panel.set_model(
        URDFModel.from_element(
            _robot(
                _box_link("base", size="2 2 2"),
                _box_link("tool", size="1 1 1"),
                _joint("base_to_tool", "base", "tool", origin_xyz="0.25 0 0"),
            )
        )
    )

    messages = [
        panel.validation_list.item(index).text()
        for index in range(panel.validation_list.count())
    ]
    assert any("geometry_overlap" in message for message in messages)
