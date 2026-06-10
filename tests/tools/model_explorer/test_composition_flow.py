"""Headless composition UX flow tests for the model explorer."""

from __future__ import annotations

import xml.etree.ElementTree as ET
import os
import sys

import pytest
from PyQt6.QtWidgets import QApplication

from src.tools.model_explorer.composition_flow import (
    AttachmentSelection,
    CompositionFlowController,
    CompositionFlowError,
)
from src.tools.model_explorer.frankenstein_editor.model import URDFModel

pytestmark = [pytest.mark.unit]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _model(xml: str) -> URDFModel:
    return URDFModel.from_element(ET.fromstring(xml))


_HUMAN = """<robot name="human">
  <link name="pelvis"/>
  <link name="right_hand"/>
  <joint name="pelvis_to_hand" type="fixed">
    <parent link="pelvis"/>
    <child link="right_hand"/>
    <origin xyz="0 0 1" rpy="0 0 0"/>
  </joint>
</robot>"""

_ARM = """<robot name="arm">
  <link name="base"/>
  <link name="tool"/>
  <joint name="base_to_tool" type="fixed">
    <parent link="base"/>
    <child link="tool"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
  </joint>
</robot>"""


def test_declared_attachment_composes_source_model_and_exports_urdf() -> None:
    target = _model(_HUMAN)
    source = _model(_ARM)
    controller = CompositionFlowController()

    result = controller.attach_source_model(
        target_model=target,
        source_model=source,
        selection=AttachmentSelection(
            target_link="right_hand",
            attachment_name="right hand mount",
            interface_xyz=(0.1, 0.2, 0.3),
            interface_rpy=(0.0, 1.57, 0.0),
            source_prefix="arm_",
        ),
    )

    assert result.validation.ok
    assert result.source_root_link == "base"
    assert result.mapped_root_link == "arm_base"
    assert target.links.keys() >= {"pelvis", "right_hand", "arm_base", "arm_tool"}
    attachment_joint = target.joints["attach_right_hand_arm_base"]
    assert attachment_joint.find("parent").get("link") == "right_hand"  # type: ignore[union-attr]
    assert attachment_joint.find("child").get("link") == "arm_base"  # type: ignore[union-attr]
    origin = attachment_joint.find("origin")
    assert origin is not None
    assert origin.get("xyz") == "0.1 0.2 0.3"
    assert origin.get("rpy") == "0.0 1.57 0.0"

    exported = controller.export_model(target, export_format="urdf")

    assert 'link name="arm_base"' in exported.content
    assert exported.format == "urdf"


def test_composed_model_exports_mjcf_preview() -> None:
    target = _model(_HUMAN)
    source = _model(_ARM)
    controller = CompositionFlowController()
    controller.attach_source_model(
        target_model=target,
        source_model=source,
        selection=AttachmentSelection(target_link="right_hand", source_prefix="arm_"),
    )

    exported = controller.export_model(target, export_format="mjcf")

    assert exported.format == "mjcf"
    assert "<mujoco" in exported.content
    assert "arm_base" in exported.content


def test_export_refuses_validation_errors_without_force() -> None:
    cyclic = _model(
        """<robot name="bad">
          <link name="base"/>
          <link name="arm"/>
          <joint name="base_to_arm" type="fixed">
            <parent link="base"/>
            <child link="arm"/>
          </joint>
          <joint name="arm_to_base" type="fixed">
            <parent link="arm"/>
            <child link="base"/>
          </joint>
        </robot>"""
    )

    with pytest.raises(CompositionFlowError, match="validation errors"):
        CompositionFlowController().export_model(cyclic, export_format="urdf")


def test_attach_fails_for_unknown_target_link() -> None:
    with pytest.raises(CompositionFlowError, match="target link"):
        CompositionFlowController().attach_source_model(
            target_model=_model(_HUMAN),
            source_model=_model(_ARM),
            selection=AttachmentSelection(target_link="missing"),
        )


@pytest.fixture
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    return app


def test_frankenstein_editor_attach_action_exports_mjcf(qapp: QApplication) -> None:
    del qapp
    from src.tools.model_explorer.frankenstein_editor.editor import FrankensteinEditor

    editor = FrankensteinEditor()
    editor.left_panel.model = _model(_ARM)
    editor.right_panel.model = _model(_HUMAN)
    editor.left_panel._refresh_tree()
    editor.right_panel._refresh_tree()

    attached = editor.attach_source_model_to_working(
        AttachmentSelection(target_link="right_hand", source_prefix="arm_")
    )

    assert attached is True
    assert "arm_base" in editor.right_panel.model.links  # type: ignore[union-attr]
    exported = editor.export_working_model("mjcf")
    assert exported is not None
    assert "<mujoco" in exported
    assert "arm_base" in exported
