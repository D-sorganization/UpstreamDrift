"""Headless composition UX flow tests for the model explorer."""

from __future__ import annotations

import xml.etree.ElementTree as ET
import os
import sys
from pathlib import Path

import pytest
from PyQt6.QtWidgets import QApplication

from src.tools.model_explorer.composition_flow import (
    AttachmentSelection,
    CompositionFlowController,
    CompositionFlowError,
)
from src.tools.model_explorer.composition_ux import (
    CompositionDragPayload,
    CompositionUxController,
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


def test_composition_ux_preview_is_non_mutating_and_highlights_ghost() -> None:
    target = _model(_HUMAN)
    source = _model(_ARM)
    payload = CompositionDragPayload(
        category="robotic",
        key="simple_arm",
        name="Simple Arm",
        format_badge="URDF",
        source_prefix="simple_arm_",
    )

    preview = CompositionUxController().preview_drop(
        payload=payload,
        target_model=target,
        source_model=source,
        selection=AttachmentSelection(
            target_link="right_hand",
            source_prefix=payload.source_prefix,
        ),
    )

    assert preview.state == "ready"
    assert preview.source_root_link == "base"
    assert preview.mapped_root_link == "simple_arm_base"
    assert preview.highlighted_links == ("right_hand", "simple_arm_base")
    assert preview.validation is not None
    assert preview.validation.ok
    assert "simple_arm_base" not in target.links
    assert "base" in source.links


def test_composition_ux_commit_updates_target_and_exports_choices() -> None:
    target = _model(_HUMAN)
    source = _model(_ARM)
    payload = CompositionDragPayload(
        category="robotic",
        key="simple_arm",
        name="Simple Arm",
        format_badge="URDF",
        source_prefix="simple_arm_",
    )
    controller = CompositionUxController()

    committed = controller.commit_drop(
        payload=payload,
        target_model=target,
        source_model=source,
        selection=AttachmentSelection(
            target_link="right_hand",
            source_prefix=payload.source_prefix,
        ),
    )

    assert committed.result.validation.ok
    assert "simple_arm_base" in target.links
    choices = {choice.format: choice for choice in controller.export_choices(target)}
    assert choices["urdf"].enabled is True
    assert choices["mjcf"].enabled is True
    assert choices["sdf"].enabled is False
    assert choices["osim"].enabled is False
    assert choices["sdf"].reason == "writer not available"


@pytest.fixture
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    return app


def test_model_panel_facade_manages_selection_and_dirty_state(
    qapp: QApplication,
) -> None:
    del qapp
    from src.tools.model_explorer.frankenstein_editor.panel import ModelPanel

    panel = ModelPanel("Working Model")
    panel.set_model(_model(_ARM), dirty=False)
    model = panel.get_model()

    assert model is not None
    assert model.is_modified is False

    links_item = panel.tree.topLevelItem(0)
    assert links_item is not None
    base_item = links_item.child(0)
    panel.tree.setCurrentItem(base_item)

    selection = panel.selected_component()

    assert selection is not None
    assert selection.comp_type == "link"
    assert selection.name == "base"
    assert selection.element.tag == "link"
    assert panel.selected_link_name() == "base"

    panel.set_dirty()
    assert model.is_modified is True

    panel.mark_clean()
    assert model.is_modified is False

    panel.set_model(None)
    assert panel.get_model() is None
    assert panel.selected_component() is None


def test_frankenstein_editor_uses_model_panel_facade_for_panel_state() -> None:
    editor_source = Path(
        "src/tools/model_explorer/frankenstein_editor/editor.py"
    ).read_text(encoding="utf-8")

    forbidden_panel_internals = (
        "left_panel.tree",
        "right_panel.tree",
        "._refresh_tree(",
        ".save_btn",
        ".file_label",
        "left_panel.model =",
        "right_panel.model =",
    )

    assert [
        pattern for pattern in forbidden_panel_internals if pattern in editor_source
    ] == []


def test_frankenstein_editor_attach_action_exports_mjcf(qapp: QApplication) -> None:
    del qapp
    from src.tools.model_explorer.frankenstein_editor.editor import FrankensteinEditor

    editor = FrankensteinEditor()
    editor.left_panel.set_model(_model(_ARM))
    editor.right_panel.set_model(_model(_HUMAN))

    preview = editor.preview_source_model_attachment(
        AttachmentSelection(target_link="right_hand", source_prefix="arm_")
    )
    assert preview is not None
    assert preview.state == "ready"
    working_model = editor.right_panel.get_model()
    assert working_model is not None
    assert "arm_base" not in working_model.links

    attached = editor.attach_source_model_to_working(
        AttachmentSelection(target_link="right_hand", source_prefix="arm_")
    )

    assert attached is True
    assert "arm_base" in working_model.links
    choices = {choice.format: choice for choice in editor.export_working_choices()}
    assert choices["urdf"].enabled is True
    assert choices["mjcf"].enabled is True
    exported = editor.export_working_model("mjcf")
    assert exported is not None
    assert "<mujoco" in exported
    assert "arm_base" in exported
