"""Offscreen pytest-qt tests for drag-and-drop assembly (CMB-9, #11660)."""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.unit

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    pytest.importorskip("PyQt6.QtWidgets")
    pytest.importorskip("pytestqt")
except (ImportError, OSError) as exc:  # pragma: no cover - env dependent
    pytest.skip(f"PyQt6/pytest-qt not loadable: {exc}", allow_module_level=True)

from PyQt6.QtCore import QByteArray, QMimeData, QPointF, Qt  # noqa: E402
from PyQt6.QtGui import QDropEvent  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.tools.model_explorer.frankenstein_editor.assembly_canvas import (  # noqa: E402
    PART_MIME,
)
from src.tools.model_explorer.frankenstein_editor.assembly_panel import (  # noqa: E402
    AssemblyPanel,
)


def _drop(panel: AssemblyPanel, part_id: str, port: str) -> None:
    """Deliver a real QDropEvent to the canvas viewport over ``port``."""
    canvas = panel.canvas
    mime = QMimeData()
    mime.setData(PART_MIME, QByteArray(part_id.encode()))
    pos = QPointF(canvas.port_viewport_pos(port))
    from PyQt6.QtGui import QDragEnterEvent

    enter = QDragEnterEvent(
        pos.toPoint(),
        Qt.DropAction.CopyAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(canvas.viewport(), enter)
    drop = QDropEvent(
        pos,
        Qt.DropAction.CopyAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(canvas.viewport(), drop)


@pytest.fixture()
def panel(qtbot) -> AssemblyPanel:
    widget = AssemblyPanel()
    qtbot.addWidget(widget)
    widget.resize(1000, 600)
    widget.show()
    return widget


def test_library_lists_parts_by_category_and_searches(panel: AssemblyPanel) -> None:
    tree = panel.library
    labels = [tree.topLevelItem(i).text(0) for i in range(tree.topLevelItemCount())]
    assert {"Golf Clubs", "Limbs", "Heads", "Shoes", "Robot Arms"} <= set(labels)
    panel.search.setText("iron")
    assert tree.topLevelItemCount() == 1
    assert tree.topLevelItem(0).child(0).text(0) == "7 Iron"


def test_valid_drop_attaches_part(panel: AssemblyPanel) -> None:
    before = len(panel.session.model.links)
    _drop(panel, "leg_left", "hip_left")
    assert len(panel.session.model.links) == before + 3
    assert "hip_left" in panel.session.occupied_ports()
    rows = [
        panel.validation_list.item(i).text()
        for i in range(panel.validation_list.count())
    ]
    assert not any(row.startswith("ERROR") for row in rows)
    assert panel.undo_btn.isEnabled()


def test_invalid_drop_is_rejected_with_a_reason(panel: AssemblyPanel) -> None:
    before = panel.session.to_urdf()
    _drop(panel, "head", "hip_left")
    assert panel.session.to_urdf() == before
    assert panel.status_label.text().startswith("Rejected:")
    assert "type mismatch" in panel.status_label.text()
    assert not panel.undo_btn.isEnabled()


def test_dropping_off_a_socket_is_rejected(panel: AssemblyPanel, qtbot) -> None:
    messages: list[str] = []
    panel.canvas.drop_rejected.connect(messages.append)
    mime = QMimeData()
    mime.setData(PART_MIME, QByteArray(b"head"))
    pos = QPointF(2.0, 2.0)
    from PyQt6.QtGui import QDragEnterEvent

    QApplication.sendEvent(
        panel.canvas.viewport(),
        QDragEnterEvent(
            pos.toPoint(),
            Qt.DropAction.CopyAction,
            mime,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        ),
    )
    QApplication.sendEvent(
        panel.canvas.viewport(),
        QDropEvent(
            pos,
            Qt.DropAction.CopyAction,
            mime,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        ),
    )
    assert messages == ["Drop the part on a socket"]


def test_undo_and_redo_buttons(panel: AssemblyPanel) -> None:
    initial = panel.session.to_urdf()
    _drop(panel, "head", "neck")
    dropped = panel.session.to_urdf()
    panel.undo_btn.click()
    assert panel.session.to_urdf() == initial
    assert panel.redo_btn.isEnabled() and not panel.undo_btn.isEnabled()
    panel.redo_btn.click()
    assert panel.session.to_urdf() == dropped


def test_detach_selected_part(panel: AssemblyPanel, qtbot) -> None:
    _drop(panel, "head", "neck")
    instance = panel.session.placed_parts[-1].instance_id
    panel.canvas._selected = instance
    panel.detach_btn.click()
    assert [p.part_id for p in panel.session.placed_parts] == ["humanoid_torso"]
    panel.detach_btn.click()
    assert "Click a part" in panel.status_label.text()


def test_drop_emits_urdf_for_the_3d_preview(panel: AssemblyPanel, qtbot) -> None:
    with qtbot.waitSignal(panel.assembly_changed, timeout=1000) as blocker:
        _drop(panel, "head", "neck")
    assert 'name="head_1__head"' in blocker.args[0]


def test_new_assembly_switches_base(panel: AssemblyPanel) -> None:
    panel.base_combo.setCurrentIndex(1)
    panel.new_btn.click()
    assert panel.session.placed_parts[0].part_id == "pedestal"
    assert panel.canvas.port_names() == ("top_mount",)


def test_canvas_screenshot_renders(panel: AssemblyPanel, tmp_path) -> None:
    _drop(panel, "leg_left", "hip_left")
    image = panel.grab()
    assert not image.isNull() and image.width() > 400
    assert image.save(str(tmp_path / "assembly.png"))


def test_drop_refreshes_the_editor_window_3d_preview(qtbot) -> None:
    """A drop in the assembly tab must reach the 3D preview (#8259 stays green)."""
    from src.tools.model_explorer.urdf_editor_window import URDFEditorWindow

    window = URDFEditorWindow()
    qtbot.addWidget(window)
    window.show()
    assembly = window.frankenstein.assembly_panel
    window.frankenstein.mode_tabs.setCurrentWidget(assembly)
    _drop(assembly, "head", "neck")
    assert "head_1__head" in window.visualization.urdf_content
    assert window.urdf_content == window.visualization.urdf_content
    assert "head_1__head" in window.code_editor.get_content()
    window._is_modified = False  # avoid the modal "save changes?" dialog on close
