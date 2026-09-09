"""Coaching references survive UI edits, view transforms and rendered exports."""

from pathlib import Path
from time import monotonic
import json

import cv2
import numpy as np
import pytest
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QMessageBox

from src.motion_capture.coaching import Drawing, DrawingLayer
from src.motion_capture.coaching.storage import load_layer
from src.motion_capture.rig.edits import CropRect, SessionEdits, ViewEdit, save_edits
from src.tools.capture_rig.coaching_canvas import CoachingCanvas
from src.tools.capture_rig.coaching_dialog import CoachingDialog
from src.tools.capture_rig.coaching_export import export_still
from src.tools.capture_rig.player import VideoReader
from src.tools.capture_rig.swing_export import export_swing
from src.tools.capture_rig.session import load_session
from tests.motion_capture.rig.test_ingest import _bundle
from tests.tools.capture_rig.test_pane_layout import _app

pytestmark = [pytest.mark.unit, pytest.mark.ui]


def test_keyboard_creation_numeric_edit_undo_and_reopen(tmp_path: Path) -> None:
    app = _app()
    root = _bundle(tmp_path)
    dialog = CoachingDialog(root, "a")
    dialog.show()
    app.processEvents()
    dialog.tool.setCurrentText("Ellipse")
    dialog.add_center()
    assert len(dialog.canvas.layer.shapes) == 1
    dialog.geometry_fields["x1"].setValue(10)
    dialog.first.setValue(2)
    dialog.last.setValue(4)
    dialog.apply_properties()
    shape = dialog.canvas.selection()
    assert shape is not None and shape.start[0] == 10 and not shape.at(0)
    QTest.keyClick(dialog.canvas, Qt.Key.Key_Right)
    assert dialog.canvas.selection().start[0] == 11
    dialog.undo_button.click()
    assert dialog.canvas.selection().start[0] == 10
    assert dialog.minimumSizeHint().width() <= 900
    assert dialog.save()
    dialog.close()
    reopened = CoachingDialog(root, "a")
    assert reopened.canvas.layer.shapes[0] == shape
    reopened.close()


@pytest.mark.parametrize(
    "size,zoom", [((640, 240), 1), ((320, 500), 1), ((640, 480), 2)]
)
def test_drawing_move_and_resize_use_source_pixels(
    size: tuple[int, int], zoom: int
) -> None:
    app = _app()
    canvas = CoachingCanvas(DrawingLayer(view="a", width=100, height=80, frames=5))
    canvas.resize(*size)
    canvas.set_frame(np.zeros((80, 100, 3), dtype=np.uint8), 0)
    canvas.show()
    canvas.set_zoom(zoom)
    app.processEvents()

    def point(x: float, y: float) -> QPoint:
        scale = min(canvas.width() / 100, canvas.height() / 80) * zoom
        return QPoint(
            round((canvas.width() - 100 * scale) / 2 + (x + 0.5) * scale),
            round((canvas.height() - 80 * scale) / 2 + (y + 0.5) * scale),
        )

    canvas.tool = "rectangle"
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=point(30, 25))
    QTest.mouseMove(canvas, point(65, 55))
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=point(65, 55))
    assert len(canvas.layer.shapes) == 1
    shape = canvas.layer.shapes[0]
    assert shape.start == pytest.approx((30, 25), abs=1)
    assert shape.end == pytest.approx((65, 55), abs=1)
    canvas.tool = "select"
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=point(48, 25))
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=point(50, 30))
    assert canvas.layer.shapes[0].start == pytest.approx((32, 30), abs=1)
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=point(67, 60))
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=point(72, 62))
    assert canvas.layer.shapes[0].end == pytest.approx((72, 62), abs=1)
    canvas.undo()
    canvas.undo()
    assert canvas.layer.shapes[0] == shape
    canvas.close()


def test_cancel_unsaved_close_and_corrupt_layer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _app()
    root = _bundle(tmp_path)
    dialog = CoachingDialog(root, "a")
    dialog.show()
    app.processEvents()
    dialog.tool.setCurrentText("Line")
    dialog.add_center()
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Cancel
    )
    dialog.close()
    assert dialog.isVisible()
    QTest.keyClick(dialog.tool, Qt.Key.Key_Escape)
    assert dialog.isVisible()
    assert dialog.save()
    dialog.close()
    assert load_layer(root, "a", 64, 48, 6).shapes
    with pytest.raises(ValueError):
        load_layer(root, "a", 128, 96, 6)


def test_still_and_trimmed_video_draw_before_crop(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    source = load_session(root).view("a").recording
    assert source is not None
    original = source.read_bytes()
    shape = Drawing(
        kind="line",
        start=(10, 20),
        end=(50, 20),
        colour="#ff0000",
        stroke=4,
        first=1,
        last=3,
    )
    layer = DrawingLayer(view="a", width=64, height=48, frames=6, shapes=(shape,))
    save_edits(
        root,
        SessionEdits(
            views={
                "a": ViewEdit(
                    first=1, last=3, crop=CropRect(x=8, y=6, width=32, height=24)
                )
            }
        ),
    )
    still = tmp_path / "reference.png"
    export_still(root, layer, 2, still)
    image = cv2.imread(str(still))
    assert image.shape == (24, 32, 3)
    assert image[14, 20, 2] > 230 and image[14, 20, 0] < 30
    output = tmp_path / "coaching.avi"
    result = export_swing(root, "a", output, drawings=layer)
    assert result["drawings"] == layer.model_dump(mode="json")
    with VideoReader(output) as reader:
        image = reader.read(1)
        assert image is not None and image.shape == (24, 32, 3)
        assert image[14, 20, 2] > image[14, 20, 0] + 100
    assert source.read_bytes() == original


def test_dialog_exports_a_snapshot_in_the_existing_worker(tmp_path: Path) -> None:
    app = _app()
    root = _bundle(tmp_path)
    dialog = CoachingDialog(root, "a")
    dialog.tool.setCurrentText("Arrow")
    dialog.add_center()
    output = tmp_path / "annotated.avi"
    snapshot = dialog.canvas.layer.model_dump(mode="json")
    dialog.exporter.start(output)
    assert dialog.exporter.busy
    dialog.canvas.clear()
    deadline = monotonic() + 15
    while dialog.exporter.busy and monotonic() < deadline:
        app.processEvents()
    assert not dialog.exporter.busy
    metadata = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["drawings"] == snapshot
    assert dialog.save()
    dialog.close()


def test_sidecar_publication_failure_removes_only_our_media(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os
    from src.tools.capture_rig.swing_export import publish_export

    staged = tmp_path / "staged.png"
    staged.write_bytes(b"new media")
    staged.with_suffix(".json").write_bytes(b"new metadata")
    output = tmp_path / "destination.png"
    sidecar = output.with_suffix(".json")
    link = os.link

    def concurrent_sidecar(source: Path, target: Path) -> None:
        if Path(target) == sidecar:
            sidecar.write_bytes(b"other exporter")
        link(source, target)

    monkeypatch.setattr(os, "link", concurrent_sidecar)
    with pytest.raises(FileExistsError):
        publish_export(staged, output)
    assert not output.exists()
    assert sidecar.read_bytes() == b"other exporter"
    assert staged.read_bytes() == b"new media"
