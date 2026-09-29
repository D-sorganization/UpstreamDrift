"""Pan gestures preserve original pixels and cannot create calibration marks."""

import numpy as np
import pytest
from PyQt6.QtCore import QPoint, Qt

from src.tools.capture_rig.annotate_widget import ImageCanvas
from src.tools.capture_rig.reference_calibration.point_editor import (
    ReferencePointEditor,
)

pytestmark = [pytest.mark.unit, pytest.mark.ui]


def test_panning_reaches_image_edges_and_fit_restores_center(qtbot):
    canvas = ImageCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(400, 300)
    image = np.zeros((300, 400, 3), dtype=np.uint8)
    image[:30, :30] = (20, 80, 220)
    canvas.set_image(image)
    canvas.show()
    canvas.set_zoom(4)
    canvas.pan_by(10000, 10000)
    assert canvas.image_point_from_widget(QPoint(20, 20)) == pytest.approx(
        (4.625, 4.625)
    )
    pixel = canvas.grab().toImage().pixelColor(20, 20)
    assert (pixel.red(), pixel.green(), pixel.blue()) == (220, 80, 20)
    canvas.set_image(image.copy())
    assert canvas.image_point_from_widget(QPoint(20, 20)) == pytest.approx(
        (4.625, 4.625)
    )
    canvas.set_zoom(1)
    assert canvas.image_point_from_widget(QPoint(20, 20)) == pytest.approx((20, 20))


def test_pan_drag_does_not_mark_a_point_and_mark_mode_still_uses_source_pixels(qtbot):
    editor = ReferencePointEditor(
        np.zeros((300, 400, 3), dtype=np.uint8),
        ("origin", "along-arrow"),
        context="Practice · Front",
    )
    qtbot.addWidget(editor)
    editor.show()
    editor.canvas.set_zoom(4)
    before = editor.canvas.image_point_from_widget(QPoint(240, 150))
    editor.pan.setChecked(True)
    qtbot.mousePress(editor.canvas, Qt.MouseButton.LeftButton, pos=QPoint(160, 100))
    qtbot.mouseMove(editor.canvas, QPoint(240, 150))
    qtbot.mouseRelease(editor.canvas, Qt.MouseButton.LeftButton, pos=QPoint(240, 150))
    assert editor.points == {}
    assert editor.canvas.image_point_from_widget(QPoint(240, 150)) != before
    editor.pan.setChecked(False)
    point = editor.canvas.image_point_from_widget(QPoint(240, 150))
    qtbot.mouseClick(editor.canvas, Qt.MouseButton.LeftButton, pos=QPoint(240, 150))
    assert editor.points["origin"] == pytest.approx(point)
    editor.fit.click()
    assert editor.canvas.zoom == 1
    assert not editor.pan.isChecked()
