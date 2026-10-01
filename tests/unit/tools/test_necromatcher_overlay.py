"""Native overlays preserve the original image and image-space coordinates."""

import pytest

pytestmark = pytest.mark.unit


def test_native_overlay_is_detached_and_handles_missing_landmarks(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PyQt6.QtWidgets")
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtGui import QColor, QPixmap
    from src.tools.necromatcher.image_review import landmark_overlay

    app = QApplication.instance() or QApplication([])
    original = QPixmap(100, 100)
    original.fill(QColor("black"))
    marked = landmark_overlay(
        original, {"hip": {"x": 0.5, "y": 0.5, "visibility": None}}
    )
    assert marked.toImage().pixelColor(50, 50) != QColor("black")
    assert original.toImage().pixelColor(50, 50) == QColor("black")
    missing = landmark_overlay(original, {})
    assert missing.toImage() == original.toImage()
    app.processEvents()
