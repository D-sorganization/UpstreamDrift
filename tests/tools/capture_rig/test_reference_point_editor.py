"""Physical point identities survive editing, undo and cancellation."""

import numpy as np
import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QDialogButtonBox

from src.tools.capture_rig.reference_calibration.point_editor import (
    ReferencePointEditor,
)

pytestmark = [pytest.mark.unit, pytest.mark.ui]


def test_marking_advances_by_physical_identity_and_supports_undo(qtbot) -> None:
    dialog = ReferencePointEditor(
        np.zeros((480, 640, 3), dtype=np.uint8),
        ("origin", "along-arrow", "opposite", "across-width"),
        context="Front Camera · Placement Near Ball · Frame 12",
    )
    qtbot.addWidget(dialog)
    dialog.canvas.clicked.emit(90.0, 110.0)
    assert dialog.points == {"origin": (90.0, 110.0)}
    assert dialog.point.currentData() == "along-arrow"
    dialog.canvas.clicked.emit(180.0, 130.0)
    assert dialog.points["along-arrow"] == (180.0, 130.0)
    dialog.undo_stack.undo()
    assert "along-arrow" not in dialog.points
    assert "1 of 4" in dialog.status.text()
    dialog.undo_stack.redo()
    assert dialog.points["along-arrow"] == (180.0, 130.0)


def test_keyboard_coordinates_and_cancel_preserve_the_original_points(qtbot) -> None:
    original = {"origin": (10.0, 20.0)}
    dialog = ReferencePointEditor(
        np.zeros((240, 320, 3), dtype=np.uint8),
        ("origin", "endpoint"),
        context="Yardstick",
        points=original,
    )
    qtbot.addWidget(dialog)
    dialog.point.setCurrentIndex(1)
    dialog.pixel_x.setValue(200.0)
    dialog.pixel_y.setValue(100.0)
    dialog.place.click()
    assert dialog.points["endpoint"] == (200.0, 100.0)
    dialog.reject()
    assert original == {"origin": (10.0, 20.0)}
    assert dialog.result() == 0


def test_empty_and_partial_observations_have_clear_save_state(qtbot) -> None:
    dialog = ReferencePointEditor(
        np.zeros((480, 640, 3), dtype=np.uint8),
        ("origin", "along-arrow", "opposite", "across-width"),
        context="Paper",
    )
    qtbot.addWidget(dialog)
    save = dialog.buttons.button(QDialogButtonBox.StandardButton.Save)
    assert not save.isEnabled()
    dialog.canvas.clicked.emit(10.0, 20.0)
    assert save.isEnabled()
    assert "Partial" in dialog.status.text()
    assert "not camera calibration" in dialog.instructions.text()
