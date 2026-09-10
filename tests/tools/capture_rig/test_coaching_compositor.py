"""The shared drawing canvas preserves comparison overlay order."""

import numpy as np
import pytest
from pathlib import Path

from src.motion_capture.coaching import Drawing, DrawingLayer, render_layer
from src.tools.capture_rig.coaching_canvas import CoachingCanvas
from tests.tools.capture_rig.test_pane_layout import _app
from src.tools.capture_rig.coaching_dialog import CoachingDialog
from tests.motion_capture.rig.test_ingest import _bundle

pytestmark = [pytest.mark.unit, pytest.mark.ui]


def test_overlay_runs_after_drawings_and_before_selection_handles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _app()
    layer = DrawingLayer(view="a", width=100, height=80, frames=3).with_shapes(
        (Drawing(kind="line", start=(10, 40), end=(90, 40)),)
    )
    source = np.zeros((80, 100, 3), dtype=np.uint8)
    observed = []
    displayed = []

    def finalize(image: np.ndarray, index: int) -> np.ndarray:
        observed.append((image.copy(), index))
        return np.full_like(image, 17)

    canvas = CoachingCanvas(layer, finalize=finalize)
    monkeypatch.setattr(canvas, "set_image", lambda image: displayed.append(image))
    canvas.set_frame(source, 2)
    assert observed[-1][1] == 2
    np.testing.assert_array_equal(observed[-1][0], render_layer(source, layer, 2))
    assert np.all(displayed[-1] == 17)
    canvas.select(layer.shapes[0].id)
    assert np.any(displayed[-1] != 17)
    assert not np.any(source)
    canvas.close()


def test_overlay_must_preserve_source_pixel_grid_and_type() -> None:
    _app()
    layer = DrawingLayer(view="a", width=100, height=80, frames=1)
    for invalid in (np.zeros((80, 99, 3), np.uint8), np.zeros((80, 100, 3), float)):
        canvas = CoachingCanvas(
            layer, finalize=lambda image, index, result=invalid: result
        )
        with pytest.raises(ValueError, match="original BGR pixel grid"):
            canvas.set_frame(np.zeros((80, 100, 3), np.uint8), 0)
        canvas.close()


def test_common_dialog_passes_overlay_to_the_drawing_canvas(tmp_path: Path) -> None:
    _app()
    indices = []

    def overlay(image: np.ndarray, index: int) -> np.ndarray:
        indices.append(index)
        return image

    dialog = CoachingDialog(_bundle(tmp_path), "a", finalize=overlay)
    dialog.slider.setValue(3)
    assert indices[-1] == 3
    dialog.close()
