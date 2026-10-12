"""Native frame selectors use exact fit indices, never the whole capture range."""

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("indices", [tuple(range(191)), (10, 75, 190)])
def test_native_slider_maps_fit_positions_to_original_indices(
    tmp_path, monkeypatch, indices
):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PyQt6.QtWidgets")
    from PyQt6.QtWidgets import QApplication
    from src.shared.python.workspace import NecromatcherLibrary
    from src.tools.necromatcher.gui import NecromatcherWidget

    app = QApplication.instance() or QApplication([])
    widget = NecromatcherWidget(
        library=NecromatcherLibrary.create(tmp_path / "library")
    )
    calls = []
    monkeypatch.setattr(widget, "_request_projection", calls.append)
    review = SimpleNamespace(frame_count=210, close=lambda: None)
    try:
        widget._fit_id = "restricted-fit"
        widget._capture_loaded(review, widget._generation, indices)
        assert widget.slider.maximum() == len(indices) - 1
        assert calls == [indices[0]]
        widget.slider.setValue(len(indices) - 1)
        app.processEvents()
        assert calls[-1] == 190
        assert all(index in indices for index in calls)
        assert not {191, 209}.intersection(calls)
    finally:
        widget.cleanup()
        widget.close()


def test_native_capture_only_retains_full_capture_domain(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PyQt6.QtWidgets")
    from PyQt6.QtWidgets import QApplication
    from src.shared.python.workspace import NecromatcherLibrary
    from src.tools.necromatcher.gui import NecromatcherWidget

    app = QApplication.instance() or QApplication([])
    widget = NecromatcherWidget(
        library=NecromatcherLibrary.create(tmp_path / "library")
    )
    calls = []
    monkeypatch.setattr(widget, "_paint_frame", lambda index: calls.append(index))
    try:
        widget._capture_loaded(
            SimpleNamespace(frame_count=210, close=lambda: None), widget._generation
        )
        widget.slider.setValue(209)
        app.processEvents()
        assert widget.slider.maximum() == 209
        assert calls == [0, 209]
    finally:
        widget.cleanup()
        widget.close()
