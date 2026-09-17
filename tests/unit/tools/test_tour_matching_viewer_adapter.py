"""Tests for Tour Matching Viewer embed adapter (Step 3)."""

from __future__ import annotations

import os
import pytest

from src.shared.python.engine_core.engine_availability import skip_if_unavailable
from src.shared.python.launcher_embed import EmbedCapabilities, EmbeddableTool

pytestmark = [
    skip_if_unavailable("pyqt6"),
    pytest.mark.unit,
]

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():  # noqa: ANN201
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def test_adapter_satisfies_embeddable_tool_protocol() -> None:
    from src.tools.tour_matching_viewer._embed_adapter import (
        _TourMatchingViewerEmbedAdapter,
    )

    adapter = _TourMatchingViewerEmbedAdapter()
    assert isinstance(adapter, EmbeddableTool)
    assert adapter.tool_id == "tour_matching_viewer"


def test_adapter_embed_capabilities() -> None:
    from src.tools.tour_matching_viewer._embed_adapter import (
        _TourMatchingViewerEmbedAdapter,
    )

    adapter = _TourMatchingViewerEmbedAdapter()
    caps = adapter.embed_capabilities()
    assert isinstance(caps, EmbedCapabilities)
    assert caps.supports_embedded is True
    assert caps.prefers_dock is False
    assert caps.min_size == (900, 650)
    assert caps.requires_separate_qapplication is False


def test_adapter_lifecycle(qapp) -> None:  # noqa: ANN001
    from PyQt6.QtWidgets import QWidget
    from src.tools.tour_matching_viewer._embed_adapter import (
        _TourMatchingViewerEmbedAdapter,
    )

    adapter = _TourMatchingViewerEmbedAdapter()
    assert adapter.is_dirty() is False

    # Cleanup before handing out widgets is safe
    adapter.cleanup()

    parent = QWidget()
    widget = adapter.create_main_widget(parent)
    try:
        assert isinstance(widget, QWidget)
        assert widget.parent() is parent
    finally:
        adapter.cleanup()
        parent.deleteLater()


def test_import_registers_adapter_in_registry() -> None:
    from src.shared.python.launcher_embed import (
        EMBEDDABLE_TOOL_REGISTRY,
        get_embeddable_tool,
    )
    import src.tools.tour_matching_viewer  # noqa: F401

    assert "tour_matching_viewer" in EMBEDDABLE_TOOL_REGISTRY
    tool = get_embeddable_tool("tour_matching_viewer")
    assert tool is not None
    assert tool.tool_id == "tour_matching_viewer"


def test_verified_simscape_bundle_uses_retained_model_and_status(
    qapp, tmp_path
) -> None:  # noqa: ANN001
    from pathlib import Path
    from src.tools.tour_matching_viewer.gui import TourMatchingViewerWidget

    root = Path(__file__).resolve().parents[3]
    manifest = (
        root
        / "docs/development/simscape_tour_matching/native_evidence/simscape_returned102.replay.json"
    )
    widget = TourMatchingViewerWidget()
    try:
        widget.load_file(manifest)
        assert widget._engine_name == "simscape"
        assert len(widget._spec["coordinate_order"]) == 27
        assert widget._replay.frame_count == 307
        assert "rejected" in widget._title_label.text().lower()
        assert "unavailable" in widget._title_label.text().lower()
        widget._slider.setValue(306)
        assert "0.850 s" in widget._frame_label.text()
        before = widget._replay
        invalid = tmp_path / "invalid.json"
        invalid.write_text('{"schema_version": "bad"}')
        with pytest.raises(ValueError):
            widget.load_file(invalid)
        assert widget._replay is before
    finally:
        widget.cleanup()
        widget.close()
