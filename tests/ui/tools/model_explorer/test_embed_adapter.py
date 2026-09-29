"""Tests for the Model Explorer embeddable-tool adapter.

Verifies the adapter satisfies the
:class:`~src.shared.python.launcher_embed.EmbeddableTool` protocol,
exposes the capabilities documented in Subtask 5 / #4998, hands out
real :class:`QWidget` instances from :meth:`create_main_widget`, and
registers itself with the embeddable-tool registry on import.
"""

from __future__ import annotations

import os

import pytest

from src.shared.python.engine_core.engine_availability import (
    skip_if_unavailable,
)

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


# --- Protocol conformance -----------------------------------------------


def test_adapter_satisfies_embeddable_tool_protocol() -> None:
    from src.shared.python.launcher_embed import EmbeddableTool
    from src.tools.model_explorer._embed_adapter import (
        _ModelExplorerEmbedAdapter,
    )

    adapter = _ModelExplorerEmbedAdapter()
    assert isinstance(adapter, EmbeddableTool)


def test_adapter_tool_id_is_model_explorer() -> None:
    from src.tools.model_explorer._embed_adapter import (
        _ModelExplorerEmbedAdapter,
    )

    adapter = _ModelExplorerEmbedAdapter()
    assert adapter.tool_id == "model_explorer"


# --- Capabilities values -------------------------------------------------


def test_embed_capabilities_match_spec() -> None:
    from src.shared.python.launcher_embed import EmbedCapabilities
    from src.tools.model_explorer._embed_adapter import (
        _ModelExplorerEmbedAdapter,
    )

    caps = _ModelExplorerEmbedAdapter().embed_capabilities()
    assert isinstance(caps, EmbedCapabilities)
    assert caps.supports_embedded is True
    assert caps.prefers_dock is False
    assert caps.min_size == (700, 500)
    assert caps.requires_separate_qapplication is False


def test_is_dirty_default_is_false() -> None:
    from src.tools.model_explorer._embed_adapter import (
        _ModelExplorerEmbedAdapter,
    )

    assert _ModelExplorerEmbedAdapter().is_dirty() is False


def test_cleanup_is_idempotent() -> None:
    from src.tools.model_explorer._embed_adapter import (
        _ModelExplorerEmbedAdapter,
    )

    adapter = _ModelExplorerEmbedAdapter()
    # Cleanup is allowed before any widget has been handed out.
    adapter.cleanup()
    adapter.cleanup()


# --- create_main_widget returns a real QWidget --------------------------


def test_create_main_widget_returns_qwidget(qapp) -> None:  # noqa: ANN001
    from PyQt6.QtWidgets import QWidget

    from src.tools.model_explorer._embed_adapter import (
        _ModelExplorerEmbedAdapter,
    )

    adapter = _ModelExplorerEmbedAdapter()
    widget = adapter.create_main_widget(None)
    try:
        assert isinstance(widget, QWidget)
    finally:
        adapter.cleanup()
        widget.deleteLater()


def test_create_main_widget_accepts_parent(qapp) -> None:  # noqa: ANN001
    from PyQt6.QtWidgets import QWidget

    from src.tools.model_explorer._embed_adapter import (
        _ModelExplorerEmbedAdapter,
    )

    parent = QWidget()
    adapter = _ModelExplorerEmbedAdapter()
    try:
        widget = adapter.create_main_widget(parent)
        assert widget.parent() is parent
    finally:
        adapter.cleanup()
        parent.deleteLater()


def test_is_dirty_reflects_widget_state(qapp) -> None:  # noqa: ANN001
    from src.tools.model_explorer._embed_adapter import (
        _ModelExplorerEmbedAdapter,
    )

    adapter = _ModelExplorerEmbedAdapter()
    widget = adapter.create_main_widget(None)
    try:
        # Fresh widget: no segments, no path → clean.
        assert adapter.is_dirty() is False
        # Simulate an unsaved edit by adding a segment directly to the
        # builder. ``add_segment`` only requires a non-empty ``name``;
        # everything else has sensible defaults.
        widget.urdf_builder.add_segment({"name": "stub"})
        assert adapter.is_dirty() is True
    finally:
        adapter.cleanup()
        widget.deleteLater()


# --- Registry side-effect on import -------------------------------------


def test_import_registers_model_explorer_in_registry() -> None:
    """Importing :mod:`src.tools.model_explorer` registers the adapter."""
    from src.shared.python.launcher_embed import (
        EMBEDDABLE_TOOL_REGISTRY,
        get_embeddable_tool,
    )

    # The tools/__init__.py side-effect runs at first import; subsequent
    # imports are a no-op thanks to the ``get_embeddable_tool`` guard in
    # :mod:`src.tools.model_explorer`.
    import src.tools.model_explorer  # noqa: F401

    assert "model_explorer" in EMBEDDABLE_TOOL_REGISTRY
    tool = get_embeddable_tool("model_explorer")
    assert tool is not None
    assert tool.tool_id == "model_explorer"
    assert tool.embed_capabilities().supports_embedded is True
