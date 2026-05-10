"""Headless tests for the Data Explorer embeddable-tool adapter.

Run with ``QT_QPA_PLATFORM=offscreen``. PyQt6 is gated via
``importorskip`` so the test module is harmless on Qt-less runners.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QWidget  # noqa: E402

from src.shared.python.launcher_embed import (  # noqa: E402
    EMBEDDABLE_TOOL_REGISTRY,
    EmbedCapabilities,
    EmbeddableTool,
)
from src.tools.data_explorer._embed_adapter import (  # noqa: E402
    _DataExplorerEmbedAdapter,
)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def _preserve_registry():
    """Snapshot and restore the embeddable-tool registry per test.

    The package-level import in ``src.tools.data_explorer`` already
    populates the registry, but individual tests register / unregister
    fresh instances and we want a clean slate around each one.
    """
    snapshot = dict(EMBEDDABLE_TOOL_REGISTRY)
    EMBEDDABLE_TOOL_REGISTRY.clear()
    try:
        yield
    finally:
        EMBEDDABLE_TOOL_REGISTRY.clear()
        EMBEDDABLE_TOOL_REGISTRY.update(snapshot)


@pytest.fixture
def adapter() -> _DataExplorerEmbedAdapter:
    return _DataExplorerEmbedAdapter()


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


def test_adapter_satisfies_embeddable_tool_protocol(
    adapter: _DataExplorerEmbedAdapter,
) -> None:
    """The adapter must satisfy :class:`EmbeddableTool` at runtime."""
    assert isinstance(adapter, EmbeddableTool)


def test_tool_id_is_data_explorer(adapter: _DataExplorerEmbedAdapter) -> None:
    assert adapter.tool_id == "data_explorer"


def test_embed_capabilities_returns_expected_values(
    adapter: _DataExplorerEmbedAdapter,
) -> None:
    caps = adapter.embed_capabilities()
    assert isinstance(caps, EmbedCapabilities)
    assert caps.supports_embedded is True
    # Tabs work better than docks for table-heavy UIs.
    assert caps.prefers_dock is False
    assert caps.min_size == (640, 480)
    assert caps.requires_separate_qapplication is False


def test_create_main_widget_returns_qwidget(
    qapp,  # noqa: ANN001 - injected from tests/launchers/conftest.py
    adapter: _DataExplorerEmbedAdapter,
) -> None:
    widget = adapter.create_main_widget(None)
    try:
        assert isinstance(widget, QWidget)
    finally:
        widget.deleteLater()


def test_is_dirty_returns_false(adapter: _DataExplorerEmbedAdapter) -> None:
    assert adapter.is_dirty() is False


def test_cleanup_is_idempotent(
    qapp,  # noqa: ANN001
    adapter: _DataExplorerEmbedAdapter,
) -> None:
    """Cleanup must tolerate being called repeatedly."""
    adapter.create_main_widget(None)
    adapter.cleanup()
    adapter.cleanup()  # second call must not raise


def test_importing_package_registers_adapter(_preserve_registry) -> None:
    """Importing :mod:`src.tools.data_explorer` registers the adapter."""
    # The package may already be imported earlier in the test session;
    # clear the registry (done by the fixture) and re-run the
    # registration logic by re-executing the package init's body.
    import importlib

    import src.tools.data_explorer as data_explorer_pkg

    importlib.reload(data_explorer_pkg)

    assert "data_explorer" in EMBEDDABLE_TOOL_REGISTRY
    registered = EMBEDDABLE_TOOL_REGISTRY["data_explorer"]
    assert registered.tool_id == "data_explorer"
    assert isinstance(registered, EmbeddableTool)
