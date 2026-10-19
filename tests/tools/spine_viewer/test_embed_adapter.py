"""Embed-adapter tests for the Spine Viewer (SPN-15b, #12772)."""

from __future__ import annotations

import importlib
import os
import sys

import pytest

from src.shared.python.launcher_embed import (
    EMBEDDABLE_TOOL_REGISTRY,
    EmbeddableTool,
)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = pytest.mark.unit

_ADAPTER_MODULE = "src.tools.spine_viewer._embed_adapter"


def test_importing_adapter_registers_without_loading_pyqt6(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    had_qt = "PyQt6" in sys.modules
    import src.tools.spine_viewer as spine_pkg

    # Import the adapter fresh: if an earlier test imported it inside a
    # registry snapshot, the cached module would make this a no-op (#13314).
    monkeypatch.delitem(sys.modules, _ADAPTER_MODULE, raising=False)
    if hasattr(spine_pkg, "_embed_adapter"):
        monkeypatch.delattr(spine_pkg, "_embed_adapter")
    monkeypatch.delitem(EMBEDDABLE_TOOL_REGISTRY, "spine_viewer", raising=False)
    _embed_adapter = importlib.import_module(_ADAPTER_MODULE)

    assert "spine_viewer" in EMBEDDABLE_TOOL_REGISTRY
    assert isinstance(EMBEDDABLE_TOOL_REGISTRY["spine_viewer"], EmbeddableTool)
    assert hasattr(_embed_adapter, "SpineViewerEmbedAdapter")
    if not had_qt:
        assert "PyQt6" not in sys.modules
        assert "src.tools.spine_viewer.gui" not in sys.modules


def test_capabilities_and_cleanup_without_widget() -> None:
    from src.tools.spine_viewer._embed_adapter import SpineViewerEmbedAdapter

    adapter = SpineViewerEmbedAdapter()
    assert adapter.tool_id == "spine_viewer"
    assert adapter.embed_capabilities().supports_embedded is True
    assert adapter.is_dirty() is False
    adapter.cleanup()
    adapter.cleanup()


def test_create_main_widget_and_idempotent_cleanup() -> None:
    try:
        import matplotlib  # noqa: F401
        from PyQt6.QtWidgets import QApplication
    except (ImportError, OSError) as exc:
        pytest.skip(f"PyQt6/matplotlib not loadable: {exc}")
    from src.tools.spine_viewer._embed_adapter import SpineViewerEmbedAdapter

    _app = QApplication.instance() or QApplication([])
    adapter = SpineViewerEmbedAdapter()
    widget = adapter.create_main_widget(None)
    assert widget is not None
    adapter.cleanup()
    adapter.cleanup()


def test_registration_surfaces_agree() -> None:
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[3]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert 'spine_viewer = "src.tools.spine_viewer._embed_adapter"' in pyproject
    models = yaml.safe_load((root / "src/config/models.yaml").read_text("utf-8"))
    tiles = [m for m in models["models"] if m.get("id") == "spine_viewer"]
    assert len(tiles) == 1
    assert tiles[0]["launcher"]["category"] == "tool"
    assert tiles[0]["launcher"]["status"] == "beta"
